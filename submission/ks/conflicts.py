"""Contradictions and which value wins.

Two kinds: table facts printed differently on two pages or changed between crawls,
and press-release boilerplate that drifts as articles age ("...has 23 branch
office locations", "the fourth time Forbes recognized Dupaco").

Who wins, in order (see NOTES.md):
  1. for "now" only the newest snapshot counts, older ones are history
  2. rate sheet > product page > info page > news > blog/promo > calculator;
     expired offers never win
  3. same kind of page: the later "Rates are effective as of" date wins
  4. boilerplate in articles: the most recently published article wins
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from .pages import KIND_RANK, url_path
from .store import FactObs, Store
from .tables import proper, tier_text
from .textutil import fmt_date

PRIMARY_METRIC = {"APY": 0, "APR": 0, "rate": 1, "dividend rate": 1}


def obs_rank(o: FactObs) -> tuple:
    path = url_path(o.url)
    return (
        KIND_RANK.get(o.kind, 0),
        o.fact.effective or date.min,
        path in ("/rates", "/business-rates"),
        -len(path),
    )


@dataclass
class ResolvedFact:
    key: tuple
    value: str
    obs: FactObs
    snap: int


@dataclass
class ConflictRec:
    topic: str
    kind: str  # "stale_page" | "changed_over_time" | "narrative" | "sibling"
    values: list[str]
    sources: list[tuple[str, int, str]]  # (url, snap, quote)
    resolution: str | None
    reason: str
    importance: float = 1.0
    product: str = ""
    label: str = ""


_ORD = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
NARRATIVE_PATTERNS = [
    (
        "number of branch office locations",
        ("branch", "branches", "locations", "offices"),
        re.compile(r"(\d{1,3}) branch office locations", re.I),
        "has {} branch office locations",
    ),
    (
        "membership size (number of members)",
        ("members", "membership", "member-owners"),
        re.compile(r"membership has grown to (?:more than|over|almost|nearly|about) ([\d,]{5,})", re.I),
        "has more than {} members",
    ),
    (
        "total assets",
        ("assets", "asset", "billion"),
        re.compile(r"assets (?:exceeding|of more than|of nearly|of almost|of) \$([\d.]+) billion", re.I),
        "has assets exceeding ${} billion",
    ),
    (
        "number of employees",
        ("employees", "employee", "staff"),
        re.compile(r"\bIt has (?:over|more than|nearly|almost|about) (\d{3,4}) employees", re.I),
        "has over {} employees",
    ),
    (
        "number of counties served",
        ("counties", "county"),
        re.compile(r"serves residents in (\d{2,3}) counties", re.I),
        "serves residents in {} counties",
    ),
    (
        "Forbes recognition count",
        ("forbes",),
        re.compile(
            r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) time Forbes "
            r"(?:has )?(?:recognized|named|ranked)",
            re.I,
        ),
        "the {} time Forbes recognized Dupaco",
    ),
]


@dataclass
class NarrativeObs:
    topic: str
    value: str
    sort_value: float
    statement: str
    published: date
    url: str
    cid: int


def narrative_observations(store: Store, snap: int) -> list[NarrativeObs]:
    out = []
    for ch in store.chunks:
        if snap not in ch.snaps or ch.kind not in ("news", "blog") or not ch.published:
            continue
        for topic, _kw, rx, tmpl in NARRATIVE_PATTERNS:
            for m in rx.finditer(ch.text):
                raw = m.group(1)
                if topic.startswith("Forbes"):
                    sv, val = float(_ORD[raw.lower()]), raw.lower()
                else:
                    sv, val = float(raw.replace(",", "")), raw
                out.append(NarrativeObs(topic, val, sv, tmpl.format(val), ch.published, ch.url, ch.cid))
    return out


def latest_narrative(store: Store, snap: int) -> dict[str, NarrativeObs]:
    """Per topic, what the most recently published article says (rule 4)."""
    best: dict[str, NarrativeObs] = {}
    for o in narrative_observations(store, snap):
        cur = best.get(o.topic)
        if cur is None or (o.published, o.sort_value) > (cur.published, cur.sort_value):
            best[o.topic] = o
    return best


def _short(url: str) -> str:
    return url_path(url)


def _combo(by_metric: dict[str, FactObs], primary: str) -> str:
    if primary == "APR" and "rate" in by_metric and "APR" in by_metric:
        return f"{by_metric['rate'].fact.value} rate / {by_metric['APR'].fact.value} APR"
    o = by_metric.get(primary) or next(iter(by_metric.values()))
    return f"{o.fact.value} {o.fact.metric}"


def _join_metrics(metric_vals: dict[str, str], primary: str) -> str:
    if primary == "APY" and "dividend rate" in metric_vals and "APY" in metric_vals:
        return f"{metric_vals['APY']} APY ({metric_vals['dividend rate']} dividend rate)"
    if primary == "APR" and "rate" in metric_vals and "APR" in metric_vals:
        return f"{metric_vals['rate']} rate ({metric_vals['APR']} APR)"
    return f"{metric_vals.get(primary, '')} {primary}".strip()


def _topic(product: str, label: str, metric: str) -> str:
    m = re.search(r"(\d+)\s*-?\s*(month|year)s?", label, re.I)
    term = f"{m.group(1)}-{m.group(2).lower()} " if m else ""
    metric = "rate/APR" if product == "mortgage" else metric
    detail = tier_text(product, label)
    return f"{term}{proper(product)} {metric}" + (f" ({detail})" if detail else "")


class ConflictEngine:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.winners: dict[int, dict[tuple, ResolvedFact]] = {}
        self.stale_chunks: dict[int, set[int]] = defaultdict(set)
        self.stale_notes: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.by_key: dict[tuple, list[FactObs]] = defaultdict(list)
        self._built_for = -1

    def build(self) -> None:
        """Winning value of every fact in every snapshot (rules 2-3). Chunks that print a
        losing value get marked stale for that snapshot."""
        st = self.store
        if self._built_for == len(st.snapshots):
            return
        self.winners, self.by_key = {}, defaultdict(list)
        self.stale_chunks = defaultdict(set)
        self.stale_notes = defaultdict(lambda: defaultdict(list))
        for o in st.fact_obs:
            self.by_key[o.fact.key].append(o)
        for key, obs in self.by_key.items():
            per_snap: dict[int, list[FactObs]] = defaultdict(list)
            for o in obs:
                per_snap[o.snap].append(o)
            for s, lst in per_snap.items():
                win = max(lst, key=obs_rank)
                self.winners.setdefault(s, {})[key] = ResolvedFact(key, win.fact.value, win, s)
                for o in lst:
                    if o.fact.value != win.fact.value and o.url != win.url:
                        self.stale_chunks[s].add(o.cid)
                        note = (
                            f"OUTDATED: this page shows {o.fact.label} {o.fact.metric} = {o.fact.value}; "
                            f"the official {_short(win.url)} page says {win.fact.value}"
                        )
                        if note not in self.stale_notes[s][o.cid]:
                            self.stale_notes[s][o.cid].append(note)
        self._built_for = len(st.snapshots)

    def _table_candidates(self) -> list[ConflictRec]:
        self.build()
        st = self.store
        groups: dict[tuple, list[tuple]] = defaultdict(list)
        for key in self.by_key:
            groups[(key[0], key[1])].append(key)
        out = []
        for (product, _tier), keys in groups.items():
            keys.sort(key=lambda k: (PRIMARY_METRIC.get(k[2], 2), k[2]))
            primary = keys[0]
            obs = self.by_key[primary]
            snaps = sorted({o.snap for o in obs})
            win = {s: self.winners[s][primary] for s in snaps}
            history = []
            for s in snaps:
                if not history or history[-1][0] != win[s].value:
                    history.append((win[s].value, s))
            final = win[snaps[-1]]
            stale_now = [o for o in obs if o.snap == final.snap and o.fact.value != final.value]
            stale_any = [o for o in obs if o.fact.value != win[o.snap].value]
            changed = len(history) > 1
            if not changed and not stale_any:
                continue
            if KIND_RANK.get(final.obs.kind, 0) < KIND_RANK["info"]:
                # only weak pages (calculators, promos) still print it; the product
                # itself is gone from the rate sheet, so it's not a live conflict
                continue
            # one value per distinct rate/APR combination, oldest first
            by_page: dict[tuple, dict[str, FactObs]] = defaultdict(dict)
            for k in keys:
                for o in self.by_key[k]:
                    by_page[(o.snap, o.url)][k[2]] = o
            values, sources, seen = [], [], set()
            for (s, url), bm in sorted(
                by_page.items(),
                key=lambda kv: (kv[0][0], -obs_rank(kv[1].get(primary[2]) or next(iter(kv[1].values())))[0]),
            ):
                if primary[2] not in bm:
                    continue
                v = _combo(bm, primary[2])
                if v in seen:
                    continue
                seen.add(v)
                values.append(v)
                o = bm[primary[2]]
                sources.append((url, s, f"{o.fact.caption or product}, {o.fact.label}: {v}"))
            metric_vals = {
                k[2]: self.winners[final.snap][k].value for k in keys if k in self.winners.get(final.snap, {})
            }
            resolution = _join_metrics(metric_vals, primary[2])
            reasons = []
            if stale_now:
                pages = sorted({_short(o.url) for o in stale_now})
                old_values = ", ".join(sorted({o.fact.value for o in stale_now}))
                reasons.append(
                    f"{', '.join(pages)} still print(s) {old_values}"
                    f" ({final.obs.fact.label}); the {_short(final.obs.url)} page is authoritative"
                    + (
                        f" with rates effective {fmt_date(final.obs.fact.effective)}"
                        if final.obs.fact.effective
                        else ""
                    )
                )
            if changed:
                hist = " -> ".join(f"{v} ({st.snapshots[s].taken_on.isoformat()})" for v, s in history)
                reasons.append(
                    f"the official value changed between snapshots: {hist}; the newest snapshot is current"
                )
            kind = "stale_page" if stale_now else ("changed_over_time" if changed else "stale_page")
            importance = 1.0 + (1.0 if stale_now else 0.0) + (0.5 if changed else 0.0)
            out.append(
                ConflictRec(
                    _topic(product, final.obs.fact.label, primary[2]),
                    kind,
                    values,
                    sources,
                    resolution,
                    "; ".join(reasons),
                    importance,
                    product,
                    final.obs.fact.label,
                )
            )
        return out

    def table_conflicts(self) -> list[ConflictRec]:
        recs = self._table_candidates()
        recs = _merge_mortgage_purpose(recs)
        recs = _merge_duplicates(recs)
        recs += self._sibling_conflicts()
        return recs

    def _sibling_conflicts(self) -> list[ConflictRec]:
        """Roth and Traditional versions of an IRA special are priced the same
        everywhere else, so a row that differs in the newest snapshot is suspicious."""
        self.build()
        latest = self.store.latest
        wins = self.winners.get(latest, {})
        pairs = defaultdict(dict)
        for key, rf in wins.items():
            product, tier, metric = key
            if not product.startswith("ira") or metric != "APY":
                continue
            m = re.match(r"(roth|traditional):\s*(.*)", tier)
            if m:
                pairs[(product, m.group(2))][m.group(1)] = rf
        out = []
        for (product, rest), d in pairs.items():
            if len(d) == 2 and d["roth"].value != d["traditional"].value:
                # break the tie with the matching certificate special
                cert_vals = [
                    rf.value
                    for k, rf in wins.items()
                    if k[0] == "certificate special"
                    and k[2] == "APY"
                    and rest.replace(" special", "") in k[1]
                ]
                vals = [d["roth"].value, d["traditional"].value]
                resolution = next((v for v in vals if v in cert_vals), None)
                reason = (
                    f"on {_short(d['roth'].obs.url)} the Roth row says {d['roth'].value} but the "
                    f"Traditional row says {d['traditional'].value} for the same {rest} special"
                )
                if resolution:
                    reason += (
                        f"; the matching certificate special is {resolution}, so the other row looks stale"
                    )
                out.append(
                    ConflictRec(
                        f"{rest.title()} IRA APY (Roth vs Traditional)",
                        "sibling",
                        vals,
                        [
                            (d["roth"].obs.url, latest, f"Roth {rest}: {vals[0]}"),
                            (d["traditional"].obs.url, latest, f"Traditional {rest}: {vals[1]}"),
                        ],
                        resolution,
                        reason,
                        1.8,
                        product,
                        rest,
                    )
                )
        return out

    def narrative_conflicts(self) -> list[ConflictRec]:
        st = self.store
        latest = st.latest
        by_topic: dict[str, list[NarrativeObs]] = defaultdict(list)
        for o in narrative_observations(st, latest):
            by_topic[o.topic].append(o)
        out = []
        for topic, lst in by_topic.items():
            if len({o.value for o in lst}) < 2:
                continue
            best = max(lst, key=lambda o: (o.published, o.sort_value))
            values, sources, seen = [], [], set()
            for o in sorted(lst, key=lambda o: o.published):
                if o.value in seen:
                    continue
                seen.add(o.value)
                values.append(o.value)
                sources.append((o.url, latest, f"{o.statement} (published {fmt_date(o.published)})"))
            # newest value last
            values = [v for v in values if v != best.value] + [best.value]
            reason = (
                f"press releases published at different times disagree; the most recent one "
                f"({_short(best.url)}, {fmt_date(best.published)}) says '{best.statement}'"
            )
            out.append(
                ConflictRec(topic, "narrative", values, sources, best.value, reason, 2.2, "narrative", topic)
            )
        return out

    def all_conflicts(self) -> list[ConflictRec]:
        recs = self.table_conflicts() + self.narrative_conflicts()
        recs.sort(key=lambda r: (-r.importance, r.topic))
        return recs


def _merge_mortgage_purpose(recs: list[ConflictRec]) -> list[ConflictRec]:
    """Purchase and refinance rows for the same term/amount move together; report once."""
    out, groups = [], defaultdict(list)
    for r in recs:
        if r.product == "mortgage":
            base = re.sub(r"^(purchase|refinance)\s*:\s*", "", r.label, flags=re.I)
            base = re.sub(r"(?<=\d),(?=\d{3})", "", base)
            base = re.sub(r"\$(\d{4,})", lambda m: f"${int(m.group(1)):,}", base)  # same key, readable
            groups[base].append(r)
        else:
            out.append(r)
    for base, rs in groups.items():
        rs.sort(key=lambda r: "purchase" not in r.label.lower())
        if len(rs) == 1:
            out.append(rs[0])
            continue
        values, sources, seen = [], [], set()
        for r in rs:
            for v, s in zip(r.values, r.sources):
                if v not in seen:
                    seen.add(v)
                    values.append(v)
                    sources.append(s)
        res = "; ".join(
            f"{'Purchase' if 'purchase' in r.label.lower() else 'Refinance'}: {r.resolution}" for r in rs
        )
        reason = " | ".join(dict.fromkeys(r.reason for r in rs))
        out.append(
            ConflictRec(
                _topic("mortgage", f"{base}, purchase & refinance", "APR"),
                rs[0].kind,
                values,
                sources,
                res,
                reason,
                max(r.importance for r in rs),
                "mortgage",
                base,
            )
        )
    return out


def _variant_rank(r: ConflictRec) -> tuple:
    p = r.product
    return ("business" in p, p.startswith("ira"), "traditional" in r.label.lower(), len(p))


def _merge_duplicates(recs: list[ConflictRec]) -> list[ConflictRec]:
    """Identical value histories on the personal, business and IRA versions of a product
    are one contradiction: keep the personal one and mention the rest."""
    groups = defaultdict(list)
    for r in recs:
        groups[(tuple(r.values), r.resolution)].append(r)
    out = []
    for rs in groups.values():
        rs.sort(key=_variant_rank)
        head = rs[0]
        if len(rs) > 1:
            others = sorted({f"{r.product} ({r.label})" for r in rs[1:]})
            head.reason += f"; the same values appear for: {', '.join(others)}"
            head.importance += 0.2
        out.append(head)
    return out

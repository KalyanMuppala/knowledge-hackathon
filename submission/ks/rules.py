"""Answers for the common question shapes: rate-table lookups (with tiers, history
and minimums), counts from press releases, routing number, hours, addresses, links,
and refusals for products the site doesn't price."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .conflicts import ConflictEngine, ResolvedFact, latest_narrative
from .pages import KIND_RANK, url_path
from .query import QueryInfo
from .store import Store
from .tables import proper, tier_text

# question wording -> product families from ks.tables
PRODUCT_PATTERNS = [
    ("auto loan", r"\b(auto|car|cars|vehicle|truck|suv|motorcycle)\b"),
    (
        "mortgage",
        r"\b(mortgages?|home loans?|home purchase|refinanc\w*|\d\d[- ]?(year|yr)s?( fixed)?|fixed[- ]rate)\b",
    ),
    ("heloc", r"\b(heloc|home equity line)\b"),
    ("home equity loan", r"\bhome equity loan\b"),
    ("home improvement loan", r"\bhome improvement\b"),
    ("personal loan", r"\bpersonal loans?\b"),
    ("boat & rv loan", r"\b(boat|rv|camper|motorhome)s?\b"),
    ("atv, snowmobile & jet ski loan", r"\b(atv|snowmobile|jet ?ski|utv)s?\b"),
    ("certificate special", r"\bspecials?\b"),
    ("certificate", r"\b(certificates?|cds?|share certificates?|term share)\b"),
    ("ira special", r"\b(ira|roth|traditional)\b.*\bspecial\b|\bspecial\b.*\b(ira|roth)\b"),
    ("ira", r"\b(ira|iras|roth|traditional)\b"),
    ("ira variable", r"\bvariable iras?\b"),
    (
        "share savings",
        r"\b(share savings|savings accounts?|regular savings|primary savings|savings tiers?)\b",
    ),
    ("bright savings", r"\b(bright savings|online savings|online-only savings)\b"),
    ("you name it savings", r"\byou[- ]name[- ]it\b"),
    ("holiday club savings", r"\b(holiday club|christmas club)\b"),
    ("launchpad savings", r"\blaunchpad\b"),
    ("investor's choice money market", r"\b(investor'?s choice|money markets?)\b"),
    ("achieve money market", r"\bachieve\b"),
    ("high-yield savings", r"\bhigh[- ]yield\b"),
    ("health savings account", r"\b(hsa|health savings)\b"),
    ("1st rate checking", r"\b(1st rate|first rate|checking)\b"),
    ("free checking", r"\b(free checking|checking)\b"),
    ("platinum visa", r"\b(platinum|credit cards?|visa)\b"),
    ("rewards visa", r"\b(rewards|credit cards?|visa)\b"),
]
# products the site describes but never prices: a rate question about them gets "not listed"
UNPRICED = re.compile(
    r"\b(adjustable|arm|\d/\d|construction|land loan|lot loan|fha|va loan|usda|"
    r"commercial|equipment|business loans?|business line of credit|student loans?|"
    r"reverse mortgage|jumbo mortgage|bridge loan)\b",
    re.I,
)


@dataclass
class RateMatch:
    score: float
    rf: ResolvedFact
    specific: bool = False  # matched a tier the member named (term, amount, year...)

    @property
    def fact(self):
        return self.rf.obs.fact


_NUM_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "ten": 10,
    "twelve": 12,
    "fifteen": 15,
    "eighteen": 18,
    "twenty": 20,
    "thirty": 30,
}


def _months(q: str) -> set[int]:
    out = set()
    low = re.sub(r"\b(" + "|".join(_NUM_WORDS) + r")\b", lambda m: str(_NUM_WORDS[m.group(1)]), q.lower())
    for m in re.finditer(r"(\d+)[- ]?(month|mo\b|year|yr)", low):
        n = int(m.group(1))
        out.add(n * 12 if m.group(2).startswith("y") else n)
    return out


def _amounts(q: str) -> list[float]:
    out = []
    for m in re.finditer(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|million)?", q.lower()):
        v = float(m.group(1).replace(",", ""))
        if m.group(2) in ("k", "thousand"):
            v *= 1000
        elif m.group(2) == "million":
            v *= 1_000_000
        out.append(v)
    return out


def _label_range(label: str) -> tuple[float, float] | None:
    """'$10,000 - $24,999.99' -> (10000, 24999.99); '$25,000+' -> (25000, inf);
    '$100,000 or less' -> (0, 100000); '$10,000-24,999' -> (10000, 24999)."""
    lab = label.replace(",", "")
    m = re.search(r"\$(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)", lab)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"\$(\d+(?:\.\d+)?)\s*\+", lab)
    if m:
        return float(m.group(1)), float("inf")
    m = re.search(r"\$(\d+(?:\.\d+)?)\s*or less", lab)
    if m:
        return 0.0, float(m.group(1))
    return None


def _label_years(label: str) -> tuple[int, int] | None:
    m = re.search(r"((?:19|20)\d{2})\s*-\s*(\d{2,4})", label)
    if m and "model year" in label.lower():
        a, b = int(m.group(1)), int(m.group(2))
        b = b + 2000 if b < 100 else b
        return min(a, b), max(a, b)
    m = re.search(r"((?:19|20)\d{2}) model year", label.lower())
    if m:
        y = int(m.group(1))
        return (1900, y) if "older" in label.lower() else (y, y)
    return None


def _label_months(label: str) -> set[int]:
    out = set()
    for m in re.finditer(r"(\d+)[- ]?(month|mo\b|year)", label.lower()):
        n = int(m.group(1))
        out.add(n * 12 if m.group(2) == "year" else n)
    return out


def detect_products(q: str) -> dict[str, float]:
    low = q.lower()
    found = {}
    for product, rx in PRODUCT_PATTERNS:
        if re.search(rx, low):
            found[product] = found.get(product, 0) + 1.0
    # a more specific product beats its generic cousin
    if "certificate special" in found and not re.search(r"\b(certificate|cd|special)", low):
        found.pop("certificate special")
    if "certificate special" in found and re.search(r"\bspecials?\b", low):
        found.pop("certificate", None)  # they said "special": regular terms don't qualify
    if "ira special" in found:
        found.pop("ira", None)
        found.pop("certificate special", None)
    if "ira" in found:
        found.pop("certificate", None)
    for specific in (
        "bright savings",
        "you name it savings",
        "holiday club savings",
        "launchpad savings",
        "health savings account",
        "high-yield savings",
        "achieve money market",
    ):
        if specific in found:
            found.pop("share savings", None)
    if "achieve money market" in found:
        found.pop("investor's choice money market", None)
    if "free checking" in found and "free" in low:
        found.pop("1st rate checking", None)
    if "1st rate checking" in found and re.search(r"1st|first", low):
        found.pop("free checking", None)
    if re.search(r"\bplatinum\b", low) and not re.search(r"\brewards?\b", low):
        found.pop("rewards visa", None)
    if re.search(r"\brewards?\b", low) and not re.search(r"\bplatinum\b", low):
        found.pop("platinum visa", None)
    if "home equity loan" in found or "heloc" in found or "home improvement loan" in found:
        found.pop("mortgage", None)
    return found


def _sort_key(c: RateMatch) -> tuple:
    f, product = c.fact, c.rf.key[0]
    months = min(_label_months(f.label), default=0)
    rng = _label_range(f.label) or (0.0, 0.0)
    if product == "mortgage":
        return (-months, rng[1], f.label)  # 30-year first, smallest loan tier first
    if "visa" in product:
        return ("business" in product, "rewards" in product, f.label)
    return (product, months, -rng[0], f.label)


def match_rates(
    store: Store, engine: ConflictEngine, qi: QueryInfo, snap: int, limit: int = 16
) -> tuple[list[RateMatch], str]:
    """Rows from the official tables that answer the question, plus a mode:
    'specific' (one tier), 'max' / 'min', or 'list' (general question)."""
    engine.build()
    q = qi.question
    low = q.lower()
    products = detect_products(q)
    if not products:
        return [], "none"
    if "certificate" in products and not re.search(r"\b(special|jumbo|regular)\b", low):
        products.setdefault("certificate special", 0.9)  # "certificate rates" covers the specials too
    cards_only = all("visa" in p for p in products)
    business = "business" in low
    months, amounts = _months(q), _amounts(q)
    years = [int(y) for y in re.findall(r"\b(19[89]\d|20[0-3]\d)\b", q)]
    if not years and re.search(r"\b(new|brand[- ]new)\s+(car|vehicle|truck|suv|auto)", low):
        years = [store.snapshots[snap].taken_on.year]  # a new car is the newest model year
    no_jumbo = bool(re.search(r"\bnon-?jumbo\b|\bnot (a )?jumbo\b", low))
    wants = {
        "jumbo": "jumbo" in low and not no_jumbo,
        "roth": "roth" in low,
        "traditional": "traditional" in low,
        "refi": bool(re.search(r"refinanc", low)),
        "intro": bool(re.search(r"\b(intro|introductory|balance transfer)", low)),
    }
    metric = None
    if "minimum" in qi.intents:
        metric = "minimum to earn" if re.search(r"\bearn", low) else "minimum to open"
    elif re.search(r"\bapy\b|\byield\b", low):
        metric = "APY"
    elif re.search(r"\bdividend\b", low):
        metric = "dividend rate"
    elif re.search(r"\bapr\b", low):
        metric = "APR"
    mode = "list"
    if re.search(r"\d\s?%", q) and re.match(r"\s*(do|does|is|are|can)\b", low):
        mode = "check"  # "do any certificates pay 4.20%?"
    elif re.search(r"\b(highest|best|top|maximum|max|most)\b", low):
        mode = "max"
    elif re.search(r"\b(lowest|least|cheapest)\b", low) or (
        re.search(r"\bminimum\b", low) and "minimum" not in qi.intents
    ):
        mode = "min"
    cands: list[RateMatch] = []
    for key, rf in engine.winners.get(snap, {}).items():
        product, tier, m = key
        if KIND_RANK.get(rf.obs.kind, 0) < KIND_RANK["info"]:
            continue  # only official / product / info pages answer rates
        base = product.replace("business ", "").replace(" (business)", "")
        if base not in products and product not in products:
            continue
        is_biz = "business" in product
        general_cards = cards_only and not re.search(r"\b(personal|platinum|rewards)\b", low)
        if is_biz != business and not (general_cards and is_biz):
            continue
        f = rf.obs.fact
        lab = f.label.lower()
        s = products.get(product, products.get(base, 0))
        specific = False
        lm = _label_months(f.label)
        if months:
            if lm & months:
                s, specific = s + 3, True
            elif lm:
                s -= 3
        rng = _label_range(f.label)
        if amounts and rng:
            if any(rng[0] <= a <= rng[1] or rng[0] <= a + 0.99 <= rng[1] for a in amounts):
                s, specific = s + 3, True
            else:
                s -= 2
        yrs = _label_years(f.label)
        if years and yrs:
            if any(yrs[0] <= y <= yrs[1] for y in years):
                s, specific = s + 3, True
            else:
                s -= 3
        for flag in ("jumbo", "roth", "traditional"):
            if flag in lab and not (flag == "jumbo" and mode in ("max", "min", "check")):
                s += 1.5 if wants[flag] else -1.5
                specific = specific or wants[flag]
        if product == "mortgage":
            s += 1 if (("refinance" in lab) == wants["refi"]) else -2
        if "visa" in product:
            intro = "introductory" in lab or "intro" in lab
            s += 2 if intro == wants["intro"] else -2
            specific = specific or (wants["intro"] and intro)
        if metric:
            if m == metric:
                s += 1
            elif metric.startswith("minimum") and m.startswith("minimum"):
                s += 0.5  # certificates only print "Min. to Earn"
            elif metric.startswith("minimum") or m.startswith("minimum"):
                continue
            elif not (metric == "APR" and m == "rate" and product == "mortgage"):
                s -= 1.5
        elif m.startswith("minimum"):
            continue
        elif m == "dividend rate" or (m == "APR" and product == "mortgage"):
            s -= 0.5  # shown alongside the APY / rate anyway
        cands.append(RateMatch(s, rf, specific))
    if not cands:
        return [], "none"
    if amounts:
        # nested brackets ("$100,000 or less", "$200,000 or less"...): the tightest one applies
        best_upper: dict = {}
        for c in cands:
            rng = _label_range(c.fact.label)
            if c.specific and rng:
                k = (c.rf.key[0], c.rf.key[2], tuple(sorted(_label_months(c.fact.label))))
                best_upper[k] = min(best_upper.get(k, float("inf")), rng[1])
        for c in cands:
            rng = _label_range(c.fact.label)
            k = (c.rf.key[0], c.rf.key[2], tuple(sorted(_label_months(c.fact.label))))
            if c.specific and rng and rng[1] == best_upper.get(k):
                c.score += 0.5
    if (
        months
        and not any(_label_months(c.fact.label) & months for c in cands)
        and any(_label_months(c.fact.label) for c in cands)
    ):
        return [], "no_such_term"  # e.g. a 72-month certificate the site doesn't offer
    cands.sort(key=lambda c: (-c.score, c.fact.label))
    top = cands[0]
    if mode in ("max", "min"):
        pool = [c for c in cands if c.score >= top.score - 1.5 and c.rf.key[2] == top.rf.key[2]]
        pick = max(pool, key=lambda c: c.fact.num) if mode == "max" else min(pool, key=lambda c: c.fact.num)
        return [pick], mode
    if mode == "check":
        return [c for c in cands if c.score >= top.score - 1.5], mode
    if top.specific and top.score >= 3.5:
        ties = [c for c in cands if c.specific and c.score >= top.score - 0.01]
        if len({(c.rf.key[0], c.rf.key[1]) for c in ties}) == 1:
            return [top], "specific"
        ties.sort(key=_sort_key)  # e.g. every loan-amount tier of a 30-year
        return ties[:limit], "list"
    best: dict[tuple, RateMatch] = {}
    for c in cands:
        if c.score >= top.score - 1.0:
            best.setdefault((c.rf.key[0], c.rf.key[1]), c)  # cands are sorted best first
    keep = sorted(best.values(), key=_sort_key)
    return keep[:limit], "list"


def _metric_pair(engine: ConflictEngine, snap: int, rf: ResolvedFact) -> str:
    """'7.13% rate (7.27% APR)' / '4.10% APY (4.06% dividend rate)' / '5.39% APR'."""
    product, tier, metric = rf.key
    if metric.startswith("minimum"):
        return f"{rf.value} {metric}"
    wins = engine.winners.get(snap, {})
    other = {"APY": "dividend rate", "dividend rate": "APY", "rate": "APR", "APR": "rate"}.get(metric)
    o = wins.get((product, tier, other)) if other else None
    if metric in ("rate", "APY"):
        return f"{rf.value} {metric}" + (f" ({o.value} {other})" if o else "")
    if o and other in ("rate", "APY"):
        return f"{o.value} {other} ({rf.value} {metric})"
    return f"{rf.value} {metric}"


_MORTGAGE = re.compile(r"(purchase|refinance):\s*(\d+)-year fixed conventional\s*(\$[\d,]+ or less)?", re.I)


def display_name(product: str, label: str) -> str:
    """Short name for a row in a list answer."""
    low = label.lower()
    m = _MORTGAGE.search(label)
    if m:
        amt = f" ({m.group(3)})" if m.group(3) else ""
        return f"{m.group(2)}-year fixed {m.group(1).lower()} mortgage{amt}"
    months = min(_label_months(label), default=0)
    if "visa" in product:
        return f"{proper(product)} intro balance transfer" if "intro" in low else proper(product)
    if product in ("heloc", "home equity loan"):
        return f"{months}-month home equity loan" if months else proper(product)
    if product.endswith("loan"):
        rest = re.sub(r",?\s*\d+ months?", "", label, flags=re.I).strip(" ,")
        name = f"{months}-month {product}" if months else product
        return f"{name} ({rest})" if rest else name
    if product.startswith(("certificate", "business certificate")) and re.fullmatch(
        r"\d+ months?", label, re.I
    ):
        return f"{months}-month {'business ' if 'business' in product else ''}certificate"
    if product.startswith("ira"):
        m = re.match(r"(roth|traditional):\s*(.*)", label, re.I)
        if m:
            kind, rest = m.group(1).title(), m.group(2)
            if re.fullmatch(r"\d+ months?", rest, re.I):
                return f"{months}-month {kind} IRA"
            return f"{kind} IRA {rest}" if "$" not in rest else f"{kind} variable IRA ({rest})"
        return label
    tier = tier_text(product, label)
    return (
        f"{proper(product)} ({tier})" if tier != label else (label if "special" in low else proper(product))
    )


def _pair_parts(pair: str) -> tuple[str, str]:
    """'7.13% rate (7.27% APR)' -> ('7.13%', '7.27% APR')."""
    m = re.match(r"(\S+) \w[\w ]* \((.*)\)$", pair)
    return (m.group(1), m.group(2)) if m else (pair, "")


def phrase_one(engine: ConflictEngine, snap: int, c: RateMatch, single_tier: bool) -> str:
    """One row as a sentence a member would expect to hear."""
    product, tier, metric = c.rf.key
    label = c.fact.label
    val = _metric_pair(engine, snap, c.rf)
    months = min(_label_months(label), default=0)
    term = f"{months}-month " if months else ""
    if metric.startswith("minimum"):
        return _minimum_sentence(metric, product, c.rf.value)
    m = _MORTGAGE.search(label)
    if m:
        rate, apr = _pair_parts(val)
        amt = f" for loans of {m.group(3)}" if m.group(3) else ""
        return f"The {m.group(2)}-year fixed {m.group(1).lower()} mortgage rate{amt} is {rate}" + (
            f" ({apr})." if apr else "."
        )
    yrs = _label_years(label)
    if yrs and "model year" in label.lower():
        span = re.search(r"((?:19|20)\d{2}\s*-\s*\d{2,4}|(?:19|20)\d{2})", label).group(1).replace(" ", "")
        older = " and older" if "older" in label.lower() else ""
        return f"The {term}{product} rate for {span}{older} model-year vehicles is as low as {val}."
    if "visa" in product:
        if "intro" in label.lower():
            return f"The {proper(product)} introductory balance transfer rate is {val}."
        return f"The {proper(product)} rate is {val}."
    if product in ("heloc", "home equity loan"):
        return f"The {display_name(product, label)} rate is as low as {val}."
    if product.endswith("loan"):
        rest = re.sub(r",?\s*\d+ months?", "", label, flags=re.I).strip(" ,")
        amount = f" for {rest}" if rest and "$" in rest else ""
        return f"The {term}{product} rate{amount} is as low as {val}."
    if product.startswith(("certificate", "business certificate", "ira")):
        return f"The {display_name(product, label)} pays {val}."
    tier = tier_text(product, label)
    if single_tier or tier == label:
        return f"{proper(product)} pays {val}."
    return f"{proper(product)} balances of {tier} earn {val}."


def _minimum_sentence(metric: str, product: str, value: str) -> str:
    name = proper(product)
    noun = name if re.search(r"certificate|account|ira", name, re.I) else f"{name} account"
    if metric == "minimum to earn":
        return f"The minimum balance to earn dividends on a {noun} is {value}."
    return f"The minimum to open a {noun} is {value}."


def describe_rates(engine: ConflictEngine, snap: int, matches: list[RateMatch], mode: str) -> str:
    if not matches:
        return ""
    rows, seen = [], set()
    for c in matches:
        if (c.rf.key[0], c.rf.key[1]) not in seen:
            seen.add((c.rf.key[0], c.rf.key[1]))
            rows.append(c)
    if all(c.rf.key[2].startswith("minimum") for c in rows) and len({c.rf.value for c in rows}) == 1:
        c = rows[0]  # same minimum for every tier
        return _minimum_sentence(c.rf.key[2], c.rf.key[0], c.rf.value)
    if len(rows) == 1:
        product = rows[0].rf.key[0]
        tiers = {k[1] for k in engine.winners.get(snap, {}) if k[0] == product}
        return phrase_one(engine, snap, rows[0], single_tier=len(tiers) <= 1)
    return (
        "; ".join(
            f"{display_name(c.rf.key[0], c.fact.label)}: {_metric_pair(engine, snap, c.rf)}" for c in rows
        )
        + "."
    )


def value_check(engine: ConflictEngine, snap: int, matches: list[RateMatch], qi: QueryInfo) -> str | None:
    """'Do any savings tiers pay 0.35% APY?' -> yes with the rows, or no with the top rate."""
    asked = re.findall(r"(\d{1,2}(?:\.\d+)?)\s?%", qi.question)
    if not asked or not re.match(r"\s*(do|does|is|are|can)\b", qi.question.lower()):
        return None
    want = {float(a) for a in asked}
    rows = [c for c in matches if not c.rf.key[2].startswith(("dividend", "minimum"))]
    if not rows:
        return None
    hits = [c for c in rows if c.fact.num in want]
    if hits:
        return (
            "Yes: "
            + "; ".join(
                f"{display_name(c.rf.key[0], c.fact.label)}: {_metric_pair(engine, snap, c.rf)}" for c in hits
            )
            + "."
        )
    top = max(rows, key=lambda c: c.fact.num)
    tier = tier_text(top.rf.key[0], top.fact.label)
    where = f" ({tier})" if tier != top.fact.label else ""
    return (
        f"No. The highest is {display_name(top.rf.key[0], top.fact.label).split(' (')[0]}{where} "
        f"at {top.rf.value} {top.rf.key[2]}."
    )


_CATEGORIES = {
    "savings": (
        "share savings",
        "bright savings",
        "you name it savings",
        "holiday club savings",
        "launchpad savings",
        "investor's choice money market",
        "achieve money market",
        "high-yield savings",
        "health savings account",
    ),
    "checking": ("free checking", "1st rate checking"),
    "credit cards": ("platinum visa", "rewards visa", "business platinum visa", "business rewards visa"),
    "loans": (
        "auto loan",
        "mortgage",
        "heloc",
        "home equity loan",
        "home improvement loan",
        "personal loan",
        "boat & rv loan",
        "atv, snowmobile & jet ski loan",
    ),
}
_LOAN_NAMES = {
    "auto loan": "auto & motorcycle loans",
    "mortgage": "fixed-rate home loans",
    "heloc": "home equity lines of credit",
    "home equity loan": "home equity loans",
    "home improvement loan": "home improvement loans",
    "personal loan": "personal loans",
    "boat & rv loan": "boat & RV loans",
    "atv, snowmobile & jet ski loan": "ATV, snowmobile & jet ski loans",
}


def product_list(engine: ConflictEngine, snap: int, qi: QueryInfo) -> str | None:
    """'What savings accounts do you offer?': the products that have a rate table right now."""
    from .textutil import tokenize

    low = qi.question.lower()
    cat = next((c for c in _CATEGORIES if re.search(rf"\b{c[:-1] if c != 'savings' else c}", low)), None)
    plain = {
        "type",
        "kind",
        "offer",
        "option",
        "account",
        "product",
        "available",
        "different",
        "card",
        "credit",
        "loan",
        "savings",
        "checking",
        "personal",
        "business",
        "provide",
    }
    if not cat or "list" not in qi.intents or set(tokenize(qi.question)) - plain:
        return None
    present = {k[0] for k in engine.winners.get(snap, {})}
    names = [(_LOAN_NAMES.get(p) or proper(p)) for p in _CATEGORIES[cat] if p in present]
    if not names:
        return None
    joined = ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]
    return f"Dupaco offers {joined}."


def history_answer(
    store: Store, engine: ConflictEngine, matches: list[RateMatch], snap: int, question: str = ""
) -> str | None:
    """For 'did X change / what was it before': the value timeline of the matched rows."""
    from .textutil import fmt_date

    engine.build()

    def when(a: int, b: int) -> str:
        da, db = (fmt_date(store.snapshots[i].taken_on) for i in (a, b))
        if b == snap:
            return f"since {da}"
        return da if a == b else f"{da} to {db}"

    changed, unchanged = [], []
    for c in matches[:4]:
        key = c.rf.key
        segs: list[list] = []
        for s in range(snap + 1):
            rf = engine.winners.get(s, {}).get(key)
            if not rf:
                continue
            if segs and segs[-1][0] == rf.value:
                segs[-1][2] = s
            else:
                segs.append([rf.value, s, s])
        name = display_name(key[0], c.fact.label)
        if len(segs) > 1:
            steps = ", then ".join(f"{v} ({when(a, b)})" for v, a, b in segs)
            changed.append(f"{name} {key[2]}: {steps}")
        elif segs:
            unchanged.append(f"{name}: {segs[0][0]} {key[2]}, unchanged {when(segs[0][1], snap)}")
    yes_no = re.match(r"\s*(has|have|did|does|is|was|were)\b", question.lower())
    if changed:
        return ("Yes. " if yes_no else "") + "; ".join(changed[:2]) + "."
    if unchanged:
        return ("No. " if yes_no else "") + "; ".join(unchanged[:2]) + "."
    return None


def stale_note(engine: ConflictEngine, snap: int, matches: list[RateMatch]) -> str | None:
    """If another page still prints an older value for the answered row, say so (kept out
    of the answer text itself)."""
    engine.build()
    for c in matches[:3]:
        win = c.rf
        for o in engine.by_key.get(c.rf.key, []):
            if o.snap == snap and o.fact.value != win.value:
                return (
                    f"{url_path(o.url)} still shows {o.fact.value}; the official "
                    f"{url_path(win.obs.url)} page ({win.value}) was used."
                )
    return None


ROUTING = re.compile(r"routing number:?\**\s*(\d{9})", re.I)


def routing_number(store: Store, snap: int) -> tuple[str, str] | None:
    votes = Counter()
    where = {}
    for ch in store.chunks:
        if snap in ch.snaps and ch.kind in ("info", "product", "rates"):
            for m in ROUTING.finditer(ch.text):
                votes[m.group(1)] += 1
                where.setdefault(m.group(1), ch.url)
    if not votes:
        return None
    num = votes.most_common(1)[0][0]
    return num, where[num]


PHONE = re.compile(r"phone:?\**\s*\[?(\d{3}-\d{3}-\d{4})", re.I)


def phone_number(store: Store, snap: int) -> tuple[str, str] | None:
    votes, where = Counter(), {}
    for ch in store.chunks:
        if snap in ch.snaps and ch.kind == "info":
            for m in PHONE.finditer(ch.text):
                votes[m.group(1)] += 1
                where.setdefault(m.group(1), ch.url)
    if not votes:
        return None
    num = votes.most_common(1)[0][0]
    return num, where[num]


_HOURS = re.compile(r"\*\*lobby\*\*(.*?)\*\*drive-?up\*\*(.*?)(?:\*\*|$)", re.I | re.S)


def branch_hours(store: Store, snap: int) -> list[tuple[str, str, str]]:
    """(lobby, drive-up, url) schedules printed in branch press releases."""
    out = []
    for ch in store.chunks:
        if snap not in ch.snaps or "lobby" not in ch.text.lower():
            continue
        for m in _HOURS.finditer(ch.text):
            lobby = re.sub(r"\s+", " ", m.group(1)).strip(" .")
            drive = re.sub(r"\s+", " ", m.group(2)).strip(" .")
            drive = re.split(r"\s{2,}|ABOUT|Dupaco Community", drive)[0].strip()
            if lobby and drive:
                out.append((lobby, drive, ch.url))
    return out


def hours_known_for(store: Store, snap: int, names: set[str]) -> bool:
    for ch in store.chunks:
        if snap in ch.snaps and _HOURS.search(ch.text):
            text = (ch.text + " " + ch.url).lower()
            if all(n in text for n in names):
                return True
    return False


_DAYS = r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"


def _schedule(text: str) -> str:
    """'Monday-Thursday: 8:30 a.m.–5 p.m. Friday: ...' -> 'Monday-Thursday 8:30 a.m.-5 p.m., Friday ...'"""
    t = re.sub(r"\s*[–—]\s*", "-", text)
    t = re.sub(_DAYS + r"(-" + _DAYS + r")?:\s*", lambda m: m.group(0).rstrip(": ") + " ", t)
    t = re.sub(r"\s+(?=" + _DAYS + r"\b)", ", ", t)
    return re.sub(r"\bClosed\b", "closed", t).strip(" ,.")


def hours_answer(store: Store, qi: QueryInfo, snap: int) -> str | None:
    sched = branch_hours(store, snap)
    if not sched:
        return None
    lobby = Counter(_schedule(s[0]) for s in sched)
    drive = Counter(_schedule(s[1]) for s in sched)
    low = qi.question.lower()
    if re.search(r"drive[- ]?(up|thr)", low):
        top = drive.most_common(2)
        text = f"Drive-up hours at most branches: {top[0][0]}."
        if len(top) > 1:
            text += f" Some branches differ, e.g. {top[1][0]}."
        return text + " Hours vary by branch, so let me know which one and I can check."
    top = lobby.most_common(3)
    if re.search(r"\b(all|same|every|differ)\b", low):
        return (
            "No, lobby hours vary by branch. Schedules on file include: " + "; ".join(t[0] for t in top) + "."
        )
    text = f"Typical lobby hours: {top[0][0]}."
    if len(top) > 1:
        text += f" Some branches differ, e.g. {top[1][0]}."
    return text + " Hours vary by branch, so let me know which one and I can check."


def narrative_answer(store: Store, qi: QueryInfo, snap: int) -> tuple[str, str] | None:
    low = qi.question.lower()
    latest = latest_narrative(store, snap)
    if "forbes" in low and "Forbes recognition count" in latest:
        o = latest["Forbes recognition count"]
        return (
            f"Forbes has named Dupaco one of the top (Best-in-State) credit unions; the most recent "
            f"release says it was the {o.value} time Forbes recognized Dupaco.",
            o.url,
        )
    if "how many" not in low and "number of" not in low:
        return None
    from .textutil import tokenize

    plain = {
        "many",
        "number",
        "branch",
        "location",
        "office",
        "member",
        "employee",
        "staff",
        "people",
        "work",
        "asset",
        "total",
        "currently",
        "today",
        "now",
        "operate",
        "physical",
        "county",
        "counties",
        "serve",
        "resident",
    }
    if set(tokenize(qi.question)) - plain:
        return None  # "how many employees used VTO" is not a headcount question
    for topic, words in (
        ("number of branch office locations", ("branch", "location", "office")),
        ("number of counties served", ("count",)),
        ("membership size (number of members)", ("member",)),
        ("number of employees", ("employee", "staff", "people work")),
        ("total assets", ("asset",)),
    ):
        if any(w in low for w in words) and topic in latest:
            o = latest[topic]
            return f"Dupaco {o.statement}.", o.url
    return None


_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_LINK_SYNONYMS = {
    "cd": "certificate",
    "cds": "certificate",
    "locator": "location",
    "atm": "location",
    "branch": "location",
    "join": "member",
    "membership": "member",
    "apply": "application",
}


def rates_page_link(store: Store, engine: ConflictEngine, qi: QueryInfo, snap: int) -> str | None:
    """For 'where can I see X rates': the product page that hosts X's rate table, in the
    form other pages link to ("...#rates" if the site uses that anchor)."""
    products = detect_products(qi.question)
    if not products:
        return None
    engine.build()
    hosts = Counter()
    for key, rf in engine.winners.get(snap, {}).items():
        if key[0] in products and rf.obs.kind == "product":
            hosts[rf.obs.url.rstrip("/")] += 1
    for o in store.fact_obs:
        if o.snap == snap and o.fact.product in products and o.kind == "product":
            hosts[o.url.rstrip("/")] += 1
    if not hosts:
        return None
    inbound = Counter()
    for ch in store.chunks:
        if snap not in ch.snaps:
            continue
        for m in _LINK.finditer(ch.text):
            base, _, frag = m.group(2).partition("#")
            if base.rstrip("/") in hosts:
                inbound[(base.rstrip("/"), frag)] += 1
    if not inbound:
        return hosts.most_common(1)[0][0]
    rated = [(n, b) for (b, f), n in inbound.items() if f == "rates"]
    if rated:
        return max(rated)[1] + "#rates"
    return max((n, b) for (b, _f), n in inbound.items())[1]


def link_answer(
    store: Store, qi: QueryInfo, snap: int, chunks: list, engine: ConflictEngine | None = None
) -> tuple[str, str] | None:
    """'What link should I use for X?': score links by how well anchor text and URL match
    the question, favoring links that sit inside the most relevant chunks."""
    from .textutil import tokenize

    if engine is not None and "rate" in qi.intents:
        url = rates_page_link(store, engine, qi, snap)
        if url:
            return url, url.split("#")[0]
    generic = {"link", "url", "page", "website", "online", "use", "see", "dupaco", "find", "go", "do"}
    qt = set(qi.tokens) - generic
    qt |= {_LINK_SYNONYMS[t] for t in list(qt) if t in _LINK_SYNONYMS}
    if re.search(r"\b(join|membership|become a member)\b", qi.question.lower()):
        qt |= {"join", "member"}
    pool = [(rank, ch) for rank, ch in enumerate(chunks[:6])]
    pool += [
        (9, ch)
        for ch in store.chunks
        if snap in ch.snaps and ch.kind in ("info", "product", "rates") and "](http" in ch.text
    ]
    best = None
    for rank, ch in pool:
        for m in _LINK.finditer(ch.text):
            anchor, url = m.group(1), m.group(2)
            if "dupaco.com" not in url:
                continue
            path = re.sub(r"https?://", "", url).replace("/", " ").replace("-", " ").replace("#", " ")
            toks = set(tokenize(f"{anchor} {path}"))
            s = len(qt & toks) + 1.2 / (rank + 1)
            if "rate" in qt:
                if "#rates" in url or "rate" in anchor.lower():
                    s += 1.0
                if re.search(r"//(join|shine)\.", url):
                    s -= 0.6  # application / login links don't show rates
            if "/calculators" in url:
                s -= 1.0  # a calculator is not where rates are published
            if s > 1.5 and (best is None or s > best[0]):
                best = (s, url, ch.url)
    return (best[1], best[2]) if best else None


_ADDRESS = re.compile(r"\b\d{2,6} [A-Z0-9][\w .'-]+?, [A-Z][\w .'-]+, (IA|IL|WI|MN), \d{5}\b")


def address_answer(store: Store, qi: QueryInfo, snap: int) -> tuple[str, str] | None:
    from .search import entity_tokens

    names = entity_tokens(qi.question) - {"branch", "office", "address", "location"}
    if not names:
        return None
    for ch in store.chunks:
        if snap not in ch.snaps or ch.kind not in ("info", "product"):
            continue
        for m in _ADDRESS.finditer(ch.text):
            line = m.group(0)
            before = ch.text[max(0, m.start() - 80) : m.start()].lower()
            if "branch" in qi.question.lower() and "branch" not in before:
                continue  # skip ATMs when the member asked about a branch
            if all(n in line.lower() for n in names):
                return line, ch.url
    return None


def looks_unpriced(qi: QueryInfo, matches: list[RateMatch]) -> bool:
    """A rate question about something no rate table covers (a 7/1 ARM, a construction
    loan). Matching rows only count if they are actually about that thing."""
    m = UNPRICED.search(qi.question) if "rate" in qi.intents else None
    if not m:
        return False
    term = m.group(0).lower()
    return not any(term in f"{c.rf.key[0]} {c.fact.label}".lower() for c in matches)

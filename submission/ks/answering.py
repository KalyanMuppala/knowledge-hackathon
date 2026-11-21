"""Turns a question into an answer: retrieve evidence from the right snapshot,
annotate it (authority, dates, stale values), then let the LLM phrase the reply.
Without an API key the rule-based answers in ks.rules are used instead."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import rules
from .conflicts import NARRATIVE_PATTERNS, ConflictEngine, latest_narrative
from .llm import LLM
from .query import QueryInfo
from .search import Searcher, coverage, entity_tokens, expand_query
from .store import ChunkRec, Store
from .textutil import fmt_date, tokenize

REFUSAL = ("I don't know. That isn't listed on our website, so please contact us directly "
           "and a team member can help.")

KIND_LABEL = {
    "rates": "Official rate sheet",
    "product": "Product page",
    "info": "Dupaco information page",
    "news": "Press release",
    "blog": "Blog article",
    "promo": "Promotional landing page",
    "calculator": "Calculator page (embedded rates are often not updated)",
    "listing": "Index page",
    "document": "Uploaded document",
    "expired": "EXPIRED offer page (do not use)",
}

SYSTEM_PROMPT = """You are the member-help assistant for Dupaco Community Credit Union. Answer members' questions using only the website excerpts supplied with each question.

Deciding what is true:
1. "Now" is the capture date given in the request. If the request asks about an earlier date, answer only for that date.
2. When excerpts disagree, trust them in this order: official rate sheet; product pages; other Dupaco pages; press releases (newest first); blog articles; promotional pages; calculators. Never use a value an excerpt marks as OUTDATED, and never use an expired offer.
3. For facts that drift over time in press releases (number of branches, members, employees, awards), use the most recently published statement. The request lists these when relevant.
4. Respect qualifiers exactly: term (48 vs 60 months), balance or loan-amount tier, model year, standard vs jumbo, regular vs special, personal vs business, Roth vs Traditional, rate vs APR vs APY vs dividend rate. They are different facts, not conflicts.

Answering:
- Lead with the direct answer in one to three sentences, with the exact figure, its unit and its qualifier (e.g. "5.39% APR for a 60-month loan on a 2024-27 model-year vehicle").
- If the member names a specific tier, give only that tier's figure. If they ask generally (e.g. "mortgage rates", "certificate rates", "checking rates", "what credit cards do you offer"), list each product or term with its figure, compactly. For mortgages lead with the 30-year fixed for loans of $100,000 or less.
- Never mention old, superseded or outdated values, other capture dates, or that pages disagree, unless the member explicitly asks how something changed or what it used to be.
- Do not add figures nobody asked for (dates, phone numbers, minimums, payment examples) unless they are needed to answer.
- Hours: give the typical schedule from the excerpts, say hours vary by branch, and offer to look up a specific branch. Never ask for a ZIP code.
- Links: when asked for a link or page, give the full URL exactly as it appears in the excerpts.
- Never ask for account details, member numbers or personal information.
- If the excerpts do not answer the question (e.g. a rate or branch the site does not list), say you don't know and that it isn't listed, and suggest contacting Dupaco. Use no numbers in that reply and set confident to false.
- Speak as Dupaco's assistant. Never mention excerpts, sources, snapshots, context or "the provided information".

Return JSON: "answer" (the reply), "confident" (false only when you could not answer), "sources" (the [S#] numbers you relied on), "conflict_note" (one short sentence if excerpts disagreed and why you chose one value, otherwise "")."""


@dataclass
class Evidence:
    chunks: list[ChunkRec]
    also_urls: dict[int, list[str]]
    notes: list[str]
    snap: int
    coverage: float                      # best share of the question's key terms in one chunk
    scores: list[float] = field(default_factory=list)


class Answerer:
    def __init__(self, store: Store, searcher: Searcher, engine: ConflictEngine, llm: LLM) -> None:
        self.store, self.searcher, self.engine, self.llm = store, searcher, engine, llm

    def answer(self, qi: QueryInfo, snap: int) -> dict:
        ev = self.gather(qi, snap)
        # Table lookups, the routing number, headcounts and "we don't price that" are
        # answered straight from the resolved facts: exact, instant and repeatable.
        fast = self.fast_answer(qi, ev)
        if fast:
            return fast
        if self.llm.available:
            data = self.llm.complete_json(SYSTEM_PROMPT, self.build_prompt(qi, ev))
            if data and isinstance(data.get("answer"), str) and data["answer"].strip():
                return self._finish_llm(data, ev)
        return self.rule_answer(qi, ev)

    def gather(self, qi: QueryInfo, snap: int, k: int = 9, budget: int = 15000) -> Evidence:
        boost = {}
        if "rate" in qi.intents:
            boost = {"rates": 1.25, "product": 1.1, "calculator": 0.7, "promo": 0.85, "blog": 0.85}
        elif "count" in qi.intents or "hours" in qi.intents:
            boost = {"news": 1.15}
        results = self.searcher.search(qi.question, snap, k=k + 6, per_url=2, kinds_boost=boost)
        chunks, scores, also, seen, used = [], [], {}, {}, 0
        for ch, s in results:
            if ch.kind == "expired":
                continue
            key = re.sub(r"\s+", " ", ch.text)[:600]
            if key in seen:                       # same table printed on another page
                also.setdefault(seen[key], []).append(ch.url)
                continue
            if used + len(ch.text) > budget and chunks:
                continue
            seen[key] = ch.cid
            chunks.append(ch)
            scores.append(s)
            used += len(ch.text)
            if len(chunks) >= k:
                break
        notes = self._hints(qi, snap, chunks)
        query = expand_query(qi.question)
        cov = max((coverage(query, c.tokens, self.searcher.bm25.idf) for c in chunks[:3]), default=0.0)
        return Evidence(chunks, also, notes, snap, cov, scores)

    def _hints(self, qi: QueryInfo, snap: int, chunks: list[ChunkRec]) -> list[str]:
        notes = []
        if "rate" in qi.intents or rules.detect_products(qi.question):
            rows = rules.rate_hints(self.store, self.engine, qi, snap)
            if rows:
                notes.append("Rows from the official rate tables that look relevant (check the tier "
                             "against the question):\n  " + "\n  ".join(rows))
        if "hours" in qi.intents:
            sched = rules.branch_hours(self.store, snap)
            if sched:
                seen = list(dict.fromkeys(f"Lobby {a} | Drive-up {b}" for a, b, _u in sched))
                notes.append("Branch schedules printed in branch press releases:\n  " +
                             "\n  ".join(seen[:6]))
        if "url" in qi.intents and "rate" in qi.intents:
            link = rules.rates_page_link(self.store, self.engine, qi, snap)
            if link:
                notes.append(f"Product page that hosts these rates: {link}")
        notes += self._narrative_notes(qi, snap)
        if qi.asks_history:
            notes += self._history_notes(qi, chunks, snap)
        return notes

    def _narrative_notes(self, qi: QueryInfo, snap: int) -> list[str]:
        low = qi.question.lower()
        if "count" not in qi.intents and not re.search(r"forbes|recogni|award|ranked|how (big|large)", low):
            return []                    # keep headcounts out of unrelated answers
        words = set(qi.tokens) | set(re.findall(r"[a-z-]+", low))
        wanted = [t for t, kws, _rx, _tmpl in NARRATIVE_PATTERNS if words & set(kws)]
        if not wanted:
            return []
        latest = latest_narrative(self.store, snap)
        out = []
        for topic in wanted:
            o = latest.get(topic)
            if not o:
                continue
            if topic.startswith("Forbes"):
                out.append(f"Most recent Forbes press release ({fmt_date(o.published)}, {o.url}): "
                           f"this was {o.statement}.")
            else:
                out.append(f"Most recent press release ({fmt_date(o.published)}, {o.url}): "
                           f"Dupaco {o.statement}.")
        return out

    def _history_notes(self, qi: QueryInfo, chunks: list[ChunkRec], snap: int) -> list[str]:
        self.engine.build()
        qtok = set(qi.tokens)
        keys = []
        for ch in chunks:
            for f in ch.facts:
                if f.metric in ("APY", "APR", "rate") and f.key not in keys and \
                        len(qtok & set(tokenize(f"{f.product} {f.label}"))) >= 2:
                    keys.append(f.key)
        notes = []
        for key in keys[:6]:
            hist = []
            for s in range(snap + 1):
                rf = self.engine.winners.get(s, {}).get(key)
                if rf and (not hist or hist[-1][0] != rf.value):
                    hist.append((rf.value, self.store.snapshots[s].taken_on))
            if len(hist) > 1:
                rf = self.engine.winners.get(snap, {}).get(key)
                label = rf.obs.fact.label if rf else key[1]
                notes.append(f"History of {key[0]} ({label}) {key[2]} on the official pages: "
                             + " -> ".join(f"{v} (seen {d.isoformat()})" for v, d in hist))
        return notes

    def build_prompt(self, qi: QueryInfo, ev: Evidence) -> str:
        st = self.store
        snap = st.snapshots[ev.snap]
        self.engine.build()
        stale = self.engine.stale_notes.get(ev.snap, {})
        carried = st.carried(ev.snap)
        lines = []
        if qi.as_of and qi.as_of < st.snapshots[-1].taken_on:
            lines.append(f"The member is asking about {fmt_date(qi.as_of)}. Use the website as it stood "
                         f"then (captured {fmt_date(snap.taken_on)}); later changes do not count.")
            if qi.as_of < st.snapshots[0].taken_on:
                lines.append(f"The earliest capture is {fmt_date(st.snapshots[0].taken_on)}; if the "
                             "excerpts can't establish the value for the requested date, say so.")
        else:
            lines.append(f"The website was captured on {fmt_date(snap.taken_on)}; treat that as now.")
        lines += ["", "Website excerpts:"]
        for i, ch in enumerate(ev.chunks, 1):
            label = KIND_LABEL.get(ch.kind, ch.kind)
            if ch.published and ch.kind in ("news", "blog"):
                label += f" published {fmt_date(ch.published)}"
            meta = [label, ch.url]
            if ch.effective:
                meta.append(f"rates effective {fmt_date(ch.effective)}")
            head = f"[S{i}] " + " | ".join(meta)
            if ch.heading:
                head += f"\nSection: {ch.heading}"
            for note in stale.get(ch.cid, [])[:4]:
                head += "\n" + note
            if ch.cid in carried:
                head += (f"\nNOTE: this page was not in the capture for this date (last seen "
                         f"{fmt_date(st.snapshots[carried[ch.cid]].taken_on)}); it may be outdated.")
            lines += [head, ch.text.strip(), ""]
        if ev.notes:
            lines.append("Notes:")
            lines += [f"- {n}" for n in ev.notes]
            lines.append("")
        lines.append(f"Member question: {qi.question}")
        return "\n".join(lines)

    def _sources(self, ev: Evidence, cited: list[int]) -> list[tuple[str, Optional[str]]]:
        cited_idx = [i - 1 for i in cited if 1 <= i <= len(ev.chunks)]
        order = cited_idx + [i for i in range(len(ev.chunks)) if i not in cited_idx]
        out, seen = [], set()
        for i in order:
            ch = ev.chunks[i]
            for u in [ch.url] + ev.also_urls.get(ch.cid, []):
                if u not in seen:
                    seen.add(u)
                    out.append((u, _quote(ch.text) if i in cited_idx else None))
        return out[:8]

    def _finish_llm(self, data: dict, ev: Evidence) -> dict:
        text = data["answer"].strip()
        confident = bool(data.get("confident", True))
        if re.search(r"\b(i (do not|don't) know|isn't listed|is not listed)\b", text, re.I):
            confident = False
        if not confident and re.search(r"\d", text):
            text = REFUSAL                        # don't let a refusal carry stray numbers
        cited = [int(x) for x in data.get("sources", []) if isinstance(x, (int, float))]
        note = (data.get("conflict_note") or "").strip() or None
        return {"text": text, "confident": confident, "sources": self._sources(ev, cited),
                "conflict_note": note, "snap": ev.snap}

    def _result(self, ev: Evidence, text: str, urls=(), confident=True, note=None) -> dict:
        srcs = [(u, None) for u in urls] + [x for x in self._sources(ev, []) if x[0] not in urls]
        return {"text": text, "confident": confident, "sources": srcs[:8],
                "conflict_note": note, "snap": ev.snap}

    def fast_answer(self, qi: QueryInfo, ev: Evidence) -> Optional[dict]:
        st, snap = self.store, ev.snap
        if not ev.chunks:
            return self._result(ev, REFUSAL, confident=False)
        matches, mode = rules.match_rates(st, self.engine, qi, snap)
        if "rate" not in qi.intents:
            matches, mode = [], "none"
        if mode == "no_such_term" or rules.looks_unpriced(qi, matches):
            return self._result(ev, REFUSAL, confident=False)
        nar = rules.narrative_answer(st, qi, snap)
        if nar:
            return self._result(ev, nar[0], [nar[1]])
        if "hours" in qi.intents:
            names = entity_tokens(qi.question) - {"lobby", "branch", "hour", "drive-up", "drive"}
            if names and not rules.hours_known_for(st, snap, names):
                return self._result(ev, REFUSAL, confident=False)   # don't borrow another branch's hours
        if re.search(r"\brouting\b|\baba\b", qi.question.lower()):
            rn = rules.routing_number(st, snap)
            if rn:
                return self._result(ev, f"Dupaco's routing number is {rn[0]}.", [rn[1]])
        if "location" in qi.intents and "hours" not in qi.intents:
            addr = rules.address_answer(st, qi, snap)
            if addr:
                return self._result(ev, f"{addr[0]}.", [addr[1]])
        if matches and qi.asks_history:
            text = rules.history_answer(st, self.engine, matches, snap)
            if text:
                return self._result(ev, text, list(dict.fromkeys(m.rf.obs.url for m in matches)))
        if matches and "url" not in qi.intents:
            text = rules.describe_rates(self.engine, snap, matches, mode)
            urls = list(dict.fromkeys(m.rf.obs.url for m in matches))
            return self._result(ev, text, urls, note=rules.stale_note(self.engine, snap, matches))
        return None

    def rule_answer(self, qi: QueryInfo, ev: Evidence) -> dict:
        """Fallback when there is no LLM (or it failed)."""
        st, snap = self.store, ev.snap
        if "url" in qi.intents:
            link = rules.link_answer(st, qi, snap, ev.chunks, self.engine)
            if link:
                return self._result(ev, f"You can use {link[0]}", [link[1]])
        if ev.coverage < 0.35:             # nothing on the site talks about this
            return self._result(ev, REFUSAL, confident=False)
        if "hours" in qi.intents:
            names = entity_tokens(qi.question) - {"lobby", "branch", "hour", "drive-up", "drive"}
            if names and not rules.hours_known_for(st, snap, names):
                return self._result(ev, REFUSAL, confident=False)
            text = rules.hours_answer(st, qi, snap)
            if text:
                return self._result(ev, text)
        if "contact" in qi.intents and re.search(r"\b(phone|call|number)\b", qi.question.lower()):
            ph = rules.phone_number(st, snap)
            if ph:
                return self._result(ev, f"You can call Dupaco at {ph[0]}.", [ph[1]])
        text, url = self._best_sentences(qi, ev)
        return self._result(ev, text, [url])

    def _best_sentences(self, qi: QueryInfo, ev: Evidence) -> tuple[str, str]:
        """Pick the sentences that overlap the question most. Footnotes and table rows
        are skipped; 'what is X' gets a couple of sentences instead of one."""
        qtok = set(qi.tokens)
        ents = entity_tokens(qi.question)
        cands = []
        for rank, ch in enumerate(ev.chunks[:4]):
            title = set(tokenize(ch.title.split("|")[0]))
            # a name the whole chunk is about ("Platinum Visa" under a Platinum Visa
            # caption) shouldn't pull in every sentence that repeats it
            ents_here = ents - set(tokenize(f"{ch.title} {ch.heading} {ch.caption}"))
            prev_toks: set = set()
            for i, sent in enumerate(re.split(r"(?<=[.!?])\s+|\n+", ch.text)):
                window, prev_toks = prev_toks, set(tokenize(sent))
                sent = sent.strip(" -")
                if not 25 < len(sent) < 400 or sent.startswith(("|", "_", "[")) or \
                        sent.endswith("?") or re.match(r"^\d[A-Z]", sent):
                    continue             # tables, disclaimers, FAQ questions, footnotes
                toks = set(tokenize(sent))
                score = len(qtok & toks) + 1.5 * len(ents_here & toks) + 0.3 * len(title & toks) \
                    + 0.5 * len((qtok | ents) & (window - toks)) - 0.5 * rank - 0.02 * i
                if "status" in qi.intents and re.search(r"\b(clos|open)", sent, re.I) and \
                        re.search(r"\b(19|20)\d\d\b", sent):
                    score += 8           # "This location will close March 31, 2026."
                if "contact" in qi.intents and re.search(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b", sent):
                    score += 3
                if "how much" in qi.question.lower() and re.search(r"\$\d", sent):
                    score += 2
                cands.append((score, rank, i, sent.replace("**", ""), ch))
        if not cands:
            return ev.chunks[0].text[:300], ev.chunks[0].url
        cands.sort(key=lambda c: -c[0])
        score, rank, i, sent, ch = cands[0]
        low = qi.question.lower()
        if re.match(r"(what is|what are|tell me about|explain|describe)\b", low):
            nxt = sorted((c for c in cands if c[4] is ch and i <= c[2] <= i + 2), key=lambda c: c[2])
            sent = " ".join(c[3] for c in nxt)
        elif "status" in qi.intents:
            prev = [c for c in cands if c[4] is ch and c[2] == i - 1]
            if prev:
                sent = f"{prev[0][3]} {sent}"
        if re.search(r"\bhow (much|many)\b", low) and not re.search(r"\d", sent):
            nxt = [c for c in sorted(cands, key=lambda c: c[2])
                   if c[4] is ch and i < c[2] <= i + 2 and re.search(r"\d", c[3])]
            if nxt:
                sent = f"{sent} {nxt[0][3]}"
        if re.search(r"\b(do you (have|offer)|can i get)\b", low):
            name = ch.title.split("|")[0].strip()
            if name and name.lower() not in sent.lower():
                sent = f"Yes - {name}: {sent}"
        return sent, ch.url


def _quote(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:220]

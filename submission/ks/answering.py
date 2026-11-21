"""Turns a question into an answer: retrieve from the right snapshot, answer table
questions from the resolved facts, and everything else from the best sentences."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import rules
from .conflicts import ConflictEngine
from .query import QueryInfo
from .search import Searcher, coverage, entity_tokens, expand_query
from .store import ChunkRec, Store
from .textutil import tokenize

REFUSAL = ("I don't know. That isn't listed on our website, so please contact us directly "
           "and a team member can help.")


@dataclass
class Evidence:
    chunks: list[ChunkRec]
    also_urls: dict[int, list[str]]      # same text found on other pages
    snap: int
    coverage: float                      # best share of the question's key terms in one chunk
    scores: list[float] = field(default_factory=list)


class Answerer:
    def __init__(self, store: Store, searcher: Searcher, engine: ConflictEngine) -> None:
        self.store, self.searcher, self.engine = store, searcher, engine

    def answer(self, qi: QueryInfo, snap: int) -> dict:
        ev = self.gather(qi, snap)
        # Table lookups, the routing number, headcounts and "we don't price that" come
        # straight from the resolved facts; the rest from the most relevant text.
        return self.fast_answer(qi, ev) or self.rule_answer(qi, ev)

    def gather(self, qi: QueryInfo, snap: int, k: int = 9, budget: int = 15000) -> Evidence:
        boost = {}
        if "rate" in qi.intents:
            boost = {"rates": 1.25, "product": 1.1, "calculator": 0.7, "promo": 0.85, "blog": 0.85}
        elif "count" in qi.intents or "hours" in qi.intents:
            boost = {"news": 1.15}
        elif re.search(r"\b(do you (have|offer)|can i get|options?)\b", qi.question.lower()):
            boost = {"product": 1.3, "news": 0.8, "blog": 0.8}     # asking about a product
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
        query = expand_query(qi.question)
        cov = max((coverage(query, c.tokens, self.searcher.bm25.idf) for c in chunks[:3]), default=0.0)
        return Evidence(chunks, also, snap, cov, scores)

    def _result(self, ev: Evidence, text: str, urls=(), confident=True, note=None) -> dict:
        """Cite the page(s) the answer came from first, then the other retrieved pages."""
        srcs, seen = [], set()
        for u in urls:
            if u not in seen:
                seen.add(u)
                srcs.append((u, _quote(text)))
        for ch in ev.chunks:
            for u in [ch.url] + ev.also_urls.get(ch.cid, []):
                if u not in seen:
                    seen.add(u)
                    srcs.append((u, None))
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
        st, snap = self.store, ev.snap
        if "url" in qi.intents:
            link = rules.link_answer(st, qi, snap, ev.chunks, self.engine)
            if link:
                return self._result(ev, f"You can use {link[0]}", [link[1]])
        if ev.coverage < 0.35:             # nothing on the site talks about this
            return self._result(ev, REFUSAL, confident=False)
        if "hours" in qi.intents:
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

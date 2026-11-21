"""Turns a question into an answer: retrieve from the right snapshot, answer table
questions from the resolved facts, and everything else from the best sentences."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import rules
from .conflicts import ConflictEngine
from .query import QueryInfo
from .search import Searcher, coverage, entity_tokens, expand_query
from .store import ChunkRec, Store
from .textutil import parse_long_date, split_sentences, strip_links, tokenize

REFUSAL = (
    "I don't know. That isn't listed on our website, so please contact us directly "
    "and a team member can help."
)


@dataclass
class Evidence:
    chunks: list[ChunkRec]
    also_urls: dict[int, list[str]]  # same text found on other pages
    snap: int
    coverage: float  # best share of the question's key terms in one chunk
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
        # lean on the rate sheet for rate questions, on press releases for counts and hours
        boost = {}
        if "rate" in qi.intents:
            boost = {"rates": 1.25, "product": 1.1, "calculator": 0.7, "promo": 0.85, "blog": 0.85}
        elif "count" in qi.intents or "hours" in qi.intents:
            boost = {"news": 1.15}
        elif re.search(r"\b(do you (have|offer)|can i get|options?)\b", qi.question.lower()):
            boost = {"product": 1.3, "news": 0.8, "blog": 0.8}  # asking about a product
        results = self.searcher.search(qi.question, snap, k=k + 6, per_url=2, kinds_boost=boost)
        chunks, scores, also, seen, used = [], [], {}, {}, 0
        for ch, s in results:
            if ch.kind == "expired":
                continue
            key = re.sub(r"\s+", " ", ch.text)[:600]
            if key in seen:  # same table printed on another page
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
        return {
            "text": text,
            "confident": confident,
            "sources": srcs[:8],
            "conflict_note": note,
            "snap": ev.snap,
        }

    def fast_answer(self, qi: QueryInfo, ev: Evidence) -> dict | None:
        st, snap = self.store, ev.snap
        if not ev.chunks:
            return self._result(ev, REFUSAL, confident=False)
        matches, mode = rules.match_rates(st, self.engine, qi, snap)
        if "rate" not in qi.intents:
            matches, mode = [], "none"
        if mode == "no_such_term" or rules.looks_unpriced(qi, matches):
            return self._result(ev, REFUSAL, confident=False)
        listed = rules.product_list(self.engine, snap, qi) if "rate" not in qi.intents else None
        if listed:
            return self._result(ev, listed)
        nar = rules.narrative_answer(st, qi, snap)
        if nar:
            return self._result(ev, nar[0], [nar[1]])
        if "hours" in qi.intents:
            names = entity_tokens(qi.question) - {"lobby", "branch", "hour", "drive-up", "drive"}
            if names and not rules.hours_known_for(st, snap, names):
                return self._result(ev, REFUSAL, confident=False)  # don't borrow another branch's hours
        if re.search(r"\brouting\b|\baba\b", qi.question.lower()):
            rn = rules.routing_number(st, snap)
            if rn:
                return self._result(ev, f"Dupaco's routing number is {rn[0]}.", [rn[1]])
        if "location" in qi.intents and "hours" not in qi.intents:
            addr = rules.address_answer(st, qi, snap)
            if addr:
                return self._result(ev, f"{addr[0]}.", [addr[1]])
        if matches and qi.asks_history:
            text = rules.history_answer(st, self.engine, matches, snap, qi.question)
            if text:
                return self._result(ev, text, list(dict.fromkeys(m.rf.obs.url for m in matches)))
        if matches and "url" not in qi.intents:
            text = rules.value_check(self.engine, snap, matches, qi) or rules.describe_rates(
                self.engine, snap, matches, mode
            )
            if "exists" in qi.intents:
                text = f"Yes. {text}"
            urls = list(dict.fromkeys(m.rf.obs.url for m in matches))
            return self._result(ev, text, urls, note=rules.stale_note(self.engine, snap, matches))
        return None

    def rule_answer(self, qi: QueryInfo, ev: Evidence) -> dict:
        st, snap = self.store, ev.snap
        if "url" in qi.intents:
            link = rules.link_answer(st, qi, snap, ev.chunks, self.engine)
            if link:
                url = re.sub(r"#[\w-]*\d{5,}[\w-]*$", "", link[0])  # drop widget ids from anchors
                return self._result(ev, f"You can find it at {url}", [link[1]])
        unknown = [e for e in entity_tokens(qi.question) if e not in self.searcher.bm25.idf]
        if ev.coverage < 0.35 or unknown:  # nothing on the site talks about this (or about "Paris")
            return self._result(ev, REFUSAL, confident=False)
        if "hours" in qi.intents:
            text = rules.hours_answer(st, qi, snap)
            if text:
                return self._result(ev, text)
        low = qi.question.lower()
        lost_card = re.search(r"\b(lost|stolen)\b.*\bcard\b|\bcard\b.*\b(lost|stolen)\b", low)
        wants_phone = "contact" in qi.intents and re.search(r"\b(phone|call|number)\b", low)
        if lost_card or wants_phone:
            ph = rules.phone_number(st, snap)
            if ph:
                text = (
                    f"Call us right away at {ph[0]} to report a lost or stolen card."
                    if lost_card
                    else f"You can call Dupaco at {ph[0]}."
                )
                return self._result(ev, text, [ph[1]])
        best = self._best_sentences(qi, ev)
        if best is None:  # nothing that actually overlaps the question
            return self._result(ev, REFUSAL, confident=False)
        return self._result(ev, best[0], [best[1]])

    def _sentence_candidates(self, qi: QueryInfo, ev: Evidence) -> list[tuple]:
        """(score, rank, index, sentence, chunk) for every presentable sentence in the
        top chunks. Headings, table rows, disclaimers, FAQ questions and footnotes are skipped."""
        qtok = set(qi.tokens)
        ents = entity_tokens(qi.question)
        how_much = "how much" in qi.question.lower()
        cands = []
        for rank, ch in enumerate(ev.chunks[:4]):
            title = set(tokenize(ch.title.split("|")[0]))
            # a name the whole chunk is about ("Platinum Visa" under a Platinum Visa
            # caption) shouldn't pull in every sentence that repeats it
            ents_here = ents - set(tokenize(f"{ch.title} {ch.heading} {ch.caption}"))
            prev_toks: set = set()
            for i, raw in enumerate(split_sentences(ch.text)):
                window, prev_toks = prev_toks, set(tokenize(raw))
                sent = _tidy(raw)
                if (
                    not 25 < len(sent) < 400
                    or len(sent.split()) < 5
                    or sent.startswith(("|", "_", "["))
                    or sent.endswith("?")
                    or re.match(r"^\d[A-Z]", sent)
                ):
                    continue
                toks = set(tokenize(sent))
                score = (
                    len(qtok & toks)
                    + 1.5 * len(ents_here & toks)
                    + 0.3 * len(title & toks)
                    + 0.5 * len((qtok | ents) & (window - toks))  # context from the sentence before
                    - 0.5 * rank
                    - 0.02 * i
                )
                if "status" in qi.intents and re.search(r"\b(clos|open)", sent, re.I) and _has_year(sent):
                    score += 8  # "This location will close March 31, 2026."
                if "contact" in qi.intents and re.search(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b", sent):
                    score += 3
                if how_much and re.search(r"\$\d", sent):
                    score += 2
                cands.append((score, rank, i, sent, ch))
        return cands

    def _best_sentences(self, qi: QueryInfo, ev: Evidence, min_score: float = 2.0) -> tuple[str, str] | None:
        """The best sentence plus whatever context the question shape needs, or None if
        even the best one barely overlaps the question."""
        cands = self._sentence_candidates(qi, ev)
        if not cands:
            return None
        score, rank, i, sent, ch = max(cands, key=lambda c: c[0])
        if score < min_score:
            return None
        same_chunk = sorted((c for c in cands if c[4] is ch), key=lambda c: c[2])
        low = qi.question.lower()
        if re.match(r"(what is|what are|tell me about|explain|describe)\b", low):
            sent = " ".join(c[3] for c in same_chunk if i <= c[2] <= i + 2)
        elif "status" in qi.intents:
            prev = [c[3] for c in same_chunk if c[2] == i - 1]
            if prev:
                sent = f"{prev[0]} {sent}"  # which location / what is closing
        if re.search(r"\bhow (much|many)\b", low) and not re.search(r"\d", sent):
            nxt = [c[3] for c in same_chunk if i < c[2] <= i + 2 and re.search(r"\d", c[3])]
            if nxt:
                sent = f"{sent} {nxt[0]}"  # the figure is often in the next sentence
        if re.search(r"\b(do you (have|offer)|can i get)\b", low):
            sent = f"Yes. {sent}"
        elif "status" in qi.intents and re.search(
            r"\b(still|currently|today|now)\b.*\bopen\b|\bis .* open\b", low
        ):
            closed = parse_long_date(sent)
            if (
                closed
                and re.search(r"\bclos", sent, re.I)
                and closed <= self.store.snapshots[ev.snap].taken_on
            ):
                sent = f"No. {sent}"
        return sent, ch.url


_DATELINE = re.compile(r"^[A-Z][A-Za-z .]+,\s*(Iowa|Ill\.|Illinois|Wis\.|Wisconsin)\s*[—–-]+\s*")


def _tidy(sent: str) -> str:
    """Make an extracted sentence presentable: no markdown links, bold or press-release dateline."""
    sent = strip_links(sent).replace("**", "").strip(" -")
    return _DATELINE.sub("", sent)


def _has_year(text: str) -> bool:
    return bool(re.search(r"\b(19|20)\d\d\b", text))


def _quote(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:220]

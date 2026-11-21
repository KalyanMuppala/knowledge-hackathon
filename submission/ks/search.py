"""BM25 over chunks, limited to one snapshot and weighted by page trust
(rate sheet > product page > blog > calculator) and article age."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Callable, Iterable, Optional

from .store import ChunkRec, Store
from .textutil import tokenize

# how members ask -> how the site words it (added to the query at a lower weight)
SYNONYMS: dict[str, list[str]] = {
    "cd": ["certificate"], "cds": ["certificate"], "certificate": ["certificate"],
    "mortgage": ["home", "loan", "fixed", "conventional", "purchase"],
    "heloc": ["home", "equity", "line", "credit"],
    "car": ["auto", "vehicle"], "vehicle": ["auto"], "truck": ["auto", "vehicle"],
    "auto": ["vehicle", "model"], "motorcycle": ["auto"],
    "interest": ["rate", "apr", "apy", "dividend"],
    "yield": ["apy"], "apy": ["dividend", "rate"], "apr": ["rate"],
    "card": ["visa", "credit"], "visa": ["card"],
    "routing": ["routing", "number", "aba"],
    "atm": ["atm", "locator", "location"],
    "hour": ["lobby", "drive-up", "open"], "lobby": ["hour"], "drive-up": ["hour"],
    "open": ["hour"], "branch": ["location", "office"],
    "location": ["branch"], "address": ["branch", "location"],
    "join": ["membership", "member", "become"], "eligible": ["membership", "join", "county"],
    "eligibility": ["membership", "join", "county"], "membership": ["join", "member"],
    "phone": ["call", "contact"], "contact": ["call", "phone"],
    "boat": ["rv", "recreational"], "rv": ["boat", "recreational"],
    "atv": ["snowmobile", "recreational"], "snowmobile": ["atv"],
    "ira": ["roth", "traditional", "retirement"], "retirement": ["ira"],
    "jumbo": ["jumbo"], "special": ["special"],
    "checking": ["checking"], "savings": ["share", "savings"],
    "student": ["student", "loan"], "wallet": ["apple", "google", "samsung", "pay"],
    "link": ["url", "page"], "url": ["link", "page"], "website": ["page", "online"],
    "employee": ["employee", "staff"], "volunteer": ["volunteer", "vto"],
    "fee": ["fee", "charge"], "overdraft": ["overdraft", "nsf"],
}

_DOMAIN_STOP = frozenset({"rate", "dupaco", "credit", "union", "current", "currently", "today",
                          "now", "right", "offer", "website"})
# generic verbs/fillers that carry little topical signal in a member question
_WEAK = frozenset({"received", "receive", "recognition", "available", "provide", "offered",
                   "use", "used", "using", "find", "see", "show", "list", "explain", "mean",
                   "work", "help", "make", "take", "keep", "go", "still", "say", "said", "kind",
                   "type", "different", "same", "all", "every", "general", "anything", "something",
                   "information", "info", "detail", "details", "again", "repeat", "thing"})


def entity_tokens(question: str) -> set[str]:
    """Capitalized words that aren't sentence-initial: "Forbes", "Galena", "Hy-Vee"."""
    out = set()
    for m in re.finditer(r"(?<![.?!]\s)(?<!^)\b([A-Z][a-zA-Z$-]{2,})", question):
        w = m.group(1)
        if w.lower() in ("dupaco", "what", "how", "when", "where", "which", "who", "can", "does",
                         "i", "apy", "apr", "ira", "cd", "visa"):
            continue
        out.update(tokenize(w))
    return out


def expand_query(question: str) -> list[tuple[str, float]]:
    toks = tokenize(question)
    ents = entity_tokens(question)
    weights: dict[str, float] = {}
    for t in toks:
        w = 0.6 if t in _DOMAIN_STOP else 0.3 if t in _WEAK else 1.0
        if t in ents:
            w = 1.6
        weights[t] = max(weights.get(t, 0.0), w)
    for t in toks:
        for s in SYNONYMS.get(t, []):
            for st in tokenize(s):
                weights.setdefault(st, 0.45)
    # terms like "60-month" matter more than anything else in a rate question
    for m in re.finditer(r"(\d+)\s*-?\s*(month|year)", question.lower()):
        weights[m.group(1)] = max(weights.get(m.group(1), 0), 1.4)
    return list(weights.items())


class BM25:
    def __init__(self, k1: float = 1.3, b: float = 0.72) -> None:
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: dict[int, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl = 1.0

    def add(self, cid: int, tokens: list[str]) -> None:
        self.doc_len[cid] = len(tokens)
        for t, tf in Counter(tokens).items():
            self.postings[t].append((cid, tf))

    def finalize(self) -> None:
        n = max(1, len(self.doc_len))
        self.avgdl = sum(self.doc_len.values()) / n
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def scores(self, query: Iterable[tuple[str, float]],
               allow: Callable[[int], bool]) -> dict[int, float]:
        """BM25 with a coverage factor, so a chunk matching most of the query beats one
        that repeats a single term ("received" x 10)."""
        out: dict[int, float] = defaultdict(float)
        covered: dict[int, float] = defaultdict(float)
        k1, b, avg = self.k1, self.b, self.avgdl
        total = 0.0
        for term, qw in query:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf.get(term, 0.0) * qw
            total += idf
            for cid, tf in plist:
                if not allow(cid):
                    continue
                dl = self.doc_len[cid]
                out[cid] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg))
                covered[cid] += idf
        if total > 0:
            for cid in out:
                out[cid] *= 0.4 + 0.6 * (covered[cid] / total)
        return out


def recency_factor(published: Optional[date], view_date: date) -> float:
    """Older articles slowly lose weight, so on identical boilerplate the newest wins."""
    if not published:
        return 1.0
    years = max(0.0, (view_date - published).days / 365.25)
    return max(0.55, 1.0 - 0.075 * years)


def coverage(query: list[tuple[str, float]], tokens: list[str], idf: dict[str, float]) -> float:
    """Fraction (idf-weighted) of the question's key terms found in a chunk."""
    key = [(t, idf.get(t, 0.0)) for t, w in query if w >= 1.0]
    total = sum(v for _, v in key)
    if not total:
        return 1.0
    toks = set(tokens)
    return sum(v for t, v in key if t in toks) / total


class Searcher:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.bm25 = BM25()
        self._indexed = 0
        self.stale_chunks: dict[int, set[int]] = {}    # snap -> chunk ids with superseded facts

    def refresh(self) -> None:
        for ch in self.store.chunks[self._indexed:]:
            self.bm25.add(ch.cid, ch.tokens)
        self._indexed = len(self.store.chunks)
        self.bm25.finalize()

    def search(self, question: str, snap: int, k: int = 8, per_url: int = 2,
               kinds_boost: Optional[dict[str, float]] = None) -> list[tuple[ChunkRec, float]]:
        store = self.store
        view_date = store.snapshots[snap].taken_on
        query = expand_query(question)
        carried = store.carried(snap)
        raw = self.bm25.scores(query, lambda cid: snap in store.chunks[cid].snaps or cid in carried)
        stale = self.stale_chunks.get(snap, set())
        scored = []
        for cid, s in raw.items():
            ch = store.chunks[cid]
            w = ch.authority * recency_factor(ch.published, view_date)
            if kinds_boost:
                w *= kinds_boost.get(ch.kind, 1.0)
            if cid in stale:
                w *= 0.45
            if cid in carried:
                w *= 0.5                 # page no longer seen in this capture
            scored.append((s * w, cid))
        scored.sort(reverse=True)
        out, per = [], Counter()
        for s, cid in scored:
            ch = store.chunks[cid]
            if per[ch.url] >= per_url:
                continue
            per[ch.url] += 1
            out.append((ch, s))
            if len(out) >= k:
                break
        return out

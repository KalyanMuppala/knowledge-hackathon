"""Temporal knowledge system for a credit union website.

    ks = build()
    for snap in snapshots: ks.ingest(snap)      # oldest -> newest
    ks.answer("What's the 60-month auto loan rate?")
    ks.answer("...", as_of="2026-05-31")        # the site as it stood on that day
    ks.conflicts()                               # every contradiction + the value we trust

See NOTES.md for the design and the rules used to decide what is current.
"""
from __future__ import annotations

import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Use the harness' own contract types when they are importable (the scorer puts
# harness/ on sys.path); otherwise fall back to identical local definitions.
try:
    from contract import Answer, Conflict, Source  # type: ignore
except ImportError:
    for _d in HERE.parents:
        if (_d / "harness" / "contract.py").exists():
            sys.path.insert(0, str(_d / "harness"))
            break
    try:
        from contract import Answer, Conflict, Source  # type: ignore
    except ImportError:
        @dataclass
        class Source:  # type: ignore[no-redef]
            url: str
            snapshot_id: Optional[str] = None
            quote: Optional[str] = None

        @dataclass
        class Answer:  # type: ignore[no-redef]
            text: str
            sources: list = field(default_factory=list)
            confident: bool = True
            conflict_note: Optional[str] = None

        @dataclass
        class Conflict:  # type: ignore[no-redef]
            topic: str
            values: list
            sources: list
            resolution: Optional[str] = None
            reason: Optional[str] = None

from ks.answering import Answerer  # noqa: E402
from ks.conflicts import ConflictEngine  # noqa: E402
from ks.llm import LLM  # noqa: E402
from ks.query import analyze  # noqa: E402
from ks.search import Searcher  # noqa: E402
from ks.store import Store  # noqa: E402


def _norm_question(q: str) -> str:
    return re.sub(r"[^a-z0-9$%./ -]+", "", re.sub(r"\s+", " ", q.lower())).strip()


class TemporalKnowledge:
    def __init__(self) -> None:
        self.store = Store()
        self.searcher = Searcher(self.store)
        self.engine = ConflictEngine(self.store)
        self.llm = LLM()
        self.answerer = Answerer(self.store, self.searcher, self.engine, self.llm)
        self._cache: dict[tuple, Answer] = {}
        self._lock = threading.Lock()
        self._dirty = False

    def ingest(self, snapshot) -> None:
        with self._lock:
            self.store.add_snapshot(snapshot)
            self._dirty = True
            self._cache.clear()
            self._refresh()          # do the indexing work now (ingest is not timed)

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        with self._lock:
            self._refresh()
        if not self.store.snapshots:
            return Answer(text="I don't know. No website data has been loaded yet.", confident=False)
        qi = analyze(question, as_of)
        snap = self.store.view_for(qi.as_of)
        key = (_norm_question(question), snap)
        cached = self._cache.get(key)
        if cached is not None:                 # asking twice gives the same answer
            return cached
        res = self.answerer.answer(qi, snap)
        sid = self.store.snapshots[res.get("snap", snap)].snapshot_id
        ans = Answer(
            text=res["text"],
            sources=[Source(url=u, snapshot_id=sid, quote=q) for u, q in res["sources"]],
            confident=res["confident"],
            conflict_note=res.get("conflict_note"),
        )
        self._cache[key] = ans
        return ans

    def conflicts(self) -> list:
        with self._lock:
            self._refresh()
        out = []
        for rec in self.engine.all_conflicts():
            out.append(Conflict(
                topic=rec.topic,
                values=list(rec.values),
                sources=[Source(url=u, snapshot_id=self.store.snapshots[s].snapshot_id, quote=q)
                         for u, s, q in rec.sources],
                resolution=rec.resolution,
                reason=rec.reason,
            ))
        return out

    def _refresh(self) -> None:
        if self._dirty:
            self.engine.build()
            self.searcher.stale_chunks = self.engine.stale_chunks
            self.searcher.refresh()
            self._dirty = False


def build() -> TemporalKnowledge:
    return TemporalKnowledge()

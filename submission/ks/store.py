"""Keeps every chunk and fact together with the snapshots it appeared in.

Consecutive crawls are mostly identical, so a chunk is stored once (keyed by url,
heading and text) with the list of snapshots that contained it. "The site as of
date D" is then just the chunks present in the last snapshot taken on or before D.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlsplit, urlunsplit

from .pages import KIND_AUTHORITY, chunk_page, prepare_page
from .tables import Fact, prose_facts, table_facts
from .textutil import normalize_url, parse_iso, strip_links, tokenize


def canonical_url(url: str) -> str:
    url = normalize_url(url.strip())
    p = urlsplit(url)
    path = p.path if p.path not in ("", "/") else "/"
    return urlunsplit((p.scheme or "https", p.netloc.lower(), path, p.query, ""))


@dataclass
class SnapInfo:
    idx: int
    snapshot_id: str
    taken_on: date
    n_pages: int


@dataclass
class ChunkRec:
    cid: int
    url: str
    title: str
    heading: str
    text: str
    kind: str
    published: date | None
    effective: date | None
    is_table: bool
    caption: str
    facts: list[Fact]
    snaps: list[int] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)

    @property
    def authority(self) -> float:
        return KIND_AUTHORITY.get(self.kind, 1.0)


@dataclass
class FactObs:
    fact: Fact
    url: str
    kind: str
    snap: int
    cid: int


@dataclass
class PageVersion:
    url: str
    title: str
    kind: str
    published: date | None
    chunk_ids: list[int]


class Store:
    def __init__(self) -> None:
        self.snapshots: list[SnapInfo] = []
        self.chunks: list[ChunkRec] = []
        self._chunk_ids: dict[str, int] = {}
        self._page_cache: dict[str, tuple] = {}
        self.pages: dict[tuple[str, int], PageVersion] = {}  # (url, snap) -> version
        self.page_snaps: dict[str, list[int]] = {}
        self.fact_obs: list[FactObs] = []
        self.new_chunks: list[int] = []  # chunk ids added by the last ingest
        self._carried: dict[int, dict[int, int]] = {}

    def add_snapshot(self, snapshot) -> SnapInfo:
        idx = len(self.snapshots)
        taken = parse_iso(snapshot.taken_on) or date.min
        info = SnapInfo(idx, snapshot.snapshot_id, taken, len(snapshot.pages))
        self.snapshots.append(info)
        self.new_chunks = []
        self._carried = {}
        seen_urls: set[str] = set()
        for page in snapshot.pages:
            url = canonical_url(page.url)
            if url in seen_urls:  # same page under two URLs (?hsLang=en etc.)
                continue
            seen_urls.add(url)
            self._add_page(idx, url, page.title or "", page.markdown or "")
        return info

    def _add_page(self, snap: int, url: str, title: str, markdown: str) -> None:
        h = hashlib.sha1(markdown.encode("utf-8", "ignore")).hexdigest()
        cached = self._page_cache.get(h + url)
        if cached is None:
            kind, text, pub, title = prepare_page(url, title, markdown)
            raw = chunk_page(text)
            cached = (kind, pub, title, raw)
            self._page_cache[h + url] = cached
        kind, pub, title, raw = cached
        ids = []
        for rc in raw:
            key = hashlib.sha1(f"{url}\x00{rc.heading}\x00{rc.text}".encode()).hexdigest()
            cid = self._chunk_ids.get(key)
            if cid is None:
                cid = len(self.chunks)
                eff = max(rc.effective) if rc.effective else None
                biz = urlsplit(url).path.lower().startswith("/business")
                facts = (
                    table_facts(rc.table, rc.caption, eff, rc.context, biz)
                    if rc.is_table
                    else prose_facts(rc.text, rc.heading, eff)
                )
                index_text = " ".join([title, rc.heading, strip_links(rc.text)])
                ch = ChunkRec(
                    cid,
                    url,
                    title,
                    rc.heading,
                    rc.text,
                    kind,
                    pub,
                    eff,
                    rc.is_table,
                    rc.caption,
                    facts,
                    [],
                    tokenize(index_text),
                )
                self.chunks.append(ch)
                self._chunk_ids[key] = cid
                self.new_chunks.append(cid)
            ch = self.chunks[cid]
            if not ch.snaps or ch.snaps[-1] != snap:
                ch.snaps.append(snap)
                for f in ch.facts:
                    self.fact_obs.append(FactObs(f, url, kind, snap, cid))
            ids.append(cid)
        self.pages[(url, snap)] = PageVersion(url, title, kind, pub, ids)
        self.page_snaps.setdefault(url, []).append(snap)

    @property
    def latest(self) -> int:
        return len(self.snapshots) - 1

    def view_for(self, as_of: date | None) -> int:
        """Snapshot that shows the site as it was on `as_of` (latest if None)."""
        if as_of is None or not self.snapshots:
            return self.latest
        best = 0
        for s in self.snapshots:
            if s.taken_on <= as_of:
                best = s.idx
        return best

    def carried(self, snap: int) -> dict[int, int]:
        """Chunks of pages that are missing from this snapshot but were seen earlier
        (chunk id -> snapshot last seen in). Crawls miss pages sometimes, so we keep
        them searchable, just down-weighted."""
        if snap not in self._carried:
            out = {}
            for url, snaps in self.page_snaps.items():
                earlier = [x for x in snaps if x < snap]
                if snap in snaps or not earlier:
                    continue
                last = max(earlier)
                for cid in self.pages[(url, last)].chunk_ids:
                    out[cid] = last
            self._carried[snap] = out
        return self._carried[snap]

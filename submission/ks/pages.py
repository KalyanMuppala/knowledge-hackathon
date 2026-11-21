"""Page types, article dates and chunking.

Most of the "which value is right" problem comes down to which page you trust, so
every page gets a kind from its URL/content:

    rates       /rates, /business-rates (the official rate sheets)
    product     /personal/..., /business/..., /savings-account/...
    info        home, about, contact, membership, policies
    news, blog  dated articles (newsroom, /learn, old /YYYY/MM/DD/ posts)
    promo       campaign landing pages (*-lp, /switch-and-earn)
    calculator  calculators; their embedded rate tables are rarely updated
    listing     index pages full of teasers for other articles
    document    files under /hubfs
    expired     "Oops ... This Offer Has Expired"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlsplit

from .textutil import clean_markdown, effective_dates, parse_long_date, safe_date

KIND_AUTHORITY = {
    "rates": 1.30,
    "product": 1.15,
    "info": 1.10,
    "news": 1.0,
    "document": 0.85,
    "blog": 0.85,
    "promo": 0.80,
    "calculator": 0.55,
    "listing": 0.35,
    "expired": 0.10,
}
# when two pages disagree about the same fact, the higher rank wins
KIND_RANK = {
    "rates": 6,
    "product": 5,
    "info": 4,
    "news": 3,
    "document": 3,
    "blog": 2,
    "promo": 2,
    "calculator": 1,
    "listing": 0,
    "expired": -1,
}

_INFO_PREFIXES = (
    "/about",
    "/contact",
    "/membership",
    "/careers",
    "/disclosures",
    "/privacy",
    "/digital-privacy-policy",
    "/community",
    "/scholarships",
    "/not-for-sale",
    "/financial-services",
    "/foundation",
    "/thankuse",
    "/subscription-preferences",
    "/photo-release-form",
    "/manage-account-alerts",
)
_PRODUCT_PREFIXES = (
    "/personal",
    "/business",
    "/savings",
    "/checking",
    "/loans",
    "/personal-loans",
    "/savings-account",
    "/online-and-mobile-banking",
    "/connect",
    "/shine",
    "/brighttrack",
    "/mobile",
    "/protection",
    "/homebuying-guidance",
    "/mortgage-refi",
    "/first-time-homebuyer-grant",
    "/hardship-grant",
    "/gift-of-innovation-grant",
    "/non-profit-grant",
    "/kccc",
    "/earn-more",
)
_PROMO_HINTS = (
    "-lp",
    "/switch-and-earn",
    "/money-makeover",
    "/sprint-to-savings-challenge",
    "/21-day-sprint-to-savings-challenge",
    "/offer-expired",
)
_ARTICLE_TAIL = re.compile(
    r"^#{1,6}\s*(handpicked for you|recent news|related (articles|posts|stories)|"
    r"you may also (like|want to consider)|more (news|stories))\b.*$",
    re.I | re.M,
)


def url_path(url: str) -> str:
    try:
        return urlsplit(url).path.rstrip("/").lower() or "/"
    except ValueError:
        return url.lower()


def classify(url: str, title: str, text: str) -> str:
    path = url_path(url)
    tl = (title or "").lower()
    head = text[:400].lower()
    if "offer has expired" in tl or head.startswith("# oops") or "this offer has expired" in head:
        return "expired"
    if path in ("/rates", "/business-rates") or path.startswith("/rates/"):
        return "rates"
    if path.startswith("/calculators") or path == "/business-calculators":
        return "calculator"
    if path.startswith("/hubfs"):
        return "document"
    if (
        path in ("/newsroom", "/resources", "/resources/dupaco-news", "/learn", "/learn/blog")
        or "/tag/" in path
        or "/author/" in path
        or ("load more" in text[-3000:].lower() and "#### filter" in text.lower())
    ):
        return "listing"
    if path.startswith("/newsroom"):
        return "news"
    if (
        path.startswith(("/learn", "/business/resources/", "/dupacodaily", "/resources/"))
        or re.match(r"^/\d{4}/", path)
        or path.startswith("/covid-19")
    ):
        return "blog"
    if any(h in path for h in _PROMO_HINTS):
        return "promo"
    if path == "/" or path.startswith(_INFO_PREFIXES):
        return "info"
    if path.startswith(_PRODUCT_PREFIXES):
        return "product"
    return "info"


def published_date(url: str, kind: str, text: str) -> date | None:
    """Date in an old-style URL (/2018/09/25/) or the byline date at the top of the article."""
    m = re.search(r"/(20\d{2}|19\d{2})/(\d{2})/(\d{2})/", url)
    if m:
        return safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if kind not in ("news", "blog"):
        return None
    d = parse_long_date(text[:700])
    if d:
        return d
    m = re.search(r"/(20\d{2})/(\d{2})/", url)
    if m:
        return safe_date(int(m.group(1)), int(m.group(2)), 1)
    return None


def drop_teaser_sections(text: str) -> str:
    """Remove "Handpicked for You" / "Recent News" blocks. They show other articles'
    dates and snippets, which would otherwise be read as part of this page."""
    while True:
        m = _ARTICLE_TAIL.search(text)
        if not m:
            return text
        nxt = re.search(r"^#{1,2}\s+\S", text[m.end() :], re.M)
        end = m.end() + nxt.start() if nxt else len(text)
        text = (text[: m.start()].rstrip() + "\n\n" + text[end:]).strip()


@dataclass
class Block:
    kind: str  # 'h' | 'p' | 'table'
    text: str
    level: int = 0


@dataclass
class RawChunk:
    heading: str
    text: str
    is_table: bool = False
    caption: str = ""
    table: str = ""
    effective: list = field(default_factory=list)
    context: str = ""  # text just before a table (for product detection)


def split_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    buf: list[str] = []

    def flush():
        if buf:
            blocks.append(Block("p", "\n".join(buf).strip()))
            buf.clear()

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("|"):
            flush()
            j = i
            rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(lines[j].strip())
                j += 1
            blocks.append(Block("table", "\n".join(rows)))
            i = j
            continue
        hm = re.match(r"^(#{1,6})\s+(.*)$", s)
        if hm:
            flush()
            title = hm.group(2).strip().strip("*").strip()
            if title:
                blocks.append(Block("h", title, len(hm.group(1))))
            i += 1
            continue
        if not s:
            flush()
        else:
            buf.append(s)
        i += 1
    flush()
    return blocks


def _is_caption(b: Block) -> bool:
    t = b.text.strip()
    if b.kind != "p" or "\n" in t or len(t) > 90:
        return False
    if t.startswith("- ") and "[" not in t and len(t) < 60:
        return True  # tab label, e.g. "- Term-Share Certificate Rates"
    return (
        not t.startswith(("[", "-", "*", "_"))
        and not t.endswith((".", ":", "?", "!"))
        and not re.match(r"^\d+([A-Z\s.]|$)", t)
    )


def _real_table(table: str) -> bool:
    rows = [r for r in table.split("\n") if r.strip()]
    if len(rows) < 3:
        return False
    return any(re.search(r"\d", r) for r in rows[2:])


MAX_PROSE = 1500
MAX_FOOTNOTES = 1400


def chunk_page(text: str) -> list[RawChunk]:
    """Section-sized chunks. Each table becomes its own chunk together with its
    caption above and the footnotes / "Rates are effective as of" line below it."""
    blocks = split_blocks(text)
    chunks: list[RawChunk] = []
    heads: dict[int, str] = {}
    prose: list[str] = []

    def heading() -> str:
        return " > ".join(heads[k] for k in sorted(heads) if k <= 3)

    def emit_prose():
        body = "\n\n".join(p for p in prose if p.strip())
        if body.strip():
            chunks.append(RawChunk(heading(), body, effective=effective_dates(body)))
        prose.clear()

    # mark which paragraph blocks are captions of the table right after them
    caption_of: dict[int, int] = {}
    for t, b in enumerate(blocks):
        if b.kind != "table":
            continue
        k, taken = t - 1, 0
        while k >= 0 and taken < 2 and _is_caption(blocks[k]) and k not in caption_of:
            caption_of[k] = t
            k -= 1
            taken += 1

    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.kind == "h":
            if b.level <= 3:
                emit_prose()
                heads = {k: v for k, v in heads.items() if k < b.level}
                heads[b.level] = b.text[:120]
            else:
                prose.append(b.text)
            i += 1
            continue
        if b.kind == "p" and i in caption_of:
            i += 1  # consumed by its table below
            continue
        if b.kind == "p":
            if sum(len(p) for p in prose) + len(b.text) > MAX_PROSE and prose:
                emit_prose()
            prose.append(b.text)
            i += 1
            continue
        # table
        if not _real_table(b.text):
            prose.append(b.text)
            i += 1
            continue
        emit_prose()
        caps = [blocks[k].text for k in sorted(caption_of) if caption_of[k] == i]
        foot: list[str] = []
        j = i + 1
        while (
            j < len(blocks)
            and blocks[j].kind == "p"
            and j not in caption_of
            and sum(len(x) for x in foot) < MAX_FOOTNOTES
        ):
            foot.append(blocks[j].text)
            j += 1
        body = "\n".join(caps + [b.text] + foot)
        # product clues: short lines / headings in the preceding blocks (tab labels,
        # captions), then the start of the nearest paragraph
        prev = [blocks[k] for k in range(max(0, i - 12), i) if blocks[k].kind != "table"]
        shorts = [x.text for x in prev if len(x.text) <= 80 or x.kind == "h"]
        near = prev[-1].text[:200] if prev else ""
        context = " ".join([heading()] + shorts + [near])[-600:]
        chunks.append(
            RawChunk(
                heading(),
                body,
                is_table=True,
                caption=" / ".join(c.lstrip("- ").strip() for c in caps),
                table=b.text,
                effective=effective_dates("\n".join(foot)),
                context=context,
            )
        )
        i = j
    emit_prose()
    return chunks


def prepare_page(url: str, title: str, markdown: str) -> tuple[str, str, date | None, str]:
    """Return (kind, cleaned_text, published_date, title)."""
    text = clean_markdown(markdown)
    kind = classify(url, title, text)
    text = drop_teaser_sections(text)
    pub = published_date(url, kind, text)
    return kind, text, pub, title or ""

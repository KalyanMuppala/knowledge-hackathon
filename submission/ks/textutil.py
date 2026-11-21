"""Text helpers: cleaning the crawler's markdown, tokenizing, numbers and dates."""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

_FLAG = re.compile(r"[\U0001F1E6-\U0001F1FF]")
_PAREN_URL = r"((?:[^()\s]|\([^()\s]*\))*)"          # URL allowing one level of (...)
_DATA_IMG = re.compile(r"!\[[^\]]*\]\(data:[^)]*\)")
_IMG = re.compile(r"!\[[^\]]*\]\(" + _PAREN_URL + r"(?:\s+\"[^\"]*\")?\)")
_LINK = re.compile(r"\[([^\]]*)\]\(" + _PAREN_URL + r"(?:\s+\"[^\"]*\")?\)")
_ESCAPED = re.compile(r"\\([*#_\[\]()$.!+\-`>|])")
_TRACKING_PARAMS = {"hslang", "utm_source", "utm_medium", "utm_campaign", "utm_content",
                    "utm_term", "hsctatracking", "_hsenc", "_hsmi", "redirect"}
_MODAL_LINES = (
    "heads up! this link leads to a different website",
    "we only do this when it's helpful for you",
    "we only do this when it’s helpful for you",
)


def normalize_url(url: str) -> str:
    """Drop tracking params (?hsLang=en etc.) so a page always has one URL."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.query:
        kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                if k.lower() not in _TRACKING_PARAMS]
        parts = parts._replace(query=urlencode(kept))
    return urlunsplit(parts)


def _link_sub(m: re.Match) -> str:
    text, url = m.group(1).strip(), m.group(2).strip()
    low = url.lower()
    if not url or low.startswith(("tel:", "mailto:", "javascript:")) or low.endswith("#"):
        return text
    url = normalize_url(url)
    if not text or text == url:
        return url
    return f"[{text}]({url})"


def clean_markdown(md: str) -> str:
    """Strip images, tracking params, modal boilerplate and form widgets; keep links and tables."""
    md = md.replace("\r\n", "\n")
    md = re.sub(r"(?:\\+r)?\\+n", "\n", md)          # CMS text with literal "\r\n" escapes
    md = re.sub(r"\\{1,2}\n", "\n", md)                 # markdown hard breaks "\\\n"
    md = _DATA_IMG.sub("", md)
    md = _IMG.sub("", md)
    md = _LINK.sub(_link_sub, md)
    md = _LINK.sub(_link_sub, md)                        # nested [![..](..)](..) leftovers
    md = _ESCAPED.sub(r"\1", md)
    md = md.replace("\u00a0", " ").replace("\u200b", "")
    out = []
    for line in md.split("\n"):
        s = line.rstrip()
        st = s.strip()
        if _FLAG.search(s):                              # country-code pickers in forms
            continue
        low = st.lower()
        if st in ("✕", "* * *", "---", "Cancel", "Continue Cancel", "Previous", "Next") or \
                low.startswith(_MODAL_LINES):
            continue
        if low.startswith("but we must inform you that dupaco isn't responsible") or \
           low.startswith("but we must inform you that dupaco isn’t responsible"):
            continue
        if re.fullmatch(r"\[continue\]\([^)]*\)\s*cancel", low):
            continue
        out.append(s)
    text = "\n".join(out)
    text = _drop_state_pickers(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    # carousels repeat the same paragraphs; keep the first copy of long paragraphs
    seen, paras = set(), []
    for para in re.split(r"\n\s*\n", text):
        key = para.strip()
        if len(key) > 40 and not key.startswith("|"):
            if key in seen:
                continue
            seen.add(key)
        paras.append(para)
    text = "\n\n".join(paras)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_STATES = {"alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
           "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas",
           "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota",
           "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey",
           "new mexico", "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
           "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah",
           "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming"}


def _drop_state_pickers(text: str) -> str:
    # forms render the state <select> as a 50-item list
    lines = text.split("\n")
    keep, run = [], []
    for line in lines + [""]:
        if line.strip().lstrip("- ").lower() in _STATES:
            run.append(line)
            continue
        if len(run) < 10:
            keep.extend(run)
        run = []
        keep.append(line)
    return "\n".join(keep[:-1])


def strip_links(text: str) -> str:
    """Links -> anchor text (for the search index)."""
    return _LINK.sub(lambda m: m.group(1) or m.group(2), text)


_TOKEN = re.compile(r"\$?\d+(?:[.,]\d+)*%?|[a-z][a-z0-9'’]*")
STOPWORDS = frozenset("""
a an and are as at be by can could do does for from had has have how i if in into is it its
me my of on or our so than that the their them then there these this those to too us was we
were what when where which who whom why will with would you your yours dupaco dupaco's
please tell give get let know want like need just also any some much many about
""".split())
_IRREGULAR = {"rates": "rate", "loans": "loan", "months": "month", "years": "year",
              "certificates": "certificate", "accounts": "account", "branches": "branch",
              "hours": "hour", "savings": "savings", "taxes": "tax", "fees": "fee",
              "cds": "cd", "iras": "ira", "atms": "atm", "cards": "card", "terms": "term",
              "options": "option", "employees": "employee", "members": "member",
              "businesses": "business", "lives": "life", "mortgages": "mortgage"}


def stem(tok: str) -> str:
    if tok in _IRREGULAR:
        return _IRREGULAR[tok]
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is", "'s")):
        return tok[:-1]
    if tok.endswith(("'s", "’s")):
        return tok[:-2]
    return tok


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    """Lowercase tokens with light stemming. "5.39%" yields both "5.39%" and "5.39"."""
    out = []
    low = text.lower().replace("–", "-").replace("—", "-")
    for raw in _TOKEN.findall(low):
        if raw[0].isdigit() or raw[0] == "$":
            num = raw.lstrip("$").replace(",", "")
            out.append(num)
            if num.endswith("%"):
                out.append(num[:-1])
            continue
        raw = raw.replace("’", "'")
        if not keep_stop and raw in STOPWORDS:
            continue
        tok = stem(raw)
        if not keep_stop and tok in STOPWORDS:
            continue
        if len(tok) < 2 and not tok.isdigit():
            continue
        out.append(tok)
    return out


PCT = re.compile(r"(\d{1,2}(?:\.\d{1,3})?)\s?%")


def clean_cell(cell: str) -> str:
    # footnote markers get glued to values: "0.20%1", "5.39% APR1"
    c = (cell or "").strip().replace("\\", "")
    c = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡]+", "", c)
    c = re.sub(r"(%)\s*\d\b", r"\1", c)
    c = re.sub(r"\b(APR|APY)\d\b", r"\1", c)
    c = c.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", c).strip()


def norm_label(s: str) -> str:
    """Row label used in fact keys: no dashes/commas variants, singular units."""
    s = clean_cell(s).lower()
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\bmonths\b", "month", s)
    s = re.sub(r"\byears\b", "year", s)
    s = re.sub(r"[^a-z0-9$%.+:&'/ -]", "", s)
    return re.sub(r"\s+", " ", s).strip()


MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
               "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12})
_MONTH_RE = (r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
             r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
LONG_DATE = re.compile(_MONTH_RE + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.I)
MONTH_YEAR = re.compile(_MONTH_RE + r"\.?,?\s+(\d{4})", re.I)
NUM_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def month_end(y: int, m: int) -> date:
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return nxt - timedelta(days=1)


def parse_iso(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    m = ISO_DATE.search(str(s))
    return safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def parse_num_date(s: str) -> Optional[date]:
    m = NUM_DATE.search(s or "")
    if not m:
        return None
    mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000
    return safe_date(y, mo, d)


def month_num(name: str) -> Optional[int]:
    return MONTHS.get(name.lower().rstrip("."))


def parse_long_date(s: str) -> Optional[date]:
    m = LONG_DATE.search(s or "")
    if not m:
        return None
    mo = month_num(m.group(1))
    return safe_date(int(m.group(3)), mo, int(m.group(2))) if mo else None


EFFECTIVE = re.compile(r"effective\s+(?:as\s+of\s+)?(\d{1,2}/\d{1,2}/\d{2,4})", re.I)


def effective_dates(text: str) -> list[date]:
    out = []
    for m in EFFECTIVE.finditer(text or ""):
        d = parse_num_date(m.group(1))
        if d:
            out.append(d)
    return out


def fmt_date(d: Optional[date]) -> str:
    return d.strftime("%b %d, %Y").replace(" 0", " ") if d else "unknown"

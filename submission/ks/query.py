"""Question analysis: which date the member is asking about, and what kind of
answer they want (a rate, hours, a link, a count, a change over time...)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .textutil import (
    ISO_DATE,
    LONG_DATE,
    MONTH_YEAR,
    NUM_DATE,
    month_end,
    month_num,
    parse_date_loose,
    parse_iso,
    safe_date,
    tokenize,
)

_DATE_PREP = r"(?:as of|as at|in|on|during|back in|by|at the end of|around)\s+"
_HISTORY = re.compile(
    r"\b(chang(e|ed|es|ing)|used to|previous(ly)?|before|history|historical|trend|went (up|down)|"
    r"go(ne)? (up|down)|increase[ds]?|decrease[ds]?|drop(ped)?|rais(e|ed)|lower(ed)?|"
    r"difference|compared?|originally|old(er)? rate|last time|since|over time|new rate)\b",
    re.I,
)


@dataclass
class QueryInfo:
    question: str
    as_of: date | None
    as_of_source: str  # "param" | "question" | "none"
    asks_history: bool
    intents: set = field(default_factory=set)
    tokens: list = field(default_factory=list)


def date_in_question(q: str) -> date | None:
    """Dates the member pins the question to: 'as of May 2026', 'on June 9, 2026',
    'in 5/2026'... (only when introduced by a temporal preposition)."""
    low = q.lower()
    for rx, kind in ((LONG_DATE, "long"), (NUM_DATE, "num"), (ISO_DATE, "iso"), (MONTH_YEAR, "my")):
        for m in rx.finditer(low):
            pre = low[max(0, m.start() - 22) : m.start()]
            if not re.search(_DATE_PREP + r"$", pre):
                continue
            if kind == "long":
                mo = month_num(m.group(1))
                return safe_date(int(m.group(3)), mo, int(m.group(2))) if mo else None
            if kind == "num":
                y = int(m.group(3))
                return safe_date(y + 2000 if y < 100 else y, int(m.group(1)), int(m.group(2)))
            if kind == "iso":
                return parse_iso(m.group(0))
            mo = month_num(m.group(1))
            if mo:
                return month_end(int(m.group(2)), mo)
    return None


def analyze(question: str, as_of: str | None) -> QueryInfo:
    d, src = None, "none"
    if as_of:
        d, src = parse_date_loose(str(as_of)), "param"
    if d is None:
        d = date_in_question(question)
        src = "question" if d else "none"
    low = question.lower()
    intents = set()
    if re.search(r"\b(rates?|apys?|aprs?|dividends?|interest|yields?|percent)\b|%", low) or re.search(
        r"\b(what|how much) (does|do|did|will|would|can)\b.*\b(pay|earn)\b|\bearn(s|ing)?\b", low
    ):
        intents.add("rate")
    if re.search(r"\b(minimum|min)\b.*\b(open|deposit|balance|earn)|\bhow much .*\bto open\b", low):
        intents.update({"rate", "minimum"})
    if re.match(
        r"\s*(is there|are there|do you (have|offer)|does dupaco (have|offer)|can i get)\b", low
    ) and re.search(
        r"\d+[- ]?(month|year|yr)|\b(one|two|three|four|five|six|seven|ten)[- ](month|year)", low
    ):
        intents.update({"rate", "exists"})  # "is there a 5-year certificate special?"
    if re.search(r"\b(hours?|what time|lobby|drive[- ]?up|drive[- ]?thr|open on|open until)", low):
        intents.add("hours")
    elif re.search(r"\b(open|close[sd]?|closing|closure)\b", low):
        intents.add("status")
    if re.search(r"\b(link|url|website|web page|webpage|online|site)\b", low):
        intents.add("url")
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", low):
        intents.add("count")
    if re.search(r"\b(address|located|location|where)\b", low):
        intents.add("location")
    if re.search(r"\b(phone|call|contact|email)\b", low):
        intents.add("contact")
    if re.search(r"\b(what|which) .*(do you|does dupaco) (offer|have)\b|\ball\b|\blist\b|\boptions\b", low):
        intents.add("list")
    return QueryInfo(question, d, src, bool(_HISTORY.search(question)), intents, tokenize(question))

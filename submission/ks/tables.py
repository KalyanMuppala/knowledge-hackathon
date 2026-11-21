"""Rate tables -> facts keyed by (product, tier, metric).

e.g. ("auto loan", "tier1|60 month", "APR") -> "5.39%"

The key has to be the same when one fact is printed on several pages (rate sheet,
product page, calculator, landing page), and different for things that only look
alike (48 vs 60 months, standard vs jumbo, rate vs APY, personal vs business).
Same key + different value = a contradiction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .textutil import PCT, clean_cell, norm_label

_SEP = re.compile(r"^:?-{2,}:?$")
_RANGE = re.compile(r"(\d{1,2}(?:\.\d{1,3})?)\s?%?\s*(?:-|–|to)\s*(\d{1,2}(?:\.\d{1,3})?)\s?%")


@dataclass(frozen=True)
class Fact:
    product: str  # canonical product family, e.g. "auto loan", "certificate special"
    tier: str  # normalized tier key, e.g. "tier1|60 month", "$25000+"
    metric: str  # "APY" | "dividend rate" | "APR" | "rate" (+ qualifier)
    value: str  # display value, e.g. "5.39%" or "12.50%-19.50%"
    num: float  # first number in value (for sorting / comparisons)
    label: str  # human readable row label, e.g. "2024-27 Model Year, 60 Months"
    caption: str  # table caption as printed
    effective: date | None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.product, self.tier, self.metric)

    def describe(self) -> str:
        return f"{self.product} ({self.label}) {self.metric}"


def split_row(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [c.strip() for c in row.split("|")]


def parse_table(md: str) -> tuple[list[str], list[list[str]]]:
    rows = [split_row(r) for r in md.split("\n") if r.strip().startswith("|")]
    if not rows:
        return [], []
    header = [clean_cell(h) for h in rows[0]]
    body = [r for r in rows[1:] if not all(_SEP.match(c.replace(" ", "")) or not c for c in r)]
    return header, body


_METRIC_WORDS = re.compile(
    r"annual percentage (yield|rate)|\(?\bap[ry]\b\)?|dividend rate|dividend|interest rate|\brate\b|"
    r"\(?as low as\)?|[*()]|\d+$",
    re.I,
)


def header_metric(h: str) -> str | None:
    """ "Annual Percentage Yield (APY)1" -> "APY", "APY* After Promotional Period" ->
    "APY (after promotional period)". Payment and points columns aren't facts."""
    hl = h.lower()
    if re.search(r"\bmin(\.|imum)?\b.*\bopen", hl):
        return "minimum to open"
    if re.search(r"\bmin(\.|imum)?\b.*\bearn", hl):
        return "minimum to earn"
    if "payment" in hl or "min." in hl or "minimum" in hl or "points" in hl:
        return None
    if "annual percentage yield" in hl or re.search(r"\bapy\b", hl):
        base = "APY"
    elif "dividend" in hl:
        base = "dividend rate"
    elif "annual percentage rate" in hl or re.search(r"\bapr\b", hl):
        base = "APR"
    elif re.fullmatch(r"(interest )?rate\s*\d?", hl.strip()):
        base = "rate"
    else:
        return None
    qual = _METRIC_WORDS.sub(" ", hl)
    qual = re.sub(r"\s+", " ", qual).strip(" -:,")
    return f"{base} ({qual})" if re.search(r"[a-z]{3}", qual) else base


_LABEL_SKIP = re.compile(r"payment|min\.|minimum|points", re.I)
_DOLLARS = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")

# product names that appear as the first cell of rate-sheet rows
_ROW_PRODUCTS = (
    "business share savings",
    "share savings",
    "bright savings",
    "you name it savings",
    "holiday club savings",
    "launchpad savings",
    "investor's choice money market",
    "achieve money market",
    "high-yield savings",
    "health savings account",
    "business 1st rate checking",
    "1st rate checking",
    "free checking",
    "business platinum visa",
    "business rewards visa",
)


def _family(text: str, label: str, special_ok: bool = True) -> str | None:
    """Keyword rules. When applied to the surrounding text (not the table's own
    caption) "special" is ignored: that text often includes the previous table's caption."""
    business = "business" in text
    special = special_ok and "special" in text and "regular" not in text
    if "model year" in text:
        if any(w in text for w in ("atv", "snowmobile", "jet ski")):
            return "atv, snowmobile & jet ski loan"
        return "auto loan"
    if "heloc" in text or "line of credit" in text:
        return "heloc"
    if "home equity loan" in text:
        return "home equity loan"
    if "home improvement" in text:
        return "home improvement loan"
    if "personal loan" in text:
        return "personal loan"
    if "boat" in text or re.search(r"\brv\b", text):
        return "boat & rv loan"
    if "atv" in text or "snowmobile" in text:
        return "atv, snowmobile & jet ski loan"
    if re.search(r"\bira", text) or label.startswith(("roth", "traditional")):
        if special or "special" in label:
            return "ira special"
        if "variable" in text or "$" in label:
            return "ira variable"
        return "ira"
    if "certificate special" in label or (special and "certificate" in text):
        return "business certificate special" if business else "certificate special"
    if "certificate" in text:
        return "business certificate" if business else "certificate"
    if "platinum" in text and ("visa" in text or "credit card" in text):
        return "business platinum visa" if business else "platinum visa"
    if "rewards" in text and ("visa" in text or "credit card" in text):
        return "business rewards visa" if business else "rewards visa"
    if "1st rate checking" in text:
        return "business 1st rate checking" if business else "1st rate checking"
    if "free checking" in text:
        return "free checking"
    if "achieve" in text:
        return "achieve money market"
    if "investor" in text and "choice" in text:
        return "investor's choice money market" + (" (business)" if business else "")
    if "high-yield" in text or "high yield" in text:
        return "high-yield savings" + (" (business)" if business else "")
    if "health savings" in text or re.search(r"\bhsa\b", text):
        return "health savings account"
    return None


def canonical_product(
    caption: str, label: str, header_label: str = "", context: str = "", business_page: bool = False
) -> str:
    name = _canonical_product(caption, label, header_label, context)
    if (
        business_page
        and "business" not in name
        and name != "unknown"
        and not name.endswith("loan")
        and name not in ("mortgage", "heloc")
    ):
        name = "business " + name  # rates on /business pages are business rates
    return name


def _canonical_product(caption: str, label: str, header_label: str, context: str) -> str:
    cap = caption.lower()
    lab = label.lower()
    head = header_label.lower()
    if re.search(r"(purchase|refinance)\s*:\s*\d+-year fixed", lab):
        return "mortgage"
    if "relief program" in lab:
        return "mortgage rate relief program"
    if "model year" in lab or "model year" in cap:
        return _family(f"{cap} {lab}", lab) or "auto loan"
    for name in _ROW_PRODUCTS:
        if lab.startswith(name):
            return name + (" (business)" if "(business)" in lab and "business" not in name else "")
    fam = _family(f"{cap} {head} {lab}", lab)
    if fam:
        return fam
    fam = _family(context.lower(), lab, special_ok=False)
    if fam:
        return fam
    if re.fullmatch(r"\d+ months?", lab.strip()):
        # a deposit table keyed only by term ("60 Month") is a certificate table
        return "business certificate" if "business" in f"{cap} {context.lower()}" else "certificate"
    name = re.sub(r"\b(rates?|accounts?)\b", "", cap or head)
    name = re.sub(r"\s+", " ", name).strip(" -/")
    return name or "unknown"


_MODEL_YEAR = re.compile(r"((?:19|20)\d{2})(?:\s*-\s*(\d{2,4}))?", re.I)


def _model_year_top(label: str) -> int | None:
    """2027 for "2024-27 Model Year", 2026 for "Model Year 2026-23"."""
    if "model year" not in label.lower():
        return None
    m = _MODEL_YEAR.search(label)
    if not m:
        return None
    a = int(m.group(1))
    b = m.group(2)
    if b:
        b = int(b) + (2000 if len(b) == 2 else 0)
        return max(a, b)
    return a


def parse_value(cell: str) -> tuple[str, float]:
    rm = _RANGE.search(cell)
    if rm and cell.find("%") >= rm.start(1):
        return f"{rm.group(1)}%-{rm.group(2)}%", float(rm.group(1))
    m = PCT.search(cell)
    return f"{m.group(1)}%", float(m.group(1))


def table_facts(
    table_md: str, caption: str, effective: date | None, context: str = "", business_page: bool = False
) -> list[Fact]:
    header, body = parse_table(table_md)
    if not header or not body:
        return []
    metrics = [header_metric(h) for h in header]
    if not any(metrics):
        return []
    header_label = " ".join(h for h, m in zip(header, metrics) if not m and not _LABEL_SKIP.search(h))
    rows: list[tuple[list[str], list[tuple[str, str]]]] = []
    for r in body:
        cells = [clean_cell(c) for c in r]
        labels, values = [], []
        for i, c in enumerate(cells):
            h = header[i] if i < len(header) else ""
            m = metrics[i] if i < len(metrics) else None
            if m and m.startswith("minimum"):
                if _DOLLARS.search(c):
                    values.append((m, c))
            elif m and PCT.search(c):
                vm = re.search(r"\b(APR|APY)\b", c)
                if vm and m in ("APR", "APY"):
                    m = vm.group(1)  # trust the cell over a mislabeled header
                values.append((m, c))
            elif c and c not in ("—", "-", "N/A") and not _LABEL_SKIP.search(h):
                labels.append(c)
        if labels and any(not m.startswith("minimum") for m, _ in values):
            rows.append((labels, values))
    # Key model-year rows by rank (newest = tier1), not by label: when the rate sheet
    # moved from "2023-26" to "2024-27", the calculator kept the old label and rate.
    tops = sorted({t for labels, _ in rows for t in [_model_year_top(" ".join(labels))] if t}, reverse=True)
    rank = {t: i + 1 for i, t in enumerate(tops)}
    facts = []
    for labels, values in rows:
        label = ", ".join(labels)
        product = canonical_product(caption, label, header_label, context, business_page)
        tier_parts = []
        for lab in labels:
            nl = norm_label(lab)
            if product.startswith(nl) or nl in product:
                continue  # the product name, not a tier
            top = _model_year_top(lab)
            if top is not None and top in rank:
                tier_parts.append(f"tier{rank[top]}")
            else:
                tier_parts.append(nl)
        tier = "|".join(tier_parts) or "standard"
        seen_metric = set()
        for metric, val in values:
            if metric in seen_metric:
                continue
            seen_metric.add(metric)
            if metric.startswith("minimum"):
                num = float(_DOLLARS.search(val).group(1).replace(",", ""))
                disp = f"${num:,.2f}"
            else:
                disp, num = parse_value(val)
            facts.append(Fact(product, tier, metric, disp, num, label, caption, effective))
    return facts


_PROSE = re.compile(
    r"^[-*\s]*([A-Za-z0-9][^:\n|]{2,60}?)\s*:\s*(\$\s?\d[\d,]*(?:\.\d+)?|\d{1,2}(?:\.\d{1,3})?\s?%)\s*$", re.M
)


def prose_facts(text: str, heading: str, effective: date | None) -> list[Fact]:
    """Facts written as "Label: value" lines ("6-month certificate APY: 3.50%")."""
    out = []
    for m in _PROSE.finditer(text):
        label, value = m.group(1).strip(" *"), m.group(2).replace(" ", "")
        low = label.lower()
        if re.search(r"\bapy\b|annual percentage yield", low):
            metric = "APY"
        elif re.search(r"\bapr\b|annual percentage rate", low):
            metric = "APR"
        elif "dividend" in low:
            metric = "dividend rate"
        elif "rate" in low:
            metric = "rate"
        elif re.search(r"\b(fee|charge|cost|price)\b", low):
            metric = "fee"
        else:
            continue
        name = re.sub(r"\b(apy|apr|dividend rate|interest rate|rate)\b", "", label, flags=re.I).strip(" -,")
        product = canonical_product("", name, "", heading)
        num = float(re.sub(r"[^\d.]", "", value) or 0)
        out.append(Fact(product, norm_label(name), metric, value, num, name, heading, effective))
    return out


_PROPER = {
    "share savings": "Share Savings",
    "bright savings": "Bright Savings",
    "you name it savings": "You Name It Savings",
    "holiday club savings": "Holiday Club Savings",
    "launchpad savings": "Launchpad Savings",
    "investor's choice money market": "Investor's Choice Money Market",
    "achieve money market": "Achieve Money Market",
    "high-yield savings": "High-Yield Savings",
    "health savings account": "Health Savings Account",
    "1st rate checking": "1st Rate Checking",
    "free checking": "Free Checking",
    "platinum visa": "Platinum Visa",
    "rewards visa": "Rewards Visa",
    "heloc": "HELOC",
    "certificate": "certificate",
    "certificate special": "certificate special",
}


def proper(product: str) -> str:
    biz = product.startswith("business ") or product.endswith(" (business)")
    base = product.replace("business ", "").replace(" (business)", "")
    name = _PROPER.get(base, base)
    return f"Business {name}" if biz else name


def tier_text(product: str, label: str) -> str:
    """The part of a row label that isn't the product name: '$10,000 - $24,999.99'."""
    head, _, rest = label.partition(", ")
    base = product.split(" (")[0].replace("business ", "")
    return rest if rest and head.lower().startswith(base[:6]) else label

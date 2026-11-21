"""Tier matching, staleness and refusals on a small made-up site with two crawls.

The rate sheet changes between the crawls (and relabels the newest model-year tier),
while a calculator page keeps last year's table, the same mess the real snapshots have.
"""

import pytest
import solution
from contract import Page, Snapshot

RATES_MAY = """# Rates

Auto & Motorcycle Loans

| Loan Program | Term | Annual Percentage Rate (As Low As) |
| --- | --- | --- |
| 2023-26 Model Year | 60 Months | 5.19% APR1 |
| 2023-26 Model Year | 84 Months | 5.39% APR1 |
| 2021-22 Model Year | 60 Months | 5.44% APR1 |

Rates are effective as of 05/01/2026.

Home Loans - Purchase

| Program | Rate | Points | APR |
| --- | --- | --- | --- |
| Purchase: 30-Year Fixed Conventional $832,750 or less | 6.500% | 0 | 6.538% |
| Purchase: 30-Year Fixed Conventional $100,000 or less | 6.250% | 0 | 6.388% |

Rates are effective as of 05/01/2026.

Special Certificates

| Certificate | Dividend Rate | Annual Percentage Yield (APY) | Min. to Earn |
| --- | --- | --- | --- |
| 10 Month Fixed Jumbo Certificate Special | 3.86% | 3.90%1 | $100,000.00 |
| 10 Month Fixed Certificate Special | 3.76% | 3.80%2 | $1,000.00 |

Rates are effective as of 05/01/2026.
"""

RATES_JUNE = (
    RATES_MAY.replace("2023-26 Model Year | 60 Months | 5.19%", "2024-27 Model Year | 60 Months | 5.39%")
    .replace("2023-26 Model Year | 84 Months | 5.39%", "2024-27 Model Year | 84 Months | 5.59%")
    .replace("6.500% | 0 | 6.538%", "7.25% | 0 | 7.29%")
    .replace("6.250% | 0 | 6.388%", "7.13% | 0 | 7.27%")
    .replace("3.86% | 3.90%1", "3.96% | 4.00%1")
    .replace("3.76% | 3.80%2", "3.86% | 3.90%2")
    .replace("05/01/2026", "06/01/2026")
)

CALCULATOR = """# Vehicle Affordability Calculator

Auto Loan Rates

| Account Type | Balance Tier | Annual Percentage Yield |
| --- | --- | --- |
| 2023–26 Model Year | 60 Months | 5.34% APR |

Rates are effective as of 1/1/2026.
"""


def snap(sid, day, rates):
    pages = [
        Page("https://cu.test/rates", "Rates", rates, day),
        Page("https://cu.test/calculators/vehicle-affordability", "Calculator", CALCULATOR, day),
    ]
    return Snapshot(sid, day, pages)


@pytest.fixture(scope="module")
def ks():
    system = solution.build()
    system.ingest(snap("01_may", "2026-05-12", RATES_MAY))
    system.ingest(snap("02_june", "2026-06-09", RATES_JUNE))
    return system


def test_current_value_beats_stale_calculator(ks):
    text = ks.answer("What's the APR for a 60-month auto loan on a 2025 model-year vehicle?").text
    assert "5.39%" in text and "5.34%" not in text


def test_as_of_uses_the_older_crawl(ks):
    text = ks.answer("What was the 60-month auto rate for a 2025 car?", as_of="2026-05-20").text
    assert "5.19%" in text and "5.39%" not in text


def test_tightest_loan_bracket(ks):
    text = ks.answer("whats the rate on a 30 yr fixed for a $90k loan").text
    assert "7.13%" in text and "7.25" not in text


def test_jumbo_vs_regular(ks):
    assert "3.90%" in ks.answer("What does the non-jumbo 10-month certificate special pay?").text
    assert "4.00%" in ks.answer("What's the 10-month jumbo special APY?").text


def test_value_check_and_missing_term(ks):
    assert ks.answer("Do any certificates pay 4.00%?").text.startswith("Yes")
    assert ks.answer("Do any certificates pay 5%?").text.startswith("No")
    refusal = ks.answer("What's the rate on a 72-month certificate special?")
    assert not refusal.confident and not any(ch.isdigit() for ch in refusal.text)


def test_history(ks):
    text = ks.answer("Has the 10-month jumbo certificate special rate changed?").text
    assert text.startswith("Yes") and "3.90%" in text and "4.00%" in text


def test_conflict_reported_with_winner(ks):
    auto = [c for c in ks.conflicts() if "auto loan" in c.topic and "60-month" in c.topic]
    assert auto, "stale calculator rate not reported"
    values = " ".join(auto[0].values)
    assert "5.34%" in values and "5.39%" in values
    assert auto[0].resolution.startswith("5.39%")
    assert "calculators" in auto[0].reason

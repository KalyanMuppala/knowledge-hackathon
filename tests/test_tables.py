from ks.tables import canonical_product, header_metric, prose_facts, table_facts

AUTO = """| Loan Program | Term | Annual Percentage Rate (As Low As) |
| --- | --- | --- |
| 2024-27 Model Year | 60 Months | 5.39% APR1 |
| 2024-27 Model Year | 84 Months | 5.59% APR1 |
| 2021-23 Model Year | 60 Months | 5.64% APR1 |"""

STALE_AUTO = """| Account Type | Balance Tier | Annual Percentage Yield |
| --- | --- | --- |
| 2023–26 Model Year | 60 Months | 5.34% APR |"""


def test_footnotes_and_model_year_rank():
    facts = table_facts(AUTO, "Auto & Motorcycle Loans", None)
    by_tier = {f.tier: f.value for f in facts}
    assert by_tier["tier1|60 month"] == "5.39%"  # "APR1" footnote stripped
    assert by_tier["tier2|60 month"] == "5.64%"
    assert {f.product for f in facts} == {"auto loan"}


def test_relabelled_stale_table_lines_up():
    # a calculator page with a mislabeled header and last year's tier label
    stale = table_facts(STALE_AUTO, "Auto Loan Rates", None)
    assert stale[0].key == ("auto loan", "tier1|60 month", "APR")


def test_metric_qualifiers_keep_promo_columns_apart():
    assert header_metric("Annual Percentage Yield (APY)1") == "APY"
    assert header_metric("APY* After Promotional Period") == "APY (after promotional period)"
    assert header_metric("Min. to Open") == "minimum to open"
    assert header_metric("*Payment Example") is None


def test_product_detection():
    assert canonical_product("Regular", "60 Month") == "certificate"
    assert (
        canonical_product("Special Certificates", "17 Month Fixed Jumbo Certificate Special")
        == "certificate special"
    )
    assert (
        canonical_product("", "Variable Interest Rate", "Business Platinum Visa") == "business platinum visa"
    )
    assert (
        canonical_product("Home Loans", "Purchase: 30-Year Fixed Conventional $100,000 or less") == "mortgage"
    )
    assert canonical_product("Regular", "60 Month", business_page=True) == "business certificate"


def test_ranges_and_minimums():
    table = """| Account Type | Balance Tier | Dividend Rate | Annual Percentage Yield (APY)1 | Min. to Open |
| --- | --- | --- | --- | --- |
| Share Savings | $25,000+ | 0.20% | 0.20%1 | $25 |"""
    facts = {f.metric: f for f in table_facts(table, "Share Savings", None)}
    assert facts["APY"].value == "0.20%" and facts["minimum to open"].value == "$25.00"
    card = "| Card Type | APR |\n| --- | --- |\n| Platinum Visa Rate | 12.50% – 19.50%2 |"
    assert table_facts(card, "Platinum Visa", None)[0].value == "12.50%-19.50%"


def test_prose_facts():
    facts = prose_facts(
        "6-month certificate APY: 3.50%\n\nLost debit card replacement fee: $2.50", "Rates", None
    )
    assert [(f.metric, f.value) for f in facts] == [("APY", "3.50%"), ("fee", "$2.50")]

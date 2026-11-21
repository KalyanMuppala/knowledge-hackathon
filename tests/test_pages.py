from datetime import date

from ks.pages import chunk_page, classify, drop_teaser_sections, published_date
from ks.query import analyze, date_in_question
from ks.textutil import clean_markdown, effective_dates, tokenize


def test_classify():
    assert classify("https://www.dupaco.com/rates", "Rates", "") == "rates"
    assert classify("https://www.dupaco.com/calculators/vehicle-affordability", "", "") == "calculator"
    assert classify("https://www.dupaco.com/newsroom/forbes-ranks-dupaco", "", "") == "news"
    assert classify("https://www.dupaco.com/makethemost-lp", "Oops ... This Offer Has Expired | Dupaco", "") == "expired"
    assert classify("https://www.dupaco.com/switch-and-earn", "", "") == "promo"


def test_dates():
    text = "2 min read\n[Emily](x)\nJune 18, 2024\n# Forbes ranks Dupaco"
    assert published_date("https://www.dupaco.com/newsroom/x", "news", text) == date(2024, 6, 18)
    assert published_date("https://www.dupaco.com/2018/09/25/deductibles/", "blog", "") == date(2018, 9, 25)
    assert effective_dates("Rates are effective as of 08/31/2026. Rates effective 1/1/26") == \
        [date(2026, 8, 31), date(2026, 1, 1)]


def test_cleaning():
    md = ("Intro ![](data:image/svg+xml,abc)\n\n- 🇺🇸 United States +1\n\nwithout notice.\\\\r\\\\n\\\\r\\\\nSavings Rates"
          "\n\n[Compare rates](https://www.dupaco.com/x?hsLang=en)")
    out = clean_markdown(md)
    assert "data:image" not in out and "United States" not in out
    assert "\\r" not in out and "Savings Rates" in out
    assert "[Compare rates](https://www.dupaco.com/x)" in out


def test_teasers_removed():
    text = "# Story\nBody.\n\n## Handpicked for You\n#### [Old story](x)\nOld teaser\n\n## FAQ\nQ?"
    out = drop_teaser_sections(text)
    assert "Old teaser" not in out and "FAQ" in out


def test_table_chunk_keeps_caption_and_effective_date():
    text = ("Special Certificates\n\n| Certificate | APY |\n| --- | --- |\n| 10 Month | 3.90% |\n\n"
            "Rates are effective as of 09/01/2026.")
    tables = [c for c in chunk_page(text) if c.is_table]
    assert tables[0].caption == "Special Certificates"
    assert tables[0].effective == [date(2026, 9, 1)]


def test_query_dates_and_intents():
    assert date_in_question("What was the rate as of May 2026?") == date(2026, 5, 31)
    assert date_in_question("What was it on June 9, 2026?") == date(2026, 6, 9)
    assert date_in_question("Is the branch still open today (October 2026)?") is None
    assert "status" in analyze("When does the Hy-Vee branch close?", None).intents
    assert "rate" in analyze("What does Launchpad Savings pay?", None).intents
    assert analyze("x", "2026-05-31").as_of == date(2026, 5, 31)


def test_tokenizer():
    assert tokenize("How many branches does Dupaco have?") == ["branch"]
    assert "5.39" in tokenize("5.39% APR")

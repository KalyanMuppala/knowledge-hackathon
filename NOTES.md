# Notes

Code is in `submission/` (`solution.py` plus the `ks` package). To run:

```
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

## Approach

I started by following a handful of facts (the 60-month auto rate, the 30-year mortgage,
the certificate specials, branch counts) through all five crawls. Nearly every wrong answer
comes from one of three places: a page that never gets updated (calculators, landing pages,
a couple of old guide pages), a value that legitimately changed between crawls, or
press-release boilerplate that ages ("...has 23 branch office locations"). So the system is
built around knowing which page to trust and when a value was true, rather than around search.

### Ingest

Pages get cleaned first: images and data URIs, tracking params, the "leaving our site"
modal, country and state pickers from forms, and the "Handpicked for You" / "Recent News"
blocks, which carry other articles' dates and snippets. Each page gets a kind from its URL
and content (rate sheet, product, info, news, blog, promo, calculator, listing, expired
offer) and is split into section chunks. A table becomes its own chunk together with its
caption and the "Rates are effective as of" line under it.

Consecutive crawls are mostly identical, so a chunk is stored once along with the snapshots
it appeared in. "The site as of date D" is the set of chunks in the last snapshot taken on or
before D. If a page drops out of a crawl it stays searchable, down-weighted. Ingesting all
five snapshots takes about 2.5 seconds.

### Facts and conflicts

Rate tables are parsed into facts keyed by `(product, tier, metric)`, for example
`("auto loan", "tier1|60 month", "APR") -> 5.39%`. Getting the key right is most of the work.
The same fact printed on two pages has to collide; things that only look alike must not:
48 vs 60 months, rate vs APR vs APY, standard vs jumbo, regular vs special, personal vs
business, Roth vs Traditional. Model-year rows are keyed by rank (newest tier first)
instead of the label, because the rate sheet went from "2023-26" to "2024-27" while the
vehicle-affordability calculator kept the old label and the old 5.34%.

For every fact and snapshot there's a winner. Pages that print a losing value are marked
stale for that snapshot, which also pushes them down in retrieval. `conflicts()` reports 30
contradictions with the values (oldest first), where each came from, the winner and why.
They include the stale calculator, switch-and-earn and guide-page tables (auto loans,
mortgages), rates that changed between crawls (certificate specials, 1st Rate Checking
3.00% to 2.50%), drifting boilerplate (branches, members, assets, counties, Forbes), and one
in-page slip: the Traditional 10-month jumbo IRA special still shows 3.90% while the Roth row
and the certificate special say 4.00%. Purchase and refinance rows, and personal, business
and IRA copies with the same history, are reported once rather than three times.

### Answering

The question picks the snapshot (`as_of`, or a date in the text like "as of May 2026"), then
BM25 retrieves from that snapshot only, weighted by page trust and article age.

Most questions are table lookups: rates, APYs, minimums, "has this changed?", plus the
routing number, headcount-type facts and products the site never prices (a 7/1 ARM, a
construction loan). Those come straight from the resolved facts, 24 of the 40 public
questions. The lookup understands the usual ways people name a tier: "five-year",
"$90k" (the tightest "$X or less" mortgage bracket), "a new car" (newest model year),
"non-jumbo", refinance vs purchase. It also handles yes/no checks ("do any savings tiers pay
0.35%?" gets "No. The highest is Share Savings ($25,000+) at 0.20% APY.") and product lists.

Hours come from the schedules printed in branch press releases. A named branch with no
published schedule gets "I don't know" instead of another branch's hours. Link questions
use the site's own links. Anything else is answered from the best-matching sentences in the
top chunks, and if nothing really covers the question, the answer is "I don't know".

An answer never repeats a stale value; if the pages disagreed, that goes in `conflict_note`.
Refusals contain no numbers. Answers are cached per question and snapshot, so asking twice
gives the same text.

## What counts as current

Applied in this order, for answers and for `conflicts()` alike:

1. Newest snapshot wins. Older snapshots are history, unless `as_of` asks for them.
2. Then page authority: the rate sheet beats product pages, which beat other site pages,
   then news, then blog and promo pages, then calculators. Expired offers never win.
3. Same kind of page: the later "Rates are effective as of" date wins.
4. Facts in articles come from the most recently published article. That's why the answer
   is 22 branches and "fifth time" for Forbes.
5. A different tier, term or metric is a different fact, not a conflict.

## API keys

None. No LLM or paid API, no network calls, standard library only (`requirements.txt` is
intentionally empty). The question shapes in this data can be answered exactly from the
parsed facts and the text, and being deterministic helps the consistency score and keeps
answers well under a tenth of a second.

## Things that bit me

- The vehicle-affordability calculator labels its APR column "Annual Percentage Yield".
  Cell text wins over the header when the two disagree.
- Promo tables have two APY columns ("for 1st 6 months" and "after promotional period"),
  so header qualifiers become part of the metric.
- The CMS leaves literal `\r\n` escapes in some text, and a naive sentence splitter breaks
  "call 800-373-7600, ext." and "South Locust St." in half.
- Certificate tables only have "Min. to Earn", not "Min. to Open".
- "Do any certificates pay 4.20%?" first came back "No", because jumbo rows were being
  penalized when the question didn't say "jumbo". Value checks now look at every tier.

## Local results

| set | questions | accuracy | stale-value hits | p95 latency |
|---|---|---|---|---|
| `eval/public_eval.jsonl` | 40 | 0.938 | 0 | 0.10 s |
| `tools/dev_eval.jsonl` (questions I wrote) | 46 | 1.00 | 0 | 0.10 s |
| `tools/dev_eval2.jsonl` (second batch) | 25 | 1.00 | 0 | 0.10 s |
| `harness/selftest` with its conflicts gold | 5 | 1.00, conflicts F1 1.00 | 0 | - |

Paraphrase and repeat consistency are 1.0 on the public set. The two public misses:
`dup-005` has no `must_include`, so it can't score without the scorer's judge, and `dup-081`
(who can join) needs the charter area and the three states from two different pages, which
sentence extraction can't combine. I wrote the two extra question files against the
snapshots (paraphrases, other products and tiers, as-of dates, history, refusals) to check
the rules weren't just fitted to the public set.

`pytest` runs without the snapshots. It covers parsing, classification, dates, the
self-test fixture end to end, and a small made-up two-crawl site with a stale calculator.
Formatting and linting are ruff (`pyproject.toml`).

## Limitations

- The product vocabulary (`PRODUCT_PATTERNS` in `ks/rules.py`, `_family` in `ks/tables.py`)
  is specific to this credit union.
- The conflict list is every real contradiction I found. If the graders' list is shorter,
  the older auto-loan tiers and the asset/membership drift are what I'd drop first.
- Retrieval is BM25 with a synonym table; embeddings would catch paraphrases the table misses.
- Open-ended answers are extracted sentences, so they can't merge facts from two pages. With
  an API budget, that's where I'd put a model: phrasing those answers from the same
  annotated chunks, while table lookups, counts and refusals stay deterministic.

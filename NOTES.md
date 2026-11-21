# Notes

The solution lives in `submission/` (`solution.py` + the `ks` package). Run it with

```
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

## How it works

**Ingest.** Every page is cleaned (images, data URIs, tracking params, modal boilerplate,
country/state pickers from forms, and the "Handpicked for You" teaser blocks that carry
*other* articles' dates). Each page gets a kind from its URL and content: rate sheet,
product page, info page, news, blog, promo landing page, calculator, listing or expired
offer. Pages are split into section chunks; every table becomes its own chunk together
with its caption and the "Rates are effective as of ..." line under it. Consecutive crawls
are mostly identical, so a chunk is stored once with the list of snapshots it appeared in.
"The site as of date D" is then just the chunks present in the latest snapshot taken on or
before D. A page that disappears from a crawl is still searchable but flagged as "last seen
on ..." and down-weighted. Ingesting all five snapshots takes about 2.5 seconds.

**Facts.** Rate tables are parsed into facts keyed by `(product, tier, metric)`, e.g.
`("auto loan", "tier1|60 month", "APR") -> 5.39%`. The key is built so the same fact printed
on different pages collides, while things that only look alike don't: 48 vs 60 months, rate
vs APR vs APY, standard vs jumbo, regular vs special, personal vs business (rates on
`/business` pages are business rates), Roth vs Traditional. Model-year rows are keyed by
rank (newest tier = `tier1`) instead of the literal label. When the rate sheet moved from
"2023-26" to "2024-27", the vehicle-affordability calculator kept the old label and the old
5.34%; keying by rank lines it up with the new 5.39% row so it shows up as stale.

**Answering.** Pick the snapshot for `as_of` (or a date written in the question, like "as of
May 2026"), retrieve with BM25 restricted to that snapshot, weighted by page trust and
article age. Then:

- Table lookups (rates, APYs, minimums), the routing number, headcount-type facts
  (branches, members, counties...) and questions about products the site never prices
  (7/1 ARM, construction loans) are answered straight from the resolved facts. That's 24 of
  the 40 public questions; it's exact, takes milliseconds and gives the same answer every time.
- Everything else (hours, links, branch closures, eligibility, "how do I..." questions) goes
  to an LLM with the top chunks. Each chunk is labeled with its page kind, publication or
  effective date and an explicit `OUTDATED` note if it prints a value the official page
  contradicts. The prompt also gets the matching rate-sheet rows, the latest press-release
  figures and branch schedules as hints.
- Without an API key (or if the call fails), a rule-based fallback answers instead: hours
  summaries from branch press releases, link lookup, and best-sentence extraction.

Answers never mention the stale value. If the site disagreed, that goes in `conflict_note`.
Refusals ("I don't know, that isn't listed") contain no numbers, so they can't accidentally
quote a rate. Answers are cached per (question, snapshot), so asking twice gives the same text.

## How I decide what's current

These are applied in order, both for answers and for `conflicts()`:

1. **Time.** For "now" only the newest snapshot counts; older snapshots are history. With
   `as_of`, the newest snapshot on or before that date is "now".
2. **Authority.** Official rate sheet (`/rates`, `/business-rates`) > product page > other
   site pages > news > blog / promo pages > calculators. Expired-offer pages never win.
   Calculators, landing pages and old guide pages (`/homebuying-guidance`, `/mortgage-refi`,
   `/switch-and-earn`) are where most stale numbers live.
3. **Freshness.** Between pages of the same kind, the later "Rates are effective as of"
   date wins.
4. **Articles.** Boilerplate that drifts as press releases age ("has 23 branch office
   locations", "the fourth time Forbes recognized Dupaco") is taken from the most recently
   published article. That gives 22 branches and "fifth time" for Forbes.
5. **Qualifiers are not conflicts.** Different tiers, terms or metrics of the same product
   are separate facts.

`conflicts()` reports 30 contradictions. Each has the competing values (oldest first), the
pages and snapshots they came from, the winning value and the reason. Examples: stale
calculator, switch-and-earn and old guide-page tables vs the rate sheet (auto loans,
mortgages); rate changes between crawls (certificate specials, 1st Rate Checking 3.00% ->
2.50%); press-release boilerplate (branches, members, assets, counties, Forbes); and one
in-page inconsistency (Traditional 10-month jumbo IRA special still at 3.90% while the Roth
row and the certificate special say 4.00%). Purchase/refinance pairs and personal/business/IRA
copies with identical histories are merged so the same story isn't reported several times.

## API keys

- `ANTHROPIC_API_KEY` (paid Anthropic API) enables the LLM step for open-ended questions.
  The default model is `claude-sonnet-5-5`, with thinking off and low effort to keep latency
  down. `KS_MODEL` overrides it (e.g. `claude-opus-5-5` if latency matters less). If the API
  rejects an option for a given model, the client retries without it.
- No key: everything still runs on the rule-based path. Nothing else is needed, and the only
  third-party package is `anthropic`.

## Local results

Rule-based path only (no key), `harness/run_eval.py`:

| set | questions | accuracy | stale-value hits | p95 latency |
|---|---|---|---|---|
| `eval/public_eval.jsonl` | 40 | 0.915 | 0 | 0.18 s |
| `tools/dev_eval.jsonl` (my held-out questions) | 46 | 1.00 | 0 | 0.18 s |
| `tools/dev_eval2.jsonl` (second held-out batch) | 25 | 1.00 | 0 | 0.18 s |
| `harness/selftest` (+ conflicts gold) | 5 | 1.00, conflicts F1 1.00 | 0 | - |

One public item (`dup-005`) has no `must_include`, so it scores 0 for everyone unless the
LLM judge is on. The two held-out files are questions I wrote against the snapshots
(paraphrases, other products and tiers, as-of dates, history and refusals) to check the
rules weren't just fitted to the public set. `tests/` has unit tests for parsing,
classification, dates and the LLM wrapper, plus an end-to-end run on the self-test fixture
(`pytest tests`).

## Limitations / next steps

- Product vocabulary (`ks/rules.py` `PRODUCT_PATTERNS`, `ks/tables.py` `_family`) is written
  for this credit union. Pointing it at another site means extending those lists.
- The conflict list is every genuine contradiction I could find. If the grader's list is
  shorter, the older auto-loan tiers and the asset/membership drift are the first things
  I'd drop to raise precision.
- Retrieval is plain BM25 with a synonym table. Embeddings would help paraphrases the
  synonym table doesn't cover, at the cost of a heavier install.

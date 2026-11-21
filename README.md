# Knowledge hackathon: what's true *now*

A question-answering system over five crawls of a credit union's website (May to Sep
2026). It answers with the current value (or the value as of a given date), picks the right
tier, cites the page, says "I don't know" when the site doesn't say, and lists every place
the site contradicts itself.

Design notes, the rules for deciding what's current, and local results are in
[NOTES.md](NOTES.md).

## Running

Python 3.9+ (the graders use 3.11). Standard library only, no API keys.

```bash
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

The website snapshots are not in this repo (they were shared for the event only). Copy the
kit's `snapshots_zip/` folder into the repo root; the harness unpacks it into `snapshots/`
on the first run.

Tests don't need the snapshots:

```bash
pip install pytest
pytest tests
```

## Layout

```
submission/solution.py   build() -> ingest / answer / conflicts
submission/ks/
  pages.py       cleaning, page kinds, article dates, chunking
  tables.py      rate tables -> (product, tier, metric) facts
  store.py       chunks and facts with the snapshots they appeared in
  search.py      BM25 limited to one snapshot, weighted by page trust and age
  conflicts.py   per-snapshot winners, stale pages, contradictions
  rules.py       rate lookup, counts, hours, addresses, links, refusals
  answering.py   retrieval for one question + picking the answer
  query.py       as-of dates in the question, intents
tools/           dev scripts and my held-out question sets
tests/           unit tests + end-to-end run on harness/selftest
harness/, eval/, starter/   from the hackathon kit, unchanged
```

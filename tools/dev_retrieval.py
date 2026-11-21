"""Show what retrieval and the fast path do for each question in an eval file.

python tools/dev_retrieval.py eval/public_eval.jsonl [question ids...]
"""

import json
import sys

from dev_load import build_system
from ks.query import analyze

path = sys.argv[1] if len(sys.argv) > 1 else "eval/public_eval.jsonl"
only = set(sys.argv[2:])
ks = build_system()
for line in open(path):
    if not line.strip():
        continue
    it = json.loads(line)
    if only and it["id"] not in only:
        continue
    qi = analyze(it["question"], it.get("as_of"))
    snap = ks.store.view_for(qi.as_of)
    ev = ks.answerer.gather(qi, snap)
    fast = ks.answerer.fast_answer(qi, ev)
    print(f"\n{it['id']}  {it['question']}  (snapshot {snap}, coverage {ev.coverage:.2f})")
    print("   fast path:", fast["text"][:150] if fast else "-")
    for ch, score in zip(ev.chunks, ev.scores):
        print(f"   {score:6.2f} {ch.kind:10s} {ch.url[:80]}")

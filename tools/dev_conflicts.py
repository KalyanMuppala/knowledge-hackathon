"""Print what conflicts() returns.   python tools/dev_conflicts.py [-v]"""

import sys

from dev_load import build_system

ks = build_system()
recs = ks.engine.all_conflicts()
print(len(recs), "conflicts")
for r in recs:
    print(f"[{r.kind:17s}] {r.topic[:80]:80s} {r.values} -> {r.resolution}")
    if "-v" in sys.argv:
        print("    reason:", r.reason)
        for url, snap, quote in r.sources:
            print("    source:", url, ks.store.snapshots[snap].snapshot_id, "|", quote[:100])

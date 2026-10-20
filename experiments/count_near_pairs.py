"""E21: (parent, child) pairs that differ in at most K coefficients, mined from a heights database.

    uv run python experiments/count_near_pairs.py <heights.sqlite> <p> <K> <out.npz> [max_per_parent]

The scored table is read from the database once and cached next to the output (`scored_p<p>.npz`).
"""

from pathlib import Path
import sys
import time

import numpy as np

from rlmath.pairs import NearPairs, Scored, find_near_pairs, load_scored

db, p, k, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cap = int(sys.argv[5]) if len(sys.argv) > 5 else 0
cache = Path(out).parent / f"scored_p{p}.npz"
t = time.time()
if cache.exists():
    z = np.load(cache)
    data = Scored(z["digits"], z["scores"])
else:
    data = load_scored(db, p)
    np.savez(cache, digits=data.digits, scores=data.scores)
print(f"{len(data.digits):,} polynomials ({time.time() - t:.0f}s); by score:",
      {int(s): int((data.scores == s).sum()) for s in np.unique(data.scores)}, flush=True)

t = time.time()
pairs = find_near_pairs(data, p, k, min_parent=2, require_ge=True, max_per_parent=cap, log=lambda s: print(s, flush=True))
print(f"{len(pairs.parent):,} pairs with 1 <= distance <= {k}, child >= parent, per-parent cap {cap or 'none'} ({time.time() - t:.0f}s)")
print("by distance:", {int(d): int((pairs.dist == d).sum()) for d in range(1, k + 1)})
print("parents with a pair:", len(np.unique(pairs.parent)))
ps, cs = pairs.parent_score, pairs.child_score
for h in range(2, int(ps.max()) + 1):
    m = ps == h
    if m.any():
        print(f"parent {h}: {int(m.sum()):,} pairs, {len(np.unique(pairs.parent[m])):,} parents, child > parent {int((cs[m] > h).sum()):,}")
pairs.save(out)
print("saved", out)

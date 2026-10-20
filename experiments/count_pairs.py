"""E0: how many usable (parent, one-coefficient change) pairs does a heights database hold?

    uv run python experiments/count_pairs.py <heights.sqlite> <p> <out.npz>
"""

import sys
import time

import numpy as np

from rlmath.pairs import find_pairs, load_scored

db, p, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t = time.time()
data = load_scored(db, p)
print(f"{len(data.digits):,} distinct polynomials with a finite height ({time.time() - t:.0f}s)")
print("by score:", {int(s): int((data.scores == s).sum()) for s in np.unique(data.scores)})

t = time.time()
pairs = find_pairs(data, p, min_parent=2)
print(f"{len(pairs.parent):,} (parent, child) pairs with parent score >= 2 and a known child ({time.time() - t:.0f}s)")

ps, cs = pairs.parent_score, pairs.child_score
print(f"\n{'parent':>6} {'parents':>10} {'w/ pair':>9} {'pairs':>10} {'child>':>9} {'child=':>9} {'child<':>10} {'>=,%':>6} "
      f"{'parents w/ >= pair':>19}")
for h in range(2, int(ps.max()) + 1):
    all_h = int((data.scores == h).sum())
    m = ps == h
    if not m.any():
        continue
    up, same, down = int((cs[m] > h).sum()), int((cs[m] == h).sum()), int((cs[m] < h).sum())
    with_pair = len(np.unique(pairs.parent[m]))
    good = len(np.unique(pairs.parent[m & (cs >= ps)]))
    print(f"{h:6d} {all_h:10,} {with_pair:9,} {int(m.sum()):10,} {up:9,} {same:9,} {down:10,} {100 * (up + same) / m.sum():6.2f} {good:19,}")
tot = int((cs >= ps).sum())
print(f"\ntotal pairs with child >= parent: {tot:,}  (child > parent: {int((cs > ps).sum()):,})")
pairs.save(out)
print("saved", out)

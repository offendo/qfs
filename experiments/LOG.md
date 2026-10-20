# Experiment log

A running notebook: one entry per experiment, in the order they were run. Each entry says what was changed, why,
the command, and what came out. The summary of everything is in `../experiments-notes.md`.

Common setup unless stated otherwise: p = 3 quartics (35 coefficients), the 12.1M-row p=3 part of the user's
`runs/heights.sqlite` (read-only snapshot, copied per run so no run writes into the user's file), GPU 1 (H200),
branch `worktree-policy-gradient`. "Ceiling c" means pool parents of height at most c, sampled with
P(height level) proportional to exp(height / 3), as in training.

Research choices made on my own authority (the user allowed this for the night) are marked **[choice]**.

---

## E0. How many one-coefficient (parent, child) pairs does the database hold?

**Why.** Before fine-tuning on stored edits we need to know whether there are enough examples, and at which parent
heights. The user's objection to supervised learning was that data may not exist for larger p, n, d, so this also
tells how much we depend on the database.

**What.** New module `src/rlmath/pairs.py`: encodes every polynomial as a base-p integer, then finds, with vectorized
arithmetic and a binary search, every (parent, child) such that the child differs from the parent in exactly one
coefficient and is also in the database with a known score (a polynomial of height >= 2 only counts if its
smoothness was checked; singular ones score 0, as in the environment). The database stores polynomials scaled to start
with a 1, so each child is looked up under every scalar multiple. Tested against a brute force (`tests/test_pairs.py`).
Script: `experiments/count_pairs.py`.

```
uv run python experiments/count_pairs.py p3_snapshot.sqlite 3 pairs_p3.npz
```

**Result.** 12,109,603 distinct polynomials with a finite height (6.9M of height 1, 2.1M singular, 3.1M smooth of
height >= 2). 2,542,921 pairs with a parent of score >= 2. Of those, 1,371,669 have child >= parent and 467,375 have
child > parent.

| parent height | parents | pairs | child > | child = | child < | % child >= | parents with a good pair |
|---|---|---|---|---|---|---|---|
| 2 | 2,098,286 | 1,677,380 | 412,558 | 802,984 | 461,838 | 72.5 | 926,344 |
| 3 | 698,940 | 562,035 | 48,037 | 89,476 | 424,522 | 24.5 | 124,259 |
| 4 | 232,577 | 190,226 | 5,874 | 10,342 | 174,010 | 8.5 | 15,614 |
| 5 | 76,703 | 66,650 | 746 | 1,264 | 64,640 | 3.0 | 1,985 |
| 6 | 25,868 | 25,658 | 142 | 196 | 25,320 | 1.3 | 333 |
| 7 | 8,442 | 10,581 | 17 | 22 | 10,542 | 0.37 | 39 |
| 8 | 2,800 | 5,176 | 1 | 8 | 5,167 | 0.17 | 9 |
| 9 | 912 | 3,181 | 0 | 2 | 3,179 | 0.06 | 2 |
| 10 | 196 | 2,034 | 0 | 0 | 2,034 | 0.00 | 0 |

**Takeaways.**
- Plenty of positive examples overall (1.37M), but 89% of them are at parent height 2. Parents of height >= 4 have
  only about 18.6k positives, height >= 6 only a few hundred, height >= 8 about ten. Fine-tuning on this data will
  mostly teach low-height edits.
- The share of single-coefficient changes that keep or raise the height falls by roughly 3x per height level
  (72% -> 24% -> 8.5% -> 3.0% -> 1.3%), matching the "each level costs a factor of p" picture.
- The pairs come from the earlier search's local-search neighbourhoods, so they are all single-coefficient changes
  and are biased toward whatever that search explored.

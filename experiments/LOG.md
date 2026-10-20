# Experiment log

A running notebook: one entry per experiment, in the order they were run. Each entry says what was changed, why,
the command, and what came out. The summary of everything is in `../experiments-notes.md`.

Common setup unless stated otherwise: p = 3 quartics (35 coefficients), the 12.1M-row p=3 part of the user's
`runs/heights.sqlite` (read-only snapshot, copied per run so no run writes into the user's file), GPU 1 (H200),
branch `worktree-policy-gradient`. "Ceiling c" means pool parents of height at most c, sampled with
P(height level) proportional to exp(height / 3), as in training.

Research choices made on my own authority (the user allowed this for the night) are marked **[choice]**.

> **Erratum (found while analysing evaluation E).** Two conclusions I first wrote in E9/E10 and E11 were wrong and are
> corrected below. The evaluation redraws children that equal their parent, but only up to a cap (20 times at first).
> A policy that copies its parent most of the time leaves some exact copies after the cap, and an exact copy trivially
> "keeps the height". That inflated the **retention at high parents** of `r10_free3_s3` (5.4% copies), `r7_free1` (1.1%),
> `r9_free3_s2` (1.1%), `r11` (0.9%) and `r12_free2` (0.8%); it does not affect the other policies (about 0%) and barely
> affects overall improvement (a copy counts as 0). The tell was that `r10`'s retention was flat at about 5.5% for
> every parent height from 4 to 8, whereas real retention decays about 3x per level. The evaluation now redraws up to 100
> times and reports retention with exact copies excluded (`retain_nocopy`); evaluation F redoes the comparison that way.

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

---

## E1. Random-edit baselines and a common evaluation harness

**Why.** Earlier comparisons used baselines measured on different parents, which made them hard to trust, and a policy
that copies its parent gets a free "same" outcome. I wanted every method scored on exactly the same parents, with
copies not counting for free.

**What.** `experiments/evaluate.py`: one environment with the parent pool frozen (no new polynomials enter it) and a
fixed seed per ceiling, so every method sees identical parents. Each cell is 40 batches x 1024 = 40,960 children, one
per parent. **[choice]** Children that equal their parent are redrawn (up to 20 times) for the policies, since the
training environment penalizes copies (-100) and a policy that copies 35% of the time would otherwise look better than
it is. Random baselines change k = 1, 2 or 3 randomly chosen coefficients to a different value. Metrics use
`child score - parent score`, with no penalties. `experiments/tabulate.py` prints the markdown tables below.

**Result** (full tables are in E2-E5; random rows only):

| method | ceiling 2 | ceiling 3 | ceiling 4 | ceiling 5 | ceiling 6 |
|---|---|---|---|---|---|
| random-1 | -0.549 | -1.129 | -1.772 | -2.467 | -3.195 |
| random-2 | -0.716 | -1.302 | -1.940 | -2.640 | -3.367 |
| random-3 | -0.795 | -1.374 | -2.011 | -2.714 | -3.438 |

**Takeaways.**
- The child's mean score under a random edit is the same whatever the parent: 1.45 / 1.28 / 1.21 for k = 1 / 2 / 3, at
  every ceiling. Even a single random coefficient change gives a child that looks like a generic polynomial, so
  nothing of the parent's height survives (93% of children are worse at ceiling 6).
- Fewer changes is better: k = 1 beats k = 2 by about 0.17 and k = 2 beats k = 3 by about 0.08 at every ceiling.
- Mean parent score at each ceiling (pool parents, weighted toward high levels with temperature 3): 2.00, 2.58, 3.22,
  3.92, 4.64.

---

## E2. Supervised warm start (SFT) on stored one-coefficient edits

**Why.** The user asked whether we can bootstrap by supervised fine-tuning. Reinforcement learning from scratch has a
sparse reward and has to learn to copy 30+ coefficients exactly; stored (parent, child) pairs give a dense target.

**What.** New `src/rlmath/sft.py` (`rlmath-sft`): trains the same transformer to maximize log pi(child | parent) on the
pairs from E0. Held-out parents (2%), random variable-renaming augmentation (same renaming for parent and child).
Variants **[choices, in the user's words "make research direction choices as long as you document them"]**:

| id | pairs used | why | pairs | epochs | model |
|---|---|---|---|---|---|
| S1 `s1_ge_all` | child >= parent, parent >= 2 | the natural "edits that did not hurt" target | 1,371,669 | 8 | 128 wide, 4 layers |
| S2 `s2_gt_all` | child > parent only | teach improvement only | 467,375 | 8 | same |
| S3 `s3_ge_h3plus` | child >= parent, parent >= 3 | 89% of S1 is height-2 parents; focus on harder parents | 156,127 | 30 | same |
| S4 `s4_ge_all_big` | as S1 | does capacity matter for SFT? | 1,371,669 | 8 | 256 wide, 8 layers |

```
uv run python experiments/run_sft_variants.py pairs_p3.npz sft/     # CUDA_VISIBLE_DEVICES=1
```

**SFT results (held-out negative log-likelihood per pair, nats).** S1 4.151, S2 4.180, S3 4.171, S4 4.125.
References: choosing one of the 70 single changes uniformly costs ln 70 = 4.25; the best model that ignores the
parent and treats the 35 coefficients independently costs 5.23. So the models do use the parent, but not by much. All
four give probability 0.646-0.647 that at least one coefficient changes, i.e. they copy their parent 35% of the time
(the factorized output cannot express "exactly one coefficient changes").

**Evaluation (identical parents, copies redrawn).** Improvement (child score - parent score; higher is better):

| method | ceiling 2 | ceiling 3 | ceiling 4 | ceiling 5 | ceiling 6 | mean Hamming |
|---|---|---|---|---|---|---|
| random-1 | -0.549 | -1.129 | -1.772 | -2.467 | -3.195 | 1.00 |
| random-2 | -0.716 | -1.302 | -1.940 | -2.640 | -3.367 | 2.00 |
| S1 | -0.362 | -0.925 | -1.579 | -2.263 | -2.994 | 1.54 |
| S2 | -0.349 | -0.925 | -1.568 | -2.270 | -2.994 | 1.55 |
| S3 | -0.355 | -0.935 | -1.572 | -2.268 | -3.001 | 1.55 |

(S4 is evaluated in the second batch, below.)

**Takeaways.**
- Every SFT variant beats random single and double changes at every ceiling, by about 0.19 over random-1 and 0.35-0.38
  over random-2 (improvement), and the share of children at least as high as their parent is about 1.5x that of random-1
  (e.g. 65.7% vs 43.9% at ceiling 2, 11.0% vs 7.2% at ceiling 6).
- **S1, S2 and S3 are indistinguishable** (within about 0.01). Which pairs we imitate - keep-or-improve, strict
  improvement, or only harder parents - does not matter. That suggests what is learned is generic ("a small change that
  does less damage"), not height-specific.
- The children's mean score is about 1.65 at every ceiling (parent-independent, like the random edits' 1.45). The gain
  over random is therefore a roughly constant lift of the child, not preserved parent structure.

---

## E3. Reference: RL from scratch with the curriculum (the previous session's best run)

**Why.** A fair comparison for the SFT and RL-from-SFT runs needs the earlier from-scratch policy scored on the same
parents. This run existed already: 8 parents x 128 children per iteration, 100% pool parents, curriculum from ceiling
2, free Hamming budget 8, penalty 0.5, copy penalty 100, 1.5M children, seed 1 (`rf_cur`, run before tonight).

**Result.** Improvement `scratch`: -0.455 / -1.032 / -1.674 / -2.383 / -3.086 at ceilings 2-6, mean Hamming 1.75, mean
child 1.55. Better than random-1 (by about 0.09-0.11) and random-2 (about 0.26-0.28), worse than every SFT variant at
every ceiling (SFT is better by about 0.09-0.10, with no reinforcement learning at all).

---

## E4. RL from scratch with a free Hamming budget of 3 (`r1_free3`)

**Why.** The earlier run had a free budget of 8 and ended with Hamming distance 1.8, but its early iterations used
edits of size about 6, which a random 2-change beats. A smaller free budget should push the policy toward small edits
sooner. Only `--hamming-free 3` differs from E3.

```
uv run python experiments/run_rl.py r1_free3 p3_snapshot.sqlite rl/ --total-timesteps 1500000 --hamming-free 3
```

**Result.** Improvement -0.293 / -0.891 / -1.523 / -2.221 / -2.942 at ceilings 2-6; mean Hamming 1.15; mean child 1.70.
**The best of this first batch at every ceiling**, ahead of S1 by 0.05-0.07 and ahead of random-1 by about 0.25
(ceiling 2: -0.293 vs -0.549; ceiling 6: -2.942 vs -3.195). Retention (child >= parent): 68.0% vs 43.9% at ceiling 2,
11.4% vs 7.2% at ceiling 6.

**Takeaway.** A tighter edit budget helps even though the penalty never binds (the policy's distance is already below
3): the budget changes the early trajectory. One seed.

---

## E5. RL from scratch with 32 parents x 32 children (`r2_groups32`)

**Why.** The KL spikes suggested that 8 distinct parents per update lets a step fit those parents too closely; more
parents with a smaller group might generalize better. Only `--children-per-parent 32` differs from E3 (same 1024
children per iteration, so 32 parents).

**Result.** Improvement -0.558 / -1.146 / -1.778 / -2.473 / -3.195: **indistinguishable from random-1 and worse than
E3**. Mean Hamming 3.25 (it never learned small edits), mean child 1.44.

**Takeaway.** With 1.5M children, the smaller group (32 vs 128) did not learn. The larger group baseline appears to
matter more than parent diversity. One seed.

---

## Second evaluation (B): SFT bigger model, RL from the SFT init, free Hamming budgets 1 and 3

All of E6-E10 below are scored in one evaluation on identical parents (same protocol as E1; the numbers for the methods
that were also in the first evaluation agree to within about 0.01-0.02, which gives a feeling for the evaluation noise:
`scratch` -0.454 vs -0.455 at ceiling 2, `s1_ge_all` -0.342 vs -0.362).

Improvement (child score - parent score, higher is better):

| method | ceiling 2 | ceiling 3 | ceiling 4 | ceiling 5 | ceiling 6 | mean Hamming |
|---|---|---|---|---|---|---|
| random-1 | -0.549 | -1.129 | -1.772 | -2.467 | -3.195 | 1.00 |
| random-2 | -0.716 | -1.302 | -1.940 | -2.640 | -3.367 | 2.00 |
| scratch (E3) | -0.454 | -1.028 | -1.674 | -2.380 | -3.083 | 1.75 |
| S1 | -0.342 | -0.929 | -1.561 | -2.262 | -2.998 | 1.55 |
| S4 (bigger SFT model) | -0.338 | -0.916 | -1.557 | -2.274 | -2.990 | 1.55 |
| r1_free3 (E4) | -0.291 | -0.891 | -1.523 | -2.221 | -2.941 | 1.15 |
| r3_s1_kl0 (E6) | -0.480 | -1.056 | -1.701 | -2.394 | -3.113 | 4.51 |
| r5_s1_lr1e4 (E6) | -0.487 | -1.065 | -1.692 | -2.403 | -3.123 | 4.51 |
| r4_s1_kl01 (E7) | -0.350 | -0.941 | -1.578 | -2.282 | -3.004 | 1.66 |
| r6_s1_free3 (E8) | -0.298 | -0.897 | -1.517 | -2.217 | -2.942 | 1.98 |
| r7_free1 (E9) | -0.355 | -0.930 | -1.554 | -2.247 | -2.966 | 1.00 |
| r8_s1_free1 (E10) | -0.297 | -0.879 | -1.512 | -2.218 | -2.932 | 1.04 |

Share of children that are at least as high as their parent, by exact parent height (the ceiling-6 pool; each row is
the same parents). This is the number that tells whether a policy preserves the structure of high-height parents:

| method | parent 2 | parent 3 | parent 4 | parent 5 | parent 6 |
|---|---|---|---|---|---|
| random-1 | 42.4 | 14.1 | 5.1 | 1.4 | 0.5 |
| random-2 | 29.6 | 9.8 | 3.4 | 1.1 | 0.4 |
| scratch | 55.8 | 19.5 | 6.5 | 2.1 | 0.7 |
| S1 | 66.6 | 22.8 | 7.1 | 2.5 | 0.8 |
| S4 | 67.1 | 22.5 | 7.7 | 2.3 | 0.8 |
| r1_free3 | 67.6 | 22.8 | 7.5 | 2.5 | 0.7 |
| r3_s1_kl0 | 60.2 | 19.8 | 6.6 | 2.5 | 0.8 |
| r4_s1_kl01 | 66.4 | 21.7 | 7.3 | 2.7 | 0.9 |
| r5_s1_lr1e4 | 61.6 | 19.8 | 6.8 | 2.3 | 0.7 |
| r6_s1_free3 | 67.4 | 22.5 | 7.9 | 2.9 | 1.0 |
| r7_free1 | 64.9 | 21.8 | 8.0 | **3.7** | **1.8** |
| r8_s1_free1 | 66.1 | 22.9 | 7.4 | 3.0 | 1.1 |

(parents per height: 3790, 5248, 7219, 10237, 14466.)

---

### E6. RL from the SFT init, default settings (`r3_s1_kl0`, `r5_s1_lr1e4`)

**Why.** The user's question was whether SFT can *bootstrap* RL: start the policy gradient from the SFT policy.
**What.** `rlmath-reinforce --init sft/s1_ge_all.pt` (new `--init` option; model size must match), otherwise the E3
settings (free Hamming budget 8). `r3`: learning rate 3e-4. `r5`: 1e-4, to protect the init. Both 1.5M children.

**Result.** Both are *worse than the SFT policy they started from* at every ceiling (ceiling 2: -0.480 and -0.487 vs
-0.342) and even worse than `scratch`; mean Hamming distance grew from 1.55 to 4.51. A lower learning rate did not
help (it only slowed the same drift).

**Takeaway.** With a free Hamming budget of 8 there is no pressure against larger edits (they cost nothing up to 8
changed coefficients), and the policy gradient moves toward them even though they are worse on average. The SFT policy
is a good starting point that RL with this budget actively damages. (Explains why the earlier from-scratch runs had
Hamming ~1.75: they were pulled down from ~6 by the group baseline, not up.)

### E7. RL from the SFT init with a KL tether (`r4_s1_kl01`)

**Why.** Keep the RL policy near the SFT policy. **What.** E6 plus `--kl-coef 0.1` (new option): penalty
0.1 * KL(policy || SFT policy), summed over coefficients. **[choice]** 0.1 because per-step policy changes were 0.01-1 nats.

**Result.** Mean Hamming 1.66 and improvement within 0.01 of S1 at every ceiling (-0.350 vs -0.342 at ceiling 2,
-3.004 vs -2.998 at ceiling 6). The tether stops the damage from E6 but gives **no gain over the SFT policy**.

### E8. SFT init + free Hamming budget 3 (`r6_s1_free3`)

**Why.** E4 showed a budget of 3 helps from scratch; E6 showed a budget of 8 hurts from the SFT init. Combine.
**Result.** Improvement -0.298 / -0.897 / -1.517 / -2.217 / -2.942, essentially equal to E4 (`r1_free3`, from scratch)
and slightly better than S1 (by 0.04-0.06). Mean Hamming 1.98. Keeping the parent's height at heights 4-6: 7.9% / 2.9% /
1.0% vs 7.1% / 2.5% / 0.8% for S1.

### E9. RL from scratch with a free budget of 1 (`r7_free1`), and E10: SFT init + budget 1 (`r8_s1_free1`)

**Why.** Following the trend that fewer free changes is better; with a budget of 1 the policy is pushed to exactly one
change (each extra changed coefficient costs 0.5). **Result.** Mean Hamming 1.00 (`r7`) and 1.04 (`r8`).
- Overall improvement: `r8` is the best method in this evaluation at ceilings 3-6 (-0.879 / -1.512 / -2.218 / -2.932),
  within 0.01-0.02 of `r1_free3`; `r7` (from scratch) is a little worse at low ceilings (-0.355 at ceiling 2).
- ~~High parents: `r7_free1` is clearly the best at parent heights 5 and 6~~ **(withdrawn, see the erratum at the top.)**
  The table above shows 3.7% (parent 5) and 1.8% (parent 6) of children keeping the height for `r7`, against 1.4% / 0.5%
  for a random single change, but 1.1% of `r7`'s children were exact copies of the parent after the redraw cap, so most
  of its advantage at parent 6 is copies. The corrected numbers are in evaluation F.

**Takeaway.** Overall, a budget of 1 gives the same band of results as budgets 2-3 (within about 0.04 in improvement).
I originally read the retention table as the first sign of parent-specific structure; that reading is withdrawn.

### S4. Bigger SFT model (256 wide, 8 layers; evaluated here)

Held-out NLL 4.125 vs 4.151 for the small model; evaluation within 0.01 of S1 at every ceiling (e.g. -0.338 vs -0.342).
**Capacity does not matter for SFT** at this data size, consistent with the earlier offline learnability result.

### Observation across E1-E10: children do not depend on the parent's height

Every method's mean child score is flat across ceilings (e.g. S1 1.65-1.66; `r1_free3` 1.69-1.71; random-1 exactly 1.45),
although the mean parent score goes from 2.0 to 4.6. The policies mostly lift a parent-independent baseline (child
score about 1.7 instead of 1.45) and change as little as possible. Only `r7_free1` shows a hint of dependence (child
1.65 at ceiling 2, 1.68 at ceiling 6; retention at parent 6 of 1.8%).

---

## Third evaluation (C): seeds, a longer run, and a free budget of 2

Same protocol and parents as before (S1 and `r1_free3` appear again as references: S1 -0.363 / -2.984 vs -0.342 / -2.998
in B, `r1_free3` -0.291 / -2.941 vs -0.291 / -2.941, so the evaluation is stable to about 0.02).

Improvement (child score - parent score):

| method | ceiling 2 | ceiling 3 | ceiling 4 | ceiling 5 | ceiling 6 | mean Hamming |
|---|---|---|---|---|---|---|
| random-1 | -0.549 | -1.129 | -1.772 | -2.467 | -3.195 | 1.00 |
| random-2 | -0.716 | -1.302 | -1.940 | -2.640 | -3.367 | 2.00 |
| S1 | -0.363 | -0.934 | -1.569 | -2.267 | -2.984 | 1.55 |
| r1_free3 (seed 1) | -0.291 | -0.892 | -1.524 | -2.222 | -2.941 | 1.15 |
| r9_free3_s2 (seed 2) | -0.334 | -0.911 | -1.551 | -2.216 | -2.945 | 2.09 |
| r10_free3_s3 (seed 3) | -0.713 | -1.262 | -1.865 | -2.521 | -3.214 | 1.96 |
| r11_free3_long (5M children) | -0.417 | -1.006 | -1.633 | -2.340 | -3.053 | 1.39 |
| r12_free2 | -0.329 | -0.895 | -1.536 | -2.241 | -2.954 | 1.05 |

Share of children at least as high as their parent, by exact parent height (ceiling-6 pool):

| method | parent 2 | parent 3 | parent 4 | parent 5 | parent 6 |
|---|---|---|---|---|---|
| random-1 | 42.4 | 14.1 | 5.1 | 1.4 | 0.5 |
| S1 | 66.8 | 22.5 | 7.7 | 2.4 | 0.8 |
| r1_free3 | 67.5 | 22.8 | 7.6 | 2.5 | 0.7 |
| r9_free3_s2 | 67.5 | 22.3 | 8.8 | 3.4 | 1.9 |
| **r10_free3_s3** | 28.2 | 12.4 | 7.9 | **6.1** | **5.7** |
| r11_free3_long | 56.3 | 18.9 | 7.5 | 2.9 | 1.5 |
| r12_free2 | 65.7 | 22.4 | 8.3 | 3.3 | 1.6 |

### E11. Seeds 2 and 3 of the free-budget-3 run (`r9_free3_s2`, `r10_free3_s3`)

**Why.** Every claim so far came from one seed. **What.** Exactly `r1_free3` (E4) with `--seed 2` and `--seed 3`.

**Result.** The recipe is **seed-fragile**. Seed 2 matches seed 1 on overall improvement (-0.334 vs -0.291 at ceiling 2,
-2.945 vs -2.941 at ceiling 6) with a larger mean Hamming distance (2.1 vs 1.15). Seed 3 ends at the level of a random
double change overall (-0.713 at ceiling 2). Its retention table looks remarkable (7.9% / 6.1% / 5.7% of children keep the
height at parent heights 4 / 5 / 6, against 0.5-0.8% for a random single change at height 6), and I first read that as a
policy that had learned to preserve high parents. **That reading was wrong (erratum at the top):** 5.4% of this
policy's children are exact copies of the parent even after redrawing, and its retention is flat at about 5.5% for every
parent height from 4 to 8 (evaluation E), which is the copy rate, not structure. Seed 3 is a policy that has collapsed
toward copying its parent, which also explains its poor overall numbers (28% retention at parent height 2 vs 42% for a
random single change).

**Takeaway.** Seed fragility is real: of three seeds of the same recipe, one collapsed toward copying. And a retention
rate that does not fall with the parent's height is a red flag for exact copies, not for structure. Retention by exact
parent height with copies excluded is the metric to use.

### E12. The same recipe with 3.3x more training (`r11_free3_long`, 5M children)

**Why.** Does the best recipe keep improving with training? **What.** `r1_free3` with `--total-timesteps 5000000`.
**Result.** Worse than the 1.5M run on every ceiling (-0.417 vs -0.291 at ceiling 2; retention at parent 2: 56% vs 68%),
with a bit better retention at the top (parent 6: 1.5% vs 0.7%). The mean per-step KL in the last 100 iterations is 0.0000
for it and for every other run in this batch: **the policies freeze** (become near-deterministic) well before training
ends, so extra steps cannot help, and the drift before freezing is not monotone improvement.

### E13. Free budget of 2 (`r12_free2`)

**Why.** Fill in the budget sweep 1 / 2 / 3 / 8. **Result.** -0.329 / -0.895 / -1.536 / -2.241 / -2.954; mean Hamming 1.05;
retention 8.3% / 3.3% / 1.6% at parents 4 / 5 / 6. Between `r7_free1` and `r1_free3`, and about equal to seed 2 of free3.
**Takeaway.** Budgets of 1-3 all land in the same band (within about 0.04 overall), with seed noise of the same size.
(The odd last-iteration line "child 1.00" printed in the run log is a single batch of 8 parents, not the final policy.)

---

## E14. A single-edit policy, and its supervised warm start

**Why.** Everything so far suggested the best policies make exactly one change, but the coefficient-by-coefficient output
cannot guarantee that: the SFT policies copy their parent 35% of the time (a wasted sample and a -100 penalty in RL),
the number of changed coefficients is only controlled indirectly by a penalty, and the 35-dimensional action space makes
the policy-gradient signal sparse. A one-softmax output over the 35 x (p-1) = 70 (position, new value) pairs fixes all
three, keeps the same network and the one-shot "embed, encode, project" structure, and makes a group of 128 children per
parent nearly cover the whole action space. **[choice, a deviation from the output head the user described, made under the
user's allowance for the night and documented here.]**

**What.** `--single` in `rlmath-reinforce`, `rlmath-sft` and the evaluation: the per-coefficient logits are masked where a
value equals the parent's own, flattened to one 70-way softmax, and one pair is sampled. Exactly one coefficient changes;
the log-probability is exact. Entropy and KL needed NaN-safe versions for the masked logits (found by a test: a
`0 * NaN` gradient). SFT (`sf1_single`): the same pairs as S1 (child >= parent, 1.37M), 8 epochs, 128 wide, 4 layers.

**SFT result.** Held-out NLL **3.182** nats per pair, against ln 70 = 4.248 for a uniform choice among the 70 single
changes: the model gives the held-out good edit about 2.9x the probability a uniform pick would. (The factorized SFT's
4.15 is not comparable: that model also pays to say "keep" at 34 positions.)

**Evaluation** (single-edit policies always change exactly one coefficient, so compare them with random-1):

| method | ceiling 2 | ceiling 3 | ceiling 4 | ceiling 5 | ceiling 6 | ceiling 7 | ceiling 8 |
|---|---|---|---|---|---|---|---|
| random-1 | -0.549 | -1.129 | -1.772 | -2.467 | -3.195 | -3.958 | -4.783 |
| sf1_single | -0.306 | -0.881 | -1.525 | -2.223 | -2.945 | -3.710 | -4.533 |

`sf1_single` is ahead of random-1 by a nearly constant 0.24-0.25 at every ceiling, and ahead of the factorized SFT (S1) by
about 0.05 (e.g. -0.306 vs -0.352 at ceiling 2, from the same evaluation).

## E15. RL with the single-edit policy: from scratch (`q1`), from the SFT init (`q2`), and with a KL tether (`q3`)

**Why.** Test the single-edit policy under RL, with and without the supervised warm start. **What.** `--single`
with the E3 settings; `q1` from scratch; `q2` with `--init sf1_single.pt`; `q3` as `q2` plus `--kl-coef 0.1`. 1.5M children each.

| method | ceiling 2 | ceiling 3 | ceiling 4 | ceiling 5 | ceiling 6 | ceiling 7 | ceiling 8 |
|---|---|---|---|---|---|---|---|
| random-1 | -0.549 | -1.129 | -1.772 | -2.467 | -3.195 | -3.958 | -4.783 |
| r1_free3 | -0.292 | -0.891 | -1.524 | -2.221 | -2.940 | -3.704 | -4.522 |
| sf1_single | -0.306 | -0.881 | -1.525 | -2.223 | -2.945 | -3.710 | -4.533 |
| q1_single_scratch | -0.417 | -1.017 | -1.625 | -2.340 | -3.054 | -3.830 | -4.661 |
| q2_single_sf1 | -0.278 | -0.863 | -1.489 | -2.200 | -2.923 | -3.682 | -4.504 |
| q3_single_sf1_kl | -0.297 | -0.852 | -1.511 | -2.215 | -2.940 | -3.716 | -4.515 |

**Result.** `q2` (RL from the single-edit SFT) is **the best or within 0.01 of the best policy of the night** at every ceiling
(e.g. -0.278 at ceiling 2 and -4.504 at ceiling 8; `q3` is best at ceiling 3), 0.27-0.28 ahead of random-1, and ahead of the
best factorized policy `r1_free3` by 0.01-0.04. `q3` (KL tether) is roughly the SFT policy plus a little, and `q1` (no SFT) is clearly worse than any SFT-based policy
(-0.417 at ceiling 2), as the SFT-vs-scratch result for the factorized policy suggested. With a single edit, the SFT
warm start matters (0.14 at ceiling 2 between `q1` and `q2`), unlike with the factorized policy and a small Hamming budget.
`q2` has not been repeated across seeds yet (see E19).

## E16. Starting the curriculum at the top (`h1_top_s1..s3`, `--curriculum-start 6`)

**Why.** I (wrongly, see the erratum) took `r10`'s high-parent retention as a sign that training on high parents teaches
structure; this tests the hypothesis directly. **What.** `r1_free3` with the curriculum ceiling starting at 6 instead of 2
(so parents are mostly heights 4-6), seeds 1-3. **Result** (evaluation E; ceiling 8 has parents up to height 8, which none of
the policies saw as parents in training):

| method | ceiling 2 | ceiling 4 | ceiling 6 | ceiling 7 | ceiling 8 |
|---|---|---|---|---|---|
| random-1 | -0.549 | -1.774 | -3.190 | -3.954 | -4.774 |
| r1_free3 | -0.292 | -1.523 | -2.942 | -3.703 | -4.523 |
| h1_top_s1 | -0.346 | -1.571 | -2.993 | -3.752 | -4.580 |
| h1_top_s2 | -0.372 | -1.588 | -2.990 | -3.755 | -4.586 |
| h1_top_s3 | -0.348 | -1.581 | -3.001 | -3.756 | -4.585 |

All three seeds behave like S1 and a little behind `r1_free3` (0.05-0.08 behind at ceiling 2, 0.06 at ceiling 8), none better,
and none of the three collapsed (unlike seed 3 of E11). Retention at parent heights 6 / 7 / 8 (copies excluded, evaluation F)
is 0.8% / 0.1-0.3% / 0.1-0.2%, the same as the other policies. **Starting on high parents neither helps nor hurts here**; the hypothesis is not supported.

## E17. Does a policy use its input? An edit chosen for another parent

**Why.** Every policy has a mean child score that is the same at every ceiling, which suggests the gain over random edits
is a parent-independent habit, not parent-specific skill. **What.** `experiments/parent_dependence.py`: each parent gets the
edit (same positions, same shifts mod p) the policy produced for a different parent in the same batch. If that
"mismatched" edit does as well as the policy's own, the policy ignores its input.

| ceiling | policy | own edit | edit chosen for another parent | difference (use of the parent) |
|---|---|---|---|---|
| 2 | random-1 | -0.549 | - | - |
| 2 | sf1_single | -0.296 | -0.339 | +0.043 |
| 2 | **q2_single_sf1** | **-0.282** | **-0.479** | **+0.197** |
| 2 | q1_single_scratch | -0.406 | -0.440 | +0.034 |
| 2 | s1_ge_all | -0.354 | -0.405 | +0.051 |
| 2 | r1_free3 | -0.292 | -0.315 | +0.023 |
| 6 | random-1 | -3.197 | - | - |
| 6 | sf1_single | -2.948 | -2.989 | +0.041 |
| 6 | **q2_single_sf1** | **-2.908** | **-3.121** | **+0.213** |
| 6 | q1_single_scratch | -3.063 | -3.081 | +0.018 |
| 6 | s1_ge_all | -3.010 | -3.046 | +0.036 |
| 6 | r1_free3 | -2.939 | -2.957 | +0.018 |

(ceilings 4 and 8 give the same pattern: `q2` +0.195 and +0.198; every other policy +0.02 to +0.08.)

**Result.** For every policy except `q2`, **74-91% of the gain over a random single change survives when the edit was
chosen for the wrong parent** (e.g. `sf1_single` -0.339 vs random -0.549, own -0.296): they learned a good fixed
distribution over edits. For `q2` an edit chosen for another parent is only 0.07 better than random, while its own edit
is 0.27 better: **about three quarters of `q2`'s advantage comes from reading the parent.** The size of the parent
dependence is the same (+0.20) at every ceiling, so it is a skill that does not depend on how high the parent is.
My interpretation, untested: choosing, for the polynomial in front of it, a single change that makes the child
non-F-split (height >= 2), which is a condition on a single coefficient (the first step of the staircase), not climbing
the later steps.

## E18. Corrected final comparison (evaluation F)

Same parents for every method, ceilings 2 / 4 / 6 / 8 (mean parent score 2.00 / 3.22 / 4.64 / 6.23), copies redrawn up to
100 times. Improvement (child score - parent score; higher is better):

| method | ceiling 2 | ceiling 4 | ceiling 6 | ceiling 8 |
|---|---|---|---|---|
| random-1 | -0.549 | -1.774 | -3.190 | -4.777 |
| random-2 | -0.716 | -1.943 | -3.362 | -4.962 |
| scratch | -0.451 | -1.678 | -3.083 | -4.710 |
| s1_ge_all | -0.354 | -1.570 | -3.006 | -4.575 |
| s4_big | -0.331 | -1.575 | -2.990 | -4.585 |
| r1_free3 | -0.291 | -1.524 | -2.940 | -4.523 |
| r7_free1 | -0.355 | -1.556 | -2.970 | -4.560 |
| r8_s1_free1 | -0.297 | -1.513 | -2.933 | -4.525 |
| r9_free3_s2 | -0.336 | -1.562 | -2.963 | -4.555 |
| r10_free3_s3 | -0.716 | -1.881 | -3.243 | -4.739 |
| r12_free2 | -0.331 | -1.541 | -2.964 | -4.537 |
| h1_top_s1 | -0.347 | -1.571 | -2.993 | -4.583 |
| h1_top_s2 | -0.374 | -1.595 | -3.001 | -4.598 |
| h1_top_s3 | -0.348 | -1.580 | -3.004 | -4.588 |
| sf1_single | -0.298 | -1.515 | -2.939 | -4.548 |
| q1_single_scratch | -0.417 | -1.628 | -3.058 | -4.650 |
| q2_single_sf1 | -0.279 | -1.501 | -2.912 | -4.510 |
| q3_single_sf1_kl | -0.296 | -1.507 | -2.936 | -4.536 |

Share of children at least as high as their parent, by exact parent height, **exact copies not counted** (ceiling-8 pool):

| method | parent 2 | parent 3 | parent 4 | parent 5 | parent 6 | parent 7 | parent 8 |
|---|---|---|---|---|---|---|---|
| random-1 | 43.8 | 13.6 | 5.3 | 1.7 | 0.7 | 0.2 | 0.1 |
| random-2 | 28.5 | 10.5 | 3.9 | 1.1 | 0.4 | 0.1 | 0.0 |
| random-3 | 23.4 | 8.3 | 2.7 | 0.8 | 0.3 | 0.0 | 0.0 |
| scratch | 55.4 | 17.6 | 5.6 | 1.9 | 0.8 | 0.2 | 0.0 |
| s1_ge_all | 65.7 | 23.8 | 7.1 | 2.3 | 0.8 | 0.3 | 0.1 |
| s4_big | 67.0 | 21.0 | 7.2 | 2.5 | 0.9 | 0.2 | 0.1 |
| r1_free3 | 68.2 | 23.6 | 7.5 | 2.3 | 1.0 | 0.4 | 0.1 |
| r7_free1 | 62.9 | 20.3 | 6.5 | 2.5 | 1.0 | 0.3 | 0.0 |
| r8_s1_free1 | 69.2 | 22.8 | 7.6 | 2.6 | 1.1 | 0.3 | 0.1 |
| r9_free3_s2 | 68.2 | 22.3 | 6.7 | 2.8 | 0.9 | 0.1 | 0.1 |
| r10_free3_s3 | 23.6 | 7.5 | 2.1 | 0.7 | 0.2 | 0.1 | 0.0 |
| r12_free2 | 65.1 | 21.8 | 6.8 | 2.6 | 0.6 | 0.2 | 0.1 |
| h1_top_s1 | 66.8 | 21.7 | 6.7 | 2.6 | 0.8 | 0.1 | 0.1 |
| h1_top_s2 | 64.6 | 20.8 | 7.1 | 2.4 | 0.8 | 0.1 | 0.2 |
| h1_top_s3 | 64.5 | 21.3 | 7.1 | 2.5 | 0.8 | 0.3 | 0.1 |
| sf1_single | 68.3 | 20.9 | 7.1 | 2.5 | 0.7 | 0.3 | 0.1 |
| q1_single_scratch | 56.9 | 19.5 | 5.8 | 2.1 | 0.8 | 0.3 | 0.1 |
| q2_single_sf1 | 68.8 | 22.6 | 8.1 | 2.2 | 1.0 | 0.2 | 0.1 |
| q3_single_sf1_kl | 70.8 | 22.1 | 7.2 | 2.6 | 0.9 | 0.2 | 0.1 |

**Takeaways.**
- Apart from the collapsed `r10`, the from-scratch single-edit `q1` and the earlier `scratch` run, every policy is within about
  0.1 of the best (`q2_single_sf1`) and 0.2-0.3 ahead of random-1 at ceiling 2 (about 0.2 ahead at ceiling 8).
- The `r10` (seed 3) failure is confirmed: with copies excluded its retention is below random at every parent height.
- **No policy preserves the height of high parents.** At parent height 6 random single changes keep the height 0.7% of the
  time and the best policies 1.0-1.1%; at heights 7-8 everything is at the random level (0.0-0.4%). The gains over random
  are at parent heights 2-4 (e.g. 68-71% vs 44% at height 2, 7-8% vs 5% at height 4).

---

## E19. Is `q2` robust? Two more seeds, and a high-parent curriculum start with the single-edit init

**Why.** Every headline so far came from one seed, and the factorized recipe lost one seed of three (E11). **What.**
`q2_single_sf1` (E15) with `--seed 2` and `--seed 3` (`q2_seed2`, `q2_seed3`; `q2_seed1` is the E15 run), and `q4_top`: the
same with `--curriculum-start 6`. **[choice for `q4`:** since starting at 6 did nothing for the factorized policy (E16) but
the single-edit SFT init changes the starting point.] 1.5M children each; evaluated together in evaluation G.

Improvement (child score - parent score; higher is better):

| method | ceiling 2 | ceiling 4 | ceiling 6 | ceiling 8 |
|---|---|---|---|---|
| random-1 | -0.549 | -1.765 | -3.194 | -4.791 |
| random-2 | -0.716 | -1.941 | -3.371 | -4.954 |
| r1_free3 | -0.292 | -1.526 | -2.939 | -4.522 |
| sf1_single | -0.314 | -1.522 | -2.952 | -4.532 |
| q2_seed1 | -0.285 | -1.487 | -2.906 | -4.499 |
| q2_seed2 | -0.264 | -1.485 | -2.915 | -4.506 |
| q2_seed3 | -0.278 | -1.495 | -2.942 | -4.518 |
| q4_top | -0.267 | -1.494 | -2.929 | -4.514 |

Share of children at least as high as their parent, by exact parent height (ceiling-8 pool, copies excluded; the
single-edit policies never copy):

| method | parent 2 | parent 3 | parent 4 | parent 5 | parent 6 | parent 7 | parent 8 |
|---|---|---|---|---|---|---|---|
| random-1 | 42.5 | 14.1 | 4.6 | 1.7 | 0.6 | 0.1 | 0.0 |
| random-2 | 29.0 | 10.5 | 4.0 | 1.1 | 0.4 | 0.1 | 0.0 |
| r1_free3 | 67.9 | 23.6 | 7.6 | 2.3 | 1.0 | 0.4 | 0.1 |
| sf1_single | 68.8 | 24.6 | 8.1 | 2.4 | 0.8 | 0.2 | 0.1 |
| q2_seed1 | 71.9 | 21.8 | 7.9 | 2.5 | 0.8 | 0.3 | 0.1 |
| q2_seed2 | 67.9 | 22.9 | 7.8 | 2.6 | 0.9 | 0.2 | 0.1 |
| q2_seed3 | 68.2 | 23.6 | 7.1 | 2.5 | 0.8 | 0.4 | 0.1 |
| q4_top | 70.1 | 21.9 | 7.8 | 2.7 | 0.9 | 0.3 | 0.1 |

Use of the parent (E17 test): improvement with the policy's own edit, minus improvement with an edit it chose for another
parent. A larger number means more of the policy's advantage comes from reading its input.

| policy | ceiling 2 | ceiling 4 | ceiling 6 | ceiling 8 |
|---|---|---|---|---|
| q2_seed1 | +0.190 | +0.201 | +0.205 | +0.204 |
| q2_seed2 | +0.141 | +0.139 | +0.129 | +0.141 |
| q2_seed3 | +0.122 | +0.109 | +0.118 | +0.106 |
| q4_top | +0.087 | +0.088 | +0.078 | +0.097 |

(In the first run of this test, `q2_seed1` gave +0.197 / +0.195 / +0.213 / +0.198, so the test itself varies by about 0.01.)

**Result and takeaways.**
- **All three `q2` seeds land in the same band** (ceiling 2: -0.285 / -0.264 / -0.278; ceiling 6: -2.906 / -2.915 / -2.942;
  ceiling 8: -4.499 / -4.506 / -4.518), none collapsed. Mean of the three: -0.276 / -1.489 / -2.921 / -4.508 at ceilings
  2 / 4 / 6 / 8, i.e. 0.27-0.28 ahead of random-1, 0.02-0.04 ahead of the SFT policy it started from, and 0.01-0.04 ahead of
  the best factorized policy. The seed spread (0.02-0.04) is as large as these last two gaps, so they are small but
  consistent (every seed and ceiling is ahead of `sf1_single`; for `r1_free3` three of the four ceilings show all seeds ahead).
- Compared with the factorized recipe (one seed of three collapsed), the single-edit SFT-then-RL recipe was stable here.
  Three seeds is still few.
- **How much each seed uses the parent differs a lot (+0.09 to +0.20) while the score is the same.** Different RL runs
  end in different mixes of "a good edit for this parent" and "a good edit in general". `q4_top` has the lowest
  dependence and the same score.
- Retention at parent heights 6-8 is at the level of all other policies (0.8-0.9% / 0.2-0.4% / 0.1%): `q2` does not
  preserve high parents either.

---

## E20. Repeat penalty instead of a Hamming penalty (new-polynomial discovery)

**Why.** Idea from the user: drop the Hamming penalty, keep a high copy penalty, and give a child that was already generated
earlier in the run no reward. **What (implemented in `PolyEnv`, flag `--repeat-penalty`).** The seen set is a hash set of
scaled children, empty at the start of each run (the database is *not* used for it; the parent pool still comes from the
database). A repeat (earlier step, or an earlier copy in the same batch) gets reward `-repeat_penalty` instead of its
improvement; the first sighting keeps `child - parent`. **[choices:** flat -10, not 0, because 0 would beat an unseen
low-height child (its improvement is usually negative) and pay the policy to repeat; no Hamming penalty; copy penalty 100.**]**
The old "improvement over the parent" metric does not fit large edits, so the runs are compared by **new (first-seen)
children of height >= 6 / >= 8 per 1024 children**, logged as `charts/new_ge6`, `charts/new_ge8`, plus `charts/repeated`.
The untrained policy's rate (first quarter of `n0`) is the reference: 2.5 / 0.27 per 1024. 1.5M children each, p = 3.

| run | setup | share repeated (last quarter) | new>=6 per 1024 (first -> last quarter) | new>=8 per 1024 | total new>=6 / >=8 |
|---|---|---|---|---|---|
| `n0_norepeat` | factorized, scratch, no repeat penalty (control) | 100% | 2.5 -> 0.0 | 0.27 -> 0.00 | 2034 / 211 |
| `n1_repeat` | factorized, scratch, repeat 10 | 0.1% | 2.5 -> 2.8 | 0.27 -> 0.29 | 3977 / 403 |
| `n3_repeat_nocurriculum` | as `n1`, curriculum off (parents average height 8) | 0.6% | 2.6 -> 2.9 | 0.31 -> 0.28 | 4087 / 440 |
| `n2_repeat_single` | single-edit, from `sf1_single`, repeat 10 | 54% | 2.5 -> 2.5 | 0.27 -> 0.26 | 3719 / 372 |
| `n4_repeat_from_sft` | factorized, from `s1_ge_all`, repeat 10 | 0.0% | 7.2 -> 7.4 | 0.74 -> 0.78 | 10602 / 1108 |

**Result and takeaways.**
- **Without the penalty the policy collapses to one polynomial** (83% repeats in the third quarter, 100% in the last): the
  known failure. With it, repeats stay at about 0.1-0.6% and nothing collapses, so the penalty does what it is for.
- **But it does not make the policy learn to find high polynomials.** From scratch (`n1`, `n3`) the rate of new height >= 6
  children moves from 2.5-2.6 to 2.8-2.9 per 1024, i.e. about 10%, with 1 seed each; the mean edit shrinks only from 23 to
  19 coefficients. There is little signal: the reward of a random 35-coefficient child is nearly always 0 or 1.
- **The high rate in `n4` comes entirely from the supervised start** (about 3x the untrained rate from the first iteration,
  flat afterwards), and RL with the repeat penalty did not move it (7.2 -> 7.4).
- **`n2` (single-edit) can't avoid repeats by design:** a parent has only 70 one-coefficient changes and each group draws 128
  children, so at least ~45% of children are repeats whatever the policy does (observed 54%). Policy frozen at its starting
  point (KL 0).
- Every run is a single seed; differences of 0.1-0.3 per 1024 are not trustworthy.
- On the worry that the policy can't see the seen set: it does not need to. A stochastic policy gets a diminishing reward for
  concentrating mass, which is what stopped the collapse here. What it can't do is create a gradient toward high
  polynomials when almost nothing it samples is rewarded.

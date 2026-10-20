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

# Overnight experiments: bootstrapping the policy with supervised fine-tuning

Branch `worktree-policy-gradient`. Everything here is at p = 3 quartics (35 coefficients), using a read-only snapshot of
`runs/heights.sqlite` (12.1M distinct p=3 polynomials with a known height). The per-experiment record (what was changed,
why, the command, the tables) is `experiments/LOG.md`; this file is the summary. Checkpoints and result files are in
`experiments/checkpoints/` and `experiments/results/`.

## In one paragraph

Supervised fine-tuning on stored one-coefficient edits works as a bootstrap, and a **single-edit policy** (one softmax
over the 70 possible (position, new value) changes) fine-tuned with reinforcement learning from that start is the best
and most reliable policy found: **`q2`**, which beat a random single change by 0.27-0.28 in expected height change at every
parent height ceiling, across three seeds. It is also the only family that clearly *reads its input*. But the gains are
modest and live at low parent heights: **no policy keeps the height of a high parent (6 or more) noticeably better than a
random single change**, and on average every policy still loses height (the child is about 1.7 on average whatever the parent).
Two of my own early conclusions were wrong and are corrected below (see "Mistakes").

## How results are measured

`experiments/evaluate.py`: pool parents (smooth polynomials from the database, level weighted by exp(height / 3)) capped
at a ceiling c = 2, 4, 6, 8 (mean parent height 2.0 / 3.2 / 4.6 / 6.2). Every method sees **exactly the same parents**
(same seed, frozen pool), 40,960 children per cell, one child per parent. The metric is **improvement = child score -
parent score** (the score is the height of a smooth polynomial, 0 if singular), no penalties. Children equal to their
parent are redrawn (up to 100 times). Baselines: change k random coefficients (random-1, random-2, random-3). Evaluation
noise is about 0.02, seed-to-seed spread of one recipe about 0.02-0.04 (single-edit) up to a complete failure
(factorized, 1 seed of 3).

## Results: the policies, ranked

Improvement (higher is better); all numbers from evaluations F and G (`experiments/results/`; the two runs agree to about
0.01-0.02). The mean parent score at each ceiling is 2.00 / 3.22 / 4.64 / 6.23.

| policy | what it is | ceiling 2 | ceiling 4 | ceiling 6 | ceiling 8 |
|---|---|---|---|---|---|
| random-1 | change one random coefficient | -0.549 | -1.765 | -3.194 | -4.791 |
| random-2 | change two | -0.716 | -1.941 | -3.371 | -4.954 |
| `scratch` (E3) | RL from scratch, factorized, free budget 8 | -0.451 | -1.678 | -3.083 | -4.710 |
| `s1_ge_all` (E2) | supervised only, factorized | -0.354 | -1.570 | -3.006 | -4.575 |
| `r1_free3` (E4) | RL from scratch, factorized, free budget 3 (1 seed of 3 collapsed) | -0.292 | -1.526 | -2.939 | -4.522 |
| `sf1_single` (E14) | supervised only, single-edit | -0.314 | -1.522 | -2.952 | -4.532 |
| **`q2` (E15, E19)** | **single-edit: supervised, then RL** (mean of 3 seeds) | **-0.276** | **-1.489** | **-2.921** | **-4.508** |

**Most successful.**
1. **`q2` = single-edit SFT, then RL.** Ahead of random-1 by 0.27-0.28 at every ceiling, 0.02-0.04 ahead of its own SFT start,
   0.01-0.04 ahead of the best factorized policy. Three seeds (spread 0.02-0.04), none collapsed. About three quarters of
   its advantage over random disappears if the edit it chose for a *different* parent is used (E17), so it reads its input.
2. **`sf1_single`, supervised only.** About three minutes of training, no oracle calls during training, and already about
   +0.25 over random-1 (E14).
3. **`r1_free3`**, the best factorized policy: equal to the supervised one, but one of three seeds collapsed (E11).

## What was tried (index)

| id | change | headline |
|---|---|---|
| E0 | count stored (parent, one-change child) pairs | 1.37M pairs with child >= parent, 89% at parent height 2; only ~19k at parent >= 4 |
| E1 | random-edit baselines + common evaluation | even a single random change gives a child independent of the parent's height (mean 1.45) |
| E2 | SFT variants S1-S4 (keep-or-improve, strict improvement, parents >= 3, bigger model) | all beat random; **S1 = S2 = S3 = S4 within 0.01**: what is imitated and the model size do not matter |
| E3 | reference: RL from scratch (previous session) | beats random-1 by 0.07-0.10, worse than any SFT policy |
| E4 | free Hamming budget 3 (was 8) | **best from scratch**; ahead of the factorized SFT policy by 0.05-0.06 on one seed |
| E5 | 32 parents x 32 children (was 8 x 128) | did not learn (= random-1) |
| E6 | RL from the SFT init, default budget 8 (two learning rates) | **worse than the SFT start**: the policy drifts to 4.5 changes |
| E7 | RL from SFT + KL tether 0.1 | preserves the SFT policy, no gain |
| E8 | RL from SFT + budget 3 | = E4 |
| E9, E10 | budget 1, from scratch / from SFT | in the same band (within 0.07 of `r1_free3`) |
| E11 | free3 recipe, seeds 2 and 3 | **seed-fragile**: seed 3 collapsed toward copying the parent |
| E12 | 5M children instead of 1.5M | worse (the policy freezes early; mean per-step KL ~0 over the last 100 iterations of the runs I checked, `r9`-`r12`) |
| E13 | budget 2 | in the same band |
| E14 | **single-edit policy** + SFT | held-out NLL 3.18 vs 4.25 for uniform; +0.25 over random-1 |
| E15 | RL with the single-edit policy: scratch / from SFT / SFT + KL | **from SFT is best**; scratch is 0.13-0.14 worse |
| E16 | curriculum starting at ceiling 6 (3 seeds) | no effect (hypothesis not supported) |
| E17 | edit chosen for another parent | every policy except `q2` keeps 74-91% of its gain over random: mostly parent-independent |
| E18 | corrected final comparison | confirms the ranking; no policy keeps high parents better than random |
| E19 | `q2` seeds 2 and 3, `q4` (SFT init, curriculum from 6) | robust across seeds; parent dependence varies 0.08-0.20 |
| E21 | SFT on pairs within K = 1, 3, 5, 7 coefficients (22.7M pairs mined) | lift over a same-size random edit: +0.30 / +0.30 / +0.22 / +0.12; absolute improvement worsens with K; 4x more data changes nothing |
| E22 | group hit rate, 128 children per parent | per child the trained models are 1.6x random; per parent `q2` is worse than random flips (diversity narrowed by RL) |
| E23 | hill-climb (accept if not lower) vs best-of-k, no training | chaining adds nothing for random proposals; the SFT prior gives +27-32% relative |
| E24 | free-walk game (always move, reward max(0, h - parent), repeat penalty), RL K = 4 / 8 / 16 | RL does not beat its SFT start; free walk beats independent tries for the SFT prior, loses for random |
| E25 | no-replacement mask on the existing models, no training | **best no-training rule so far**: SFT best-of-16 without repeats beats every RL free-walk model by 0.12-0.16 at ceilings 2-4; `q2` was only hurt by repeats |
| E26 | 10M-parameter model, 40-step free walk, 12 h of RL (61M oracle calls) | barely moves from its SFT start (+0.06-0.10 at ceilings 2-4, level at 6-8); matches but does not beat SFT best-of-40 without repeats |
| E20 | repeat penalty (seen set from scratch, no Hamming penalty, copy 100) | stops collapse; no learning from scratch (new>=6: 2.5 -> 2.8 per 1024); SFT start gives 3x but RL adds nothing |

## What I learned

1. **Supervised bootstrapping works**, and for the single-edit policy it matters (0.14 at ceiling 2 against RL from scratch).
   For the factorized policy with a small Hamming budget, RL from scratch reaches the same level without it.
2. **The edit budget matters more than the algorithm.** Free budget 8 lets RL drift to bigger, worse edits (it ruins an
   SFT start); budgets 1-3 are fine. The single-edit head removes the question: exactly one change, no copies.
3. **The one-softmax single-edit head is better than the coefficient-by-coefficient head** for this task (E14, E15), and
   it never produces copies (the factorized SFT copied its parent 35% of the time).
4. **Most of the "gain over random" is a parent-independent lift.** All policies produce children of mean height about
   1.7 whatever the parent's height (random single changes: 1.45); the gap to random is the same (~0.25) at every ceiling.
5. **Nothing preserves high parents.** With copies excluded, children keep the height of a height-6 parent 0.6-0.7% of the
   time for random single changes and 0.8-1.1% for the best policies; at heights 7-8 everyone is at the random level.
   The gains are at parent heights 2-4 (68-72% vs 43% at height 2).
6. **`q2` is the only policy that uses the parent much** (E17, E19), by +0.08 to +0.20 depending on seed. My guess
   (untested): it picks the single coefficient change that makes the child non-F-split (height >= 2), a condition on one
   coefficient; climbing further needs several conditions at once.
7. **RL runs freeze** (mean per-step KL ~0 in the last 100 iterations of the runs I checked, `r9`-`r12`) and are
   seed-fragile in the factorized setting; the single-edit SFT-then-RL recipe was stable over 3 seeds.

## Research choices I made (you allowed this for the night)

- SFT variants S1-S4 (E2), and imitating "child >= parent" as the default target.
- Redrawing exact copies in the evaluation (up to 100 times), and reporting retention without copies.
- The single-edit policy class (E14): a deviation from the output head you described (still one forward pass: embed, encode,
  project, but sampling one pair from a single softmax). Reasons are in the log. The default factorized policy is unchanged.
- KL coefficient 0.1; free Hamming budgets 1 / 2 / 3; curriculum start 6; 32 x 32 groups; 5M-child run.

## Mistakes I made and corrected

- **Retention at high parents was inflated by exact copies.** I first reported that seed 3 of the free3 recipe (`r10`) and
  the budget-1 run (`r7`) preserved high parents; in fact my evaluation only redrew copies up to 20 times, so a policy that
  copies often left exact copies, which trivially keep the height (5.4% of `r10`'s children). The tell was a retention that
  was flat across parent heights. Fixed (100 redraws, copies excluded); the log entries are withdrawn and marked.
- A test caught NaN gradients from masked logits in the single-edit policy (entropy and KL); both now have safe versions.

## Caveats

- p = 3 and one database only. The database pairs are all one-coefficient edits from an earlier local search, and 89% have parent
  height 2, so the supervised signal is mostly about low parents.
- Every policy still **loses** height on average (improvement is negative at every ceiling). "Better than random" is a small
  absolute change: at ceiling 2, `q2` is -0.28 against -0.55.
- Evaluation parents are pool polynomials (weighted toward high heights), not a random sample; other weightings would change the
  absolute numbers (though not the ranking within a ceiling).
- Few seeds: 3 for the single-edit recipe, 3 for the factorized one, 1 for most other runs. The seed-to-seed spread is as
  large as several of the gaps above.

## Suggested next steps (not done)

1. **Look inside `q2`**: which coefficient does it change, and does the child become non-F-split? Check against the first
   staircase coefficient directly. If so, the next target is the second condition.
2. **More data at high parents**: the database has ~2k positives at parent height 5 and ~40 at height 7. Generate pairs by
   scoring many single changes of high-height pool polynomials (cheap at p = 3) and fine-tune on them.
3. **Chain the edits** (apply the single-edit policy several times with the oracle in between) instead of one edit.
4. **Check p = 2 and p = 5.** The pair-finding needs a different encoding at p >= 5.
5. PPO-style reuse of each batch, now that the policy class is exact.

## Reproduce

```
uv run python experiments/count_pairs.py <heights.sqlite> 3 pairs_p3.npz                    # E0
CUDA_VISIBLE_DEVICES=1 uv run rlmath-sft --pairs pairs_p3.npz --out sft_single.pt --single --require ge
uv run python experiments/run_rl.py q2 <snapshot.sqlite> out/ --single --init sft_single.pt --total-timesteps 1500000
uv run python experiments/evaluate.py --db <copy.sqlite> --ceilings 2 4 6 8 --ckpt q2=out/q2/model.pt
uv run python experiments/parent_dependence.py --db <copy.sqlite> --ckpt q2=out/q2/model.pt
```

The evaluation scripts write to the database they are given, so pass a copy. All the SFT and RL command lines used are in
`experiments/LOG.md` and `experiments/run_sft_variants.py`.

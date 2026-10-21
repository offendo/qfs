# Plan: searching for high-height Calabi–Yau polynomials with learning

Status: implemented on the `fast-height` branch (October 2026); see `README.md` for what was built.
The PatternBoost search loop described in section 5 was later replaced by a Gymnasium environment and
a CleanRL-style REINFORCE agent (`src/rlmath/env.py`, `src/rlmath/reinforce.py`); sections 5 to 7 are kept as history.
Where the code differs from this plan: heights come from HTTP servers (`Qfs.jl/Server.jl`) instead of ZMQ
workers, and supersingular polynomials are not rewarded by default, because
at p = 5 sparse supersingular quartics turned out to be easy to find and flooded the search.

## 1. The problem in plain terms

We pick a prime `p`, a number of variables `n`, and a degree `d`, and look at homogeneous polynomials
`f` of degree `d` in `n` variables with coefficients in the finite field `F_p`. The project fixes
`n = d`; the zero set of such an `f` is a Calabi–Yau variety (for `n = d = 4`, a quartic K3 surface;
for `n = d = 5`, a quintic threefold). There are `C(n+d-1, d)` monomials, so `f` is a vector of 35
coefficients for quartics and 126 for quintics, each in `{0, …, p-1}`.

Every such `f` has a **quasi-F-split height** `h(f)`, a positive integer or `∞`. For Calabi–Yau
varieties it agrees with the classical Artin–Mazur height. What matters for the search:

- **Height 1** ("F-split", ordinary) is the generic case. Fedder's criterion makes it a one-coefficient
  test: `h = 1` exactly when the coefficient of `(x_1⋯x_n)^(p-1)` in `f^(p-1)` is nonzero mod `p`.
- **Higher heights are a staircase of vanishing conditions.** The Kawakami–Takamatsu–Yoshikawa (KTY)
  algorithm in `Qfs.jl` builds a sequence `g_2, g_3, …` with `g_{k+1} = θ(g_k)` and stops at the
  first `k` whose "top" coefficient `c_k` (again the coefficient of `(x_1⋯x_n)^(p-1)`) is nonzero.
  So `h(f) ≥ k+1` exactly when `c_1(f) = c_2(f) = … = c_k(f) = 0`, where each `c_i` is a fixed
  polynomial function of the coefficients of `f`.
- **Each extra unit of height costs about a factor of `p` in rarity.** For K3 surfaces the locus
  of height `≥ h` has codimension `h-1` in moduli, so a random `f` has height `≥ h` with probability
  roughly `p^-(h-1)`. Finite heights stop at 10 for K3 surfaces; beyond that the surface is
  supersingular (`h = ∞`, codimension 10, rarer than height 10). Our random-sampling measurements
  match this (section 3).
- **Height is invariant under linear changes of variables** (`GL_n(F_p)`) and under scaling `f`.
  The 35 quartic coefficients therefore carry only 19 "real" parameters; 16 directions of the search
  space never change the height.
- **The answer is only meaningful for smooth `f`.** Singular or non-reduced polynomials also get a
  number out of the algorithm (for example the "Fermat" quartic at `p = 2` is `(x+y+z+w)^4` and
  returns `cutoff + 2`), but it is not a K3 height. Any reward must reject singular `f`, or the
  search will happily find degenerate polynomials.

The consequence for method choice: there is **no partial credit inside a level**. `c_k(f)` is an
element of `F_p`; there is no sense in which `c_k = 1` is "closer to zero" than `c_k = 3`. The only
signal is the staircase itself, and each step is a `1/p` event under random sampling. Whatever we
build has to beat the baseline "sample uniformly, about `p^(h-1)` tries per height-`h` example".

## 2. What exists today

**Julia (`Qfs.jl/src/Qfs.jl`).** `quasiFSplitHeight_CY_lift(p, f, cutoff)` does: check homogeneity;
F-split test; compute `Δ = Δ₁(f^(p-1))` by lifting to integers, taking the `p`-th power and
dividing cross terms by `p`; then iterate `g ↦ u(Δ · g)` (multiply by `Δ` and keep the
`(p-1, …, p-1)` Frobenius component) up to `cutoff` times. It returns `h`, `cutoff + 2` when the
chain hits zero (provably `∞`), or `cutoff + 1` when it ran out of steps. `qfsFromCoeffs_d4`
builds the polynomial from a coefficient vector, hard-coded to 4 variables.

**Python bridge (`src/rlmath/juliaserver.py`).** A ZMQ REQ/REP server that embeds Julia through
`juliacall`, receives one polynomial per request, calls `qfsFromCoeffs_d4`, and replies with 16
bytes. The client in `utils.py` calls it serially, one polynomial per round trip.

**RL code (`environment.py`, `model.py`, `train.py`).** TorchRL PPO with a transformer encoder over
monomial tokens. Problems worth not repeating:

- Coefficients are binary (0/1) only, so for `p > 2` most of the space is unreachable.
  `compute_num_monomials` returns `(p-1)·35` slots but the server reads exactly 35 integers.
- The policy samples a continuous `TanhNormal` action, but the environment ignores it and
  thresholds `sigmoid(logits) > 0.5`, so the log-probabilities PPO optimises are not those of the
  action actually taken.
- The reward is `±0.5` for "height went up or not", which gives no information about how rare the
  result is and rewards degenerate polynomials.
- Each step costs one serial network round trip and one height computation.

### 2.1 The bus error

`Qfs.jl` opens with

```julia
PythonCall.GC.disable() # This is necessary for use in juliacall, otherwise it crashed due to "bus error"
Base.GC.enable(false)
```

My reading (from the code and known `juliacall` behaviour, **not reproduced** here):

- The crash is the known class of failures from running Julia *inside* a Python process. Julia's
  garbage collector runs finalizers for Python-object wrappers that call into CPython without the
  right thread/GIL state, and Julia relies on its own signal handlers (it uses deliberate memory
  faults for GC safepoints). If Python owns the signal handlers, those faults kill the process. On
  macOS a fault like that is reported as a "bus error". `juliaserver.py` already sets
  `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`, which is the documented fix for the signal half.
- Disabling GC is a harmful workaround. If it takes effect, the server never frees anything, and
  each height evaluation at `p ≥ 5` allocates polynomials with tens of thousands of big-integer
  terms; a long training run then exhausts memory. In practice it may not take effect at all:
  top-level statements in a package run during precompilation, not when `using Qfs` loads the
  cached module, so these two lines probably do nothing at runtime in a normally precompiled
  setup (inferred, not verified).
- The height code itself is fine with GC on. I ran thousands of evaluations in plain Julia (Nemo,
  GC enabled, no Python) with no crash.

Whether it still happens through `juliacall` I can't say without reinstalling the Python side. The
plan removes the cause instead: **no Julia inside Python**. The oracle becomes a standalone Julia
process, and `PythonCall`, `GC.disable` and `GC.enable(false)` come out of `Qfs`.

## 3. Cost of one evaluation (measured)

Plain Julia 1.11 + Nemo on this Mac (Apple Silicon), `n = d = 4`, uniformly random coefficients,
`cutoff = 10`, using the current `Qfs` algorithm unchanged:

| p | random samples | heights seen (count) | median time | slowest | `Δ₁(f^(p-1))` alone, median |
|---|---|---|---|---|---|
| 2 | 300 | 1:150, 2:76, 3:44, 4:16, 5:8, 7:1, 9:1, ">10":4 | 21 µs | 1 ms | 48 µs |
| 3 | 300 | 1:215, 2:61, 3:15, 4:6, 5:2, 6:1 | 20 µs | 24 ms | 1.8 ms |
| 5 | 100 | 1:76, 2:19, 3:4, 4:1 | 0.2 ms | 1.8 s | 0.48 s |
| 7 | 20 | 1:18, 3:2 | 1 ms | 53 s | 23 s |

Quintics (`n = d = 5`, cutoff 30) at `p = 2`: median 1.5 ms over 30 samples, heights up to 8.
At `p = 3`: `Δ₁` took 0.1 s. Larger `p` for quintics was not tried.

The four `>10` results at `p = 2` are almost certainly singular quartics (see section 1).

Takeaways:

- Most random polynomials are height 1 and cost microseconds, because the F-split test exits early.
- The cost is almost entirely `Δ₁(f^(p-1))` for the non-F-split ones, and it grows steeply with `p`
  (a `p`-th power of a polynomial with ~`p³` terms; `Δ` itself has 72 thousand terms at `p = 5` and
  700 thousand at `p = 7`). As written, `p = 7` costs about half a minute per non-F-split polynomial.
  Section 4 item 4 removes almost all of this.
- The height distribution matches the `p^-(h-1)` heuristic: at `p = 2` the counts halve per level.
- Sanity checks pass: the Fermat quartic `Σ x_i^4` gets height 1 for `p = 5` (`p ≡ 1 mod 4`) and
  `∞` for `p = 3` (`p ≡ 3 mod 4`), as theory predicts.

What random search alone would need for one height-10 quartic (`≈ p^9` samples): about 500 at
`p = 2`, 20 thousand at `p = 3`, 2 million at `p = 5`, 40 million at `p = 7`, 2.4 billion at
`p = 11`. With the faster oracle in section 4, a `1/p` fraction of samples costs ~7 ms at `p = 5`,
so 2 million samples is a few CPU-hours. So `p ≤ 3` is a pipeline test, `p = 5` is a fair
benchmark that random search can also reach, and `p ≥ 7` is where a learned method has to earn
its keep.

## 4. Oracle (Julia server) changes

These come first, because every method below is bottlenecked by the oracle, not the model.

1. **Standalone Julia process, no `juliacall`.** A Julia script that binds a ZMQ socket (ZMQ.jl) and
   speaks msgpack (MsgPack.jl), or plain length-prefixed bytes over TCP. Delete `using PythonCall`,
   `PythonCall.GC.disable()` and `Base.GC.enable(false)` from `Qfs.jl`; drop `PythonCall` from
   `Qfs.jl/Project.toml`. This eliminates the bus-error class rather than working around it.
2. **Depend on Nemo, not all of Oscar.** Everything the height code uses (`polynomial_ring`, `GF`,
   `lift`, `map_coefficients`, `change_coefficient_ring`) is in Nemo/AbstractAlgebra. Oscar pulls in
   GAP, Singular and Polymake, which slows startup and grows memory. Keep Oscar only in a separate
   verification script if we use Singular for smoothness checks.
3. **Batched, parallel API.** One request carries a batch: `(p, n, d, coeffs::Matrix{UInt8})`; the
   reply carries per-row `(status, height)` where status is one of `finite`, `inf_proved` (chain hit
   zero), `inf_by_bound` (ran past the theoretical bound, see below), `singular`, `zero`. A pool of
   worker *processes* (Distributed.jl, one per core) handles rows in parallel. Processes rather than
   threads keep FLINT and GC behaviour simple.
4. **Never compute `Δ₁(f^(p-1))`.** `Δ₁` obeys a product rule mod `p`, which gives
   `Δ₁(f^(p-1)) = -f^(p(p-2)) · Δ₁(f)` up to terms that only matter once the chain has already
   stopped. Because the splitting map `u` pulls `p`-th powers out (`u(a^p · b) = a · u(b)`), one
   step of the chain becomes

   `θ(g) = -f^(p-2) · u(Δ₁(f) · g)`

   which only needs `Δ₁` of the 35-term `f` itself. I prototyped this as a drop-in replacement and
   compared it with `quasiFSplitHeight_CY_lift` on 666 random non-F-split quartics: **zero
   mismatches**, including heights up to 9 and `∞` at `p = 2, 3`. Timings per non-F-split
   polynomial:

   | p | current code | new step | speed-up |
   |---|---|---|---|
   | 3 | 5.6 ms | 1.2 ms | 5× |
   | 5 | 1.0 s | 6.7 ms | 150× |
   | 7 | 38 s | 25 ms | 1500× |
   | 11 | (not run) | 1.5 s | |
   | 13 | (not run) | 5.0 s | |

   A proof that it returns exactly the same value as the current code, for every `p`, `n` and
   cutoff, is in `PROOF_fast_height.md`. Empirical agreement at `p ≥ 5` was only tested up to
   height 4; the regression suite below should still cover high heights there.

   (Computing `Δ₁` with coefficients mod `p²` instead of big integers, which I tried first, only
   saves about 15%.)
5. **Then shrink the chain to a 35×35 matrix (not yet prototyped).** In the new step,
   `u(Δ₁(f) · g)` is a quartic, so it is linear in `g` with 35 outputs, and multiplying by
   `-f^(p-2)` is linear back. Composing the two, the chain runs in a 35-dimensional space: a 35×35
   matrix over `F_p` whose entries are read off coefficients of `Δ₁(f)` and `f^(p-2)`, iterated at
   most 10 times. That should make `p = 11, 13` cost milliseconds; the remaining time there is
   in the generic term-by-term Julia loops, not in the mathematics. The same structure gives the
   vector `(c_1, …, c_k)` for free, which is useful for logging and for the solver in 5.3.
6. **Status instead of magic numbers, and the right cutoff.** For smooth quartic K3s, finite
   heights are at most 10, so running past 10 means supersingular. For quintic threefolds the code
   uses 101; check this against the literature bound (I believe it is `h^{1,2} + 1 = 102` for a
   smooth quintic, but this needs confirming).
7. **Smoothness check, applied lazily.** Only polynomials that would earn a reward above height 1
   need it, so the cost is paid on a `1/p` fraction. Check that `f` and all partial derivatives have
   no common projective zero over the algebraic closure (Gröbner basis via Singular/Oscar, in a
   separate worker pool). Include `f` itself in the ideal: when `p | d` the Euler relation does not
   recover `f` from its partials.
8. **General `(n, d)`.** Generate the monomial order in Julia from `(n, d)` and send it to the client
   at connection time, replacing `qfsFromCoeffs_d4` and the hard-coded `ND_4_EXPONENT_VECTOR`.
9. **Cache and fingerprint.** Cache results by coefficient bytes. Also return a cheap
   `GL_n`-invariant fingerprint, such as the point counts `#X(F_p)` and `#X(F_{p²})`, so the search
   can tell when it has rediscovered the same surface in different coordinates.
10. **Operational:** per-request timeout (a `p = 7` evaluation can take seconds), worker restart on
    failure, a startup warm-up call, and a sysimage or PrecompileTools workload to cut load time.

Before any learning work, the oracle needs a regression suite: Fermat quartics across `p`, a stored
set of random polynomials with heights computed by the current `Qfs` code, and every height-≥2
result re-checked by the slow reference implementation.

## 5. Search method

### 5.1 Assessment of the transformer `f → f'` idea

The idea: at fixed `(p, n, d)`, a transformer reads a polynomial and outputs one of higher height.

- **What it gets right:** polynomials are short sequences of monomial tokens (35 or 126), a natural
  fit for a small transformer, and a model that edits good polynomials is a sensible way to use
  what has been found so far.
- **The weak point is conditioning on `f` with a full rewrite.** The height of `f'` does not depend
  on `f`. If the model may output any `f'`, conditioning on `f` contributes nothing and it is really
  an unconditional generator. Conditioning helps only when `f'` stays near `f`, so the model learns
  *which local edits* tend to climb the staircase.
- **PPO is a poor fit for this reward.** Each attempt is a one-shot draw with a reward that is
  almost always "height 1", so a value network has nothing to learn and the gradient is mostly noise.
  Ranking-based training (keep the best samples and imitate them) uses the same information with
  much less variance.

So I'd keep the transformer but change its job and its training: make it a **masked-coefficient
model** (BERT/MaskGIT-style) that, given an elite polynomial with `k` coefficients masked, fills them
in, conditioned on a target-height token. That is exactly "map `f` to a better `f'`" as a local
edit, and the same model also generates from scratch (mask everything). Train it by
**self-imitation on elites**, not PPO.

### 5.2 Recommended loop (PatternBoost-style)

This follows the pattern of Wagner's cross-entropy method (2021) and PatternBoost (Charton,
Ellenberg, Wagner, Williamson, 2024), both built for "find discrete objects with a property that is
expensive to check and has no gradient".

1. **Generate** a batch of candidates: some by masked resampling of elites (the `f → f'` move,
   with `k` from 1 to ~8), some from scratch, some uniformly random (keeps exploration honest).
2. **Filter cheaply**: compute `c_1` (the F-split test) first; it costs microseconds and removes a
   `(p-1)/p` fraction.
3. **Evaluate** survivors on the batched oracle, then smoothness-check anything above the current
   elite threshold.
4. **Local search** around each new elite: exhaustively try all `p` values of each coefficient
   (35·p neighbours for quartics), keep the best, and repeat while it improves. This is cheap
   because most neighbours fail the `c_1` filter.
5. **Update the elite set**: keep the top polynomials by height, de-duplicated by fingerprint, with
   a cap per height level so lower levels stay represented.
6. **Retrain** the model on the elite set for a few epochs, with data augmentation by permuting
   variables and random `GL_n(F_p)` changes of coordinates (both preserve height, and teach the
   model the symmetry for free).

**Model.** Encoder-only transformer, 4 layers, width 256, one token per monomial. Each token embeds
its coefficient (vocabulary `F_p` plus `[MASK]`) plus the monomial's identity, built as a sum of
per-variable exponent embeddings so that permuting variables is a permutation of inputs. Prepend
tokens for `p` and the target height. It is a few million parameters; the oracle, not the GPU, is
the bottleneck, so a laptop GPU or CPU is enough at first.

**If a policy-gradient method is still wanted**, use REINFORCE with a group-normalized baseline
(score each sample against the mean reward of its batch, as in GRPO) over discrete edits, with no
critic. Treat it as an ablation against self-imitation, not the default.

### 5.3 Approaches considered and not prioritised

- **Surrogate classifier that predicts height from coefficients.** Each `c_k` is a high-degree
  polynomial mod `p` in the coefficients; neural networks are notoriously bad at this kind of
  function (like learning parity). The exact `c_1` filter is cheaper and perfect. Revisit only if
  the elite data shows structure, such as sparsity patterns, that a classifier could pick up.
- **Pure algebraic solving.** "Height `≥ h`" is the system `c_1 = … = c_{h-1} = 0`, so in principle
  a Gröbner-basis solve finds points. With the step in section 4, `c_k` has degree roughly `2p·k` in the coefficients,
  so this is out of reach for `h` near 10, but solving `c_1 = 0` (degree `p-1`) exactly when proposing candidates is cheap
  and saves a factor of `p`. Worth adding to the generator once the loop works.

### 5.4 Where learning can plausibly win

Uniform coefficients are the worst case for learning: the conditions are "random" mod-`p` events.
Gains, if any, come from **structure**: high-height and supersingular examples in the literature
tend to be sparse or highly symmetric (Fermat and Delsarte-type polynomials with few monomials). A
model trained on elites can learn "which supports and patterns tend to be high height" and propose
from that region, where the per-try success rate can be far above `p^-(h-1)`. The experiments are
designed to tell quickly whether this is happening.

## 6. Reward

- **Validity first.** Singular, reducible or zero `f` gets reward 0 (or is dropped from the elite
  set). Non-homogeneous input cannot occur if the generator only emits degree-`d` coefficients.
- **Height on a rarity scale.** `r(f) = (h(f) - 1) · log p`, roughly the surprisal of hitting that
  height at random. It makes runs at different `p` comparable. For elite selection only the ranking
  matters, so this mainly matters for logging and any policy-gradient ablation.
- **Supersingular (`h = ∞`) is tracked separately.** Default: count it as height 10 for reward, so
  the search is not pulled towards it, and log every supersingular example as its own find. If you
  want supersingular surfaces as targets in their own right, give them a bonus instead; that is a
  one-line change.
- **Diversity.** A repeated fingerprint earns nothing new, so the elite set and reward don't
  collapse onto one surface in many coordinate systems.

## 7. Evaluation

**Primary metric:** distinct smooth surfaces (by fingerprint) of height `≥ h` found per CPU-hour,
for each `h` from 2 to 10, at each `p`. Report the oracle-call count too, since oracle speed will
change.

**Baselines every method must beat, under the same budget:**

1. Uniform random sampling (expected rate `≈ p^-(h-1)`, measured empirically).
2. Random sampling with exact `c_1 = 0` solving.
3. Local search / simulated annealing from random starts, without a model.

**Ladder:**

1. `p = 3`, quartics: reach height 10 and supersingular. Random search can already do this, so this
   is the end-to-end correctness test.
2. `p = 5`, quartics: the first real benchmark. Compare time-to-first-height-10 and the full
   per-level rate curve against the baselines.
3. `p = 7` and `p = 11`, quartics: where random search is hopeless (tens of millions to billions of
   samples). Any height-9 or 10 find here is a meaningful result. Also test transfer: start the
   model trained at `p = 5` and see whether it speeds up `p = 7`.
4. Quintic threefolds at small `p`: heights can go far higher (into the tens), so the staircase is
   longer and learning has more room to help. This needs the oracle work first, since the
   126-coefficient, degree-5 computation is much heavier.

**Verification of reported examples:** recompute with the reference `Qfs` implementation, check
smoothness with Singular, confirm invariance by recomputing in random coordinates, and for headline
results cross-check against an independent method (Newton polygon from point counts over
`F_{p^k}`) where it is computationally feasible.

## 8. Order of work

1. Oracle: standalone Julia server, Nemo only, batched API with status codes, the new chain step
   (then the 35×35 form), smoothness check, regression suite.
2. Baselines and their per-level rate curves at `p = 3, 5, 7`.
3. Elite loop with local search only, no model, to make sure infrastructure and dedup work.
4. Add the masked-coefficient transformer; compare against step 3 at `p = 5`.
5. Scale to `p = 7, 11`, then quintics.

## 9. Open questions for you

- Is supersingular (`h = ∞`) a target, or only large finite heights? The default above treats it
  as a separate find, not the goal.
- Which primes matter most? The plan defaults to `p = 5` as the benchmark and `p = 7, 11` as the
  real test.
- Are quintic threefolds in scope now, or after quartics work?

## Appendix: prototype of the faster chain step

Drop-in for `quasiFSplitHeight_CY_lift`, using the existing helpers in `Qfs.jl` (homogeneity check
omitted). This is what produced the timings and the zero-mismatch comparison in section 4.

```julia
function height_fast(p, f, cutoff)
    isFSplit(p, f) && return 1
    D = Δ₁l(p, f)             # Δ₁ of f itself, not of f^(p-1)
    fp2 = f^(p-2)
    θ(g) = -fp2 * polynomial_frobenius_generator(p, D * g)
    g = θ(f^(p-1))
    n = 2
    while n ≤ cutoff
        g == 0 && return cutoff + 2
        !inPowerOfVariableIdeal(p, p, g) && return n
        n += 1
        g = θ(g)
    end
    cutoff + 1
end
```

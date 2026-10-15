# rlmath

Learning-guided search for polynomials of high quasi-F-split height. See `PLAN.md` for the
methodology and `PROOF_fast_height.md` for why the height code's fast step is correct.

## Setup

You need Julia (1.11 tested) and [uv](https://docs.astral.sh/uv/).

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

```bash
uv sync
```

Check everything works. This runs the Julia height tests and the Python tests, which start a few
Julia workers:

```bash
julia --project=Qfs.jl -e 'using Pkg; Pkg.test()'
```

```bash
uv run pytest
```

## Running

```bash
uv run rlmath-train --p 5 --max-oracle-calls 100000
```

The two baselines, for comparison at the same budget:

```bash
uv run rlmath-train --p 5 --max-oracle-calls 100000 --no-use-model
```

```bash
uv run rlmath-baseline --p 5 --max-oracle-calls 100000
```

**Watching a run.** In a terminal, a progress bar shows the oracle calls used out of the budget,
the speed, the time remaining, the best height so far and what the round is doing right now. After
each round a summary line prints the new finds and the elite pool by height. The same lines go to
`runs/<name>/train.log`, so `tail -f runs/<name>/train.log` works for a run started in the
background. `uv run rlmath-summarize runs/<name>` gives the comparison table at any point,
including mid-run.

Every polynomial whose height is computed, by any run, is also saved to the SQLite file
`runs/heights.sqlite` (change with `--db`, or `--db ""` to turn it off). Each row holds the
setting, the coefficients, the status and the height. `src/rlmath/store.py` documents the tables
and shows how to read them.

Each run writes to `runs/<name>/`: `config.json`, `log.jsonl` (one line per round), `elites.jsonl`
(every new polynomial of score at least 2, with when it was found) and `model.pt`.
`uv run rlmath-train --help` lists every option. `runs_compare.sh` runs all three at p = 5.

## Debugging the height worker

The worker is an ordinary Julia script, so it can be run by hand. First check that the Julia
environment installs and loads (this shows the real error if a package is missing):

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate(); using Qfs, Nemo, Singular; println("ok")'
```

Then talk to the worker directly. It prints `READY` once loaded, then one reply per request line.
The polynomial here is the Fermat quartic `w⁴ + x⁴ + y⁴ + z⁴`, which should give `ok 1` at
p = 5, `inf` at p = 3, and `1` (smooth) for the last line:

```bash
printf 'H 5 4 4 10 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1\nH 3 4 4 10 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1\nS 5 4 4 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1\n' | julia --project=julia julia/height_worker.jl
```

Python starts workers with plain `julia` from the `PATH`. If Julia lives somewhere else, put it on
the `PATH` before running.

## How the pieces fit

| Piece | File | What it does |
|---|---|---|
| Height code | `Qfs.jl/src/Qfs.jl` | The quasi-F-split height algorithm (fast version), on Nemo |
| Height worker | `julia/height_worker.jl` | Standalone Julia process: reads polynomials on stdin, writes heights or smoothness on stdout |
| Oracle client | `src/rlmath/oracle.py` | Runs one worker per CPU core, sends batches, caches results, counts oracle calls |
| Polynomials | `src/rlmath/polynomials.py` | Coefficient vectors, monomial order, height-preserving symmetries |
| Model | `src/rlmath/model.py` | Masked-coefficient transformer (fills in hidden coefficients) |
| Search | `src/rlmath/search.py` | Scoring with the smoothness check, the elite pool, hill climbing |
| Training loop | `src/rlmath/train.py` | Propose, score, hill-climb, keep elites, retrain, repeat |
| Baseline | `src/rlmath/baseline.py` | Uniform random sampling with the same logging |

Julia never runs inside Python. Each worker is its own process, which removes the "bus error"
class of crashes, and a worker that dies is restarted.

## Why these choices and defaults

**Algorithm.** The search works like PatternBoost (Charton, Ellenberg, Wagner and Williamson,
2024) and Wagner's cross-entropy method (2021). Both target the same kind of problem: find
discrete objects with a property that is expensive to check and gives no gradient. Nothing here is
PPO or another value-based RL method. Every candidate gets a one-shot score that is almost always
"height 1", so a value network has nothing to learn. Imitating the best finds uses the same
information with far less noise.

**Model.** `torch.nn.TransformerEncoder` in plain PyTorch. That is the most common way to write a
small transformer, and the model is too small to need a framework. The defaults are 4 layers,
width 128, 4 heads, feed-forward 512, dropout 0.1 and pre-norm, about 0.8M parameters. The training
data is a few thousand sequences of 35 tokens, so a bigger model would only be slower; at width
256 a training step was twice as slow on this laptop. Each token is a coefficient plus a monomial
embedding, built as a sum of per-variable exponent embeddings. A height token in front says what
height to aim for.

**Training objective.** The MaskGIT objective: hide a random fraction of the coefficients on a
cosine schedule, at least one, and predict them with cross-entropy. 10% of the time the height
token is hidden, so the model also learns to generate without a target. Generation fills masked
positions over 8 steps and keeps the most confident samples first.

**Optimiser.** AdamW with learning rate 3e-4, betas (0.9, 0.95), weight decay 0.01, 500 warm-up
steps, gradient clipping at 1.0 and batch 256. These are the standard transformer defaults.
Training runs 100 steps per round on the elite pool. The model keeps training across rounds
rather than restarting, because the pool changes gradually.

**Data augmentation.** Each training batch has random symmetries applied that don't change the
height: permuting the variables, scaling each variable, and scaling the whole polynomial.

**Proposals per round.** 1024 candidates: 50% elite edits, 25% generated from scratch and 25%
uniform random.
- An edit re-samples 1 to 8 coefficients of an elite polynomial. This is the "f → f′" move.
- Parents are drawn favouring higher levels: weight exp(height) for each level, uniform within a
  level.
- Edits and generations ask for the best height seen so far.
- The random share keeps exploration honest and covers the case where the model is wrong.

**Local search.** Steepest-ascent hill climbing from the 8 best new finds of each round, for up
to 5 steps. Each step tries all 35·(p−1) single-coefficient changes. Most of those fail the cheap
F-split test, so this costs little.

**Score.** The score is the height. Anything scoring 2 or more must pass the smoothness check or
it scores 0, because singular polynomials give heights that mean nothing. Supersingular (infinite
height) polynomials score 1 by default, the same as an ordinary one. In a first p = 5 run that
rewarded them as height 10, the search found over a thousand sparse supersingular quartics within
a few rounds (for example `3w³x + 2wz³ + 4x⁴ + xy³ + y⁴`) and spent its effort there. Use
`--infinite-score 10` to target them deliberately.

**Elite pool.** Distinct polynomials, with a cap of 2000 per height level so lower levels stay
represented in the training data. "Distinct" means up to permuting variables, scaling variables
and scaling the polynomial. Without that, one surface written in many coordinates counted as many
finds.

**What to compare.** Distinct smooth finds at each height per oracle call, against
`--no-use-model` and `rlmath-baseline` at the same budget. Results at a single seed are noisy, so
use several seeds before concluding anything.

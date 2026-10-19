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
uv run rlmath-reinforce --p 5 --total-timesteps 1000000 --track
```

`--track` logs to Weights and Biases (run `wandb login` first, or set `WANDB_MODE=offline`). Without
it, each iteration still prints a summary line: the mean reward, the share of children that beat
or fell below their parent, the mean Hamming distance from parent to child, the best score so far
and the oracle calls used. `uv run rlmath-reinforce --help` lists every option.

Every polynomial whose height is computed, by any run, is saved to the SQLite file
`runs/heights.sqlite` (change with `--db`, or `--db ""` to turn it off). Each row holds the
setting, the coefficients, the status and the height. `src/rlmath/store.py` documents the tables
and shows how to read them. With `--parent-db-frac 0.5`, half of the parents are drawn from the
smooth polynomials of height at least `--parent-min-height` saved in that file when the run starts.

Each run writes its model to `runs/<name>/model.pt`.

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
| Polynomials | `src/rlmath/polynomials.py` | Coefficient vectors and the monomial order |
| Environment | `src/rlmath/env.py` | Gymnasium vector env: parent in, child out, reward `score(child) - score(parent)` |
| Agent and training | `src/rlmath/reinforce.py` | Transformer policy and a REINFORCE loop, laid out like a CleanRL script |

Julia never runs inside Python. Each worker is its own process, which removes the "bus error"
class of crashes, and a worker that dies is restarted.

## Design

**Environment.** One episode is one step. The observation is a parent polynomial, a vector of 35
coefficients in F_p (quartics). The action is a child polynomial, a full coefficient vector. The
reward is `score(child) - score(parent) - hamming_penalty * (coefficients that differ)`. `PolyEnv` runs `--num-envs` episodes as one batch with one
oracle call, and `step` returns the next parents. Parents are uniformly random, or partly drawn from
the database (`--parent-db-frac`).

**Score.** The score is the height. Anything scoring 2 or more must pass the smoothness check or it
scores 0, because singular polynomials give heights that mean nothing. A zero polynomial scores 0.
Supersingular (infinite height) polynomials score 1 by default, the same as an ordinary one. In an
early p = 5 run that rewarded them as height 10, the search found over a thousand sparse
supersingular quartics within a few rounds (for example `3w³x + 2wz³ + 4x⁴ + xy³ + y⁴`) and spent
its effort there. Use `--infinite-score 10` to target them deliberately.

**Agent.** Each coefficient is embedded (a table for the value in F_p, plus a monomial embedding
built as a sum of per-variable exponent embeddings), the 35 tokens go through a
`torch.nn.TransformerEncoder` (4 layers, width 128, 4 heads, feed-forward 512, pre-norm, no
dropout), and a linear layer gives p logits per coefficient. The child is sampled, coefficient by
coefficient, from those softmaxes. The log-probability of the child is the sum of the 35. There is no value head.

**Algorithm.** REINFORCE. Episodes have one step, so each iteration scores `--num-envs` children
and takes one gradient step on `-(reward - baseline) * log pi(child | parent)`, with no critic.
The reward contains `-score(parent)`, which depends on the parent alone, so subtracting it
changes nothing in expectation and only removes noise. The baseline is therefore that term plus the
batch mean of the rest; in effect the advantage is `(child score - Hamming penalty)` centred on the
batch mean, divided by its standard deviation. The layout (a dataclass of arguments parsed by
`tyro`, optional wandb, one loop) follows CleanRL's scripts, but nothing is imported from it. The
entropy bonus is off by default (`--ent-coef 0`).

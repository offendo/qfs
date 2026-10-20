# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`rlmath`: learning-guided search for homogeneous polynomials of degree `d` in `n` variables over `F_p`
(default `n = d = 4`, quartic K3 surfaces; 35 coefficients) with high quasi-F-split height. A polynomial is
a vector of coefficients in `0..p-1` over a fixed monomial order. Height 1 is generic; each extra level is
roughly a `1/p` event, and there is no partial credit inside a level.

The learner is a one-step policy: a transformer reads a parent polynomial and writes a child, trained with
REINFORCE (group baseline, no critic; deliberately not PPO) on reward `height(child) - height(parent)` plus
penalties. The cost metric is **oracle calls** (height computations). `README.md` has the design and run
instructions; `PLAN.md` the methodology; `PROOF_fast_height.md` why the fast height step equals the original;
`experiments-notes.md` the summary of the supervised/single-edit experiments and `experiments/LOG.md` every
experiment (what, why, result, including corrections). Read the notes before proposing a new experiment.

## Setup and commands

Python (uv) and Julia (1.11) are both required.

```bash
uv sync
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
uv run pytest                                   # starts real Julia workers; ~3 minutes
uv run pytest tests/test_env.py::test_name      # single test
uv run rlmath-reinforce --p 3 --total-timesteps 1500000 [--track]   # REINFORCE; --help lists every option
uv run python experiments/count_pairs.py <heights.sqlite> 3 pairs_p3.npz        # mine (parent, one-edit child) pairs
uv run rlmath-sft --pairs pairs_p3.npz --out sft.pt [--single] --require ge     # supervised warm start
uv run rlmath-reinforce --p 3 --init sft.pt [--single] ...                      # RL from the warm start
uv run python experiments/run_rl.py <name> <snapshot.sqlite> <out_dir> [args]   # one run on its own DB copy
uv run python experiments/evaluate.py --help    # common-parent comparison against random edits
```

There is no linter/formatter config. The README mentions `julia --project=Qfs.jl -e 'using Pkg; Pkg.test()'`,
but `Qfs.jl/` has no `test/` directory, so there are currently no Julia tests.

## Architecture

- **Julia never runs inside Python.** `src/rlmath/oracle.py` spawns Julia worker processes
  (`julia/height_worker.jl`) and talks to them with a line protocol over stdin/stdout (`MONOS`, `H`, `S`
  requests; see the header of the worker). Dead workers are restarted. Nothing else may print to the
  worker's stdout. The old juliacall/ZMQ bridge was removed (it caused "bus error" crashes).
- **Monomial order is a cross-language contract.** `rlmath.polynomials.monomials` must match the
  worker's `monomials` (descending lex); the oracle checks this at startup. Change both together.
- **Height algorithm** lives in the `Qfs.jl` git submodule (`Qfs.jl/src/Qfs.jl`, Nemo/Singular). The
  `julia/` project points at it via `[sources]`. It is a separate repo (`offendo/Qfs.jl`), so changes
  there are committed and pushed in the submodule, then the pointer is bumped here.
  The pinned commit's `quasiFSplitHeight_CY_lift` still uses the original slow step
  (`Δ₁l(p, f^(p-1))`, despite the README calling it the fast version). The faster step
  `θ(g) = -f^(p-2)·u(Δ(f)·g)` exists as a prototype in the `PLAN.md` appendix, and
  `PROOF_fast_height.md` proves it equivalent; keep that argument intact if you change the chain step.
  In a fresh clone or worktree run `git submodule update --init Qfs.jl` first.
- **Scoring (`env.py`, `score`):** score = height; anything scoring >= 2 must pass the smoothness check
  (`S` request) or it scores 0. Infinite height scores 1 by default (`--infinite-score` to change),
  because sparse supersingular quartics flood the search otherwise.
- **Environment (`env.py`):** `PolyEnv` is a Gymnasium `VectorEnv` of one-step episodes. Observation = parent,
  action = child, reward = `score(child) - score(parent)` minus optional penalties (Hamming beyond a free
  budget, exact copy, repeat of a child already generated this run). Parents come from a `ParentPool` seeded
  from the SQLite database once it is big enough, weighted toward higher heights (`parent_temperature`) and
  capped by a curriculum ceiling. Each parent is repeated `children_per_parent` times so the group mean
  can serve as the baseline.
- **Policy and training (`reinforce.py`):** `Agent` = coefficient + per-variable exponent embeddings, a
  transformer encoder, a linear head over `p` values. Default factorized policy samples every coefficient
  independently; `--single` samples one (position, new value) pair from a single softmax (exactly one
  change, exact log-prob; masked logits need the NaN-safe entropy and `policy_kl` already in the file).
  Logs include exact per-step KL, KL to the `--init` policy, and pool/random parent splits.
- **Supervised warm start (`pairs.py`, `sft.py`):** pairs are found in a heights DB by integer-encoding
  coefficient vectors in base `p` (p = 2 or 3 only, `p^35 < 2^63`); SFT maximizes log-prob of the child
  given the parent and can augment with variable renaming (height-preserving).
- **Equivalence:** polynomials are the same up to scaling the whole polynomial; the pool, the seen set and
  the pair finder store the scaled form (first nonzero coefficient 1). Renaming variables also preserves height.
- **Persistence:** every computed height goes to a SQLite file (`store.py`, default `runs/heights.sqlite`);
  each run writes `runs/<name>/model.pt`. `runs/` is gitignored.

## Working notes

- Results at a single seed are noisy (RL runs here froze or collapsed on some seeds). Compare several
  seeds before concluding a change helps, and use the common-parent evaluation, not training curves.
- A run writes every scored polynomial into `--db`: point it at a copy, never at a database another job
  is reading. Delete the `-wal`/`-shm` files together with a database you replace.
- Check an evaluation's exact-copy handling before trusting a "retention" number (an earlier result was an
  artifact of this; see the erratum in `experiments/LOG.md`).
- When debugging the worker, run it by hand (README section "Debugging the height worker" has a
  known-answer input: Fermat quartic gives `ok 1` at p=5, `inf` at p=3).
- Tests share Julia workers via the session-scoped `oracle_factory` fixture in `tests/conftest.py`;
  reuse it instead of constructing `HeightOracle` per test (workers take seconds to start).
- `.python-version` and the `.venv` are uv-managed; the `MMPSingularities.jl` entry in `.gitmodules`
  has no checked-out directory and isn't used.

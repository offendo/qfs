# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`rlmath`: learning-guided search (PatternBoost / cross-entropy style, **not** PPO) for homogeneous
polynomials of degree `d` in `n` variables over `F_p` (default `n = d = 4`, quartic K3 surfaces) with
high quasi-F-split height. A polynomial is a vector of coefficients in `0..p-1` over a fixed monomial
order. Height 1 is generic; each extra level is roughly a `1/p` event, and there is no partial credit
inside a level. The cost metric is **oracle calls** (height computations), so experiments are compared
at equal oracle budget. `README.md` has the design rationale and defaults; `PLAN.md` the methodology;
`PROOF_fast_height.md` why the fast height step equals the original.

## Setup and commands

Python (uv) and Julia (1.11) are both required.

```bash
uv sync
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
uv run pytest                                   # starts real Julia workers; slow first run
uv run pytest tests/test_train.py::test_name    # single test
uv run rlmath-train --p 5 --max-oracle-calls 100000   # model-guided search
uv run rlmath-train --p 5 --max-oracle-calls 100000 --no-use-model   # local search only
uv run rlmath-baseline --p 5 --max-oracle-calls 100000                # uniform random
uv run rlmath-summarize runs/<name>             # comparison table, works mid-run
./runs_compare.sh                               # all three at p=5
```

`uv run rlmath-train --help` lists all options. There is no linter/formatter config.
The README mentions `julia --project=Qfs.jl -e 'using Pkg; Pkg.test()'`, but `Qfs.jl/` has no `test/`
directory, so there are currently no Julia tests.

## Architecture

- **Julia never runs inside Python.** `src/rlmath/oracle.py` spawns one `julia/height_worker.jl`
  process per CPU core and talks to it with a line protocol over stdin/stdout (`MONOS`, `H`, `S`
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
- **Scoring (`search.py`):** score = height; anything scoring >= 2 must pass the smoothness check
  (`S` request) or it scores 0. Infinite height scores 1 by default (`--infinite-score` to change),
  because sparse supersingular quartics flood the search otherwise.
- **Equivalence:** polynomials are "the same" up to permuting variables, scaling variables and scaling
  the whole polynomial (`polynomials.py`). The elite pool dedups on this and caps 2000 per height level;
  training augmentation applies the same height-preserving symmetries.
- **Training loop (`train.py`):** propose (50% elite edits, 25% model generation, 25% uniform), score
  via oracle, hill-climb from the best new finds, update elites, train the masked-coefficient
  transformer (`model.py`, MaskGIT-style objective, height-conditioning token) for 100 steps, repeat.
- **Persistence:** every computed height goes to `runs/heights.sqlite` (`store.py`); each run writes
  `runs/<name>/{config.json,log.jsonl,elites.jsonl,train.log,model.pt}`. `runs/` is gitignored.

## Working notes

- Results at a single seed are noisy; compare several seeds before concluding a design change helps.
- When debugging the worker, run it by hand (README section "Debugging the height worker" has a
  known-answer input: Fermat quartic gives `ok 1` at p=5, `inf` at p=3).
- Tests share Julia workers via the session-scoped `oracle_factory` fixture in `tests/conftest.py`;
  reuse it instead of constructing `HeightOracle` per test (workers take seconds to start).
- `.python-version` and the `.venv` are uv-managed; the `MMPSingularities.jl` entry in `.gitmodules`
  has no checked-out directory and isn't used.

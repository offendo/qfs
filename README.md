# rlmath

Learning-guided search for polynomials of high quasi-F-split height. See `PLAN.md` for the
methodology and `PROOF_fast_height.md` for why the height step is correct.

## Setup

You need Julia and [uv](https://docs.astral.sh/uv/).

```bash
git submodule update --init Qfs.jl
julia --project=Qfs.jl -e 'using Pkg; Pkg.instantiate()'
uv sync
uv run pytest    # starts its own height servers on ports 8080 and 8081
```

## Running

Heights come from HTTP servers written in Julia (`Qfs.jl/Server.jl`). Start N of them, on ports
8080 to 8080+N-1, in a separate terminal:

```bash
Qfs.jl/serve.sh 8
```

Then train, listing the servers:

```bash
uv run rlmath-reinforce --p 5 --total-timesteps 1000000 --urls http://127.0.0.1:8080 http://127.0.0.1:8081 --track
```

`--track` logs to Weights and Biases (run `wandb login` first, or set `WANDB_MODE=offline`). Without
it, each iteration still prints a summary line: the mean reward, the share of children that beat
or fell below their parent, the mean Hamming distance from parent to child, the best score so far
and the oracle calls used. `uv run rlmath-reinforce --help` lists every option.

Every polynomial whose height is computed, by any run, is saved to the SQLite file
`runs/heights.sqlite` (change with `--db`, or `--db ""` to turn it off). Each row holds the
setting, the coefficients, the status and the height. `src/rlmath/store.py` documents the tables
and shows how to read them. By default (`--parent-db-frac 0.5`), half of the parents are drawn from a pool of smooth
polynomials of height at least `--parent-min-height`. The pool starts with the ones saved in that
file when the run starts and grows with every such polynomial the run scores. Until it holds
`--parent-db-min-size` polynomials (default 1000), all parents are random. Among pool parents, a
height level is chosen with probability proportional to `exp(height / --parent-temperature)` and
then a polynomial of that level uniformly, so a lower temperature favours higher heights (default 3;
1 puts most parents at the top level, a very large value gives equal weight to every level).

Each run writes its model to `runs/<name>/model.pt`.

## Supervised warm start, the single-edit policy and the experiments

- `experiments/count_pairs.py` finds the (parent, one-coefficient child) pairs stored in a heights database
  (`src/rlmath/pairs.py`; p = 2 or 3).
- `uv run rlmath-sft --pairs pairs.npz --out sft.pt [--single]` trains the policy to imitate pairs whose child is at least as high
  as the parent. `rlmath-reinforce --init sft.pt [--kl-coef 0.1]` starts reinforcement learning from it (the model size and
  `--single` must match).
- `--single` (in both) samples exactly one (position, new value) pair from a single softmax over the 35 x (p - 1) changes
  instead of every coefficient independently: no copies, exact log-probability.
- `experiments/evaluate.py` compares policies and random edits on identical parents; `experiments/parent_dependence.py` tests
  whether a policy uses its input. The results of the overnight batch are in `experiments-notes.md` (summary) and
  `experiments/LOG.md` (every experiment), with the key checkpoints in `experiments/checkpoints/`.

## Checking the height server

Start a server and ask for the Fermat quartic `w⁴ + x⁴ + y⁴ + z⁴` (a polynomial is one line of 35
coefficients). The height is `ok 1` at p = 5 and `inf` at p = 3, and the smoothness is `1`:

```bash
julia --project=Qfs.jl Qfs.jl/Server.jl 8080 &
F="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1"
echo "$F" | curl -s --data-binary @- "localhost:8080/height?p=5&n=4&d=4&cutoff=10"
echo "$F" | curl -s --data-binary @- "localhost:8080/smooth?p=5&n=4&d=4"
```

## How the pieces fit

| Piece | File | What it does |
|---|---|---|
| Height code | `Qfs.jl/src/Qfs.jl` | Quasi-F-split height, on Nemo |
| Height server | `Qfs.jl/Server.jl` | HTTP endpoints `/height`, `/smooth`, `/monomials` |
| Oracle client | `src/rlmath/oracle.py` | Sends batches to the servers, caches results, counts oracle calls |
| Polynomials | `src/rlmath/polynomials.py` | Coefficient vectors and the monomial order |
| Environment | `src/rlmath/env.py` | Gymnasium vector env: parent in, child out, reward `score(child) - score(parent)` |
| Agent and training | `src/rlmath/reinforce.py` | Transformer policy and a REINFORCE loop |

## Design

**Environment.** One episode is one step. The observation is a parent polynomial, a vector of 35
coefficients in F_p (quartics). The action is a child polynomial, a full coefficient vector. The
reward is `score(child) - score(parent) - hamming_penalty * max(0, distance - hamming_free) - copy_penalty * (child is exactly the parent)`,
where distance is the number of coefficients that differ. `PolyEnv` runs `--num-envs` episodes as one batch with one
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
Each iteration uses `--num-envs / --children-per-parent` parents (default 1024 / 128 = 8) and samples
`--children-per-parent` independent children for each. The baseline is the mean reward of a parent's
children, so a child is judged against other edits of the same parent, and the parent's own score,
which is the same for the whole group, cancels. The advantage is divided by its standard deviation
over the batch. A group where every child has the same reward contributes no gradient.

**Curriculum.** Pool parents are capped at a height ceiling that starts at `--curriculum-start`
(default 2; 0 turns the curriculum off). Whenever the share of children at least as high as their
parent, averaged over the last `--curriculum-window` iterations (20), reaches
`--curriculum-threshold` (0.25), the ceiling rises by one, up to the cutoff. The ceiling and the
share of parents for which some child beat the parent (`group_hit_rate`) are logged. Random parents
are not capped. The layout (a dataclass of arguments parsed by
`tyro`, optional wandb, one loop) follows CleanRL's scripts, but nothing is imported from it. The
entropy bonus is off by default (`--ent-coef 0`).

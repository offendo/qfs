"""Hill-climbing against best-of-k sampling, at equal numbers of oracle calls, on the same parents.

For every parent, `--per-parent` trajectories of `--steps` single-coefficient edits are played (one oracle call per step) with
a proposal rule, either as a hill-climb ("accept if not lower": the next edit starts from the best-so-far polynomial) or
as independent edits of the parent ("best of k"). The reported gain after k steps is  best height reached - parent height.
Proposal rules: random single flips, or a single-edit checkpoint (--ckpt name=path). Use a COPY of the database.

    uv run python experiments/climb_eval.py --db copy.sqlite --ckpt q2=model.pt --out climb.json
"""

import argparse
import json
import time

import numpy as np
import torch

from evaluate import load_agent
from rlmath.climb import ClimbEnv, play, policy_proposer, random_flips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ceilings", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--ckpt", action="append", default=[], help="name=path")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--modes", nargs="+", default=["walk", "bestof"], choices=["walk", "bestof"])
    ap.add_argument("--accept", default="always", choices=["always", "not_lower"], help="rule of the hill-climb rows")
    ap.add_argument("--per-parent", type=int, default=32)
    ap.add_argument("--parents-per-batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rules = {"random": lambda rng: (lambda obs, score, t: random_flips(rng, obs, args.p))}
    for item in args.ckpt:
        name, path = item.split("=", 1)
        agent = load_agent(path, device)
        rules[name] = lambda rng, agent=agent: policy_proposer(agent, device)

    env = ClimbEnv(
        args.per_parent * args.parents_per_batch, args.p, workers=args.workers, db=args.db, parent_db_frac=1.0,
        parent_temperature=args.temperature, children_per_parent=args.per_parent, steps=args.steps, accept=args.accept,
    )
    env._remember = lambda coeffs, scores: None  # frozen pool: identical parents for every method
    results = {}
    for ceiling in args.ceilings:
        for walk in [m == "walk" for m in args.modes]:
            env.walk = walk
            for name, make in rules.items():
                t = time.time()
                env.parent_max_height = ceiling
                env.reset(seed=1000 + ceiling)
                gain, hit, total = [], [], []
                for _ in range(args.batches):
                    game = play(env, make(np.random.default_rng(len(gain))))
                    g = game["best"] - env.parent_scores[None]  # (steps, N) gain after each step
                    gain.append(g)
                    total.append(np.maximum(game["child_score"] - env.parent_scores[None], 0).cumsum(axis=0))
                    hit.append((g > 0).reshape(args.steps, -1, args.per_parent).any(axis=2))  # some trajectory of the parent
                    env.reset()
                gain, hit, total = np.concatenate(gain, axis=1), np.concatenate(hit, axis=1), np.concatenate(total, axis=1)
                key = f"{'climb' if walk else 'bestof'}-{name}@{ceiling}"
                results[key] = {
                    "gain": gain.mean(axis=1).tolist(), "total": total.mean(axis=1).tolist(), "success": (gain > 0).mean(axis=1).tolist(),
                    "up2": (gain >= 2).mean(axis=1).tolist(), "hit": hit.mean(axis=1).tolist(), "n": int(gain.shape[1]),
                }
                r = results[key]
                print(f"ceiling {ceiling}  {'climb ' if walk else 'bestof'} {name:12s} gain@{args.steps} {r['gain'][-1]:+.3f}  summed reward {r['total'][-1]:.3f}  "
                      f"success@1/4/{args.steps} {100 * r['success'][0]:.1f}/{100 * r['success'][min(3, args.steps - 1)]:.1f}/"
                      f"{100 * r['success'][-1]:.1f}%  gain>=2 @{args.steps} {100 * r['up2'][-1]:.1f}%  "
                      f"parent hit@{args.steps} {100 * r['hit'][-1]:.1f}%  [{time.time() - t:.0f}s]", flush=True)
    env.close()
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

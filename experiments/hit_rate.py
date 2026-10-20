"""Group hit rate: sample `--children` children from each parent; how often does at least one beat the parent?

Same parents for every method (frozen pool, same seed), at several height ceilings. Use a COPY of the database.

    uv run python experiments/hit_rate.py --db copy.sqlite --ceilings 2 4 6 8 --baselines 1 2 3 --ckpt q2=model.pt --out hits.json

"hit" = some child scores strictly higher than its parent; "hit_ge" = some child that is not an exact copy scores at
least as high. "per_child" is the share of all children that beat their parent.
"""

import argparse
import json
import time

import numpy as np
import torch

from evaluate import load_agent, policy_edit, random_edit
from rlmath.env import PolyEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ceilings", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--baselines", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--ckpt", action="append", default=[], help="name=path")
    ap.add_argument("--children", type=int, default=128)
    ap.add_argument("--parents-per-batch", type=int, default=64)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PolyEnv(
        args.children * args.parents_per_batch, args.p, workers=args.workers, db=args.db, parent_db_frac=1.0,
        parent_temperature=args.temperature, children_per_parent=args.children,
    )
    env._remember = lambda coeffs, scores: None  # frozen pool: every method sees the same parents
    rng = np.random.default_rng(0)
    methods = {f"random-{k}": random_edit(rng, args.p, k) for k in args.baselines}
    for item in args.ckpt:
        name, path = item.split("=", 1)
        methods[name] = policy_edit(load_agent(path, device), device)

    results = {}
    for ceiling in args.ceilings:
        for name, act in methods.items():
            t = time.time()
            env.parent_max_height = ceiling
            obs, _ = env.reset(seed=1000 + ceiling)
            hit, hit_ge, per_child, parent = [], [], [], []
            for _ in range(args.batches):
                obs, _, _, _, info = env.step(act(obs))
                shape = (-1, args.children)
                c, q, ham = info["child_score"].reshape(shape), info["parent_score"].reshape(shape), info["hamming"].reshape(shape)
                hit.append((c > q).any(axis=1))
                hit_ge.append(((c >= q) & (ham > 0)).any(axis=1))
                per_child.append((c > q).mean(axis=1))
                parent.append(q[:, 0])
            r = {
                "hit": float(np.concatenate(hit).mean()), "hit_ge": float(np.concatenate(hit_ge).mean()),
                "per_child": float(np.concatenate(per_child).mean()), "parent": float(np.concatenate(parent).mean()),
                "parents": int(sum(len(h) for h in hit)),
            }
            results[f"{name}@{ceiling}"] = r
            print(f"ceiling {ceiling}  {name:16s} parent {r['parent']:.2f}  hit {100 * r['hit']:5.1f}%  hit_ge {100 * r['hit_ge']:5.1f}%  "
                  f"per-child {100 * r['per_child']:5.2f}%  parents {r['parents']}  [{time.time() - t:.0f}s]", flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
    env.close()


if __name__ == "__main__":
    main()

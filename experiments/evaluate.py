"""Compare edit policies with random edits, on the same pool parents, at several height ceilings.

For every ceiling, each method sees exactly the same parents (same seed, and the parent pool is frozen), samples one
child per parent, and is scored on  child score - parent score  (no penalties). Random baselines change k randomly
chosen coefficients to a different value. Use a COPY of the database: the environment writes to it.

    uv run python experiments/evaluate.py --db copy.sqlite --p 3 --ceilings 2 3 4 5 6 --baselines 1 2 3 \
        --ckpt scratch=runs/a/model.pt --ckpt sft=runs/sft.pt --out results.json
"""

import argparse
import json
import time

import numpy as np
import torch

from rlmath.env import PolyEnv
from rlmath.reinforce import Agent


def load_agent(path: str, device) -> Agent:
    ckpt = torch.load(path, map_location=device)
    a = ckpt["args"]
    agent = Agent(a["p"], a["n"], a["d"], a["d_model"], a["n_layers"], a["n_heads"], a["d_ff"], a.get("single", False)).to(device)
    agent.load_state_dict(ckpt["model"])
    return agent.eval()


def random_edit(rng: np.random.Generator, p: int, k: int):
    def act(obs: np.ndarray) -> np.ndarray:
        child = obs.copy()
        pos = np.argsort(rng.random(child.shape), axis=1)[:, :k]
        old = np.take_along_axis(child, pos, axis=1)
        np.put_along_axis(child, pos, (old + rng.integers(1, p, size=old.shape)) % p, axis=1)
        return child

    return act


def policy_edit(agent: Agent, device, resample_copies: bool = True, tries: int = 20):
    """Sample a child per parent. The environment penalizes an exact copy, so by default copies are drawn again
    (up to `tries` times); otherwise a policy that copies often would get "same" results for free."""

    def act(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=device)
            child = agent.get_action(obs_t)[0]
            for _ in range(tries if resample_copies else 0):
                copied = (child == obs_t).all(dim=1)
                if not copied.any():
                    break
                child[copied] = agent.get_action(obs_t[copied])[0]
            return child.cpu().numpy()

    return act


def run_cell(env: PolyEnv, act, ceiling: int, batches: int, seed: int) -> dict:
    env.parent_max_height = ceiling
    obs, _ = env.reset(seed=seed)
    gain, parent, child, ham = [], [], [], []
    for _ in range(batches):
        obs, _, _, _, info = env.step(act(obs))
        gain.append(info["child_score"] - info["parent_score"])
        parent.append(info["parent_score"])
        child.append(info["child_score"])
        ham.append(info["hamming"])
    gain, parent, child, ham = map(np.concatenate, (gain, parent, child, ham))
    by_parent = {  # for each exact parent height: how many, how often the child is at least as high, mean child
        str(h): {"n": int((parent == h).sum()), "retain": float((gain[parent == h] >= 0).mean()), "child": float(child[parent == h].mean())}
        for h in np.unique(parent)
    }
    return {
        "by_parent": by_parent,
        "improvement": float(gain.mean()), "up": float((gain > 0).mean()), "same": float((gain == 0).mean()),
        "down": float((gain < 0).mean()), "child": float(child.mean()), "parent": float(parent.mean()),
        "hamming": float(ham.mean()), "copied": float((ham == 0).mean()), "n": int(len(gain)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ceilings", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--baselines", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--ckpt", action="append", default=[], help="name=path")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=3.0, help="parent temperature")
    ap.add_argument("--keep-copies", action="store_true", help="do not redraw children that equal their parent")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PolyEnv(1024, args.p, workers=args.workers, db=args.db, parent_db_frac=1.0, parent_temperature=args.temperature)
    env._remember = lambda coeffs, scores: None  # freeze the pool, so every method sees identical parents
    rng = np.random.default_rng(0)
    methods = {f"random-{k}": random_edit(rng, args.p, k) for k in args.baselines}
    for item in args.ckpt:
        name, path = item.split("=", 1)
        methods[name] = policy_edit(load_agent(path, device), device, resample_copies=not args.keep_copies)

    results = {}
    for ceiling in args.ceilings:
        for name, act in methods.items():
            t = time.time()
            results[f"{name}@{ceiling}"] = run_cell(env, act, ceiling, args.batches, seed=1000 + ceiling)
            r = results[f"{name}@{ceiling}"]
            print(f"ceiling {ceiling}  {name:14s} parent {r['parent']:.2f}  child {r['child']:.2f}  improvement {r['improvement']:+.3f}  "
                  f"up {100 * r['up']:5.2f}%  same {100 * r['same']:5.2f}%  down {100 * r['down']:5.1f}%  hamming {r['hamming']:5.2f}  "
                  f"copied {100 * r['copied']:4.1f}%  [{time.time() - t:.0f}s]", flush=True)

    # each checkpoint against the random baseline whose k is closest to its mean Hamming distance
    if args.baselines:
        print("\nimprovement minus that of the distance-matched random baseline (positive = better than random):")
        for ceiling in args.ceilings:
            cells = []
            for name in methods:
                if name.startswith("random-"):
                    continue
                r = results[f"{name}@{ceiling}"]
                k = min(args.baselines, key=lambda b: abs(b - r["hamming"]))
                base = results[f"random-{k}@{ceiling}"]
                cells.append(f"{name} {r['improvement'] - base['improvement']:+.3f} (vs random-{k}, up {100 * r['up']:.1f}% vs {100 * base['up']:.1f}%)")
            print(f"  ceiling {ceiling}: " + "; ".join(cells))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
    env.close()


if __name__ == "__main__":
    main()

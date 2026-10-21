"""Does a policy use its input? Compare its own edit with an edit it chose for a different parent.

For each parent A in a batch, a "mismatched" child applies to A the edit that the policy produced for another parent B
of the same batch (the same coefficient positions, shifted by the same amounts mod p). If mismatched edits do as well
as the policy's own, the policy is just a fixed distribution over edits that ignores its input.

    uv run python experiments/parent_dependence.py --db copy.sqlite --ceilings 2 4 6 8 --ckpt q2=runs/q2/model.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import load_agent, policy_edit, random_edit, run_cell  # noqa: E402

from rlmath.env import PolyEnv  # noqa: E402


def mismatched_edit(agent, device, p: int, rng: np.random.Generator):
    own = policy_edit(agent, device)

    def act(obs: np.ndarray) -> np.ndarray:
        other = rng.permutation(len(obs))
        shift = (own(obs[other]) - obs[other]) % p  # the edit chosen for parent B, as per-coefficient shifts
        return (obs + shift) % p

    return act


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ceilings", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--ckpt", action="append", default=[], help="name=path")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--urls", nargs="+", default=["http://127.0.0.1:8080"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PolyEnv(1024, args.p, urls=tuple(args.urls), db=args.db, parent_db_frac=1.0, parent_temperature=3.0)
    env._remember = lambda coeffs, scores: None
    rng = np.random.default_rng(0)
    for ceiling in args.ceilings:
        base = run_cell(env, random_edit(rng, args.p, 1), ceiling, args.batches, seed=1000 + ceiling)
        print(f"ceiling {ceiling}  random-1: improvement {base['improvement']:+.3f}  child {base['child']:.2f}", flush=True)
        for item in args.ckpt:
            name, path = item.split("=", 1)
            agent = load_agent(path, device)
            own = run_cell(env, policy_edit(agent, device), ceiling, args.batches, seed=1000 + ceiling)
            other = run_cell(env, mismatched_edit(agent, device, args.p, rng), ceiling, args.batches, seed=1000 + ceiling)
            print(f"ceiling {ceiling}  {name:18s} own edit: {own['improvement']:+.3f} (child {own['child']:.2f}, hamming {own['hamming']:.2f})   "
                  f"edit for another parent: {other['improvement']:+.3f} (child {other['child']:.2f}, hamming {other['hamming']:.2f})   "
                  f"dependence on the parent = {own['improvement'] - other['improvement']:+.3f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()

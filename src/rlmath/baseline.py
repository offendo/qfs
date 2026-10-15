"""Uniform random sampling baseline.

Scores uniformly random polynomials in batches and logs, per batch, the number of distinct smooth
finds at each height. The output format matches rlmath.train, so the two can be compared on
"finds at height >= h per oracle call". The model-free local-search baseline is
`rlmath-train --no-use-model`.
"""

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from rlmath.oracle import HeightOracle
from rlmath.polynomials import random_polys, to_string
from rlmath.search import ElitePool, Scorer

log = logging.getLogger("rlmath.baseline")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--infinite-score", type=int, default=1)
    parser.add_argument("--max-oracle-calls", type=int, default=100_000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--db", default="runs/heights.sqlite", help='SQLite file for every scored polynomial ("" to disable)')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    out = Path(args.out or f"runs/random-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    rng = np.random.default_rng(args.seed)
    start = time.time()

    with HeightOracle(args.p, args.n, args.d, args.cutoff, workers=args.workers or None, db=args.db or None) as oracle:
        scorer = Scorer(oracle, args.infinite_score, past_cutoff_is_infinite=(args.n == 4 and args.cutoff >= 10))
        pool = ElitePool(args.p, args.n, args.d, per_level=10**9)
        r = 0
        bar = tqdm(total=args.max_oracle_calls, unit=" calls", unit_scale=True, dynamic_ncols=True, disable=None)
        with logging_redirect_tqdm():
            while oracle.calls < args.max_oracle_calls:
                polys = random_polys(rng, args.batch, args.p, args.n, args.d)
                scores, results = scorer(polys)
                new = pool.add(polys, scores, results, r)
                with open(out / "elites.jsonl", "a") as f:
                    for e in new:
                        f.write(json.dumps({
                            "score": e.score, "status": e.status, "round": r, "oracle_calls": oracle.calls,
                            "seconds": round(time.time() - start, 1),
                            "coeffs": e.coeffs.tolist(), "poly": to_string(e.coeffs, args.n, args.d),
                        }) + "\n")
                stats = {
                    "round": r, "seconds": round(time.time() - start, 1), "oracle_calls": oracle.calls,
                    "best": pool.best, "pool_by_score": pool.counts(),
                    "score_hist": {int(s): int((scores == s).sum()) for s in np.unique(scores)},
                }
                with open(out / "log.jsonl", "a") as f:
                    f.write(json.dumps(stats) + "\n")
                bar.update(oracle.calls - bar.n)
                bar.set_postfix_str(f"best {pool.best}")
                log.info("batch %d  calls %d  best %d  pool %s", r, oracle.calls, pool.best, pool.counts())
                r += 1
        bar.close()


if __name__ == "__main__":
    main()

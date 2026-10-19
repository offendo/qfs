"""Compare runs: distinct finds at each height, and the oracle calls / time to the first one.

    rlmath-summarize runs/p5-model runs/p5-nomodel runs/p5-random
"""

import argparse
import json
from pathlib import Path


def summarize(run: Path) -> dict:
    elites = [json.loads(line) for line in (run / "elites.jsonl").read_text().splitlines()]
    log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    out = {"run": run.name, "oracle_calls": log[-1]["oracle_calls"], "seconds": log[-1]["seconds"], "levels": {}}
    for s in sorted({e["score"] for e in elites}):
        at = [e for e in elites if e["score"] == s]
        out["levels"][s] = {
            "count": len(at),
            "first_calls": min(e["oracle_calls"] for e in at),
            "first_seconds": min(e["seconds"] for e in at),
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    rows = [summarize(r) for r in args.runs]
    levels = sorted({s for r in rows for s in r["levels"]})
    print("distinct smooth finds per height (first found at: oracle calls)")
    header = f"{'run':<16}{'calls':>9}{'secs':>7}" + "".join(f"{'h=' + str(s):>16}" for s in levels)
    print(header)
    for r in rows:
        cells = []
        for s in levels:
            v = r["levels"].get(s)
            cells.append(f"{v['count']:>6} @{v['first_calls']:>8}" if v else f"{'-':>16}")
        print(f"{r['run']:<16}{r['oracle_calls']:>9}{r['seconds']:>7.0f}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()

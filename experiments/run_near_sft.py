"""E21 sweep: SFT on pairs that differ in at most K coefficients, K = 1, 3, 5, 7 (jobs run in parallel on one GPU).

    uv run python experiments/run_near_sft.py <pairs_k7.npz> <out_dir>

Sweep A: the same number of pairs (as many as K = 1 has) and the same number of steps for every K, 2 seeds.
Sweep B: more pairs for larger K (up to 5.5M, fewer epochs), the same number of steps.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

pairs, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
jobs = []
for seed in (1, 2):
    for k in (1, 3, 5, 7):
        jobs.append((f"a_k{k}_s{seed}", ["--max-dist", str(k), "--max-pairs", "1371669", "--epochs", "8", "--seed", str(seed)]))
for k in (3, 5, 7):
    jobs.append((f"b_k{k}", ["--max-dist", str(k), "--max-pairs", "5500000", "--epochs", "2"]))


def run(job):
    name, extra = job
    with open(out / f"{name}.log", "w") as log:
        code = subprocess.run(
            ["uv", "run", "rlmath-sft", "--pairs", pairs, "--out", str(out / f"{name}.pt"), "--require", "ge", *extra],
            stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
        ).returncode
    print(name, "exit", code, flush=True)


with ThreadPoolExecutor(4) as pool:
    list(pool.map(run, jobs))

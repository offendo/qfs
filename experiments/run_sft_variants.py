"""E2: train the SFT variants in parallel on one GPU. Usage: run_sft_variants.py <pairs.npz> <out_dir>"""

import os
import subprocess
import sys
from pathlib import Path

pairs, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
VARIANTS = {
    "s1_ge_all": ["--require", "ge"],
    "s2_gt_all": ["--require", "gt"],
    "s3_ge_h3plus": ["--require", "ge", "--min-parent", "3", "--epochs", "30"],
    "s4_ge_all_big": ["--require", "ge", "--d-model", "256", "--n-layers", "8", "--n-heads", "8", "--d-ff", "1024"],
}
procs = []
for name, extra in VARIANTS.items():
    cmd = ["uv", "run", "rlmath-sft", "--pairs", pairs, "--out", str(out / f"{name}.pt"), *extra]
    log = open(out / f"{name}.log", "w")
    procs.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"})))
for name, proc in procs:
    print(name, "exit", proc.wait(), flush=True)

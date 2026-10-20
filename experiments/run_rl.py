"""Run one rlmath-reinforce experiment on its own copy of a database snapshot, logging to <out_dir>/<name>.log.

    uv run python experiments/run_rl.py <name> <snapshot.sqlite> <out_dir> [rlmath-reinforce arguments ...]
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

name, snapshot, out_dir, extra = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4:]
out_dir.mkdir(parents=True, exist_ok=True)
db = out_dir / f"{name}.sqlite"
for stale in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
    stale.unlink(missing_ok=True)  # leftovers from an earlier run would not match the fresh copy
shutil.copy(snapshot, db)
cmd = [
    "uv", "run", os.environ.get("RLMATH_CMD", "rlmath-reinforce"), "--p", "3", "--parent-db-frac", "1.0", "--workers", "20",
    "--db", str(db), "--out", str(out_dir / name), *extra,
]
with open(out_dir / f"{name}.log", "w") as log:
    code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"}).returncode
db.unlink(missing_ok=True)  # the copy is 1 GB+; the log and model.pt are what we keep
for ext in ("-wal", "-shm"):
    Path(f"{db}{ext}").unlink(missing_ok=True)
print(name, "exit", code)

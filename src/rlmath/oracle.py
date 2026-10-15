"""Client for the Julia height oracle.

Starts several copies of julia/height_worker.jl as subprocesses and talks to them over
stdin/stdout, one request per line. Each worker is a plain Julia process (no Julia inside
Python), so a crash in one worker cannot take down training; it is restarted on the next call.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import subprocess
import threading

import numpy as np

from rlmath.polynomials import key, monomials

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = REPO_ROOT / "julia" / "height_worker.jl"
JULIA_PROJECT = REPO_ROOT / "julia"


@dataclass(frozen=True)
class HeightResult:
    """Outcome of one height computation.

    status:
      "ok"          height is a finite number (`height`)
      "inf"         the KTY chain hit zero: provably infinite height (supersingular)
      "past_cutoff" no answer within `cutoff` steps. For smooth quartic K3 surfaces with
                    cutoff >= 10 this also means infinite height.
      "invalid"     zero polynomial, or not homogeneous of degree n = d
      "error"       the worker raised; `message` says why
    """

    status: str
    height: int | None = None
    message: str = ""

    @property
    def infinite(self) -> bool:
        return self.status == "inf"


def _parse_height(line: str) -> HeightResult:
    parts = line.split(maxsplit=1)
    if parts[0] == "ok":
        return HeightResult("ok", int(parts[1]))
    if parts[0] in ("inf", "past_cutoff", "invalid"):
        return HeightResult(parts[0])
    return HeightResult("error", message=line)


class _Worker:
    def __init__(self, julia: str, threads_env: dict):
        self.proc = subprocess.Popen(
            [julia, f"--project={JULIA_PROJECT}", "--startup-file=no", str(WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # Julia errors go to our stderr
            text=True,
            bufsize=1,
            env=threads_env,
        )
        ready = self.proc.stdout.readline().strip()
        if ready != "READY":
            raise RuntimeError(f"height worker failed to start (got {ready!r})")
        self.lock = threading.Lock()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def ask(self, lines: list[str]) -> list[str]:
        """Send a chunk of requests, then read the same number of replies."""
        with self.lock:
            self.proc.stdin.write("".join(line + "\n" for line in lines))
            self.proc.stdin.flush()
            replies = [self.proc.stdout.readline() for _ in lines]
        if any(r == "" for r in replies):
            raise RuntimeError("height worker exited unexpectedly")
        return [r.strip() for r in replies]

    def close(self):
        if self.alive():
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class HeightOracle:
    """Batched, cached, parallel quasi-F-split height computation.

    Results are cached by polynomial up to scaling, so asking twice is free. `calls` counts the
    polynomials actually sent to Julia, which is the budget every search method is measured by.
    """

    CHUNK = 64  # max requests per round trip; keeps both pipe buffers far from full

    def __init__(self, p: int, n: int, d: int, cutoff: int = 10, workers: int | None = None, julia: str = "julia"):
        self.p, self.n, self.d, self.cutoff = p, n, d, cutoff
        self.julia = julia
        self.n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
        env = dict(os.environ, JULIA_NUM_THREADS="1")
        self._env = env
        with ThreadPoolExecutor(self.n_workers) as pool:
            self.workers = list(pool.map(lambda _: _Worker(julia, env), range(self.n_workers)))
        self._check_monomial_order()
        self.height_cache: dict[bytes, HeightResult] = {}
        self.smooth_cache: dict[bytes, bool] = {}
        self.calls = 0
        self.smooth_calls = 0

    def _check_monomial_order(self):
        reply = self.workers[0].ask([f"MONOS {self.n} {self.d}"])[0]
        julia_order = np.array(reply.split(), dtype=np.int64).reshape(-1, self.n)
        if not np.array_equal(julia_order, monomials(self.n, self.d)):
            raise RuntimeError("Python and Julia disagree on the monomial order")

    def _run(self, lines: list[str]) -> list[str]:
        """Spread request lines over the workers in chunks and return replies in order."""
        # Small batches (e.g. one local-search step) are split evenly so every worker gets some.
        size = max(1, min(self.CHUNK, -(-len(lines) // self.n_workers)))
        chunks = [lines[i : i + size] for i in range(0, len(lines), size)]

        def run_chunk(i):
            w = self.workers[i % self.n_workers]
            try:
                return w.ask(chunks[i])
            except RuntimeError:
                log.warning("height worker %d died; restarting it", i % self.n_workers)
                w.close()
                self.workers[i % self.n_workers] = _Worker(self.julia, self._env)
                return [f"error worker crashed on this chunk"] * len(chunks[i])

        with ThreadPoolExecutor(self.n_workers) as pool:
            results = list(pool.map(run_chunk, range(len(chunks))))
        return [r for chunk in results for r in chunk]

    def heights(self, coeffs: np.ndarray) -> list[HeightResult]:
        """Heights for a batch of coefficient vectors, shape (B, M)."""
        coeffs = np.atleast_2d(coeffs) % self.p
        keys = [key(row, self.p) for row in coeffs]
        todo = {}
        for k, row in zip(keys, coeffs):
            if k not in self.height_cache and k not in todo:
                todo[k] = row
        if todo:
            header = f"H {self.p} {self.n} {self.d} {self.cutoff} "
            replies = self._run([header + " ".join(map(str, row)) for row in todo.values()])
            for k, reply in zip(todo, replies):
                self.height_cache[k] = _parse_height(reply)
            self.calls += len(todo)
        return [self.height_cache[k] for k in keys]

    def smooth(self, coeffs: np.ndarray) -> list[bool]:
        """Whether each polynomial defines a smooth hypersurface (Gröbner basis check in Singular)."""
        coeffs = np.atleast_2d(coeffs) % self.p
        keys = [key(row, self.p) for row in coeffs]
        todo = {}
        for k, row in zip(keys, coeffs):
            if k not in self.smooth_cache and k not in todo:
                todo[k] = row
        if todo:
            header = f"S {self.p} {self.n} {self.d} "
            replies = self._run([header + " ".join(map(str, row)) for row in todo.values()])
            for k, reply in zip(todo, replies):
                if reply not in ("0", "1"):
                    log.warning("smoothness check failed: %s", reply)
                self.smooth_cache[k] = reply == "1"
            self.smooth_calls += len(todo)
        return [self.smooth_cache[k] for k in keys]

    def close(self):
        for w in self.workers:
            w.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

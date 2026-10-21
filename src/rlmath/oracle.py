"""Client for the Julia height server (Qfs.jl/Server.jl), cached and batched."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from urllib.request import Request, urlopen

import numpy as np

from rlmath.polynomials import key, monomials
from rlmath.store import HeightStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeightResult:
    """status: "ok" (finite `height`), "inf" (chain hit zero), "past_cutoff", "invalid", or "error"."""

    status: str
    height: int | None = None

    @property
    def infinite(self) -> bool:
        return self.status == "inf"


def _parse_height(line: str) -> HeightResult:
    if line.startswith("ok "):
        return HeightResult("ok", int(line[3:]))
    return HeightResult(line if line in ("inf", "past_cutoff", "invalid") else "error")


class HeightOracle:
    """Heights and smoothness from one or more running servers, cached by polynomial up to scaling.

    `calls` counts the polynomials sent to Julia, the budget every search method is measured by.
    If `db` is a path, every result is also written to that SQLite file (see rlmath.store).
    """

    def __init__(self, p: int, n: int, d: int, cutoff: int = 10, urls: list[str] | None = None, db: str | None = None):
        self.p, self.n, self.d, self.cutoff = p, n, d, cutoff
        self.urls = urls or ["http://127.0.0.1:8080"]
        self.store = HeightStore(db) if db else None
        reply = self._post("/monomials", "", self.urls[0], f"n={n}&d={d}")
        if not np.array_equal(np.array([r.split() for r in reply], dtype=np.int64), monomials(n, d)):
            raise RuntimeError("Python and Julia disagree on the monomial order")
        self.height_cache: dict[bytes, HeightResult] = {}
        self.smooth_cache: dict[bytes, bool] = {}
        self.calls = 0
        self.smooth_calls = 0

    @staticmethod
    def _post(path: str, body: str, url: str, query: str) -> list[str]:
        with urlopen(Request(f"{url}{path}?{query}", body.encode(), method="POST")) as r:
            return r.read().decode().split("\n")

    def _run(self, path: str, query: str, rows: list[np.ndarray]) -> list[str]:
        """Split the rows evenly over the servers and return one reply line per row."""
        k = len(self.urls)
        size = -(-len(rows) // k)
        chunks = [rows[i : i + size] for i in range(0, len(rows), size)]

        def post(i):
            body = "\n".join(" ".join(map(str, r)) for r in chunks[i])
            return self._post(path, body, self.urls[i], query)

        with ThreadPoolExecutor(len(chunks)) as pool:
            return [line for lines in pool.map(post, range(len(chunks))) for line in lines]

    def _missing(self, coeffs, cache):
        coeffs = np.atleast_2d(coeffs) % self.p
        keys = [key(row, self.p) for row in coeffs]
        todo = {}
        for k, row in zip(keys, coeffs):
            if k not in cache and k not in todo:
                todo[k] = row
        return keys, todo

    def heights(self, coeffs: np.ndarray) -> list[HeightResult]:
        """Heights for a batch of coefficient vectors, shape (B, M)."""
        keys, todo = self._missing(coeffs, self.height_cache)
        if todo:
            query = f"p={self.p}&n={self.n}&d={self.d}&cutoff={self.cutoff}"
            for k, line in zip(todo, self._run("/height", query, list(todo.values()))):
                self.height_cache[k] = _parse_height(line)
            self.calls += len(todo)
            if self.store:
                rows = [(k, self.height_cache[k].status, self.height_cache[k].height) for k in todo]
                self.store.add_heights(self.p, self.n, self.d, self.cutoff, rows)
        return [self.height_cache[k] for k in keys]

    def smooth(self, coeffs: np.ndarray) -> list[bool]:
        """Whether each polynomial defines a smooth hypersurface."""
        keys, todo = self._missing(coeffs, self.smooth_cache)
        if todo:
            query = f"p={self.p}&n={self.n}&d={self.d}"
            for k, line in zip(todo, self._run("/smooth", query, list(todo.values()))):
                if line not in ("0", "1"):
                    log.warning("smoothness check failed: %s", line)
                self.smooth_cache[k] = line == "1"
            self.smooth_calls += len(todo)
            if self.store:
                self.store.set_smooth(self.p, self.n, self.d, [(k, self.smooth_cache[k]) for k in todo])
        return [self.smooth_cache[k] for k in keys]

    def close(self):
        if self.store:
            self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

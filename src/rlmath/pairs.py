"""(parent, child) pairs that differ in one coefficient, found among the polynomials in the heights database.

The database holds every polynomial whose height was computed (see rlmath.store). Wherever a polynomial and
a one-coefficient change of it are both in there, that is a free example of an edit and its outcome.

A polynomial is encoded as a base-p integer so that its neighbours can be found with arithmetic and a binary
search. The database stores each polynomial scaled to start with a 1, so a neighbour is looked up under every
scalar multiple. Codes must fit in int64, which limits this to p ** (number of monomials) < 2 ** 63
(p = 2 or 3 for quartics).
"""

from dataclasses import dataclass
import sqlite3

import numpy as np

from rlmath.polynomials import monomials


@dataclass
class Scored:
    digits: np.ndarray  # (N, M) uint8 coefficients
    scores: np.ndarray  # (N,) int16; height, 0 if known to be singular, -1 if smoothness is unknown


@dataclass
class Pairs:
    digits: np.ndarray  # (N, M) parents table (uint8)
    parent: np.ndarray  # (K,) row of the parent in `digits`
    pos: np.ndarray  # (K,) coefficient that changes
    new: np.ndarray  # (K,) its new value
    parent_score: np.ndarray  # (K,)
    child_score: np.ndarray  # (K,)

    def compact(self) -> "Pairs":
        """Keep only the rows of `digits` that are a parent of some pair."""
        used, inverse = np.unique(self.parent, return_inverse=True)
        return Pairs(self.digits[used], inverse.astype(np.int32), self.pos, self.new, self.parent_score, self.child_score)

    def save(self, path: str):
        c = self.compact()
        np.savez(path, digits=c.digits, parent=c.parent, pos=c.pos, new=c.new, parent_score=c.parent_score, child_score=c.child_score)

    @staticmethod
    def load(path: str) -> "Pairs":
        z = np.load(path)
        return Pairs(*(z[k] for k in ("digits", "parent", "pos", "new", "parent_score", "child_score")))

    def children(self, idx: np.ndarray) -> np.ndarray:
        """Child coefficient vectors for the pairs in `idx`."""
        child = self.digits[self.parent[idx]].copy()
        child[np.arange(len(idx)), self.pos[idx]] = self.new[idx]
        return child


def load_scored(db: str, p: int, n: int = 4, d: int = 4) -> Scored:
    """Every distinct polynomial with a finite height in the database, scored as the environment would."""
    m = len(monomials(n, d))
    if p**m >= 2**63:
        raise ValueError("p ** monomials does not fit in int64")
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=120) as con:
        rows = con.execute(
            "SELECT DISTINCT coeffs, height FROM heights WHERE p = ? AND n = ? AND d = ? AND status = 'ok'", (p, n, d)
        ).fetchall()
        smooth = dict(con.execute("SELECT coeffs, smooth FROM smoothness WHERE p = ? AND n = ? AND d = ?", (p, n, d)))
    digits = np.frombuffer(b"".join(c for c, _ in rows), dtype=np.uint8).reshape(len(rows), m)
    height = np.array([h for _, h in rows], dtype=np.int16)
    scores = height.copy()
    for i in np.flatnonzero(height >= 2):  # the environment only trusts these if they are smooth
        s = smooth.get(rows[i][0])
        scores[i] = -1 if s is None else (height[i] if s else 0)
    return Scored(digits, scores)


def _codes(digits: np.ndarray, p: int, scalar: int = 1, chunk: int = 1 << 20) -> np.ndarray:
    powers = p ** np.arange(digits.shape[1], dtype=np.int64)
    out = np.empty(len(digits), dtype=np.int64)
    for lo in range(0, len(digits), chunk):
        out[lo : lo + chunk] = ((digits[lo : lo + chunk].astype(np.int64) * scalar) % p * powers).sum(axis=1)
    return out


def find_pairs(data: Scored, p: int, min_parent: int = 2) -> Pairs:
    """All (parent, one-coefficient change) pairs whose child is also in the database with a known score."""
    digits, scores = data.digits, data.scores
    m = digits.shape[1]
    powers = p ** np.arange(m, dtype=np.int64)
    codes = {s: _codes(digits, p, s) for s in range(1, p)}  # codes of every scalar multiple of every row
    order = np.argsort(codes[1])
    sorted_codes = codes[1][order]
    parents = np.flatnonzero(scores >= min_parent)

    def lookup(query: np.ndarray) -> np.ndarray:
        """Row of each code in the table, or -1."""
        at = np.searchsorted(sorted_codes, query).clip(max=len(sorted_codes) - 1)
        return np.where(sorted_codes[at] == query, order[at], -1)

    out = {k: [] for k in ("parent", "pos", "new", "parent_score", "child_score")}
    for j in range(m):
        x = digits[parents, j].astype(np.int64)
        for delta in range(1, p):
            nd = (x + delta) % p
            found = np.full(len(parents), -1, dtype=np.int64)
            for s in range(1, p):  # the database may hold the child scaled by s
                query = codes[s][parents] + ((s * nd) % p - (s * x) % p) * powers[j]
                hit = lookup(query)
                found = np.where(found < 0, hit, found)
            ok = found >= 0
            child_score = scores[found[ok]]
            known = child_score >= 0
            idx = np.flatnonzero(ok)[known]
            out["parent"].append(parents[idx].astype(np.int32))
            out["pos"].append(np.full(len(idx), j, dtype=np.int8))
            out["new"].append(nd[idx].astype(np.int8))
            out["parent_score"].append(scores[parents[idx]])
            out["child_score"].append(child_score[known])
    return Pairs(digits, *(np.concatenate(out[k]) for k in ("parent", "pos", "new", "parent_score", "child_score")))

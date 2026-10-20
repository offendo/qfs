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


@dataclass
class NearPairs:
    """(parent, child) pairs whose coefficient vectors differ in at most K places, children given as rows of `digits`.

    The child vector is `(scale * digits[child]) % p`: the database stores each polynomial scaled to start with a 1, and
    the scalar multiple that is nearest to the parent is the one that differs in the fewest places.
    """

    digits: np.ndarray  # (N, M) uint8
    parent: np.ndarray  # (K,) row in digits
    child: np.ndarray  # (K,) row in digits
    scale: np.ndarray  # (K,) int8, 1 .. p - 1
    dist: np.ndarray  # (K,) int8, number of coefficients that differ
    parent_score: np.ndarray  # (K,)
    child_score: np.ndarray  # (K,)

    def save(self, path: str):
        used, inverse = np.unique(np.concatenate([self.parent, self.child]), return_inverse=True)
        k = len(self.parent)
        np.savez(
            path, digits=self.digits[used], parent=inverse[:k].astype(np.int32), child=inverse[k:].astype(np.int32),
            scale=self.scale, dist=self.dist, parent_score=self.parent_score, child_score=self.child_score,
        )

    @staticmethod
    def load(path: str) -> "NearPairs":
        z = np.load(path)
        return NearPairs(*(z[k] for k in ("digits", "parent", "child", "scale", "dist", "parent_score", "child_score")))


def _popcount(x):
    """Number of set bits of non-negative int64 tensors (here at most 35 bits)."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56 & 0xFF


def find_near_pairs(
    data: Scored, p: int, max_dist: int, min_parent: int = 2, require_ge: bool = True, max_per_parent: int = 0,
    seed: int = 0, device: str = "cuda", tile: int = 1 << 25, log=print,
) -> NearPairs:
    """All pairs of database polynomials (parent score >= min_parent) that differ in 1 .. max_dist coefficients.

    Both polynomials must have a known score of at least `min_parent`; with `require_ge` only pairs whose child is at
    least as high as the parent are kept. If two vectors differ in at most K places, then split into K + 1 blocks of
    coefficients at least one block is identical, so the candidates are the pairs that share a block. Candidates are
    compared on the GPU with the vectors packed into bit planes. A pair found in several blocks is kept once (in the first
    block they agree on). `max_per_parent` > 0 keeps at most that many children per parent, chosen at random.
    """
    import torch

    rng = np.random.default_rng(seed)
    keep = np.flatnonzero(data.scores >= min_parent)
    digits, scores = data.digits[keep], data.scores[keep]
    n, m = digits.shape
    scales = list(range(1, p))
    bits = 1 << np.arange(m, dtype=np.int64)

    def planes(d: np.ndarray):  # one int64 per vector and digit value 1 .. p - 1
        return [torch.as_tensor(((d == v) * bits).sum(axis=1), device=device) for v in range(1, p)]

    scaled = [((s * digits.astype(np.int64)) % p).astype(np.uint8) for s in scales]
    parent_planes = planes(digits)
    per_scale = [planes(d) for d in scaled]
    child_planes = [torch.cat([per_scale[s][v] for s in range(len(scales))]) for v in range(p - 1)]
    child_row = np.tile(np.arange(n), len(scales))
    child_scale = np.repeat(scales, n)
    child_digits = np.concatenate(scaled)

    order = rng.permutation(m)
    blocks = [np.sort(b) for b in np.array_split(order, max_dist + 1)]
    masks = [int(sum(1 << int(j) for j in b)) for b in blocks]
    place = p ** np.arange(m, dtype=np.int64)
    ps = torch.as_tensor(scores, device=device)
    child_row_t = torch.as_tensor(child_row, device=device)
    found = {k: [] for k in ("parent", "child", "dist")}
    for bi, block in enumerate(blocks):
        pkey = (digits[:, block].astype(np.int64) * place[: len(block)]).sum(axis=1)
        ckey = (child_digits[:, block].astype(np.int64) * place[: len(block)]).sum(axis=1)
        po, co = np.argsort(pkey, kind="stable"), np.argsort(ckey, kind="stable")
        pk, pstart, pcount = np.unique(pkey[po], return_index=True, return_counts=True)
        ck, cstart, ccount = np.unique(ckey[co], return_index=True, return_counts=True)
        _, pi, ci = np.intersect1d(pk, ck, return_indices=True)
        po_t, co_t = torch.as_tensor(po, device=device), torch.as_tensor(co, device=device)
        pp = [x[po_t] for x in parent_planes]
        cp = [x[co_t] for x in child_planes]
        pscore = ps[po_t]
        crow = child_row_t[co_t]
        cscore = ps[crow]
        total = 0
        for a, b in zip(pi, ci):
            a0, a1, b0, b1 = pstart[a], pstart[a] + pcount[a], cstart[b], cstart[b] + ccount[b]
            rows = max(1, tile // (b1 - b0))
            for lo in range(a0, a1, rows):
                hi = min(lo + rows, a1)
                diff = 0
                for v in range(p - 1):
                    diff = diff | (pp[v][lo:hi, None] ^ cp[v][None, b0:b1])
                d = _popcount(diff)
                ok = (d >= 1) & (d <= max_dist) & (po_t[lo:hi, None] != crow[None, b0:b1])  # not the polynomial itself
                if require_ge:
                    ok &= cscore[None, b0:b1] >= pscore[lo:hi, None]
                i, j = torch.nonzero(ok, as_tuple=True)
                if not len(i):
                    continue
                x = diff[i, j]
                first = torch.ones_like(i, dtype=torch.bool)  # the pair must differ in every earlier block
                for earlier in masks[:bi]:
                    first &= (x & earlier) != 0
                i, j = i[first], j[first]
                if not len(i):
                    continue
                found["parent"].append(po_t[lo + i].cpu().numpy())
                found["child"].append(co_t[b0 + j].cpu().numpy())
                found["dist"].append(_popcount(x[first]).cpu().numpy())
                total += len(i)
        log(f"block {bi + 1}/{len(blocks)}: {len(pi):,} shared keys, {total:,} new pairs")
    if not found["parent"]:
        empty = np.zeros(0, dtype=np.int32)
        return NearPairs(digits, empty, empty, empty.astype(np.int8), empty.astype(np.int8), empty.astype(np.int16), empty.astype(np.int16))
    parent = np.concatenate(found["parent"])
    cidx = np.concatenate(found["child"])
    dist = np.concatenate(found["dist"]).astype(np.int8)
    child, scale = child_row[cidx], child_scale[cidx].astype(np.int8)
    if max_per_parent:
        shuffle = rng.permutation(len(parent))
        parent, child, scale, dist = parent[shuffle], child[shuffle], scale[shuffle], dist[shuffle]
        order = np.argsort(parent, kind="stable")
        parent, child, scale, dist = parent[order], child[order], scale[order], dist[order]
        first = np.r_[0, np.flatnonzero(np.diff(parent)) + 1]
        rank = np.arange(len(parent)) - np.repeat(first, np.diff(np.r_[first, len(parent)]))
        sel = rank < max_per_parent
        parent, child, scale, dist = parent[sel], child[sel], scale[sel], dist[sel]
    return NearPairs(digits, parent.astype(np.int32), child.astype(np.int32), scale, dist, scores[parent], scores[child])

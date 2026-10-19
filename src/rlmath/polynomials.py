"""Coefficient-vector representation of homogeneous polynomials over F_p.

A polynomial of degree d in n variables is a vector of integers in [0, p) indexed by
`monomials(n, d)`. The Julia worker uses the same order (checked at startup).
"""

from functools import lru_cache
import itertools

import numpy as np


@lru_cache(maxsize=None)
def monomials(n: int, d: int) -> np.ndarray:
    """Exponent vectors of all degree-d monomials in n variables, descending lex order. Shape (M, n)."""
    if n == 1:
        return np.array([[d]], dtype=np.int64)
    rows = []
    for a in range(d, -1, -1):
        for rest in monomials(n - 1, d - a):
            rows.append([a, *rest])
    return np.array(rows, dtype=np.int64)


def random_polys(rng: np.random.Generator, count: int, p: int, n: int, d: int) -> np.ndarray:
    """Uniformly random coefficient vectors (nonzero), shape (count, M)."""
    m = len(monomials(n, d))
    out = rng.integers(0, p, size=(count, m), dtype=np.int64)
    zero = ~out.any(axis=1)
    while zero.any():
        out[zero] = rng.integers(0, p, size=(int(zero.sum()), m), dtype=np.int64)
        zero = ~out.any(axis=1)
    return out


def normalize(coeffs: np.ndarray, p: int) -> np.ndarray:
    """Scale each row so its first nonzero coefficient is 1.

    f and c*f define the same variety, so they have the same height; this picks one representative.
    """
    coeffs = np.atleast_2d(coeffs) % p
    out = coeffs.copy()
    for i, row in enumerate(coeffs):
        nz = np.flatnonzero(row)
        if len(nz):
            out[i] = (row * pow(int(row[nz[0]]), -1, p)) % p
    return out


def key(coeffs: np.ndarray, p: int) -> bytes:
    """Hashable key for one polynomial, identical for f and c*f."""
    return normalize(coeffs, p)[0].astype(np.uint8).tobytes()


@lru_cache(maxsize=None)
def _permutation_index(n: int, d: int, perm: tuple[int, ...]) -> np.ndarray:
    mons = monomials(n, d)
    lookup = {tuple(m): i for i, m in enumerate(mons)}
    return np.array([lookup[tuple(m[list(perm)])] for m in mons])


def augment(rng: np.random.Generator, coeffs: np.ndarray, p: int, n: int, d: int) -> np.ndarray:
    """Apply a random height-preserving symmetry to each row.

    Uses the symmetries that keep monomials as monomials: permuting variables, scaling each
    variable x_i -> a_i x_i (a_i nonzero), and scaling the whole polynomial. Rows shape (B, M).
    """
    coeffs = np.atleast_2d(coeffs)
    mons = monomials(n, d)
    out = np.empty_like(coeffs)
    for i, row in enumerate(coeffs):
        perm = tuple(rng.permutation(n))
        row = row[_permutation_index(n, d, perm)]
        a = rng.integers(1, p, size=n)
        scale = np.ones(len(mons), dtype=np.int64)
        for j in range(n):
            scale = scale * np.array([pow(int(a[j]), int(e), p) for e in mons[:, j]]) % p
        c = int(rng.integers(1, p))
        out[i] = row * scale * c % p
    return out


def to_string(coeffs: np.ndarray, n: int, d: int) -> str:
    names = "wxyz" if n == 4 else [f"x{i}" for i in range(n)]
    terms = []
    for c, m in zip(np.asarray(coeffs).ravel(), monomials(n, d)):
        if c == 0:
            continue
        mono = "*".join(f"{names[i]}^{e}" if e > 1 else names[i] for i, e in enumerate(m) if e)
        terms.append(mono if c == 1 else f"{c}*{mono}")
    return " + ".join(terms) if terms else "0"


def single_edits(coeffs: np.ndarray, p: int) -> np.ndarray:
    """All polynomials differing from `coeffs` in exactly one coefficient. Shape (M*(p-1), M)."""
    coeffs = np.asarray(coeffs).ravel()
    out = []
    for j, v in itertools.product(range(len(coeffs)), range(1, p)):
        row = coeffs.copy()
        row[j] = (row[j] + v) % p
        out.append(row)
    return np.array(out)


@lru_cache(maxsize=None)
def _torus_scalings(p: int, n: int, d: int) -> np.ndarray:
    """Factors prod_i a_i^{m_i} mod p for every (a_1..a_n) in (F_p^*)^n. Shape ((p-1)^n, M)."""
    mons = monomials(n, d)
    a = np.array(list(itertools.product(range(1, p), repeat=n)), dtype=np.int64)  # (T, n)
    out = np.ones((len(a), len(mons)), dtype=np.int64)
    for j in range(n):
        powers = np.array([[pow(int(x), int(e), p) for e in mons[:, j]] for x in range(p)])
        out = out * powers[a[:, j]] % p
    return out


def canonical_key(coeffs: np.ndarray, p: int, n: int, d: int, max_transforms: int = 200_000) -> bytes:
    """Key that is the same for f and any polynomial obtained from it by permuting variables,
    scaling variables, or scaling f. All of these have the same height and define isomorphic
    varieties, so the elite pool counts them once.

    When there are too many such transforms (large p or n), only permutations and overall scaling
    are used.
    """
    row = np.asarray(coeffs).ravel() % p
    perms = list(itertools.permutations(range(n)))
    use_torus = len(perms) * (p - 1) ** n <= max_transforms
    best = None
    for perm in perms:
        r = row[_permutation_index(n, d, perm)]
        variants = r[None, :] * _torus_scalings(p, n, d) % p if use_torus else r[None, :]
        variants = _normalize_rows(variants, p)
        cand = variants[np.lexsort(variants.T[::-1])[0]]
        if best is None or tuple(cand) < tuple(best):
            best = cand
    return best.astype(np.uint8).tobytes()


def _normalize_rows(rows: np.ndarray, p: int) -> np.ndarray:
    """Vectorised `normalize` for many rows."""
    first = np.argmax(rows != 0, axis=1)
    lead = rows[np.arange(len(rows)), first]
    inv = np.array([0] + [pow(x, -1, p) for x in range(1, p)])[lead]
    return rows * inv[:, None] % p

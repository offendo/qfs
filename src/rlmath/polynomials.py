"""Coefficient-vector representation of homogeneous polynomials over F_p.

A polynomial of degree d in n variables is a vector of integers in [0, p) indexed by
`monomials(n, d)`. The Julia worker uses the same order (checked at startup).
"""

from functools import lru_cache

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


def to_string(coeffs: np.ndarray, n: int, d: int) -> str:
    names = "wxyz" if n == 4 else [f"x{i}" for i in range(n)]
    terms = []
    for c, m in zip(np.asarray(coeffs).ravel(), monomials(n, d)):
        if c == 0:
            continue
        mono = "*".join(f"{names[i]}^{e}" if e > 1 else names[i] for i, e in enumerate(m) if e)
        terms.append(mono if c == 1 else f"{c}*{mono}")
    return " + ".join(terms) if terms else "0"

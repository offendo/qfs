import numpy as np

from rlmath.polynomials import key, monomials, normalize, random_polys, to_string


def test_monomial_count_and_degree():
    for n, d in [(3, 3), (4, 4), (5, 5)]:
        mons = monomials(n, d)
        assert (mons.sum(axis=1) == d).all()
        assert len({tuple(m) for m in mons}) == len(mons)
    assert len(monomials(4, 4)) == 35
    assert len(monomials(5, 5)) == 126


def test_normalize_and_key_ignore_scaling():
    rng = np.random.default_rng(0)
    f = random_polys(rng, 1, 7, 4, 4)[0]
    assert key(f, 7) == key(3 * f % 7, 7)
    assert normalize(f, 7)[0][np.flatnonzero(f)[0]] == 1


def test_to_string():
    f = np.zeros(35, dtype=np.int64)
    f[0] = 1
    f[-1] = 2
    assert to_string(f, 4, 4) == "w^4 + 2*z^4"

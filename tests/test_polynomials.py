import numpy as np

from rlmath.polynomials import augment, canonical_key, key, monomials, normalize, random_polys, single_edits, to_string


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


def test_augment_keeps_support_size_and_range():
    rng = np.random.default_rng(1)
    f = random_polys(rng, 20, 5, 4, 4)
    g = augment(rng, f, 5, 4, 4)
    assert ((g >= 0) & (g < 5)).all()
    assert ((f != 0).sum(axis=1) == (g != 0).sum(axis=1)).all()


def test_single_edits():
    f = np.zeros(35, dtype=np.int64)
    e = single_edits(f, 3)
    assert e.shape == (70, 35)
    assert ((e != f).sum(axis=1) == 1).all()


def test_to_string():
    f = np.zeros(35, dtype=np.int64)
    f[0] = 1
    f[-1] = 2
    assert to_string(f, 4, 4) == "w^4 + 2*z^4"


def test_canonical_key_identifies_symmetric_copies():
    rng = np.random.default_rng(3)
    f = random_polys(rng, 10, 5, 4, 4)
    g = augment(rng, f, 5, 4, 4)
    assert [canonical_key(a, 5, 4, 4) for a in f] == [canonical_key(b, 5, 4, 4) for b in g]
    assert len({canonical_key(a, 5, 4, 4) for a in f}) == 10

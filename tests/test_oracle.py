import numpy as np

from rlmath.polynomials import augment, monomials, random_polys


def fermat(n=4, d=4):
    mons = monomials(n, d)
    return (mons.max(axis=1) == d).astype(np.int64)


def test_fermat_heights(oracle_factory):
    # Fermat quartic: height 1 if p ≡ 1 mod 4, supersingular if p ≡ 3 mod 4
    assert oracle_factory(5).heights(fermat())[0].height == 1
    assert oracle_factory(3).heights(fermat())[0].status == "inf"


def test_smoothness(oracle_factory):
    o = oracle_factory(3)
    w4 = np.zeros(35, dtype=np.int64)
    w4[0] = 1
    assert o.smooth(np.stack([fermat(), w4])) == [True, False]


def test_cache_counts_calls_once(oracle_factory):
    o = oracle_factory(3)
    f = random_polys(np.random.default_rng(5), 10, 3, 4, 4)
    before = o.calls
    o.heights(f)
    o.heights(np.concatenate([f, 2 * f % 3]))  # same polynomials up to scaling
    assert o.calls - before == 10


def test_symmetries_preserve_height(oracle_factory):
    o = oracle_factory(3)
    rng = np.random.default_rng(2)
    f = random_polys(rng, 300, 3, 4, 4)
    hs = o.heights(f)
    hard = np.array([r.status == "ok" and r.height >= 2 for r in hs])
    f = f[hard][:30]
    g = augment(rng, f, 3, 4, 4)
    assert [r.height for r in o.heights(f)] == [r.height for r in o.heights(g)]

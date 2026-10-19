import numpy as np

from rlmath.polynomials import key, random_polys
from rlmath.store import HeightStore


def test_roundtrip(tmp_path):
    s = HeightStore(tmp_path / "h.sqlite")
    f = random_polys(np.random.default_rng(0), 3, 5, 4, 4)
    keys = [key(r, 5) for r in f]
    s.add_heights(5, 4, 4, 10, [(keys[0], "ok", 3), (keys[1], "inf", None), (keys[2], "ok", 1)])
    s.set_smooth(5, 4, 4, [(keys[0], True)])
    rows = dict(s.db.execute("SELECT coeffs, height FROM heights"))
    assert rows[keys[0]] == 3
    assert rows[keys[1]] is None
    assert s.count() == 3
    assert s.db.execute("SELECT smooth FROM smoothness WHERE coeffs = ?", (keys[0],)).fetchone() == (1,)
    assert np.frombuffer(keys[2], dtype=np.uint8).shape == (35,)

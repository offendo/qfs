import numpy as np

from rlmath.pairs import Scored, find_pairs, load_scored
from rlmath.polynomials import normalize
from rlmath.store import HeightStore


def test_find_pairs_matches_brute_force():
    rng = np.random.default_rng(0)
    p, m = 3, 6
    rows = np.unique(normalize(rng.integers(0, p, size=(400, m)), p), axis=0)
    rows = rows[rows.any(axis=1)].astype(np.uint8)
    scores = rng.integers(0, 5, size=len(rows)).astype(np.int16)
    scores[rng.random(len(rows)) < 0.1] = -1
    data = Scored(rows, scores)
    table = {r.tobytes(): i for i, r in enumerate(rows)}

    expected = set()
    for i in np.flatnonzero(scores >= 2):
        for j in range(m):
            for new in range(p):
                if new == rows[i][j]:
                    continue
                child = rows[i].astype(np.int64).copy()
                child[j] = new
                if not child.any():
                    continue
                k = table.get(normalize(child, p)[0].astype(np.uint8).tobytes())
                if k is not None and scores[k] >= 0:
                    expected.add((int(i), j, new, int(scores[k])))
    pairs = find_pairs(data, p, min_parent=2)
    got = set(zip(pairs.parent.tolist(), pairs.pos.tolist(), pairs.new.tolist(), pairs.child_score.tolist()))
    assert got == expected and len(expected) > 20
    assert (pairs.parent_score >= 2).all()

    idx = np.arange(len(pairs.parent))
    children = pairs.children(idx)
    assert ((children != rows[pairs.parent]).sum(axis=1) == 1).all()  # exactly one coefficient differs


def test_load_scored_applies_the_smoothness_rule(tmp_path):
    store = HeightStore(tmp_path / "h.sqlite")
    keys = [bytes([1] + [0] * 34), bytes([1, 1] + [0] * 33), bytes([1, 2] + [0] * 33), bytes([1, 0, 1] + [0] * 32)]
    store.add_heights(3, 4, 4, 10, [(keys[0], "ok", 3), (keys[1], "ok", 2), (keys[2], "ok", 2), (keys[3], "ok", 1)])
    store.set_smooth(3, 4, 4, [(keys[0], True), (keys[1], False)])  # keys[2] was never checked
    store.close()
    data = load_scored(str(tmp_path / "h.sqlite"), 3)
    by_key = {data.digits[i].tobytes(): int(data.scores[i]) for i in range(len(data.digits))}
    assert by_key[keys[0]] == 3  # smooth: keeps its height
    assert by_key[keys[1]] == 0  # singular
    assert by_key[keys[2]] == -1  # smoothness unknown
    assert by_key[keys[3]] == 1  # height 1 needs no check

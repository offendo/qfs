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


def test_find_near_pairs_matches_brute_force():
    from rlmath.pairs import Scored, find_near_pairs

    p, m = 3, 35
    rng = np.random.default_rng(0)
    base = rng.integers(0, p, size=(40, m))
    rows = [base]
    for k in (1, 2, 3, 5):  # near neighbours of the base rows
        near = base.copy()
        for r in near:
            pos = rng.choice(m, size=k, replace=False)
            r[pos] = (r[pos] + rng.integers(1, p, size=k)) % p
        rows.append(near)
    digits = np.concatenate(rows)
    first = (digits != 0).argmax(axis=1)
    lead = digits[np.arange(len(digits)), first]
    digits = ((digits * lead[:, None]) % p).astype(np.uint8)  # scaled to start with a 1, as in the database
    digits = np.unique(digits, axis=0)
    scores = rng.integers(2, 5, size=len(digits)).astype(np.int16)
    data = Scored(digits, scores)

    for max_dist in (2, 4):
        got = find_near_pairs(data, p, max_dist, require_ge=True, device="cpu", log=lambda *_: None)
        found = {(int(a), int(b), int(s)) for a, b, s in zip(got.parent, got.child, got.scale)}
        assert len(found) == len(got.parent)  # no pair twice
        expect = set()
        for i in range(len(digits)):
            for j in range(len(digits)):
                for s in (1, 2):
                    if i != j and scores[j] >= scores[i]:
                        d = int((digits[i] != (s * digits[j].astype(int)) % p).sum())
                        if 1 <= d <= max_dist:
                            expect.add((i, j, s))
        used = np.unique(np.concatenate([got.parent, got.child]))
        assert found == expect
        assert (got.dist >= 1).all() and (got.dist <= max_dist).all()

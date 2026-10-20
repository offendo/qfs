import numpy as np

from rlmath.env import PolyEnv


def make_env(tmp_path, **kw):
    return PolyEnv(num_envs=64, p=3, workers=2, db=str(tmp_path / "heights.sqlite"), **kw)


def test_spaces_and_reset(tmp_path):
    env = make_env(tmp_path)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (64, 35)
    assert env.observation_space.contains(obs) and env.single_observation_space.contains(obs[0])
    env.close()


def test_copying_the_parent_gives_zero_reward_without_a_copy_penalty(tmp_path):
    env = make_env(tmp_path)
    obs, _ = env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(obs.copy())
    assert (reward == 0).all() and terminated.all() and not truncated.any()
    assert (info["child_score"] == info["parent_score"]).all()
    env.close()


def test_copy_penalty_only_hits_exact_copies(tmp_path):
    env = make_env(tmp_path, hamming_penalty=0.5, copy_penalty=100.0)
    obs, _ = env.reset(seed=0)
    child = obs.copy()
    child[32:, 0] = (child[32:, 0] + 1) % 3  # half the rows change one coefficient
    _, reward, _, _, info = env.step(child)
    assert (reward[:32] == -100).all()
    assert np.allclose(reward[32:], info["child_score"][32:] - info["parent_score"][32:] - 0.5)
    env.close()


def test_zero_polynomial_scores_zero(tmp_path):
    env = make_env(tmp_path)
    env.reset(seed=0)
    parent = env.parent_scores.copy()
    _, reward, *_ = env.step(np.zeros((64, 35), dtype=np.int64))
    assert (reward == -parent).all()
    env.close()


def test_hamming_penalty(tmp_path):
    env = make_env(tmp_path, hamming_penalty=0.5)
    obs, _ = env.reset(seed=0)
    parent = env.parent_scores.copy()
    child = obs.copy()
    child[:, :3] = (child[:, :3] + 1) % 3  # change exactly three coefficients
    _, reward, _, _, info = env.step(child)
    assert (info["hamming"] == 3).all()
    assert np.allclose(reward, info["child_score"] - parent - 1.5)
    env.close()


def test_parents_from_the_database(tmp_path):
    env = make_env(tmp_path)
    env.reset(seed=0)
    for _ in range(5):  # score enough random polynomials that some have height >= 2
        env.step(env.action_space.sample())
    env.close()

    env = make_env(tmp_path, parent_db_frac=1.0, parent_min_height=2, parent_db_min_size=1)
    obs, _ = env.reset(seed=1)
    assert (env.parent_scores >= 2).all()
    _, reward, *_ = env.step(obs.copy())
    assert (reward == 0).all()  # the stored heights agree with a fresh computation
    env.close()


def test_parents_switch_to_the_pool_once_it_is_big_enough(tmp_path):
    env = make_env(tmp_path, parent_db_frac=1.0, parent_min_height=2, parent_db_min_size=40)
    env.reset(seed=0)
    assert len(env.pool) < 40 and not (env.parent_scores >= 2).all()  # pool too small: random parents
    assert not env.from_pool.any()
    for _ in range(5):
        env.step(env.action_space.sample())  # the children it scores fill the pool
    assert len(env.pool) >= 40
    assert (env.parent_scores >= 2).all()  # parents now come from the pool
    assert env.from_pool.all()
    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["from_pool"].all()  # reported for the parents the step acted on
    env.close()


def test_pool_is_not_used_when_no_fraction_is_set(tmp_path):
    env = make_env(tmp_path)
    env.reset(seed=0)
    for _ in range(3):
        env.step(env.action_space.sample())
    assert len(env.pool) == 0
    env.close()


def test_pool_sampling_favours_higher_scores():
    from rlmath.env import ParentPool

    pool = ParentPool(35)
    rng = np.random.default_rng(0)
    low = rng.integers(0, 3, size=(500, 35))
    high = rng.integers(0, 3, size=(20, 35))
    pool.add(low, np.full(500, 2), 3)
    pool.add(high, np.full(20, 6), 3)
    assert set(pool.scores[: len(pool)]) == {2, 6}

    def share_high(temperature):
        _, scores = pool.sample(rng, 4000, temperature)
        return (scores == 6).mean()

    assert share_high(1.0) > 0.95  # exp(4) = 55 to 1 in favour of the high level
    assert 0.4 < share_high(1e6) < 0.6  # equal weight per level, although the high level has 25x fewer polynomials
    assert share_high(1.0) > share_high(3.0) > share_high(1e6)


def test_hamming_penalty_has_a_free_budget(tmp_path):
    env = make_env(tmp_path, hamming_penalty=0.5, hamming_free=2)
    obs, _ = env.reset(seed=0)
    parent = env.parent_scores.copy()
    child = obs.copy()
    child[:32, :2] = (child[:32, :2] + 1) % 3  # two changes: inside the budget
    child[32:, :4] = (child[32:, :4] + 1) % 3  # four changes: two beyond it
    _, reward, _, _, info = env.step(child)
    gain = info["child_score"] - parent
    assert np.allclose(reward[:32], gain[:32])
    assert np.allclose(reward[32:], gain[32:] - 1.0)
    env.close()


def test_children_share_a_parent_within_a_group(tmp_path):
    env = PolyEnv(num_envs=48, p=3, workers=2, db=str(tmp_path / "heights.sqlite"), children_per_parent=8)
    obs, _ = env.reset(seed=0)
    groups = obs.reshape(6, 8, 35)
    assert (groups == groups[:, :1]).all()  # the 8 rows of a group are the same parent
    assert len({tuple(g[0]) for g in groups}) > 1  # different groups have different parents
    _, reward, _, _, info = env.step(env.action_space.sample())
    assert (info["parent_score"].reshape(6, 8) == info["parent_score"].reshape(6, 8)[:, :1]).all()
    env.close()
    try:
        PolyEnv(num_envs=50, p=3, workers=1, children_per_parent=8)
    except ValueError:
        pass
    else:
        raise AssertionError("num_envs not divisible by children_per_parent should be rejected")


def test_parent_max_height_caps_pool_parents(tmp_path):
    env = PolyEnv(
        num_envs=64, p=3, workers=2, db=str(tmp_path / "heights.sqlite"), parent_db_frac=1.0, parent_db_min_size=30,
        parent_max_height=3,
    )
    env.reset(seed=0)
    for _ in range(8):
        env.step(env.action_space.sample())
    assert env.from_pool.all() and len(env.pool) >= 30
    assert (env.pool.scores[: len(env.pool)] > 3).any()  # there are higher polynomials in the pool...
    assert env.parent_scores.max() <= 3  # ...but they are not used as parents
    env.parent_max_height = None
    seen = max(env.parent_scores.max(), *(env.step(env.action_space.sample())[4]["parent_score"].max() for _ in range(15)))
    assert seen > 3  # lifting the cap makes them available
    env.close()


def test_repeats_are_penalized_across_steps_and_within_a_batch(tmp_path):
    env = make_env(tmp_path, repeat_penalty=7.0)
    obs, _ = env.reset(seed=0)
    child = np.tile(obs[0], (64, 1))
    child[:, 0] = (child[:, 0] + 1) % 3  # 64 identical children
    _, reward, _, _, info = env.step(child)
    assert not info["repeated"][0] and info["repeated"][1:].all()
    assert (reward[1:] == -7).all() and reward[0] > -7
    _, reward, _, _, info = env.step(child)  # seen in an earlier step: all repeats
    assert info["repeated"].all() and (reward == -7).all()
    env.close()


def test_repeats_are_only_counted_without_a_penalty(tmp_path):
    env = make_env(tmp_path)
    obs, _ = env.reset(seed=0)
    child = np.tile(obs[0], (64, 1))
    child[:, 0] = (child[:, 0] + 1) % 3
    _, reward, _, _, info = env.step(child)
    assert info["repeated"].sum() == 63
    assert np.allclose(reward, info["child_score"] - info["parent_score"])
    env.close()

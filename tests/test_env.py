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

    env = make_env(tmp_path, parent_db_frac=1.0, parent_min_height=2)
    obs, _ = env.reset(seed=1)
    assert (env.parent_scores >= 2).all()
    _, reward, *_ = env.step(obs.copy())
    assert (reward == 0).all()  # the stored heights agree with a fresh computation
    env.close()

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


def test_copying_the_parent_gives_zero_reward(tmp_path):
    env = make_env(tmp_path)
    obs, _ = env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(obs.copy())
    assert (reward == 0).all() and terminated.all() and not truncated.any()
    assert (info["child_score"] == info["parent_score"]).all()
    env.close()


def test_zero_polynomial_scores_zero(tmp_path):
    env = make_env(tmp_path)
    env.reset(seed=0)
    parent = env.parent_scores.copy()
    _, reward, *_ = env.step(np.zeros((64, 35), dtype=np.int64))
    assert (reward == -parent).all()
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

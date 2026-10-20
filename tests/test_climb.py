import numpy as np
import torch

from rlmath.climb import Args, ClimbEnv, main, play, policy_proposer, random_flips
from rlmath.reinforce import Agent

SMALL = dict(d_model=32, n_layers=2, n_heads=4, d_ff=64)


def make_env(tmp_path, **kw):
    return ClimbEnv(64, p=3, workers=2, db=str(tmp_path / "heights.sqlite"), steps=3, **kw)


def test_trajectory_never_goes_down_and_best_is_monotone(tmp_path):
    env = make_env(tmp_path, accept="not_lower")
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    game = play(env, lambda obs, score, t: random_flips(rng, obs, 3))
    assert game["child"].shape == (3, 64, 35)
    assert (np.diff(game["score"], axis=0) >= 0).all()  # the current height never decreases
    assert (game["best"] >= game["best_before"]).all()
    assert (game["best"][-1] >= env.parent_scores).all()
    # a child is taken exactly when it is not lower than the current polynomial
    assert ((game["child_score"] >= game["score"]) == game["accepted"]).all()
    # one coefficient changes per step
    assert ((game["child"] != game["obs"]).sum(axis=2) == 1).all()
    env.close()


def test_free_walk_moves_to_every_child(tmp_path):
    env = make_env(tmp_path)  # accept="always"
    env.reset(seed=0)
    rng = np.random.default_rng(3)
    game = play(env, lambda obs, score, t: random_flips(rng, obs, 3))
    assert game["accepted"].all()
    assert (game["obs"][1:] == game["child"][:-1]).all()  # each step starts where the last one ended
    assert (game["score"][1:] == game["child_score"][:-1]).all()
    assert (np.diff(game["score"], axis=0) < 0).any()  # and the height does go down sometimes
    env.close()


def test_a_polynomial_already_in_the_trajectory_is_a_repeat(tmp_path):
    env = make_env(tmp_path)
    obs, _ = env.reset(seed=0)
    up = obs.copy()
    up[:, 0] = (up[:, 0] + 1) % 3
    _, _, _, _, first = env.step(up)
    assert not first["repeat"].any()
    _, _, _, _, second = env.step(obs)  # back to the parent: the parent is part of the trajectory
    assert second["repeat"].all()
    far = obs.copy()
    far[:, 1] = (far[:, 1] + 1) % 3
    _, _, _, _, third = env.step(far)
    assert not third["repeat"].any()
    env.close()


def test_walk_false_keeps_the_trajectory_at_the_parent(tmp_path):
    env = make_env(tmp_path, walk=False)
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    game = play(env, lambda obs, score, t: random_flips(rng, obs, 3))
    assert (game["obs"] == env.parents[None]).all()
    env.close()


def test_step_reward_is_the_gain_in_best_height(tmp_path):
    env = make_env(tmp_path, accept="not_lower")
    env.reset(seed=0)
    rng = np.random.default_rng(2)
    total = np.zeros(64)
    for _ in range(3):
        _, reward, done, _, _ = env.step(random_flips(rng, env.current, 3))
        total += reward
    assert done.all() and (total == env.best - env.parent_scores).all()
    env.close()


def test_climb_trains_and_the_context_starts_as_a_no_op(tmp_path):
    agent = Agent(3, 4, 4, 32, 2, 4, 64, single=True, context=True)
    obs = torch.randint(0, 3, (5, 35))
    plain = agent(obs).logits
    with_ctx = agent(obs, (torch.full((5,), 4), torch.full((5,), 2))).logits
    assert torch.allclose(plain, with_ctx)  # zero-initialized embeddings
    main(Args(p=3, total_timesteps=2 * 8 * 8, num_envs=16, children_per_parent=4, steps=8, workers=2, parent_db_frac=0.0,
              db=str(tmp_path / "heights.sqlite"), out=str(tmp_path / "run"), cuda=False, curriculum_start=0, **SMALL))
    ckpt = torch.load(tmp_path / "run" / "model.pt")
    assert ckpt["args"]["context"] and ckpt["args"]["single"]
    proposed = policy_proposer(agent, torch.device("cpu"))(obs.numpy(), np.array([1, 2, 3, 4, 5]), 0)
    assert ((proposed != obs.numpy()).sum(axis=1) == 1).all()

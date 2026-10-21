import math

import pytest
import torch

from rlmath.reinforce import Agent, Args, group_advantages, main, update_curriculum


def small_agent(p=3):
    return Agent(p, 4, 4, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def test_agent_shapes_and_logprob():
    agent = small_agent()
    obs = torch.randint(0, 3, (5, 35))
    action, logprob, entropy, dist = agent.get_action(obs)
    assert action.shape == (5, 35) and ((action >= 0) & (action < 3)).all()
    assert dist.logits.shape == (5, 35, 3)
    assert logprob.shape == entropy.shape == (5,)
    assert torch.allclose(agent(obs).log_prob(action).sum(-1), logprob)
    assert entropy.min() > 35 * 0.9  # starts close to uniform: 35 * ln 3 ~ 38


def test_short_run(tmp_path):
    args = Args(
        p=3, total_timesteps=512, num_envs=128, cuda=False, out=str(tmp_path),
        db=str(tmp_path / "heights.sqlite"), d_model=32, n_layers=2, n_heads=4, d_ff=64,
    )
    main(args)
    ckpt = torch.load(tmp_path / "model.pt")
    assert ckpt["args"]["p"] == 3
    small_agent().load_state_dict(ckpt["model"])


def test_reinforce_raises_a_synthetic_reward():
    """The update used in main(), with reward = number of coefficients equal to 1 and no oracle."""
    torch.manual_seed(0)
    agent = small_agent()
    opt = torch.optim.Adam(agent.parameters(), lr=3e-3)
    obs = torch.randint(0, 3, (256, 35))

    def step():
        action, logprob, _, _ = agent.get_action(obs)
        reward = (action == 1).float().sum(1)
        adv = reward - reward.mean()
        adv = adv / (adv.std() + 1e-8)
        loss = -(adv * logprob).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return reward.mean().item()

    first = step()
    for _ in range(60):
        step()
    assert step() > first + 10  # 35 coefficients at chance gives ~11.7; it should approach 35


def test_group_advantages():
    reward = torch.tensor([1.0, 3.0, 10.0, 10.0, 0.0, 4.0])
    adv = group_advantages(reward, group=2, norm=False)
    assert torch.allclose(adv, torch.tensor([-1.0, 1.0, 0.0, 0.0, -2.0, 2.0]))  # centred within each pair
    scaled = group_advantages(reward, group=2, norm=True)
    assert abs(scaled.std().item() - 1) < 1e-3
    single = group_advantages(reward, group=1, norm=False)
    assert torch.allclose(single, reward - reward.mean())  # no group to compare with: the batch mean


def test_curriculum_raises_the_ceiling_after_a_good_window():
    successes = []
    ceiling = 2
    for _ in range(4):
        ceiling = update_curriculum(ceiling, successes, 0.1, window=4, threshold=0.25, cutoff=10)
    assert ceiling == 2  # success rate too low
    for _ in range(2):  # window is now [0.1, 0.1, 0.5, 0.5]: mean 0.3 >= 0.25
        ceiling = update_curriculum(ceiling, successes, 0.5, window=4, threshold=0.25, cutoff=10)
    assert ceiling == 3 and successes == []  # raised once, and the window restarts
    assert update_curriculum(10, [0.9] * 4, 0.9, window=4, threshold=0.25, cutoff=10) == 10  # never past the cutoff


def test_short_run_with_groups_and_curriculum(tmp_path):
    args = Args(
        p=3, total_timesteps=512, num_envs=64, children_per_parent=8, cuda=False, out=str(tmp_path),
        db=str(tmp_path / "heights.sqlite"), parent_db_frac=1.0, parent_db_min_size=10, curriculum_start=2,
        curriculum_window=2, curriculum_threshold=0.0, d_model=32, n_layers=2, n_heads=4, d_ff=64,
    )
    main(args)
    assert (tmp_path / "model.pt").exists()


def test_init_and_kl_penalty(tmp_path):
    base = Args(
        p=3, total_timesteps=256, num_envs=64, children_per_parent=8, cuda=False,
        db=str(tmp_path / "heights.sqlite"), d_model=32, n_layers=2, n_heads=4, d_ff=64,
    )
    main(Args(**{**base.__dict__, "out": str(tmp_path / "a")}))
    ckpt = str(tmp_path / "a" / "model.pt")
    main(Args(**{**base.__dict__, "out": str(tmp_path / "b"), "init": ckpt, "kl_coef": 0.1}))
    assert (tmp_path / "b" / "model.pt").exists()
    with pytest.raises(ValueError):
        main(Args(**{**base.__dict__, "out": str(tmp_path / "c"), "init": ckpt, "d_model": 64}))
    with pytest.raises(ValueError):
        main(Args(**{**base.__dict__, "out": str(tmp_path / "d"), "kl_coef": 0.1}))


def test_single_edit_policy_changes_exactly_one_coefficient():
    agent = Agent(3, 4, 4, d_model=32, n_layers=2, n_heads=4, d_ff=64, single=True)
    obs = torch.randint(0, 3, (16, 35))
    child, logprob, entropy, dist = agent.get_action(obs)
    assert ((child != obs).sum(dim=1) == 1).all()
    assert torch.allclose(agent.child_log_prob(obs, child), logprob)
    assert torch.allclose(dist.probs.sum(-1), torch.ones(16))
    assert dist.probs.shape == (16, 35 * 3)
    assert (dist.probs.reshape(16, 35, 3).gather(-1, obs.unsqueeze(-1)) == 0).all()  # keeping a value is not an option
    assert entropy.min() > 0.9 * math.log(70)  # starts near uniform over the 70 changes


def test_single_edit_run_with_init(tmp_path):
    base = Args(
        p=3, total_timesteps=256, num_envs=64, children_per_parent=8, cuda=False, single=True,
        db=str(tmp_path / "heights.sqlite"), d_model=32, n_layers=2, n_heads=4, d_ff=64,
    )
    main(Args(**{**base.__dict__, "out": str(tmp_path / "a")}))
    main(Args(**{**base.__dict__, "out": str(tmp_path / "b"), "init": str(tmp_path / "a" / "model.pt"), "kl_coef": 0.1, "ent_coef": 0.01}))
    state = torch.load(tmp_path / "b" / "model.pt")["model"]
    assert all(torch.isfinite(v).all() for v in state.values())  # no NaN from the masked logits
    with pytest.raises(ValueError):  # a single-edit checkpoint cannot start a factorized policy
        main(Args(**{**base.__dict__, "out": str(tmp_path / "c"), "init": str(tmp_path / "a" / "model.pt"), "single": False}))

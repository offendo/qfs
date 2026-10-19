import torch

from rlmath.reinforce import Agent, Args, main


def small_agent(p=3):
    return Agent(p, 4, 4, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def test_agent_shapes_and_logprob():
    agent = small_agent()
    obs = torch.randint(0, 3, (5, 35))
    action, logprob, entropy = agent.get_action(obs)
    assert action.shape == (5, 35) and ((action >= 0) & (action < 3)).all()
    assert logprob.shape == entropy.shape == (5,)
    assert torch.allclose(agent(obs).log_prob(action).sum(-1), logprob)
    assert entropy.min() > 35 * 0.9  # starts close to uniform: 35 * ln 3 ~ 38


def test_short_run(tmp_path):
    args = Args(
        p=3, total_timesteps=512, num_envs=128, workers=2, cuda=False, out=str(tmp_path),
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
        action, logprob, _ = agent.get_action(obs)
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

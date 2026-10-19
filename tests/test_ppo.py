
import torch

from rlmath.ppo import Agent, Args, main


def small_agent(p=3):
    return Agent(p, 4, 4, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def test_agent_shapes_and_logprob_consistency():
    agent = small_agent()
    obs = torch.randint(0, 3, (5, 35))
    action, logprob, entropy, value = agent.get_action_and_value(obs)
    assert action.shape == (5, 35) and ((action >= 0) & (action < 3)).all()
    assert logprob.shape == entropy.shape == value.shape == (5,)
    _, logprob2, _, _ = agent.get_action_and_value(obs, action)
    assert torch.allclose(logprob, logprob2)
    assert entropy.min() > 35 * 0.9  # starts close to uniform: 35 * ln 3 ~ 38


def test_short_run(tmp_path):
    args = Args(
        p=3, total_timesteps=512, num_envs=128, workers=2, cuda=False, out=str(tmp_path),
        db=str(tmp_path / "heights.sqlite"), d_model=32, n_layers=2, n_heads=4, d_ff=64,
    )
    main(args)
    ckpt = torch.load(tmp_path / "model.pt")
    assert ckpt["args"]["p"] == 3
    agent = small_agent()
    agent.load_state_dict(ckpt["model"])

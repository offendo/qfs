import torch

from rlmath.model import MaskedPolyTransformer, ModelConfig


def small_model(p=5):
    return MaskedPolyTransformer(ModelConfig(p=p, n=4, d=4, d_model=32, n_layers=2, n_heads=4, d_ff=64))


def test_forward_shape():
    m = small_model()
    c = torch.randint(0, 6, (3, 35))
    h = torch.tensor([0, 2, 11])
    assert m(c, h).shape == (3, 35, 5)


def test_loss_backward():
    m = small_model()
    loss = m.loss(torch.randint(0, 5, (8, 35)), torch.randint(1, 10, (8,)))
    loss.backward()
    assert torch.isfinite(loss)


def test_fill_only_changes_masked_positions():
    m = small_model().eval()
    c = torch.randint(0, 5, (4, 35))
    masked = torch.zeros(4, 35, dtype=torch.bool)
    masked[:, :5] = True
    masked[3] = True  # one row fully masked
    out = m.fill(c, masked, torch.full((4,), 3))
    assert ((out >= 0) & (out < 5)).all()
    assert (out[~masked] == c[~masked]).all()


def test_learns_a_fixed_polynomial():
    torch.manual_seed(0)
    m = small_model(p=3)
    target = torch.randint(0, 3, (1, 35))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        loss = m.loss(target.repeat(32, 1), torch.full((32,), 4))
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    out = m.fill(torch.zeros(1, 35, dtype=torch.long), torch.ones(1, 35, dtype=torch.bool), torch.tensor([4]),
                 temperature=0.1)
    assert (out == target).float().mean() > 0.9


def test_trace_log_prob_matches_the_fill():
    torch.manual_seed(0)
    m = small_model(p=3).eval()
    c = torch.randint(0, 3, (6, 35))
    masked = torch.rand(6, 35) < 0.4
    masked[:, 0] = True
    h = torch.full((6,), 2)
    trace = []
    out = m.fill(c, masked, h, trace=trace)
    # every masked position was revealed exactly once, with the token that ended up in the output
    assert (sum(r.long() for _, r, _ in trace) == masked.long()).all()
    for _, reveal, sample in trace:
        assert (out[reveal] == sample[reveal]).all()
    logp = m.trace_log_prob(trace, h)
    assert logp.shape == (6,) and logp.requires_grad and (logp <= 0).all()
    logp.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_reinforce_on_traces_raises_a_synthetic_reward():
    """The same group-baseline update the trainer uses, with reward = number of coefficients equal to 1."""
    torch.manual_seed(0)
    m = small_model(p=3).eval()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    group, n_groups = 8, 16
    base = torch.randint(0, 3, (n_groups, 35)).repeat_interleave(group, 0)
    masked = torch.zeros_like(base, dtype=torch.bool)
    masked[:, :12] = True
    h = torch.full((len(base),), 2)

    def step():
        trace = []
        out = m.fill(base, masked, h, trace=trace)
        reward = (out == 1).float().sum(1)
        adv = reward.view(-1, group)
        adv = (adv - adv.mean(1, keepdim=True)).view(-1)
        adv = adv / (adv.std() + 1e-8)
        loss = -(adv * m.trace_log_prob(trace, h)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return reward.mean().item()

    first = sum(step() for _ in range(3)) / 3
    for _ in range(80):
        step()
    last = sum(step() for _ in range(3)) / 3
    assert last > first + 3  # 12 masked positions, so the reward can gain at most ~8

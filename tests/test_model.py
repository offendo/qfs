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

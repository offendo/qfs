import numpy as np
import torch

from rlmath.pairs import Pairs
from rlmath.reinforce import Args as RLArgs
from rlmath.reinforce import main as rl_main
from rlmath.sft import Args, main, select

SMALL = dict(d_model=32, n_layers=2, n_heads=4, d_ff=64)


def synthetic_pairs(path, rng, parents=300, per_parent=4):
    digits = rng.integers(0, 3, size=(parents, 35)).astype(np.uint8)
    parent = np.repeat(np.arange(parents), per_parent).astype(np.int32)
    k = len(parent)
    pairs = Pairs(
        digits, parent, rng.integers(0, 35, size=k).astype(np.int8), rng.integers(0, 3, size=k).astype(np.int8),
        rng.integers(2, 6, size=k).astype(np.int16), rng.integers(0, 6, size=k).astype(np.int16),
    )
    pairs.save(str(path))
    return pairs


def test_select_filters_by_outcome_and_parent_score(tmp_path):
    pairs = synthetic_pairs(tmp_path / "pairs.npz", np.random.default_rng(0))
    ge = select(pairs, Args(pairs="x", out="y", require="ge"))
    gt = select(pairs, Args(pairs="x", out="y", require="gt"))
    assert (pairs.child_score[ge] >= pairs.parent_score[ge]).all() and len(gt) < len(ge) <= len(pairs.parent)
    assert (pairs.child_score[gt] > pairs.parent_score[gt]).all()
    high = select(pairs, Args(pairs="x", out="y", require="all", min_parent=4))
    assert (pairs.parent_score[high] >= 4).all()


def test_sft_learns_a_fixed_edit_and_feeds_reinforce(tmp_path):
    rng = np.random.default_rng(1)
    digits = rng.integers(0, 3, size=(200, 35)).astype(np.uint8)
    # every parent has the same edit: coefficient 7 becomes (its value + 1) mod 3
    parent = np.arange(200, dtype=np.int32)
    new = ((digits[:, 7] + 1) % 3).astype(np.int8)
    Pairs(digits, parent, np.full(200, 7, dtype=np.int8), new, np.full(200, 3, dtype=np.int16), np.full(200, 3, dtype=np.int16)).save(
        str(tmp_path / "pairs.npz")
    )
    out = str(tmp_path / "sft.pt")
    main(Args(pairs=str(tmp_path / "pairs.npz"), out=out, epochs=60, batch_size=64, learning_rate=1e-3, holdout=0.0,
              augment=False, cuda=False, **SMALL))
    ckpt = torch.load(out)
    assert ckpt["args"]["d_model"] == 32

    from rlmath.reinforce import Agent

    agent = Agent(3, 4, 4, **SMALL)
    agent.load_state_dict(ckpt["model"])
    obs = torch.as_tensor(digits[:50]).long()
    with torch.no_grad():
        probs = agent(obs).probs  # (50, 35, 3)
    # it should copy elsewhere and make the learned edit at coefficient 7
    copy_prob = probs.gather(-1, obs.unsqueeze(-1)).squeeze(-1)
    assert copy_prob[:, [i for i in range(35) if i != 7]].mean() > 0.8
    assert probs[torch.arange(50), 7, torch.as_tensor(new[:50]).long()].mean() > 0.6

    rl_main(RLArgs(
        p=3, total_timesteps=128, num_envs=64, children_per_parent=8, cuda=False, out=str(tmp_path / "rl"),
        db=str(tmp_path / "heights.sqlite"), init=out, kl_coef=0.05, **SMALL,
    ))
    assert (tmp_path / "rl" / "model.pt").exists()


def test_single_edit_sft_learns_a_fixed_edit(tmp_path):
    rng = np.random.default_rng(2)
    digits = rng.integers(0, 3, size=(200, 35)).astype(np.uint8)
    new = ((digits[:, 7] + 1) % 3).astype(np.int8)
    Pairs(digits, np.arange(200, dtype=np.int32), np.full(200, 7, dtype=np.int8), new, np.full(200, 3, dtype=np.int16),
          np.full(200, 3, dtype=np.int16)).save(str(tmp_path / "pairs.npz"))
    out = str(tmp_path / "sft.pt")
    main(Args(pairs=str(tmp_path / "pairs.npz"), out=out, epochs=200, batch_size=64, learning_rate=1e-3, holdout=0.0,
              augment=False, cuda=False, single=True, **SMALL))
    from rlmath.reinforce import Agent

    ckpt = torch.load(out)
    assert ckpt["args"]["single"] is True
    agent = Agent(3, 4, 4, single=True, **SMALL)
    agent.load_state_dict(ckpt["model"])
    obs = torch.as_tensor(digits[:50]).long()
    child = obs.clone()
    child[:, 7] = torch.as_tensor(new[:50]).long()
    with torch.no_grad():
        assert agent.child_log_prob(obs, child).exp().mean() > 0.6  # the learned edit has most of the probability


def test_sft_reads_near_pairs_and_filters_by_distance(tmp_path):
    from rlmath.pairs import NearPairs

    rng = np.random.default_rng(2)
    digits = rng.integers(0, 3, size=(100, 35)).astype(np.uint8)
    k = 300
    pairs = NearPairs(
        digits, rng.integers(0, 100, size=k).astype(np.int32), rng.integers(0, 100, size=k).astype(np.int32),
        rng.integers(1, 3, size=k).astype(np.int8), rng.integers(1, 8, size=k).astype(np.int8),
        np.full(k, 3, dtype=np.int16), np.full(k, 3, dtype=np.int16),
    )
    pairs.save(str(tmp_path / "near.npz"))
    near = Args(pairs="x", out="y", max_dist=3)
    assert (pairs.dist[select(pairs, near)] <= 3).all() and len(select(pairs, near)) < k
    assert len(select(pairs, Args(pairs="x", out="y", max_pairs=50))) == 50
    out = str(tmp_path / "sft.pt")
    main(Args(pairs=str(tmp_path / "near.npz"), out=out, epochs=2, batch_size=32, max_dist=5, holdout=0.1, cuda=False, **SMALL))
    assert torch.load(out)["args"]["max_dist"] == 5

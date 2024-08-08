#!/usr/bin/env python3
# pyright: reportPrivateImportUsage=false
from collections import defaultdict

from tensordict import TensorDict
import torch
import numpy as np
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase
from tqdm import tqdm

from rlmath_pt.environment import PolynomialPredictionEnv
from rlmath_pt.model import (PolynomialModelConfig, PolynomialOutput,
                             PolynomialTransformer)
from rlmath_pt.utils import (compute_height, compute_num_monomials,
                             compute_reward, decode_polynomial)


def train(model: PolynomialTransformer, env: EnvBase):
    policy = TensorDictModule(
        model,
        in_keys=["polynomial", "height"],
        out_keys=["action"],
    )
    optim = torch.optim.Adam(policy.parameters(), lr=2e-3)
    STEPS = 10000
    pbar = tqdm(range(STEPS), total=STEPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, STEPS)
    logs = defaultdict(list)
    policy.train()

    for _ in pbar:
        init_td = env.reset()
        rollout = env.rollout(100, policy, tensordict=init_td, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()
        (-traj_return).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        pbar.set_description(
            f"reward: {traj_return: 4.4f}, "
            f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
        )
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
        scheduler.step()

if __name__ == "__main__":
    model_config = PolynomialModelConfig(
        n_monomials=compute_num_monomials(2, 4, 4),
        use_input_height=True,
        d_model=32,
        d_hidden=128,
        n_layers=2,
        n_heads=2,
        dropout=0.1,
        threshhold=0.5,
    )
    model = PolynomialTransformer(model_config)
    before = {k: v.sum().detach().numpy() for k,v in model.named_parameters()}
    env = PolynomialPredictionEnv(p=2, n=4, d=4)
    train(model, env)
    after = {k: v.sum().detach().numpy() for k,v in model.named_parameters()}

    eq = all([np.all(bv == av) for (bk, bv), (ak, av) in zip(before.items(), after.items())])
    print(f"All equal? {eq}")

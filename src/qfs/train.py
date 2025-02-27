#!/usr/bin/env python3
# pyright: reportPrivateImportUsage=false
from collections import defaultdict

from tensordict import TensorDict
import torch
import numpy as np
import pandas as pd
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch.distributions import Normal
from torchrl.data import (
    LazyTensorStorage,
    ReplayBuffer,
    SamplerWithoutReplacement,
    replay_buffers,
)
from torchrl.envs import EnvBase
from torchrl.modules import Actor, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.collectors import SyncDataCollector
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

from torchrl.objectives.ppo import GAE
from tqdm import tqdm
from icecream import ic

from rlmath.environment import PolynomialPredictionEnv
from rlmath.model import (
    PolynomialModelConfig,
    PolynomialOutput,
    PolynomialTransformer,
)
from rlmath.utils import compute_batch_height, compute_num_monomials, compute_reward

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    policy_model: PolynomialTransformer,
    value_model: PolynomialTransformer,
    env: EnvBase,
):
    total_frames = 10_000
    frames_per_batch = 128
    num_epochs = 8
    sub_batch_size = 64
    max_grad_norm = 1
    lr = 3e-4

    # Define the policy (polynomial + height --> polynomial)
    policy = TensorDictModule(
        policy_model,
        in_keys=["polynomial", "height"],
        out_keys=["loc", "scale"],
    )
    policy_module = ProbabilisticActor(
        policy,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action", "logits"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )
    value_module = ValueOperator(
        module=value_model,
        in_keys=["polynomial", "height"],
    )
    advantage_module = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=value_module,
        average_gae=True,
    )

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
    )

    # Replay buffer (???)
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    # Optimizer and loss
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch
    )
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=0.2,
        entropy_bonus=True,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    # Store the training logs
    logs = defaultdict(list)
    policy.train()

    pbar = tqdm(collector, total=total_frames)
    for i, tensordict_data in enumerate(pbar):
        for _ in range(num_epochs):
            # Compute the advantage
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(DEVICE))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()

                logs["return"].append(loss_value.item())
        pbar.update(tensordict_data.numel())
        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        cum_reward_str = f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        logs["step_count"].append(i)
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
                # execute a rollout with the trained policy
                rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(rollout["next", "reward"].sum().item())
                logs["eval step_count"].append(i)
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del rollout
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        pbar.set_description(
            ", ".join([eval_str, cum_reward_str, stepcount_str, lr_str])
        )
        scheduler.step()


if __name__ == "__main__":
    model_config = PolynomialModelConfig(
        n_monomials=compute_num_monomials(p=2, n=4, d=4),
        use_input_height=True,
        d_model=128,
        d_hidden=256,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        threshhold=0.5,
    )
    policy = PolynomialTransformer(model_config)
    value = PolynomialTransformer(model_config, value_model=True)

    # before = {k: v.sum().detach().numpy() for k, v in policy.named_parameters()}
    env = PolynomialPredictionEnv(p=2, n=4, d=4)
    train(policy, value, env)
    # after = {k: bv.sum().detach().numpy() for k, v in policy.named_parameters()}
    torch.save(policy, "policy.pt")
    torch.save(value, "value.pt")
    # eq = all(
    #     [np.all(bv == av) for (bk, bv), (ak, av) in zip(before.items(), after.items())]
    # )
    # print(f"All equal? {eq}")

#!/usr/bin/env python3
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torchrl.data.tensor_specs import BoundedDiscrete
import tqdm
from icecream import ic
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import (
    Binary,
    Composite,
    Bounded,
    UnboundedContinuous,
)
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp
from rlmath.utils import (
    compute_num_monomials,
    compute_reward,
    compute_batch_height,
    compute_max_height,
)


class PolynomialPredictionEnv(EnvBase):
    def __init__(
        self,
        p: int,
        n: int,
        d: int,
    ):
        super().__init__()
        self.p = p
        self.d = d
        self.n = n
        self.n_monomials = compute_num_monomials(p, d, n)
        self.max_height = compute_max_height(n, d)

        self.observation_spec = Composite(
            dict(
                height=Bounded(  # type:ignore
                    low=1,
                    high=self.max_height,
                    shape=torch.Size([1]),
                    dtype=torch.float32,
                    device=self.device,
                    domain="discrete",
                ),
                polynomial=Binary(
                    shape=torch.Size([self.n_monomials]),
                    dtype=torch.long,
                    device=self.device,
                ),
            )
        )

        self.action_spec = Bounded(  # type:ignore
            low=0,
            high=1,
            shape=torch.Size([self.n_monomials]),
            dtype=torch.float32,
            device=self.device,
        )

        self.reward_spec = UnboundedContinuous(  # type:ignore
            shape=torch.Size([1]),
            dtype=torch.float32,
            device=self.device,
        )

    def _reset(self, tensordict: TensorDict, **kwargs):
        random_poly = torch.randint(
            0,
            2,
            size=(self.n_monomials,),
            device=self.device,
            dtype=torch.int,
        )
        observation = TensorDict(
            {
                "polynomial": random_poly,
                "height": compute_batch_height(random_poly, self.n, self.d, self.p),
            },
            batch_size=self.batch_size,
        )

        return observation

    def _step(self, tensordict: TensorDict):
        # Previous state
        logits = tensordict["logits"]
        prev_height = tensordict["height"]

        # New state
        new_polynomial = (torch.sigmoid(logits.squeeze(-1)) > 0.5).int()

        # Reminder: Transpose to put batch dim first (N, B) --> (B, N)
        new_height = compute_batch_height(new_polynomial, n=self.n, d=self.d, p=self.p)
        reward = compute_reward(prev_height, new_height, n=self.n, d=self.d)

        # We're done if the policy generated a polynomial of max height
        done = new_height >= self.max_height

        out = TensorDict(
            {
                "height": new_height,
                "polynomial": new_polynomial,
                "reward": reward,
                "done": done,
            },
            batch_size=self.batch_size,
        )
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    _reset = _reset
    _step = _step
    _set_seed = _set_seed

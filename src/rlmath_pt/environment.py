#!/usr/bin/env python3
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import (
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultiOneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
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
from rlmath_pt.utils import (
    compute_num_monomials,
    compute_reward,
    compute_height,
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
        self.max_height = compute_max_height(p, d, n)

        self.observation_spec = CompositeSpec(
            height=BoundedTensorSpec(
                low=1,
                high=self.max_height,
                shape=torch.Size(),
                dtype=torch.float32,
                device=self.device,
            ),
            polynomial=BinaryDiscreteTensorSpec(
                n=self.n_monomials,
                shape=torch.Size([self.n_monomials]),
                dtype=torch.long,
                device=self.device,
            ),
        )

        self.action_spec = BoundedTensorSpec(
            low=0,
            high=1,
            shape=torch.Size([self.n_monomials]),
            dtype=torch.float32,
            device=self.device,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size([1]), dtype=torch.float32, device=self.device)

    def _reset(self, tensordict: TensorDict, **kwargs):
        random_poly = torch.randint(0, 2, size=(self.n_monomials,), device=self.device, dtype=torch.float32)
        observation = TensorDict(
            {
                "polynomial": random_poly.long(),
                "height": compute_height(random_poly),
            },
            batch_size=self.batch_size,
        )

        return observation

    @staticmethod
    def _step(tensordict: TensorDict):
        # Previous state
        guess = tensordict["action"]
        prev_height = tensordict["height"]

        # New state
        new_polynomial = (torch.sigmoid(guess) > 0.5).long()
        new_height = compute_height(guess)
        reward = compute_reward(prev_height, new_polynomial)
        done = torch.zeros_like(reward, dtype=torch.bool)

        out = TensorDict(
            {
                "height": new_height,
                "polynomial": new_polynomial,
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    _reset = _reset
    _step = _step
    _set_seed = _set_seed

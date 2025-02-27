#!/usr/bin/env python3
from rlmath.model import PolynomialTransformer, PolynomialModelConfig, PolynomialOutput
from tensordict.nn import TensorDictModule


def make_policy(model_config: PolynomialModelConfig):
    policy = TensorDictModule(
        PolynomialTransformer(model_config),
        in_keys=["observation"],
        out_keys=["action"],
    )
    return policy

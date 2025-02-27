#!/usr/bin/env python3
import logging
import math

import numpy as np
import torch
import zmq
import msgpack
from icecream import ic

logging.basicConfig(level=logging.INFO)


ND_4_EXPONENT_VECTOR = [
    [0, 0, 0, 4],
    [0, 0, 1, 3],
    [0, 0, 2, 2],
    [0, 0, 3, 1],
    [0, 0, 4, 0],
    [0, 1, 0, 3],
    [0, 1, 1, 2],
    [0, 1, 2, 1],
    [0, 1, 3, 0],
    [0, 2, 0, 2],
    [0, 2, 1, 1],
    [0, 2, 2, 0],
    [0, 3, 0, 1],
    [0, 3, 1, 0],
    [0, 4, 0, 0],
    [1, 0, 0, 3],
    [1, 0, 1, 2],
    [1, 0, 2, 1],
    [1, 0, 3, 0],
    [1, 1, 0, 2],
    [1, 1, 1, 1],
    [1, 1, 2, 0],
    [1, 2, 0, 1],
    [1, 2, 1, 0],
    [1, 3, 0, 0],
    [2, 0, 0, 2],
    [2, 0, 1, 1],
    [2, 0, 2, 0],
    [2, 1, 0, 1],
    [2, 1, 1, 0],
    [2, 2, 0, 0],
    [3, 0, 0, 1],
    [3, 0, 1, 0],
    [3, 1, 0, 0],
    [4, 0, 0, 0],
]


class Client:

    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

    def compute_height(self, polynomial: np.ndarray, n: int, d: int, p: int):
        data = msgpack.packb((polynomial.tobytes(), n, d, p))
        self.socket.send(data)

        #  Get the reply.
        height = int.from_bytes(self.socket.recv())
        return height


client = Client()


def compute_max_height(n: int, d: int):
    if n == 4 and d == 4:
        return 10
    if n == 5 and d == 5:
        return 101
    raise NotImplementedError(f"Polynomial with {n=} and {d=} is not supported yet.")


def compute_num_monomials(p: int, d: int, n: int) -> int:
    # simple stars & bars
    # p-1 because 0 is implicit
    return int((p - 1) * math.comb(n + d - 1, n - 1))


def get_exponent_vector(n: int, d: int):
    if n == 4 and d == 4:
        return ND_4_EXPONENT_VECTOR
    raise NotImplementedError("Only have N=D=4 right now.")


def compute_batch_height(batch: torch.Tensor, n: int, d: int, p: int) -> torch.Tensor:
    if len(batch.shape) == 1:
        return torch.tensor(
            client.compute_height(batch.numpy(), n=n, d=d, p=p),
            dtype=torch.float32,
            device=batch.device,
        )

    return torch.tensor(
        [client.compute_height(ex.numpy(), n=n, d=d, p=p) for ex in batch],
        dtype=torch.float32,
        device=batch.device,
    )


def compute_reward(old_h: torch.Tensor, new_h: torch.Tensor, n: int, d: int):
    # We encode height H+1 as infinity, but we treat that differently I think,
    # since it's "less" interesting than H???
    # So, mask out the infinities, and replace them with INF_VAL so we don't
    # over-weight infinities

    # Assertion depends on proof of bound on height for CY varieties
    assert n == d, "Not supported for n != d"
    H = compute_max_height(n=n, d=d)
    INF_VAL = H + 1  # (change to whatever value feels right for "infinity" reward)
    inf_mask = new_h > H
    masked_h = (new_h * ~(inf_mask)) + (inf_mask * INF_VAL)

    # Reward function
    # reward = torch.nn.functional.relu(masked_h - old_h) - 0.5
    reward = (
        torch.heaviside(masked_h - old_h, values=torch.zeros(1, device=old_h.device))
        - 0.5
    )
    return reward

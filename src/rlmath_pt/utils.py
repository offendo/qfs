#!/usr/bin/env python3
import torch


def compute_max_height(p: int, d: int, n: int) -> int:
    return 99999999999999


def compute_num_monomials(p: int, d: int, n: int) -> int:
    return 30


def decode_polynomial(idx2mon: dict[int, str], vector: torch.Tensor) -> str:
    return ""


def _compute_height_julia(poly: torch.Tensor):
    x = torch.round(torch.sum(poly, dim=-1))
    return x.float()


def compute_height(poly: torch.Tensor) -> torch.Tensor:
    if len(poly.shape) == 1:
        return _compute_height_julia(poly)

    return torch.tensor(
        [_compute_height_julia(p) for p in poly],
        dtype=torch.float32,
        device=poly.device,
    )


def compute_reward(h0: torch.Tensor, poly: torch.Tensor):
    h1 = compute_height(poly)
    return torch.nn.functional.elu(h1 - h0, alpha=1) - 0.5

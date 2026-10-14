"""Masked-coefficient transformer.

The model sees a polynomial as one token per monomial. Each token is the monomial's coefficient
(an element of F_p, or [MASK]) plus an embedding of which monomial it is. A height token in front
says what height the polynomial has (in training) or should have (when generating).

Trained like BERT/MaskGIT: hide a random subset of the coefficients of a good polynomial and
predict them. Used two ways:
  - edit: take a good polynomial, mask k coefficients, fill them in  (the "f -> f'" move)
  - generate: mask everything and fill it in over several steps
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlmath.polynomials import monomials


@dataclass
class ModelConfig:
    p: int
    n: int
    d: int
    max_height: int = 12  # largest height token; scores above this are clipped
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1


class MaskedPolyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mask_id = cfg.p  # coefficient vocabulary is 0..p-1, plus [MASK]
        self.null_height = 0  # height token 0 = "unconditional"

        self.coeff_emb = nn.Embedding(cfg.p + 1, cfg.d_model)
        # One embedding table per variable, indexed by exponent. A monomial's embedding is the sum
        # over its variables, so permuting variables permutes the inputs in a consistent way.
        self.exp_emb = nn.ModuleList(nn.Embedding(cfg.d + 1, cfg.d_model) for _ in range(cfg.n))
        self.height_emb = nn.Embedding(cfg.max_height + 1, cfg.d_model)
        self.register_buffer("exponents", torch.as_tensor(monomials(cfg.n, cfg.d)), persistent=False)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.p)

    @property
    def n_monomials(self) -> int:
        return self.exponents.shape[0]

    def forward(self, coeffs: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
        """coeffs: (B, M) in 0..p (p = [MASK]); height: (B,) in 0..max_height. Returns logits (B, M, p)."""
        mono = sum(emb(self.exponents[:, j]) for j, emb in enumerate(self.exp_emb))  # (M, D)
        x = self.coeff_emb(coeffs) + mono
        h = self.height_emb(height.clamp(0, self.cfg.max_height)).unsqueeze(1)  # (B, 1, D)
        x = self.encoder(torch.cat([h, x], dim=1))
        return self.head(self.norm(x[:, 1:]))

    def loss(self, coeffs: torch.Tensor, height: torch.Tensor, cond_dropout: float = 0.1) -> torch.Tensor:
        """MaskGIT objective: mask a random fraction (cosine schedule) of each row, predict it."""
        B, M = coeffs.shape
        r = torch.rand(B, device=coeffs.device)
        n_mask = (torch.cos(r * math.pi / 2) * M).ceil().clamp(1, M)  # at least one masked
        scores = torch.rand(B, M, device=coeffs.device)
        ranks = scores.argsort(dim=1).argsort(dim=1)
        masked = ranks < n_mask.unsqueeze(1)
        inputs = coeffs.masked_fill(masked, self.mask_id)
        # Sometimes hide the height so the model can also generate unconditionally.
        drop = torch.rand(B, device=coeffs.device) < cond_dropout
        height = height.masked_fill(drop, self.null_height)
        logits = self(inputs, height)
        return F.cross_entropy(logits[masked], coeffs[masked])

    @torch.no_grad()
    def fill(
        self,
        coeffs: torch.Tensor,
        masked: torch.Tensor,
        height: torch.Tensor,
        temperature: float = 1.0,
        steps: int = 8,
    ) -> torch.Tensor:
        """Fill the masked positions of each row by iterative MaskGIT decoding.

        At each step, sample every still-masked position, keep the most confident samples, and
        re-mask the rest; the number left masked follows a cosine schedule down to zero.
        """
        coeffs = coeffs.clone()
        masked = masked.clone()
        coeffs[masked] = self.mask_id
        total = masked.sum(dim=1)
        for step in range(steps):
            if not masked.any():
                break
            logits = self(coeffs, height) / max(temperature, 1e-6)
            probs = logits.softmax(-1)
            sample = torch.distributions.Categorical(probs=probs).sample()  # (B, M)
            conf = probs.gather(-1, sample.unsqueeze(-1)).squeeze(-1)
            conf = conf.masked_fill(~masked, -1.0)
            # how many should still be masked after this step
            frac = math.cos(math.pi / 2 * (step + 1) / steps)
            keep_masked = (total.float() * frac).floor().long()
            n_reveal = (masked.sum(dim=1) - keep_masked).clamp(min=1)
            order = conf.argsort(dim=1, descending=True)
            rank = order.argsort(dim=1)
            reveal = masked & (rank < n_reveal.unsqueeze(1))
            coeffs[reveal] = sample[reveal]
            masked &= ~reveal
        return coeffs

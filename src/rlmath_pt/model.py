#!/usr/bin/env python3

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PolynomialModelConfig:
    n_monomials: int
    use_input_height: bool
    d_model: int = 512
    d_hidden: int = 2048
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    threshhold: float = 0.5


class PolynomialOutput:
    logits: torch.FloatTensor | None
    predictions: torch.LongTensor | None

    def __init__(self, logits=None, predictions=None):
        self.logits = logits
        self.predictions = predictions

    def __repr__(self):
        return f"""PolynomialOutput(logits={self.logits}, predictions={self.predictions})"""

    def __getitem__(self, q):
        if isinstance(q, int):
            if q == 0:
                return self.logits
            elif q == 1:
                return self.predictions
            else:
                raise IndexError(f"No such index {q}")
        elif isinstance(q, str):
            if q == "logits":
                return self.logits
            elif q == "predictions":
                return self.predictions
            else:
                raise KeyError(f"No such key {q}")


class PolynomialTransformer(nn.Module):
    """Model designed to generate a polynomial with some qFs height `h > h_0` by transforming a monomial sequence into another."""

    # Input representation:
    # Each monomial is a possible token in the vocabulary. Then, our polynomial is a bag of words (sequence of monomials).
    # How do we handle coefficients?
    # - One possibility is to just treat 1x^n as a completely different monomial as 2x^n, etc. This is
    #   very easy to represent, but may remove some information as we know the two are linked in some way.
    # - Another possiiblity is to treat 2x^n as twice the vector of 1x^n. This causes some problems
    #   though, since we're not actually dealing with the integers 1 and 2, but rather the elements 1
    #   and 2 of the finite field. Who knows is this actually matters?
    # Output representation:
    # We want to output a polynomial. We could use a decoder, but this is slow and unnecessary since
    # polynomials are non-autoregressive. Instead, we can just output a single vector and threshold the
    # logits to determine which monomials are present in the output. This is contingent on using the
    # "each coefficient means different monomial" strategy.
    def __init__(self, config: PolynomialModelConfig):
        super().__init__()
        self.config = config

        # If we're using the input height, make sure we add one to the vocab size
        self.embedding = nn.Embedding(
            config.n_monomials + config.use_input_height,
            config.d_model,
        )

        # Pool the output using average
        self.pooler = torch.mean

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_hidden,
            dropout=config.dropout,
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)

        # Output layer
        self.monomial_head = nn.Linear(config.d_model, config.n_monomials)

    def forward(self, polynomial: torch.LongTensor, height: torch.LongTensor | None = None):
        # Embed -> encode -> project
        polynomial = self.embedding(polynomial)  # B, N, D
        polynomial = self.encoder(polynomial)  # B, N, D
        logits = self.monomial_head(polynomial)  # B, N, V

        # Pool -> sigmoid -> threshhold
        # preds = torch.sigmoid(self.pooler(logits, dim=1)) > self.config.threshhold

        return torch.mean(logits, dim=-1)

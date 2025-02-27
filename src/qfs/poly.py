import logging
import os
import numpy as np
from rlmath.utils import compute_polynomial_height, compute_max_height

class Poly:
    def __init__(
        self, p: int, n: int, degree: int, coeffs: list[int], ordering: list[list[int]]
    ):
        self.p = p
        self.degree = degree
        if self.degree > 8:
            raise NotImplementedError("Degree > 8 not supported yet")
        self.coeffs = coeffs
        self.ordering = ordering
        self.max_height = compute_max_height(p, n, degree)

    def get_height(self):
        return compute_polynomial_height(
            self.p, self.coeffs, self.ordering, cutoff=self.max_height
        )

    def __repr__(self):
        vs = "abcdwxyz"[: self.degree]
        terms = []
        for coeff, monomial in zip(self.coeffs, self.ordering):
            if coeff != 0:
                terms.append(
                    (str(coeff) if coeff != 1 else "")
                    + "*".join(
                        [f"{vs[i]}^{m}" for i, m in enumerate(monomial) if m != 0]
                    )
                )
        return " + ".join(terms)

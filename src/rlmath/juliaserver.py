#!/usr/bin/env python3
from typing import Iterable
import time
import zmq
import os
import numpy as np
import msgpack

os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
from juliacall import Main as jl

jl.seval("using Qfs;")
from rlmath.utils import get_exponent_vector, compute_max_height
import torch
import logging


def compute_polynomial_height(coeffs: Iterable, n: int, d: int, p: int):
    """Calculates qFs height (Julia wrapper)

    Parameters
    ----------
    coeffs : list[int]
        List of coefficients in the order of `monomials`
    n : int
        Number of variables the polynomial is in
    d : int
        Degree of the polynomial
    p : int
        Field dimension (prime number)

    Examples
    --------
    >>> compute_polynomial_height([1, 1], 2, 4, 4)

    """

    monomials = get_exponent_vector(n, d)
    cutoff = compute_max_height(n, d)
    if isinstance(coeffs, torch.Tensor):
        cs = coeffs.detach().int().numpy().ravel()
    else:
        cs = np.array(coeffs, dtype=int).ravel()

    # Don't try to compute the height of an empty polynomial
    if sum(cs) == 0:
        return 0

    # quasiFSplitHeight_CY_lift_sort
    R, vars, poly, height = jl.Qfs.qfsFromCoeffs_d4(p, cs, monomials, cutoff)
    return height


def run():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    logging.info("Running server on port 5555")
    while True:
        logging.debug("Waiting for message...")
        #  Wait for next request from client
        buf = socket.recv()
        poly_b, n, d, p = msgpack.unpackb(buf)
        polynomial = np.frombuffer(poly_b, dtype=np.int32)
        height = compute_polynomial_height(polynomial, n, d, p)
        #  Send reply back to client
        socket.send(height.to_bytes(length=16))
        logging.debug(f"Height of {polynomial} is {height}.")

"""Public layers: permutations, repetition/summation, plain accumulators, random matrices."""

from __future__ import annotations

import numpy as np

from ..fields import Field
from ..ops import linalg
from ..ops.operators import DenseOperator, IdentityOperator, LinearOperator, PermutationOperator
from ..ops.structured import block_sum, plain_accumulator, repetition, reversal

PUBLIC_KINDS = ("identity", "permutation", "random", "random_invertible")


def identity(field: Field, n: int) -> IdentityOperator:
    return IdentityOperator(field, n)


def random_permutation(field: Field, n: int, rng: np.random.Generator) -> PermutationOperator:
    return PermutationOperator(field, rng.permutation(n))


def random_matrix(
    field: Field, rows: int, cols: int, rng: np.random.Generator, invertible: bool = False
) -> DenseOperator:
    matrix = field.random(rng, (rows, cols))
    if invertible:
        if rows != cols:
            raise ValueError("only square matrices can be invertible")
        while not linalg.is_invertible(field, matrix):
            matrix = field.random(rng, (rows, cols))
    return DenseOperator(field, matrix)


def public_layer(kind: str, field: Field, n: int, rng: np.random.Generator) -> LinearOperator:
    """Square ``n x n`` public layer of the given kind (see ``PUBLIC_KINDS``)."""
    if kind == "identity":
        return identity(field, n)
    if kind == "permutation":
        return random_permutation(field, n, rng)
    if kind == "random":
        return random_matrix(field, n, n, rng)
    if kind == "random_invertible":
        return random_matrix(field, n, n, rng, invertible=True)
    raise ValueError(f"unknown public layer kind {kind!r}; expected one of {PUBLIC_KINDS}")


__all__ = [
    "PUBLIC_KINDS",
    "block_sum",
    "identity",
    "plain_accumulator",
    "public_layer",
    "random_matrix",
    "random_permutation",
    "repetition",
    "reversal",
]

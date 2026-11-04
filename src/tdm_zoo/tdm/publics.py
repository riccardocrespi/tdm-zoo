"""Public layers: permutations, repetition/summation, plain accumulators, random and LUOF matrices."""

from __future__ import annotations

import numpy as np

from ..fields import Field
from ..ops import linalg
from ..ops.operators import DenseOperator, IdentityOperator, LinearOperator, PermutationOperator
from ..ops.structured import block_sum, plain_accumulator, repetition, reversal
from .di14 import LUOFCodes
from .families import BlockDiagonalFamily

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


def luof_matrix(
    field: Field, codes: LUOFCodes, rng: np.random.Generator, invertible: bool = False, tries: int = 100
) -> DenseOperator:
    """One member ``A = G H P E`` of the linear uniform output family, as a public ``k x k`` matrix.

    ``G`` is ``codes.extractor``; ``P E`` is ``codes.encoder``; ``H`` is a fresh uniform
    block-diagonal matrix, revealed with the rest of ``A``. Small parameters can make every
    member singular, so ``invertible=True`` gives up after ``tries`` draws.
    """
    family = BlockDiagonalFamily(field, codes.inner_dimension, codes.block_size, invertible=False)
    left, right = codes.extractor.to_dense(), codes.encoder.to_dense()
    for _ in range(tries):
        matrix = field.matmul(field.matmul(left, family.sample(rng).matrix()), right)
        if not invertible or linalg.is_invertible(field, matrix):
            return DenseOperator(field, matrix)
    raise ValueError(f"no invertible G H P E in {tries} draws")


def luof_span_dimension(field: Field, codes: LUOFCodes) -> int:
    """``dim span{G H P E}`` over all block-diagonal ``H``; ``A`` is exactly uniform iff this is ``k^2``."""
    left, right = codes.extractor.to_dense(), codes.encoder.to_dense()
    block = codes.block_size
    a, b = np.divmod(np.arange(codes.inner_dimension * block), block)
    columns = a // block * block + b  # H[a, column] runs over the entries of the diagonal blocks
    outer = field.mul(left[:, a].T[:, :, None], right[columns][:, None, :])
    return linalg.rank(field, outer.reshape(a.size, -1))


__all__ = [
    "PUBLIC_KINDS",
    "block_sum",
    "identity",
    "luof_matrix",
    "luof_span_dimension",
    "plain_accumulator",
    "public_layer",
    "random_matrix",
    "random_permutation",
    "repetition",
    "reversal",
]

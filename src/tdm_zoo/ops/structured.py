"""Fast structured operators: block-diagonal, Toeplitz, Hankel, circulant, accumulators.

These are the building blocks of the secret layers. Their ``cost`` reports the
nominal complexity of the fast algorithm (e.g. FFT-based convolution for
Toeplitz/circulant), independently of the implementation used here.
"""

from __future__ import annotations

import math

import flint
import numpy as np

from ..fields import Field
from . import linalg
from .operators import (
    ComposedOperator,
    DenseOperator,
    DiagonalOperator,
    LinearOperator,
    PermutationOperator,
    SelectionOperator,
    SparseOperator,
)

Array = np.ndarray

_FLINT_MIN_LENGTH = 64
_FLINT_MAX_BATCH = 8


def _nominal_fft_cost(length: int) -> int:
    return 2 * length * max(1, math.ceil(math.log2(max(length, 2))))


def convolve(field: Field, kernel: Array, x: Array) -> Array:
    """Full linear convolution of ``kernel`` with every column of ``x``.

    Returns shape ``(len(kernel) + x.shape[0] - 1, x.shape[1])``.
    """
    kernel = np.asarray(kernel, dtype=np.int64)
    length = kernel.size + x.shape[0] - 1
    if (
        field.is_prime_field
        and x.shape[1] <= _FLINT_MAX_BATCH
        and min(kernel.size, x.shape[0]) >= _FLINT_MIN_LENGTH
    ):
        poly = flint.nmod_poly(kernel.tolist(), field.p)
        out = np.zeros((length, x.shape[1]), dtype=np.int64)
        for column in range(x.shape[1]):
            coefficients = (poly * flint.nmod_poly(x[:, column].tolist(), field.p)).coeffs()
            out[: len(coefficients), column] = [int(c) for c in coefficients]
        return out

    out = field.zeros((length, x.shape[1]))
    if field.is_prime_field:
        # Integer accumulation with periodic reduction.
        p = field.p
        budget = max(1, (2**63 - 1) // ((p - 1) ** 2 + p) - 1)
        for start in range(0, kernel.size, budget):
            for i in range(start, min(start + budget, kernel.size)):
                if kernel[i]:
                    out[i : i + x.shape[0]] += kernel[i] * x
            out %= p
        return out
    for i in np.flatnonzero(kernel):
        out[i : i + x.shape[0]] = field.add(out[i : i + x.shape[0]], field.mul(kernel[i], x))
    return out


class BlockDiagonalOperator(LinearOperator):
    """``diag(B_1, ..., B_t)`` with square ``b x b`` blocks given as an array ``(t, b, b)``."""

    def __init__(self, field: Field, blocks: Array) -> None:
        blocks = field.asarray(blocks)
        if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
            raise ValueError("blocks must have shape (count, b, b)")
        n = blocks.shape[0] * blocks.shape[1]
        super().__init__(field, (n, n))
        self.blocks = blocks

    def _block_apply(self, blocks: Array, x: Array) -> Array:
        count, b, _ = blocks.shape
        xr = x.reshape(count, b, -1)
        out = self.field.zeros((count, b, x.shape[1]))
        for j in range(b):
            out = self.field.add(out, self.field.mul(blocks[:, :, j, None], xr[:, None, j, :]))
        return out.reshape(count * b, -1)

    def _apply(self, x: Array) -> Array:
        return self._block_apply(self.blocks, x)

    def _apply_T(self, y: Array) -> Array:
        return self._block_apply(self.blocks.transpose(0, 2, 1), y)

    def inverse(self) -> LinearOperator:
        inverses = np.stack([linalg.inv(self.field, block) for block in self.blocks])
        return BlockDiagonalOperator(self.field, inverses)

    @property
    def cost(self) -> int:
        count, b, _ = self.blocks.shape
        return 2 * count * b * b


class ToeplitzOperator(LinearOperator):
    """``T[i, j] = diagonals[i - j + cols - 1]`` (``rows + cols - 1`` diagonals)."""

    def __init__(self, field: Field, diagonals: Array, rows: int, cols: int) -> None:
        diagonals = field.asarray(diagonals)
        if diagonals.shape != (rows + cols - 1,):
            raise ValueError("need rows + cols - 1 diagonals")
        super().__init__(field, (rows, cols))
        self.diagonals = diagonals

    def _apply(self, x: Array) -> Array:
        full = convolve(self.field, self.diagonals, x)
        return full[self.cols - 1 : self.cols - 1 + self.rows]

    def _apply_T(self, y: Array) -> Array:
        full = convolve(self.field, self.diagonals[::-1], y)
        return full[self.rows - 1 : self.rows - 1 + self.cols]

    @property
    def cost(self) -> int:
        return _nominal_fft_cost(self.rows + self.cols)


def reversal(field: Field, n: int) -> PermutationOperator:
    """The exchange matrix ``J`` (reverses coordinates)."""
    return PermutationOperator(field, np.arange(n)[::-1])


def hankel_operator(field: Field, antidiagonals: Array, rows: int, cols: int) -> LinearOperator:
    """``H[i, j] = antidiagonals[i + j]``, computed as ``Toeplitz(antidiagonals) @ J``."""
    return ComposedOperator([ToeplitzOperator(field, antidiagonals, rows, cols), reversal(field, cols)])


class CirculantOperator(LinearOperator):
    """``C[i, j] = column[(i - j) mod n]``."""

    def __init__(self, field: Field, column: Array) -> None:
        column = field.asarray(column)
        super().__init__(field, (column.size, column.size))
        self.column = column

    def _apply(self, x: Array) -> Array:
        n = self.rows
        full = convolve(self.field, self.column, x)
        tail = self.field.zeros((n, x.shape[1]))
        tail[: n - 1] = full[n:]
        return self.field.add(full[:n], tail)

    def _apply_T(self, y: Array) -> Array:
        transposed = np.concatenate([self.column[:1], self.column[1:][::-1]])
        return CirculantOperator(self.field, transposed)._apply(y)

    def inverse(self) -> LinearOperator:
        e0 = self.field.zeros(self.rows)
        e0[0] = 1
        first_column = linalg.solve(self.field, self.to_dense(), e0)
        return CirculantOperator(self.field, first_column)

    @property
    def cost(self) -> int:
        return _nominal_fft_cost(2 * self.rows)


class AccumulatorOperator(LinearOperator):
    """Weighted accumulation ``y[a] = sum_{b <= a} w[b] x[b]``, i.e. ``C_n diag(w)``."""

    def __init__(self, field: Field, weights: Array) -> None:
        weights = field.asarray(weights)
        super().__init__(field, (weights.size, weights.size))
        self.weights = weights

    def _apply(self, x: Array) -> Array:
        return self.field.cumsum(self.field.mul(self.weights[:, None], x), axis=0)

    def _apply_T(self, y: Array) -> Array:
        suffix = self.field.cumsum(y[::-1], axis=0)[::-1]
        return self.field.mul(self.weights[:, None], suffix)

    def inverse(self) -> LinearOperator:
        if np.any(self.weights == 0):
            raise ValueError("weights have zero entries")
        return ComposedOperator(
            [DiagonalOperator(self.field, self.field.inv(self.weights)), difference(self.field, self.rows)]
        )

    @property
    def cost(self) -> int:
        return 2 * self.rows


def plain_accumulator(field: Field, n: int) -> AccumulatorOperator:
    """``C_n``: the all-ones lower-triangular matrix (prefix sums)."""
    return AccumulatorOperator(field, field.ones(n))


def difference(field: Field, n: int) -> SparseOperator:
    """``C_n^{-1} = I - Z``: ``y[a] = x[a] - x[a-1]``."""
    rows = np.concatenate([np.arange(n), np.arange(1, n)])
    cols = np.concatenate([np.arange(n), np.arange(n - 1)])
    values = np.concatenate([field.ones(n), field.neg(field.ones(n - 1))])
    return SparseOperator(field, rows, cols, values, (n, n))


def repetition(field: Field, k: int, copies: int) -> SelectionOperator:
    """``D``: ``(ck x k)`` matrix repeating each coordinate ``copies`` times consecutively."""
    return SelectionOperator(field, np.repeat(np.arange(k), copies), k)


def block_sum(field: Field, k: int, copies: int) -> LinearOperator:
    """``D^T``: sums consecutive blocks of length ``copies``."""
    return repetition(field, k, copies).T


__all__ = [
    "AccumulatorOperator",
    "BlockDiagonalOperator",
    "CirculantOperator",
    "DenseOperator",
    "ToeplitzOperator",
    "block_sum",
    "convolve",
    "difference",
    "hankel_operator",
    "plain_accumulator",
    "repetition",
    "reversal",
]

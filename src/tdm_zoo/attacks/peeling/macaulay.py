"""Macaulay matrices of multi-homogeneous polynomial systems.

Given polynomials ``p_1, ..., p_K`` of multidegree ``e = (e_0, ..., e_{B-1})``
(coefficient rows in a :class:`ProductBasis`), the Macaulay matrix at multiplier
multidegree ``a`` has

* rows ``m * p_i`` for every monomial ``m`` of multidegree ``a`` and every ``i``
  (row index ``mult_index * K + i``);
* columns indexed by the monomials of multidegree ``e + a`` (the canonical
  :class:`ProductBasis` order);

and entry = coefficient of the column monomial in the row polynomial.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...fields import Field
from ...ops.multihomogeneous import ProductBasis
from ...ops.operators import LinearOperator, SparseOperator

Array = np.ndarray

DEFAULT_MAX_DENSE_ENTRIES = 2 * 10**8  # rows x cols of a dense array (int64: 1.6 GB)
DEFAULT_MAX_NONZEROS = 3 * 10**8  # stored entries of an explicit sparse matrix


@dataclass(frozen=True)
class MacaulayShape:
    polynomials: int  # K
    base: ProductBasis  # multidegree of the p_i
    multipliers: ProductBasis  # multidegree a
    columns: ProductBasis  # multidegree e + a

    @property
    def rows(self) -> int:
        return self.polynomials * self.multipliers.size

    @property
    def cols(self) -> int:
        return self.columns.size

    @property
    def entries(self) -> int:
        return self.rows * self.cols

    def describe(self) -> dict:
        return {
            "K": self.polynomials,
            "rows": self.rows,
            "cols": self.cols,
            "base_degrees": list(self.base.degrees),
            "multiplier_degrees": list(self.multipliers.degrees),
            "column_degrees": list(self.columns.degrees),
        }


def macaulay_shape(polynomials: int, base: ProductBasis, multipliers: tuple[int, ...]) -> MacaulayShape:
    if len(multipliers) != len(base.degrees):
        raise ValueError("one multiplier degree per block")
    mult = ProductBasis(base.nvars, tuple(int(a) for a in multipliers))
    cols = ProductBasis(base.nvars, tuple(e + a for e, a in zip(base.degrees, mult.degrees, strict=True)))
    return MacaulayShape(polynomials, base, mult, cols)


def column_table(shape: MacaulayShape) -> Array:
    """``table[m, c]``: column of ``mult_m * base_c`` (shape ``|mult| x |base|``).

    Row ``m * K + i`` of the Macaulay matrix holds ``coefficients[i, c]`` at column
    ``table[m, c]``; the table plus the ``K x |base|`` coefficients determine the matrix.
    """
    tables = shape.multipliers.times(shape.base)
    strides = shape.columns.strides
    blocks = len(tables)
    table = np.zeros(shape.multipliers.shape + shape.base.shape, dtype=np.int64)
    for b, block_table in enumerate(tables):
        # broadcast block b's (mult digit, base digit) table over the 2B mixed-radix axes
        axes = [1] * (2 * blocks)
        axes[b], axes[blocks + b] = block_table.shape
        table += block_table.reshape(axes) * strides[b]
    return table.reshape(shape.multipliers.size, shape.base.size)


def _check(coefficients: Array, shape: MacaulayShape) -> Array:
    coefficients = np.asarray(coefficients, dtype=np.int64)
    if coefficients.shape != (shape.polynomials, shape.base.size):
        raise ValueError(f"coefficients must have shape {(shape.polynomials, shape.base.size)}")
    return coefficients


def nonzeros(coefficients: Array, shape: MacaulayShape) -> int:
    """Exact number of nonzero entries of the Macaulay matrix."""
    return int(np.count_nonzero(coefficients)) * shape.multipliers.size


def macaulay_matrix(
    field: Field, coefficients: Array, shape: MacaulayShape, max_entries: int = DEFAULT_MAX_DENSE_ENTRIES
) -> Array:
    """Dense Macaulay matrix ``(K * |mult|) x |columns|`` (for small instances / exact elimination).

    ``max_entries`` bounds ``rows x cols``: this is the memory a dense array really needs.
    """
    coefficients = _check(coefficients, shape)
    if shape.entries > max_entries:
        raise MemoryError(
            f"dense Macaulay matrix {shape.rows} x {shape.cols} exceeds max_entries={max_entries}; "
            "use the sparse or implicit form"
        )
    table = column_table(shape)
    poly, flat = np.nonzero(coefficients)
    values = coefficients[poly, flat]
    matrix = field.zeros((shape.rows, shape.cols))
    for mult_index in range(shape.multipliers.size):
        matrix[mult_index * shape.polynomials + poly, table[mult_index, flat]] = values
    return matrix


def macaulay_sparse(
    field: Field, coefficients: Array, shape: MacaulayShape, max_nonzeros: int = DEFAULT_MAX_NONZEROS
) -> SparseOperator:
    """Explicit sparse (coordinate) Macaulay matrix; ``max_nonzeros`` bounds the stored entries."""
    coefficients = _check(coefficients, shape)
    count = nonzeros(coefficients, shape)
    if count > max_nonzeros:
        raise MemoryError(f"Macaulay matrix has {count} nonzeros, above max_nonzeros={max_nonzeros}")
    table = column_table(shape)
    poly, flat = np.nonzero(coefficients)
    mults = np.arange(shape.multipliers.size)
    rows = (mults[:, None] * shape.polynomials + poly[None, :]).ravel()
    cols = table[:, flat].ravel()
    values = np.broadcast_to(coefficients[poly, flat], (mults.size, flat.size)).ravel()
    return SparseOperator(field, rows, cols, values, (shape.rows, shape.cols))


class MacaulayOperator(LinearOperator):
    """Implicit Macaulay matrix: stores only the coefficients and the column table.

    ``T x``: gather ``X[c, m] = x[table[m, c]]`` and multiply ``coefficients @ X``
    (a dense, BLAS-backed product); ``T^T y``: ``Y @ coefficients`` and scatter-add
    back through the table. Memory is ``O(K |base| + |mult| |base|)`` instead of
    ``O(nonzeros)``; each product costs ``O(nonzeros)`` arithmetic.
    """

    def __init__(
        self, field: Field, coefficients: Array, shape: MacaulayShape, device: str | None = None
    ) -> None:
        super().__init__(field, (shape.rows, shape.cols))
        self.coefficients = _check(coefficients, shape)
        self.macaulay_shape = shape
        self.table = column_table(shape)
        self._flat_table = self.table.ravel()
        self.device = device
        if device is not None:
            self._init_torch(device)

    # -- optional GPU path (prime fields): data stays on the device between products ---------
    def _init_torch(self, device: str) -> None:
        if not self.field.is_prime_field:
            raise NotImplementedError("the torch path supports prime fields only")
        import torch

        from ...arith.backends.torch_backend import TorchBackend

        self._torch = torch
        self._backend = TorchBackend(device)
        dev = self._backend.device
        self._t_coefficients = torch.as_tensor(self.coefficients, device=dev)
        self._t_table = torch.as_tensor(self.table, device=dev)
        self._t_flat = self._t_table.reshape(-1)

    def _apply_torch(self, x: Array) -> Array:
        torch, p = self._torch, self.field.p
        out = np.empty((self.rows, x.shape[1]), dtype=np.int64)
        for column in range(x.shape[1]):
            xt = torch.as_tensor(np.ascontiguousarray(x[:, column]), device=self._backend.device)
            product = self._backend.matmul_device(xt[self._t_table], self._t_coefficients.T, p)
            out[:, column] = product.reshape(-1).cpu().numpy()
        return out

    def _apply_T_torch(self, y: Array) -> Array:
        torch, p = self._torch, self.field.p
        k, mults = self.macaulay_shape.polynomials, self.macaulay_shape.multipliers.size
        out = np.empty((self.cols, y.shape[1]), dtype=np.int64)
        for column in range(y.shape[1]):
            yt = torch.as_tensor(np.ascontiguousarray(y[:, column]), device=self._backend.device).reshape(
                mults, k
            )
            contributions = self._backend.matmul_device(yt, self._t_coefficients, p).reshape(-1)
            total = torch.zeros(self.cols, dtype=torch.int64, device=self._backend.device)
            total.index_add_(0, self._t_flat, contributions)  # < |mult| * p < 2**63
            out[:, column] = torch.remainder(total, p).cpu().numpy()
        return out

    def _apply(self, x: Array) -> Array:
        if self.device is not None:
            return self._apply_torch(x)
        k, mults = self.macaulay_shape.polynomials, self.macaulay_shape.multipliers.size
        out = np.empty((mults * k, x.shape[1]), dtype=np.int64)
        for column in range(x.shape[1]):
            gathered = x[self.table, column]  # (|mult|, |base|)
            out[:, column] = self.field.matmul(gathered, self.coefficients.T).reshape(-1)
        return out

    def _apply_T(self, y: Array) -> Array:
        if self.device is not None:
            return self._apply_T_torch(y)
        k, mults = self.macaulay_shape.polynomials, self.macaulay_shape.multipliers.size
        out = np.empty((self.cols, y.shape[1]), dtype=np.int64)
        for column in range(y.shape[1]):
            contributions = self.field.matmul(y[:, column].reshape(mults, k), self.coefficients)
            out[:, column] = self.field.segment_sum(contributions.reshape(-1), self._flat_table, self.cols)
        return out

    @property
    def nonzeros(self) -> int:
        return nonzeros(self.coefficients, self.macaulay_shape)

    @property
    def cost(self) -> int:
        return 2 * self.nonzeros

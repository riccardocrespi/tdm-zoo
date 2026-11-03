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

Array = np.ndarray

DEFAULT_MAX_ENTRIES = 2 * 10**8


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


def macaulay_matrix(
    field: Field, coefficients: Array, shape: MacaulayShape, max_entries: int = DEFAULT_MAX_ENTRIES
) -> Array:
    """Dense Macaulay matrix ``(K * |mult|) x |columns|``."""
    coefficients = np.asarray(coefficients, dtype=np.int64)
    if coefficients.shape != (shape.polynomials, shape.base.size):
        raise ValueError(f"coefficients must have shape {(shape.polynomials, shape.base.size)}")
    if shape.entries > max_entries:
        raise MemoryError(
            f"Macaulay matrix {shape.rows} x {shape.cols} exceeds max_entries={max_entries}; "
            "raise the limit deliberately"
        )
    tables = shape.multipliers.times(shape.base)
    poly, flat = np.nonzero(coefficients)
    values = coefficients[poly, flat]
    base_digits = np.unravel_index(flat, shape.base.shape) if flat.size else [flat] * len(tables)
    strides = shape.columns.strides
    matrix = field.zeros((shape.rows, shape.cols))
    for mult_index, mult_digits in enumerate(np.ndindex(*shape.multipliers.shape)):
        cols = np.zeros(flat.size, dtype=np.int64)
        for b, table in enumerate(tables):
            cols += table[mult_digits[b], base_digits[b]] * strides[b]
        matrix[mult_index * shape.polynomials + poly, cols] = values
    return matrix

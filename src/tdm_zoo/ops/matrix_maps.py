"""Matrix-valued polynomial maps ``params -> matrix``.

These describe secret families symbolically: a family's *parametrisation*
``f`` (linear, image containing every sampled secret) and, when available, its
*inverse parametrisation* ``g`` (image containing every inverse). The generic
rank attack consumes exactly these objects.

Supported shapes of map:

* :class:`LinearMatrixMap` - ``sum_i w_i B_i`` stored as sparse coordinates;
* :class:`ConstantMatrixMap` - a fixed matrix (degree 0);
* :class:`ProductSumMatrixMap` - ``sum_t c_t F_{t,1}(w) F_{t,2}(w) ...`` (matrix
  products of maps), e.g. the degree-2 Gohberg-Semencul inverse formula;
* :class:`SandwichMatrixMap` - ``L f(w) R`` for fixed matrices ``L``, ``R``.

Every map can be evaluated, and expanded into explicit polynomial entries with
:meth:`MatrixMap.entry_polynomials` (used for verification and linearisation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..fields import Field
from .poly import SparsePolynomial

Array = np.ndarray


class MatrixMap(ABC):
    def __init__(self, field: Field, shape: tuple[int, int], nparams: int) -> None:
        self.field = field
        self.shape = (int(shape[0]), int(shape[1]))
        self.nparams = int(nparams)

    @property
    @abstractmethod
    def degree(self) -> int: ...

    @abstractmethod
    def evaluate(self, params) -> Array: ...

    @abstractmethod
    def entry_polynomials(self) -> list[list[SparsePolynomial]]:
        """``rows x cols`` nested list of polynomials in the parameters."""

    def _check(self, params) -> Array:
        params = np.asarray(params, dtype=np.int64)
        if params.shape != (self.nparams,):
            raise ValueError(f"expected {self.nparams} parameters, got shape {params.shape}")
        return params

    def describe(self) -> dict:
        return {
            "type": type(self).__name__,
            "shape": self.shape,
            "nparams": self.nparams,
            "degree": self.degree,
        }


class LinearMatrixMap(MatrixMap):
    """``f(w)[r, c] = sum over terms (i, r, c, a) of a * w_i``."""

    def __init__(self, field: Field, shape, nparams: int, param, row, col, coef) -> None:
        super().__init__(field, shape, nparams)
        self.param = np.asarray(param, dtype=np.int64)
        self.row = np.asarray(row, dtype=np.int64)
        self.col = np.asarray(col, dtype=np.int64)
        self.coef = field.asarray(np.broadcast_to(np.asarray(coef, dtype=np.int64), self.param.shape))
        if not (self.param.shape == self.row.shape == self.col.shape):
            raise ValueError("term arrays must have equal length")
        if self.param.size and (self.param.max() >= nparams or self.param.min() < 0):
            raise ValueError("parameter index out of range")

    @classmethod
    def from_tensor(cls, field: Field, tensor: Array) -> LinearMatrixMap:
        """Build from a dense coefficient tensor of shape ``(nparams, rows, cols)``."""
        tensor = field.asarray(tensor)
        param, row, col = np.nonzero(tensor)
        return cls(field, tensor.shape[1:], tensor.shape[0], param, row, col, tensor[param, row, col])

    @property
    def degree(self) -> int:
        return 1

    def evaluate(self, params) -> Array:
        params = self._check(params)
        rows, cols = self.shape
        values = self.field.mul(self.coef, params[self.param])
        flat = self.field.segment_sum(values, self.row * cols + self.col, rows * cols)
        return flat.reshape(rows, cols)

    def coefficient_tensor(self) -> Array:
        rows, cols = self.shape
        index = (self.param * rows + self.row) * cols + self.col
        flat = self.field.segment_sum(self.coef, index, self.nparams * rows * cols)
        return flat.reshape(self.nparams, rows, cols)

    def entry_polynomials(self) -> list[list[SparsePolynomial]]:
        rows, cols = self.shape
        entries = [[SparsePolynomial(self.field, self.nparams) for _ in range(cols)] for _ in range(rows)]
        for i, r, c, a in zip(self.param, self.row, self.col, self.coef, strict=True):
            entries[r][c] = entries[r][c] + SparsePolynomial(self.field, self.nparams, {(int(i),): int(a)})
        return entries


class ConstantMatrixMap(MatrixMap):
    def __init__(self, field: Field, matrix: Array, nparams: int) -> None:
        matrix = field.asarray(matrix)
        super().__init__(field, matrix.shape, nparams)
        self.matrix = matrix

    @property
    def degree(self) -> int:
        return 0

    def evaluate(self, params) -> Array:
        self._check(params)
        return self.matrix.copy()

    def entry_polynomials(self) -> list[list[SparsePolynomial]]:
        return [
            [SparsePolynomial.constant(self.field, self.nparams, int(v)) for v in row] for row in self.matrix
        ]


class ProductSumMatrixMap(MatrixMap):
    """``sum_t coef_t * F_{t,1}(w) @ F_{t,2}(w) @ ...`` for maps sharing the parameters."""

    def __init__(self, field: Field, terms: list[tuple[int, list[MatrixMap]]]) -> None:
        if not terms:
            raise ValueError("at least one term is required")
        shapes, nparams = set(), {f.nparams for _, factors in terms for f in factors}
        for _, factors in terms:
            for left, right in zip(factors, factors[1:], strict=False):
                if left.shape[1] != right.shape[0]:
                    raise ValueError("incompatible factor shapes")
            shapes.add((factors[0].shape[0], factors[-1].shape[1]))
        if len(shapes) != 1 or len(nparams) != 1:
            raise ValueError("all terms must share shape and parameter count")
        super().__init__(field, shapes.pop(), nparams.pop())
        self.terms = [(int(field.from_int(c)), list(factors)) for c, factors in terms]

    @property
    def degree(self) -> int:
        return max(sum(f.degree for f in factors) for _, factors in self.terms)

    def evaluate(self, params) -> Array:
        params = self._check(params)
        total = self.field.zeros(self.shape)
        for coefficient, factors in self.terms:
            value = factors[0].evaluate(params)
            for factor in factors[1:]:
                value = self.field.matmul(value, factor.evaluate(params))
            total = self.field.add(total, self.field.scale(coefficient, value))
        return total

    def entry_polynomials(self) -> list[list[SparsePolynomial]]:
        rows, cols = self.shape
        total = [[SparsePolynomial(self.field, self.nparams) for _ in range(cols)] for _ in range(rows)]
        for coefficient, factors in self.terms:
            value = factors[0].entry_polynomials()
            for factor in factors[1:]:
                value = _poly_matmul(value, factor.entry_polynomials(), self.field, self.nparams)
            for r in range(rows):
                for c in range(cols):
                    total[r][c] = total[r][c] + value[r][c].scale(coefficient)
        return total


class SandwichMatrixMap(MatrixMap):
    """``left @ inner(w) @ right`` for fixed matrices ``left`` and ``right``."""

    def __init__(self, field: Field, left: Array | None, inner: MatrixMap, right: Array | None) -> None:
        rows = inner.shape[0] if left is None else left.shape[0]
        cols = inner.shape[1] if right is None else right.shape[1]
        super().__init__(field, (rows, cols), inner.nparams)
        self.left, self.inner, self.right = left, inner, right

    @property
    def degree(self) -> int:
        return self.inner.degree

    def _wrap(self, matrix: Array) -> Array:
        if self.left is not None:
            matrix = self.field.matmul(self.left, matrix)
        if self.right is not None:
            matrix = self.field.matmul(matrix, self.right)
        return matrix

    def evaluate(self, params) -> Array:
        return self._wrap(self.inner.evaluate(self._check(params)))

    def coefficient_tensor(self) -> Array:
        if self.inner.degree != 1:
            raise ValueError("coefficient tensors exist only for linear maps")
        inner = self.inner.coefficient_tensor()
        return np.stack([self._wrap(slab) for slab in inner])

    def entry_polynomials(self) -> list[list[SparsePolynomial]]:
        if self.inner.degree == 1:
            return LinearMatrixMap.from_tensor(self.field, self.coefficient_tensor()).entry_polynomials()
        inner = self.inner.entry_polynomials()
        n = self.nparams
        if self.left is not None:
            inner = _poly_matmul(
                ConstantMatrixMap(self.field, self.left, n).entry_polynomials(), inner, self.field, n
            )
        if self.right is not None:
            inner = _poly_matmul(
                inner, ConstantMatrixMap(self.field, self.right, n).entry_polynomials(), self.field, n
            )
        return inner


def _poly_matmul(left, right, field: Field, nparams: int) -> list[list[SparsePolynomial]]:
    rows, inner, cols = len(left), len(right), len(right[0])
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            acc = SparsePolynomial(field, nparams)
            for t in range(inner):
                if left[r][t].terms and right[t][c].terms:
                    acc = acc + left[r][t] * right[t][c]
            row.append(acc)
        out.append(row)
    return out

"""Linear operators over a field: the "fast evaluation circuit" abstraction.

An operator represents a linear map ``F^cols -> F^rows`` without necessarily
materialising its matrix. Vectors are columns: ``apply`` accepts shape
``(cols,)`` or a batch ``(cols, b)``. ``A @ B`` composes operators (apply ``B``
first). Every operator also supports its transpose, which by the transposition
principle costs about as much as the operator itself.

``cost`` is a nominal count of field operations for one matrix-vector product.
It is meant for comparing constructions (e.g. "c * k" vs "k**2"), not timing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..fields import Field
from . import linalg

Array = np.ndarray


class LinearOperator(ABC):
    def __init__(self, field: Field, shape: tuple[int, int]) -> None:
        self.field = field
        self.shape = (int(shape[0]), int(shape[1]))

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    # -- application ----------------------------------------------------------------------
    def apply(self, x) -> Array:
        return self._dispatch(x, self.cols, self._apply)

    def apply_T(self, y) -> Array:
        return self._dispatch(y, self.rows, self._apply_T)

    @staticmethod
    def _dispatch(x, expected: int, function) -> Array:
        x = np.asarray(x, dtype=np.int64)
        vector = x.ndim == 1
        batch = x[:, None] if vector else x
        if batch.ndim != 2 or batch.shape[0] != expected:
            raise ValueError(f"expected leading dimension {expected}, got shape {x.shape}")
        out = function(batch)
        return out[:, 0] if vector else out

    @abstractmethod
    def _apply(self, x: Array) -> Array: ...

    @abstractmethod
    def _apply_T(self, y: Array) -> Array: ...

    # -- derived views --------------------------------------------------------------------
    def to_dense(self) -> Array:
        return self._apply(self.field.eye(self.cols))

    @property
    def T(self) -> LinearOperator:
        return TransposedOperator(self)

    def inverse(self) -> LinearOperator:
        """Operator for the inverse map; raises ``ValueError`` if unavailable."""
        if self.rows != self.cols:
            raise ValueError("only square operators can be inverted")
        return DenseOperator(self.field, linalg.inv(self.field, self.to_dense()))

    def __matmul__(self, other):
        if isinstance(other, LinearOperator):
            return ComposedOperator([self, other])
        return self.apply(other)

    # -- bookkeeping ----------------------------------------------------------------------
    @property
    def cost(self) -> int:
        return 2 * self.rows * self.cols

    @property
    def cost_T(self) -> int:
        return self.cost

    def describe(self) -> dict:
        return {"type": type(self).__name__, "shape": self.shape, "cost": self.cost}

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self.shape}"


class DenseOperator(LinearOperator):
    def __init__(self, field: Field, matrix: Array) -> None:
        matrix = field.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        super().__init__(field, matrix.shape)
        self.matrix = matrix

    def _apply(self, x: Array) -> Array:
        return self.field.matmul(self.matrix, x)

    def _apply_T(self, y: Array) -> Array:
        return self.field.matmul(self.matrix.T, y)

    def to_dense(self) -> Array:
        return self.matrix.copy()


class IdentityOperator(LinearOperator):
    def __init__(self, field: Field, n: int) -> None:
        super().__init__(field, (n, n))

    def _apply(self, x: Array) -> Array:
        return x.copy()

    def _apply_T(self, y: Array) -> Array:
        return y.copy()

    def inverse(self) -> LinearOperator:
        return self

    @property
    def cost(self) -> int:
        return 0


class TransposedOperator(LinearOperator):
    def __init__(self, base: LinearOperator) -> None:
        super().__init__(base.field, base.shape[::-1])
        self.base = base

    def _apply(self, x: Array) -> Array:
        return self.base._apply_T(x)

    def _apply_T(self, y: Array) -> Array:
        return self.base._apply(y)

    @property
    def T(self) -> LinearOperator:
        return self.base

    def inverse(self) -> LinearOperator:
        return self.base.inverse().T

    @property
    def cost(self) -> int:
        return self.base.cost_T

    @property
    def cost_T(self) -> int:
        return self.base.cost


class ComposedOperator(LinearOperator):
    """``factors[0] @ factors[1] @ ... @ factors[-1]`` (the last factor acts first)."""

    def __init__(self, factors: list[LinearOperator]) -> None:
        flat: list[LinearOperator] = []
        for factor in factors:
            flat.extend(factor.factors if isinstance(factor, ComposedOperator) else [factor])
        if not flat:
            raise ValueError("at least one factor is required")
        for left, right in zip(flat, flat[1:], strict=False):
            if left.cols != right.rows:
                raise ValueError(f"cannot compose {left!r} with {right!r}")
        super().__init__(flat[0].field, (flat[0].rows, flat[-1].cols))
        self.factors = flat

    def _apply(self, x: Array) -> Array:
        for factor in reversed(self.factors):
            x = factor._apply(x)
        return x

    def _apply_T(self, y: Array) -> Array:
        for factor in self.factors:
            y = factor._apply_T(y)
        return y

    def inverse(self) -> LinearOperator:
        return ComposedOperator([factor.inverse() for factor in reversed(self.factors)])

    @property
    def cost(self) -> int:
        return sum(factor.cost for factor in self.factors)

    @property
    def cost_T(self) -> int:
        return sum(factor.cost_T for factor in self.factors)

    def describe(self) -> dict:
        return {**super().describe(), "factors": [factor.describe() for factor in self.factors]}


class SelectionOperator(LinearOperator):
    """``y = x[indices]``: projections, repetitions, truncations, redistributions.

    The transpose scatters and adds, e.g. the transpose of a repetition sums blocks.
    """

    def __init__(self, field: Field, indices, in_dim: int) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or (indices.size and (indices.min() < 0 or indices.max() >= in_dim)):
            raise ValueError("indices must be a vector of positions below in_dim")
        super().__init__(field, (indices.size, in_dim))
        self.indices = indices

    def _apply(self, x: Array) -> Array:
        return x[self.indices]

    def _apply_T(self, y: Array) -> Array:
        return self.field.segment_sum(y, self.indices, self.cols)

    @property
    def cost(self) -> int:
        return 0

    @property
    def cost_T(self) -> int:
        return int(self.indices.size)


class PermutationOperator(SelectionOperator):
    """Permutation matrix ``P`` with ``P e_a = e_{perm[a]}``, i.e. ``(P x)[perm[a]] = x[a]``."""

    def __init__(self, field: Field, perm) -> None:
        perm = np.asarray(perm, dtype=np.int64)
        if not np.array_equal(np.sort(perm), np.arange(perm.size)):
            raise ValueError("not a permutation")
        super().__init__(field, np.argsort(perm), perm.size)
        self.perm = perm

    def _apply_T(self, y: Array) -> Array:
        return y[self.perm]

    def inverse(self) -> LinearOperator:
        return PermutationOperator(self.field, np.argsort(self.perm))

    @property
    def cost_T(self) -> int:
        return 0


class SparseOperator(LinearOperator):
    """Coordinate-format sparse matrix."""

    def __init__(self, field: Field, rows, cols, values, shape: tuple[int, int]) -> None:
        super().__init__(field, shape)
        self.row_index = np.asarray(rows, dtype=np.int64)
        self.col_index = np.asarray(cols, dtype=np.int64)
        self.values = field.asarray(values)
        if not (self.row_index.shape == self.col_index.shape == self.values.shape):
            raise ValueError("rows, cols and values must have equal length")

    def _apply(self, x: Array) -> Array:
        products = self.field.mul(self.values[:, None], x[self.col_index])
        return self.field.segment_sum(products, self.row_index, self.rows)

    def _apply_T(self, y: Array) -> Array:
        products = self.field.mul(self.values[:, None], y[self.row_index])
        return self.field.segment_sum(products, self.col_index, self.cols)

    @property
    def cost(self) -> int:
        return 2 * int(self.values.size)


class DiagonalOperator(LinearOperator):
    def __init__(self, field: Field, diagonal) -> None:
        diagonal = field.asarray(diagonal)
        super().__init__(field, (diagonal.size, diagonal.size))
        self.diagonal = diagonal

    def _apply(self, x: Array) -> Array:
        return self.field.mul(self.diagonal[:, None], x)

    _apply_T = _apply

    def inverse(self) -> LinearOperator:
        if np.any(self.diagonal == 0):
            raise ValueError("diagonal has zero entries")
        return DiagonalOperator(self.field, self.field.inv(self.diagonal))

    @property
    def cost(self) -> int:
        return self.rows


class HStackOperator(LinearOperator):
    """``[A_1 | A_2 | ...]`` (blocks side by side)."""

    def __init__(self, blocks: list[LinearOperator]) -> None:
        rows = {block.rows for block in blocks}
        if len(rows) != 1:
            raise ValueError("blocks must have equal row counts")
        super().__init__(blocks[0].field, (rows.pop(), sum(block.cols for block in blocks)))
        self.blocks = blocks
        self._offsets = np.cumsum([0] + [block.cols for block in blocks])

    def _apply(self, x: Array) -> Array:
        out = None
        for block, start, stop in zip(self.blocks, self._offsets, self._offsets[1:], strict=False):
            part = block._apply(x[start:stop])
            out = part if out is None else self.field.add(out, part)
        return out

    def _apply_T(self, y: Array) -> Array:
        return np.concatenate([block._apply_T(y) for block in self.blocks], axis=0)

    @property
    def cost(self) -> int:
        return sum(block.cost for block in self.blocks) + self.rows * (len(self.blocks) - 1)


class VStackOperator(LinearOperator):
    """``[A_1 // A_2 // ...]`` (blocks stacked vertically)."""

    def __init__(self, blocks: list[LinearOperator]) -> None:
        cols = {block.cols for block in blocks}
        if len(cols) != 1:
            raise ValueError("blocks must have equal column counts")
        super().__init__(blocks[0].field, (sum(block.rows for block in blocks), cols.pop()))
        self.blocks = blocks

    def _apply(self, x: Array) -> Array:
        return np.concatenate([block._apply(x) for block in self.blocks], axis=0)

    def _apply_T(self, y: Array) -> Array:
        return HStackOperator([block.T for block in self.blocks])._apply(y)

    @property
    def cost(self) -> int:
        return sum(block.cost for block in self.blocks)

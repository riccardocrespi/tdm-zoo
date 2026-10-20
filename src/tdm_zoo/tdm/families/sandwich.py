"""Secrets of the form ``L S R`` with fixed public ``L``, ``R`` and a secret ``S``."""

from __future__ import annotations

import numpy as np

from ...ops.matrix_maps import MatrixMap, SandwichMatrixMap
from ...ops.operators import ComposedOperator, LinearOperator
from .base import SecretFamily


class SandwichFamily(SecretFamily):
    """``left @ inner @ right``; e.g. the DI14 matrix ``Ext @ H @ Enc``.

    Either side may be ``None`` (identity). The parametrisation stays linear when
    the inner one is. Not registered by name since it needs operator arguments.
    """

    name = "sandwich"

    def __init__(
        self,
        inner: SecretFamily,
        left: LinearOperator | None = None,
        right: LinearOperator | None = None,
        label: str | None = None,
    ) -> None:
        rows = inner.shape[0] if left is None else left.rows
        cols = inner.shape[1] if right is None else right.cols
        if left is not None and left.cols != inner.shape[0]:
            raise ValueError("left factor does not match the inner family")
        if right is not None and right.rows != inner.shape[1]:
            raise ValueError("right factor does not match the inner family")
        super().__init__(inner.field, rows, cols)
        self.inner, self.left, self.right = inner, left, right
        self.label = label

    @property
    def nparams(self) -> int:
        return self.inner.nparams

    def options(self) -> dict:
        return {
            "label": self.label,
            "inner": self.inner.describe(),
            "left": None if self.left is None else self.left.shape,
            "right": None if self.right is None else self.right.shape,
        }

    def sample_params(self, rng):
        return self.inner.sample_params(rng)

    def operator(self, params) -> LinearOperator:
        factors = [f for f in (self.left, self.inner.operator(params), self.right) if f is not None]
        return factors[0] if len(factors) == 1 else ComposedOperator(factors)

    def _param_map(self) -> MatrixMap:
        dense = lambda op: None if op is None else op.to_dense()  # noqa: E731
        return SandwichMatrixMap(self.field, dense(self.left), self.inner.param_map, dense(self.right))

    def matrix(self, params) -> np.ndarray:
        return self.param_map.evaluate(params)

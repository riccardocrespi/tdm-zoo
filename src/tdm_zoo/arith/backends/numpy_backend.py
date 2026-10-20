"""Pure numpy backend: vectorised Gauss-Jordan elimination modulo p."""

from __future__ import annotations

import numpy as np

from .. import modp
from .base import PrimeBackend


class NumpyBackend(PrimeBackend):
    name = "numpy"

    def matmul(self, left: np.ndarray, right: np.ndarray, p: int) -> np.ndarray:
        return modp.matmul(left, right, p)

    def rref(self, matrix: np.ndarray, p: int) -> tuple[np.ndarray, tuple[int, ...]]:
        reduced = np.mod(np.array(matrix, dtype=np.int64), p)
        rows, columns = reduced.shape
        pivots: list[int] = []
        row = 0
        for column in range(columns):
            if row == rows:
                break
            candidates = np.flatnonzero(reduced[row:, column])
            if candidates.size == 0:
                continue
            pivot = row + int(candidates[0])
            if pivot != row:
                reduced[[row, pivot]] = reduced[[pivot, row]]
            scale = pow(int(reduced[row, column]), -1, p)
            reduced[row, column:] = (reduced[row, column:] * scale) % p
            factors = reduced[:, column].copy()
            factors[row] = 0
            targets = np.flatnonzero(factors)
            if targets.size:
                reduced[targets, column:] = (
                    reduced[targets, column:] - factors[targets, None] * reduced[row, column:]
                ) % p
            pivots.append(column)
            row += 1
        return reduced, tuple(pivots)

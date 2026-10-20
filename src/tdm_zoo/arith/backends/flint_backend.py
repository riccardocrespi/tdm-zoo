"""FLINT backend (python-flint ``nmod_mat``): exact, C-speed linear algebra modulo p."""

from __future__ import annotations

import flint
import numpy as np

from .base import PrimeBackend


def to_flint(matrix: np.ndarray, p: int) -> flint.nmod_mat:
    rows, columns = matrix.shape
    return flint.nmod_mat(rows, columns, np.mod(matrix, p).ravel().tolist(), p)


def from_flint(matrix: flint.nmod_mat) -> np.ndarray:
    values = np.fromiter((int(x) for x in matrix.entries()), dtype=np.int64)
    return values.reshape(matrix.nrows(), matrix.ncols())


class FlintBackend(PrimeBackend):
    name = "flint"

    def matmul(self, left: np.ndarray, right: np.ndarray, p: int) -> np.ndarray:
        if left.shape[1] != right.shape[0]:
            raise ValueError(f"incompatible shapes {left.shape} and {right.shape}")
        if 0 in left.shape or 0 in right.shape:
            return np.zeros((left.shape[0], right.shape[1]), dtype=np.int64)
        return from_flint(to_flint(left, p) * to_flint(right, p))

    def rref(self, matrix: np.ndarray, p: int) -> tuple[np.ndarray, tuple[int, ...]]:
        if 0 in matrix.shape:
            return np.zeros(matrix.shape, dtype=np.int64), ()
        reduced, rank = to_flint(matrix, p).rref()
        dense = from_flint(reduced)
        pivots = tuple(int(np.flatnonzero(dense[row])[0]) for row in range(rank))
        return dense, pivots

    def rank(self, matrix: np.ndarray, p: int) -> int:
        if 0 in matrix.shape:
            return 0
        return int(to_flint(matrix, p).rank())

    def inv(self, matrix: np.ndarray, p: int) -> np.ndarray:
        try:
            return from_flint(to_flint(matrix, p).inv())
        except ZeroDivisionError as error:
            raise np.linalg.LinAlgError("matrix is singular over the field") from error

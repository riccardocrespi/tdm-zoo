"""Interface for dense linear algebra modulo a prime.

Backends exchange canonical int64 numpy arrays with the caller regardless of the
device they compute on, so higher layers never see backend-specific types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PrimeBackend(ABC):
    name: str

    @abstractmethod
    def matmul(self, left: np.ndarray, right: np.ndarray, p: int) -> np.ndarray: ...

    @abstractmethod
    def rref(self, matrix: np.ndarray, p: int) -> tuple[np.ndarray, tuple[int, ...]]:
        """Reduced row echelon form and pivot columns."""

    def rank(self, matrix: np.ndarray, p: int) -> int:
        return len(self.rref(matrix, p)[1])

    def nullspace(self, matrix: np.ndarray, p: int) -> np.ndarray:
        """Basis of the right kernel, as the columns of a ``cols x nullity`` matrix."""
        reduced, pivots = self.rref(matrix, p)
        return nullspace_from_rref(reduced, pivots, p)

    def inv(self, matrix: np.ndarray, p: int) -> np.ndarray:
        n = _square_size(matrix)
        reduced, pivots = self.rref(np.hstack([matrix, np.eye(n, dtype=np.int64)]), p)
        if pivots[:n] != tuple(range(n)):
            raise np.linalg.LinAlgError("matrix is singular over the field")
        return reduced[:, n:]

    def solve(self, matrix: np.ndarray, rhs: np.ndarray, p: int) -> np.ndarray:
        """Solve ``matrix @ X = rhs`` for square invertible ``matrix``."""
        n = _square_size(matrix)
        vector = rhs.ndim == 1
        rhs2 = rhs.reshape(n, -1)
        reduced, pivots = self.rref(np.hstack([matrix, rhs2]), p)
        if pivots[:n] != tuple(range(n)):
            raise np.linalg.LinAlgError("matrix is singular over the field")
        solution = reduced[:, n:]
        return solution[:, 0] if vector else solution


def nullspace_from_rref(reduced: np.ndarray, pivots: tuple[int, ...], p: int) -> np.ndarray:
    columns = reduced.shape[1]
    pivot_set = set(pivots)
    free = [c for c in range(columns) if c not in pivot_set]
    basis = np.zeros((columns, len(free)), dtype=np.int64)
    for index, column in enumerate(free):
        basis[column, index] = 1
        for row, pivot in enumerate(pivots):
            basis[pivot, index] = (-reduced[row, column]) % p
    return basis


def _square_size(matrix: np.ndarray) -> int:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return matrix.shape[0]

"""FLINT backend (python-flint ``nmod_mat``) for elimination; products use exact numpy BLAS."""

from __future__ import annotations

import flint
import numpy as np

from .. import modp
from .base import PrimeBackend


def to_flint(matrix: np.ndarray, p: int) -> flint.nmod_mat:
    rows, columns = matrix.shape
    return flint.nmod_mat(rows, columns, np.mod(matrix, p).ravel().tolist(), p)


def rows_to_flint(rows, shape: tuple[int, int], p: int) -> flint.nmod_mat:
    """``nmod_mat`` filled row by row from an iterable: no Python list of all the entries.

    Slower per entry than :func:`to_flint`, but memory stays at the matrix itself, which is what
    makes a ``16k x 16k`` elimination fit.
    """
    out = flint.nmod_mat(shape[0], shape[1], p)
    for i, row in enumerate(rows):
        for j, value in enumerate(np.mod(row, p).tolist()):
            if value:
                out[i, j] = value
    return out


def from_flint(matrix: flint.nmod_mat) -> np.ndarray:
    values = np.fromiter((int(x) for x in matrix.entries()), dtype=np.int64)
    return values.reshape(matrix.nrows(), matrix.ncols())


class FlintBackend(PrimeBackend):
    name = "flint"

    def matmul(self, left: np.ndarray, right: np.ndarray, p: int) -> np.ndarray:
        # Exact float64-BLAS products beat FLINT once numpy<->FLINT conversion is counted.
        return modp.matmul(left, right, p)

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

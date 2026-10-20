"""Dense linear algebra over any :class:`~tdm_zoo.fields.Field`.

Prime fields dispatch to the field's backend (numpy / FLINT / torch). Extension
fields use the generic vectorised Gauss-Jordan elimination below, which only
calls field operations.
"""

from __future__ import annotations

import numpy as np

from ..fields import Field

Array = np.ndarray


def matmul(field: Field, left: Array, right: Array) -> Array:
    return field.matmul(left, right)


def rref(field: Field, matrix: Array) -> tuple[Array, tuple[int, ...]]:
    """Reduced row echelon form and the tuple of pivot columns."""
    matrix = np.asarray(matrix, dtype=np.int64)
    if field.is_prime_field:
        return field.backend.rref(matrix, field.p)
    return _generic_rref(field, matrix)


def rank(field: Field, matrix: Array) -> int:
    matrix = np.asarray(matrix, dtype=np.int64)
    if 0 in matrix.shape:
        return 0
    if field.is_prime_field:
        return field.backend.rank(matrix, field.p)
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix.T
    return len(_generic_rref(field, matrix)[1])


def nullspace(field: Field, matrix: Array) -> Array:
    """Right kernel basis as the columns of a ``cols x nullity`` matrix."""
    reduced, pivots = rref(field, matrix)
    columns = reduced.shape[1]
    pivot_set = set(pivots)
    free = [c for c in range(columns) if c not in pivot_set]
    basis = np.zeros((columns, len(free)), dtype=np.int64)
    for index, column in enumerate(free):
        basis[column, index] = 1
        if pivots:
            basis[list(pivots), index] = field.neg(reduced[: len(pivots), column])
    return basis


def left_nullspace(field: Field, matrix: Array) -> Array:
    """Left kernel basis as the rows of a ``nullity x rows`` matrix."""
    return nullspace(field, np.asarray(matrix).T).T


def inv(field: Field, matrix: Array) -> Array:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if field.is_prime_field:
        return field.backend.inv(matrix, field.p)
    n = matrix.shape[0]
    reduced, pivots = _generic_rref(field, np.hstack([matrix, field.eye(n)]))
    if pivots[:n] != tuple(range(n)):
        raise np.linalg.LinAlgError("matrix is singular over the field")
    return reduced[:, n:]


def is_invertible(field: Field, matrix: Array) -> bool:
    matrix = np.asarray(matrix)
    return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and rank(field, matrix) == matrix.shape[0]


def solve(field: Field, matrix: Array, rhs: Array) -> Array:
    """Solve ``matrix @ X = rhs`` for square invertible ``matrix``."""
    matrix, rhs = np.asarray(matrix, dtype=np.int64), np.asarray(rhs, dtype=np.int64)
    if field.is_prime_field:
        return field.backend.solve(matrix, rhs, field.p)
    n = matrix.shape[0]
    vector = rhs.ndim == 1
    reduced, pivots = _generic_rref(field, np.hstack([matrix, rhs.reshape(n, -1)]))
    if pivots[:n] != tuple(range(n)):
        raise np.linalg.LinAlgError("matrix is singular over the field")
    solution = reduced[:, n:]
    return solution[:, 0] if vector else solution


def generalized_inverse(field: Field, matrix: Array) -> Array:
    """A matrix ``G`` with ``A @ G @ A == A`` (a {1}-inverse), for any shape and rank.

    Built from a nonsingular ``r x r`` submatrix ``W = A[rows][:, cols]``:
    ``G[cols][:, rows] = W^{-1}`` and zero elsewhere.
    """
    matrix = np.asarray(matrix, dtype=np.int64)
    _, column_pivots = rref(field, matrix)
    _, row_pivots = rref(field, matrix.T)
    rows_, cols_ = list(row_pivots), list(column_pivots)
    result = np.zeros(matrix.shape[::-1], dtype=np.int64)
    if cols_:
        result[np.ix_(cols_, rows_)] = inv(field, matrix[np.ix_(rows_, cols_)])
    return result


def _generic_rref(field: Field, matrix: Array) -> tuple[Array, tuple[int, ...]]:
    reduced = field.asarray(matrix).copy()
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
        scale = field.inv(reduced[row, column])
        reduced[row, column:] = field.mul(reduced[row, column:], scale)
        factors = reduced[:, column].copy()
        factors[row] = 0
        targets = np.flatnonzero(factors)
        if targets.size:
            reduced[targets, column:] = field.sub(
                reduced[targets, column:],
                field.mul(factors[targets, None], reduced[row, column:][None, :]),
            )
        pivots.append(column)
        row += 1
    return reduced, tuple(pivots)

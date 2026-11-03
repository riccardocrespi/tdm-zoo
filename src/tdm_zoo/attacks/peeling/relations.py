"""Base relations: degree-d polynomials annihilating the two-layer core map.

For public ``A_1, A_2, A_3`` and the parametrisations ``f_1, f_2`` of the first two
secret layers,

    F(X_1, X_2) = A_1 f_1(X_1) A_2 f_2(X_2) A_3      (n x n, i.e. n^2 entries),

we compute a basis ``Q_1, ..., Q_K`` of the degree-``d`` homogeneous polynomials in
the ``n^2`` entries with ``Q_i(F(X_1, X_2)) == 0`` identically.

Method: ``Q o F`` has bidegree ``(d, d)``, so it lives in a space of dimension
``N(v_1, d) N(v_2, d)``. Evaluating all degree-``d`` monomials of ``F`` at more
random points than that dimension, the kernel of the evaluation matrix is exactly
the space of relations (with overwhelming probability over a large field). The
result is then verified on fresh random points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...fields import Field
from ...ops import linalg
from ...ops.matrix_maps import MatrixMap
from ...ops.multihomogeneous import BlockBasis

Array = np.ndarray


@dataclass(frozen=True)
class BaseRelations:
    coefficients: Array  # (K, N(n^2, d)) in the order of ``basis``
    basis: BlockBasis  # degree-d monomials in the n^2 entries of an n x n matrix (row-major vec)

    @property
    def count(self) -> int:
        return self.coefficients.shape[0]

    @property
    def degree(self) -> int:
        return self.basis.degree

    def evaluate(self, field: Field, matrices: Array) -> Array:
        """``Q_i`` at a stack of ``n x n`` matrices: ``(P, n, n) -> (P, K)``."""
        flat = np.asarray(matrices, dtype=np.int64).reshape(len(matrices), -1)
        return field.matmul(self.basis.evaluate(field, flat), self.coefficients.T)


def core_map_values(
    field: Field, publics: list[Array], f1: MatrixMap, f2: MatrixMap, rng: np.random.Generator, count: int
) -> Array:
    """``F(X_1, X_2)`` at ``count`` random parameter pairs: shape ``(count, n, n)``."""
    a1, a2, a3 = publics
    s1 = np.stack([f1.evaluate(field.random(rng, f1.nparams)) for _ in range(count)])
    s2 = np.stack([f2.evaluate(field.random(rng, f2.nparams)) for _ in range(count)])
    left = linalg.batched_matmul(field, linalg.batched_matmul(field, a1[None], s1), a2[None])
    return linalg.batched_matmul(field, linalg.batched_matmul(field, left, s2), a3[None])


def base_relations(
    field: Field,
    publics: list[Array],
    f1: MatrixMap,
    f2: MatrixMap,
    degree: int,
    rng: np.random.Generator,
    margin: int = 32,
    verify_points: int = 32,
) -> BaseRelations:
    n = publics[0].shape[0]
    basis = BlockBasis(n * n, degree)
    bound = BlockBasis(f1.nparams, degree).size * BlockBasis(f2.nparams, degree).size
    points = min(bound, basis.size) + margin
    values = core_map_values(field, publics, f1, f2, rng, points)
    evaluation = basis.evaluate(field, values.reshape(points, -1))
    coefficients = linalg.nullspace(field, evaluation).T
    relations = BaseRelations(coefficients, basis)
    if relations.count:
        check = core_map_values(field, publics, f1, f2, rng, verify_points)
        if np.any(relations.evaluate(field, check)):
            raise RuntimeError("base relations failed verification; increase the margin or the field size")
    return relations

"""Black-box (matrix-free) linear algebra: Wiedemann rank over a prime field.

Only matrix-vector products with ``A`` and ``A^T`` are used, so sparse or implicit
matrices never fill in. Method (Chen, Eberly, Kaltofen, Saunders, Turner, Villard,
"Efficient matrix preconditioners for black box linear algebra", 2002): for
``A`` of rank ``r`` and random nonsingular diagonals ``D1, D2``, the symmetric-type
matrix ``B = D1 A^T D2 A D1`` has rank ``r`` and, with high probability over a
large field, minimal polynomial of degree ``r + 1`` if ``r < N`` (constant term 0)
and ``r`` otherwise. The minimal polynomial is recovered from the scalar sequence
``u^T B^i v`` by Berlekamp-Massey (FLINT).

The result is a lower bound that is exact with high probability; failures (an
unlucky projection or preconditioner) can only *lower* the computed rank, so
taking the maximum over independent repeats is safe.
"""

from __future__ import annotations

import flint
import numpy as np

from ..fields import Field
from .operators import LinearOperator


def wiedemann_rank(
    field: Field,
    operator: LinearOperator,
    rng: np.random.Generator,
    repeats: int = 1,
    early_termination: bool = True,
    check_every: int = 512,
    margin: int = 32,
) -> int:
    if not field.is_prime_field:
        raise NotImplementedError("Wiedemann rank is implemented over prime fields")
    return max(
        _rank_once(field, operator, rng, early_termination, check_every, margin) for _ in range(repeats)
    )


def _rank_once(field, operator, rng, early_termination, check_every, margin) -> int:
    p = field.p
    rows, cols = operator.shape
    # Work on the smaller side: B is N x N with N = min(rows, cols).
    transpose = rows < cols
    n_inner, n_outer = (rows, cols) if transpose else (cols, rows)
    apply_a = operator._apply_T if transpose else operator._apply
    apply_at = operator._apply if transpose else operator._apply_T
    d1 = field.random_nonzero(rng, n_inner)
    d2 = field.random_nonzero(rng, n_outer)
    u = field.random(rng, n_inner)
    w = field.random(rng, n_inner)[:, None]

    def apply_b(x):
        x = field.mul(d1[:, None], x)
        y = field.mul(d2[:, None], apply_a(x))
        return field.mul(d1[:, None], apply_at(y))

    ctx = flint.fmpz_mod_poly_ctx(p)
    limit = 2 * n_inner + 2
    sequence: list[int] = []
    previous_degree = -1
    for index in range(limit):
        sequence.append(int(np.sum((u * w[:, 0]) % p) % p))
        if index + 1 < limit:
            w = apply_b(w)
        if early_termination and (index + 1) % check_every == 0:
            degree = ctx.minpoly(sequence).degree()
            if degree == previous_degree and 2 * degree + margin <= len(sequence):
                break
            previous_degree = degree
    minpoly = ctx.minpoly(sequence)
    degree = minpoly.degree()
    constant = int(minpoly.coeffs()[0]) if degree > 0 else 1
    return degree - 1 if constant == 0 else degree

"""RAA-style candidate [EMVP Eq. (2)], generalised to ``l`` accumulation layers.

With ``n = c k``, repetition ``D`` (``n x k``) and weighted accumulators
``S(w) = C_n diag(w)``::

    R = D^T Pi_{l+1} S_l Pi_l ... Pi_2 S_1 Pi_1 D

Written left to right as a layered product (``A_1`` leftmost)::

    A_1 = D^T Pi_{l+1},  A_j = Pi_{l+2-j} (2 <= j <= l),  A_{l+1} = Pi_1 D,
    secrets S_l, ..., S_1.

``view="accumulator"`` uses weighted accumulators as the secret layers.
``view="diagonal"`` absorbs the fixed ``C_n`` into the public matrix on its left
(``A_j -> A_j C_n`` for ``j <= l``), so the secrets are diagonal, as in the paper's
application of the generic attack. Both views give the same distribution of ``R``.

``outer_permutations=False`` drops ``Pi_1`` and ``Pi_{l+1}``.
``secret_permutations=True`` hides every public layer from the distinguisher.
"""

from __future__ import annotations

from ...fields import Field
from ...ops.operators import ComposedOperator, LinearOperator
from ..families import DiagonalFamily, WeightedAccumulatorFamily
from ..product import LayeredProduct
from ..publics import block_sum, identity, plain_accumulator, random_permutation, repetition
from .base import CANDIDATES, Candidate

VIEWS = ("accumulator", "diagonal")


@CANDIDATES.register("raa")
class RAA(Candidate):
    def __init__(
        self,
        field: Field,
        k: int,
        ell: int = 3,
        c: int = 3,
        view: str = "accumulator",
        nonzero_weights: bool = True,
        outer_permutations: bool = True,
        secret_permutations: bool = False,
    ) -> None:
        super().__init__(field, k, ell)
        if c <= 0:
            raise ValueError("c must be positive")
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        self.c = c
        self.view = view
        self.nonzero_weights = nonzero_weights
        self.outer_permutations = outer_permutations
        self.secret_permutations = secret_permutations

    @property
    def n(self) -> int:
        return self.c * self.k

    def options(self) -> dict:
        return {
            "c": self.c,
            "view": self.view,
            "nonzero_weights": self.nonzero_weights,
            "outer_permutations": self.outer_permutations,
            "secret_permutations": self.secret_permutations,
        }

    def sample_product(self, rng) -> LayeredProduct:
        f, n, ell = self.field, self.n, self.ell
        # Pi_1, ..., Pi_{l+1} in the paper's numbering.
        perms = [random_permutation(f, n, rng) for _ in range(ell + 1)]
        if not self.outer_permutations:
            perms[0] = perms[ell] = identity(f, n)
        publics: list[LinearOperator] = [ComposedOperator([block_sum(f, self.k, self.c), perms[ell]])]
        publics += [perms[ell + 1 - j] for j in range(2, ell + 1)]
        publics.append(ComposedOperator([perms[0], repetition(f, self.k, self.c)]))

        if self.view == "diagonal":
            accumulator = plain_accumulator(f, n)
            publics = [ComposedOperator([a, accumulator]) for a in publics[:ell]] + [publics[ell]]
            secrets = [DiagonalFamily(f, n, nonzero=self.nonzero_weights) for _ in range(ell)]
        else:
            secrets = [WeightedAccumulatorFamily(f, n, nonzero=self.nonzero_weights) for _ in range(ell)]

        revealed = [not self.secret_permutations] * (ell + 1)
        return LayeredProduct(f, publics, secrets, revealed=revealed, metadata=self.describe())

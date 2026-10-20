"""LUOF candidate: products of Druk-Ishai linear-uniform-output matrices [EMVP Sec. 5.2].

Each secret layer is ``S_i = Ext H_i Enc`` (``Enc`` applied first, then the secret
block-diagonal ``H_i``, then ``Ext``; see :mod:`tdm_zoo.tdm.di14`), and layers are
interleaved with public permutations ``Pi_i`` of ``F^k``::

    M = S_1 Pi_2 S_2 ... Pi_l S_l

Two equivalent views of the same product:

* ``view="composite"`` - secrets are the ``k x k`` matrices ``Ext H_i Enc``;
* ``view="split"`` - secrets are the ``N x N`` block-diagonal ``H_i`` and the
  publics absorb the codes, as in the paper::

      A_1 = Ext,   A_i = Enc Pi_i Ext  (2 <= i <= l),   A_{l+1} = Enc.

``codes="di14"`` uses the faithful structure of [DI14]; ``codes="random"`` uses
dense random ``Enc``/``Ext`` of the same dimensions. With ``shared_codes=True``
(default, as in the paper) all layers share one ``(Enc, Ext)`` pair.
"""

from __future__ import annotations

from ...fields import Field
from ...ops.operators import ComposedOperator, LinearOperator
from ..di14 import DI14Params, LUOFCodes, sample_luof_codes
from ..families import BlockDiagonalFamily, SandwichFamily, SecretFamily
from ..product import LayeredProduct
from ..publics import identity, random_permutation
from .base import CANDIDATES, Candidate

VIEWS = ("split", "composite")


@CANDIDATES.register("luof")
class LUOF(Candidate):
    def __init__(
        self,
        field: Field,
        k: int,
        ell: int = 3,
        codes: str = "di14",
        view: str = "split",
        block_size: int = 4,
        base_length: int = 8,
        expander_degree: int = 6,
        invertible_blocks: bool = False,
        shared_codes: bool = True,
        outer_permutations: bool = False,
    ) -> None:
        super().__init__(field, k, ell)
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        self.codes = codes
        self.view = view
        self.params = DI14Params(base_length, expander_degree, block_size)
        self.invertible_blocks = invertible_blocks
        self.shared_codes = shared_codes
        self.outer_permutations = outer_permutations

    def options(self) -> dict:
        return {
            "codes": self.codes,
            "view": self.view,
            "block_size": self.params.block_size,
            "base_length": self.params.base_length,
            "expander_degree": self.params.expander_degree,
            "invertible_blocks": self.invertible_blocks,
            "shared_codes": self.shared_codes,
            "outer_permutations": self.outer_permutations,
        }

    def sample_codes(self, rng) -> list[LUOFCodes]:
        if self.shared_codes:
            return [sample_luof_codes(self.field, self.k, rng, self.params, self.codes)] * self.ell
        return [sample_luof_codes(self.field, self.k, rng, self.params, self.codes) for _ in range(self.ell)]

    def sample_product(self, rng) -> LayeredProduct:
        f, k, ell = self.field, self.k, self.ell
        codes = self.sample_codes(rng)
        perms: list[LinearOperator] = [random_permutation(f, k, rng) for _ in range(ell + 1)]
        if not self.outer_permutations:
            perms[0] = perms[ell] = identity(f, k)

        def block_family(layer: LUOFCodes) -> BlockDiagonalFamily:
            return BlockDiagonalFamily(
                f, layer.inner_dimension, layer.block_size, invertible=self.invertible_blocks
            )

        if self.view == "composite":
            secrets: list[SecretFamily] = [
                SandwichFamily(block_family(c), left=c.extractor, right=c.encoder, label="Ext H Enc")
                for c in codes
            ]
            publics = perms
        else:
            secrets = [block_family(c) for c in codes]
            publics = [ComposedOperator([perms[0], codes[0].extractor])]
            publics += [
                ComposedOperator([codes[i - 1].encoder, perms[i], codes[i].extractor]) for i in range(1, ell)
            ]
            publics.append(ComposedOperator([codes[-1].encoder, perms[ell]]))
        return LayeredProduct(f, publics, secrets, metadata=self.describe())

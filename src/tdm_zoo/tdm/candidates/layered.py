"""Generic layered product with any registered square secret family per layer.

    M = A_1 S_1 A_2 S_2 ... A_l S_l A_{l+1}

Used for experiments outside the three named candidates, e.g. block-diagonal
secrets with uniformly random public matrices (the setting of the toy rank
experiments), or mixing Toeplitz/Hankel/block-diagonal layers.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...fields import Field
from ...ops.operators import LinearOperator
from ..di14 import DI14Params, sample_luof_codes
from ..families import SECRET_FAMILIES
from ..product import LayeredProduct
from ..publics import PUBLIC_KINDS, luof_matrix, luof_span_dimension, public_layer
from .base import CANDIDATES, Candidate

FamilySpec = str | tuple[str, dict] | dict
TDM_PUBLICS = ("random_invertible", "permutation", "luof")


def _normalise(spec: FamilySpec) -> tuple[str, dict]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict):
        spec = dict(spec)
        return spec.pop("family"), spec
    name, kwargs = spec
    return name, dict(kwargs)


@CANDIDATES.register("layered")
class Layered(Candidate):
    """``families``: one spec per layer, either a name, ``(name, kwargs)`` or
    ``{"family": name, **kwargs}``; the family is built as ``family(field, k, **kwargs)``.
    ``publics``: inner public layers; ``outer``: ``A_1`` and ``A_{l+1}`` (see ``PUBLIC_KINDS``).
    """

    def __init__(
        self,
        field: Field,
        k: int,
        families: Sequence[FamilySpec],
        publics: str = "permutation",
        outer: str = "identity",
    ) -> None:
        super().__init__(field, k, len(families))
        self.families = [_normalise(spec) for spec in families]
        for name, _ in self.families:
            SECRET_FAMILIES.get(name)
        if publics not in PUBLIC_KINDS or outer not in PUBLIC_KINDS:
            raise ValueError(f"public kinds must be among {PUBLIC_KINDS}")
        self.publics = publics
        self.outer = outer

    def options(self) -> dict:
        return {
            "families": [{"family": n, **kw} for n, kw in self.families],
            "publics": self.publics,
            "outer": self.outer,
        }

    def sample_publics(self, rng) -> tuple[list[LinearOperator], dict]:
        """The public layers, and anything about them worth keeping in the product's metadata."""
        f, k = self.field, self.k
        publics = [public_layer(self.outer, f, k, rng)]
        publics += [public_layer(self.publics, f, k, rng) for _ in range(self.ell - 1)]
        publics.append(public_layer(self.outer, f, k, rng))
        return publics, {}

    def sample_product(self, rng) -> LayeredProduct:
        f, k = self.field, self.k
        secrets = [SECRET_FAMILIES.create(name, f, k, **kwargs) for name, kwargs in self.families]
        for secret in secrets:
            if secret.shape != (k, k):
                raise ValueError(f"family {secret!r} is not k x k")
        publics, info = self.sample_publics(rng)
        return LayeredProduct(f, publics, secrets, metadata={**self.describe(), **info})


@CANDIDATES.register("layered_tdm")
class LayeredTDM(Layered):
    """The layered TDM attacked by the peeling distinguisher.

    ``M = A_1 S_1 A_2 S_2 ... A_l S_l A_{l+1}`` with every ``A_i`` a uniformly random
    *invertible* public ``n x n`` matrix and every ``S_i`` an invertible secret.
    ``S_1, S_2`` (the base layers, entering the base relations through their
    parametrisation ``f``) come from ``base_family``; ``S_3, ..., S_l`` (the peeled
    layers, entering through their inverse map ``g_j``) from ``peel_family``. Pass
    ``families`` to choose every layer explicitly instead.

    ``publics`` replaces the random ``A_i``: ``"permutation"``, or ``"luof"`` for invertible
    members ``A_i = G H_i P E`` of the Druk-Ishai family with revealed ``H_i``. One ``(G, P E)``
    pair is shared by all layers unless ``shared_codes=False``; ``block_size``, ``base_length``
    and ``expander_degree`` are the :class:`DI14Params`.
    """

    def __init__(
        self,
        field: Field,
        k: int,
        ell: int = 4,
        base_family: FamilySpec = "diagonal",
        peel_family: FamilySpec = "diagonal",
        families: Sequence[FamilySpec] | None = None,
        publics: str = "random_invertible",
        block_size: int = 4,
        base_length: int = 8,
        expander_degree: int = 6,
        shared_codes: bool = True,
    ) -> None:
        if publics not in TDM_PUBLICS:
            raise ValueError(f"publics must be among {TDM_PUBLICS}")
        if families is None:
            if ell < 3:
                raise ValueError("the layered TDM needs ell >= 3")
            families = [base_family] * 2 + [peel_family] * (ell - 2)
        normalised = [_normalise(spec) for spec in families]
        kind = "random_invertible" if publics == "luof" else publics
        super().__init__(field, k, normalised, publics=kind, outer=kind)
        if self.ell < 3:
            raise ValueError("the layered TDM needs ell >= 3")
        self.publics = self.outer = publics
        self.params = DI14Params(base_length, expander_degree, block_size)
        self.shared_codes = shared_codes

    def options(self) -> dict:
        options = super().options()
        if self.publics == "luof":
            options.update(
                block_size=self.params.block_size,
                base_length=self.params.base_length,
                expander_degree=self.params.expander_degree,
                shared_codes=self.shared_codes,
            )
        return options

    def sample_publics(self, rng) -> tuple[list[LinearOperator], dict]:
        if self.publics != "luof":
            return super().sample_publics(rng)
        f, layers = self.field, self.ell + 1
        if self.shared_codes:
            codes = [sample_luof_codes(f, self.k, rng, self.params)] * layers
        else:
            codes = [sample_luof_codes(f, self.k, rng, self.params) for _ in range(layers)]
        publics = [luof_matrix(f, c, rng, invertible=True) for c in codes]
        return publics, {"luof_span": min(luof_span_dimension(f, c) for c in codes)}

    def sample_product(self, rng) -> LayeredProduct:
        product = super().sample_product(rng)
        for index, secret in enumerate(product.secrets[2:], start=3):
            if secret.homogeneous_inverse_map is None:
                raise ValueError(f"layer S_{index} ({secret!r}) has no homogeneous inverse map to peel with")
        return product

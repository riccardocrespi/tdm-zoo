"""Generic layered product with any registered square secret family per layer.

    M = A_1 S_1 A_2 S_2 ... A_l S_l A_{l+1}

Used for experiments outside the three named candidates, e.g. block-diagonal
secrets with uniformly random public matrices (the setting of the toy rank
experiments), or mixing Toeplitz/Hankel/block-diagonal layers.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...fields import Field
from ..families import SECRET_FAMILIES
from ..product import LayeredProduct
from ..publics import PUBLIC_KINDS, public_layer
from .base import CANDIDATES, Candidate

FamilySpec = str | tuple[str, dict] | dict


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

    def sample_product(self, rng) -> LayeredProduct:
        f, k = self.field, self.k
        secrets = [SECRET_FAMILIES.create(name, f, k, **kwargs) for name, kwargs in self.families]
        for secret in secrets:
            if secret.shape != (k, k):
                raise ValueError(f"family {secret!r} is not k x k")
        publics = [public_layer(self.outer, f, k, rng)]
        publics += [public_layer(self.publics, f, k, rng) for _ in range(self.ell - 1)]
        publics.append(public_layer(self.outer, f, k, rng))
        return LayeredProduct(f, publics, secrets, metadata=self.describe())


@CANDIDATES.register("layered_tdm")
class LayeredTDM(Layered):
    """The layered TDM attacked by the peeling distinguisher.

    ``M = A_1 S_1 A_2 S_2 ... A_l S_l A_{l+1}`` with every ``A_i`` a uniformly random
    *invertible* public ``n x n`` matrix and every ``S_i`` an invertible secret.
    ``S_1, S_2`` (the base layers, entering the base relations through their
    parametrisation ``f``) come from ``base_family``; ``S_3, ..., S_l`` (the peeled
    layers, entering through their inverse map ``g_j``) from ``peel_family``. Pass
    ``families`` to choose every layer explicitly instead.
    """

    def __init__(
        self,
        field: Field,
        k: int,
        ell: int = 4,
        base_family: FamilySpec = "diagonal",
        peel_family: FamilySpec = "diagonal",
        families: Sequence[FamilySpec] | None = None,
    ) -> None:
        if families is None:
            if ell < 3:
                raise ValueError("the layered TDM needs ell >= 3")
            families = [base_family] * 2 + [peel_family] * (ell - 2)
        normalised = [_normalise(spec) for spec in families]
        super().__init__(field, k, normalised, publics="random_invertible", outer="random_invertible")
        if self.ell < 3:
            raise ValueError("the layered TDM needs ell >= 3")

    def sample_product(self, rng) -> LayeredProduct:
        product = super().sample_product(rng)
        for index, secret in enumerate(product.secrets[2:], start=3):
            if secret.homogeneous_inverse_map is None:
                raise ValueError(f"layer S_{index} ({secret!r}) has no homogeneous inverse map to peel with")
        return product

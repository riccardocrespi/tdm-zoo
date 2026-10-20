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

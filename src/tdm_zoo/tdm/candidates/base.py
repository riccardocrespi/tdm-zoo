"""Candidate TDM constructions: distributions over :class:`LayeredProduct` instances."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ...fields import Field
from ...registry import Registry
from ..product import LayeredProduct

CANDIDATES: Registry[Candidate] = Registry("candidate")


class Candidate(ABC):
    """A TDM candidate producing ``k x k`` matrices over ``field``.

    :meth:`sample_product` samples the public randomness (permutations, codes,
    ...) and returns a product whose :meth:`~LayeredProduct.sample` draws the
    secrets. Keeping the two separate lets experiments reuse an expensive attack
    preprocessing across many samples with the same public matrices.
    """

    name: ClassVar[str] = "candidate"

    def __init__(self, field: Field, k: int, ell: int) -> None:
        if k <= 0 or ell <= 0:
            raise ValueError("k and ell must be positive")
        self.field = field
        self.k = k
        self.ell = ell

    @abstractmethod
    def sample_product(self, rng: np.random.Generator) -> LayeredProduct: ...

    def options(self) -> dict:
        return {}

    def describe(self) -> dict:
        return {
            "candidate": self.name,
            "field": self.field.describe(),
            "k": self.k,
            "ell": self.ell,
            **self.options(),
        }

    def __repr__(self) -> str:
        options = ", ".join(f"{k}={v}" for k, v in self.options().items())
        return f"{type(self).__name__}(k={self.k}, ell={self.ell}{', ' if options else ''}{options})"

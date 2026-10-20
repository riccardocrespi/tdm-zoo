"""Distinguisher interface.

A distinguisher is given the public view of a product (public layers that are
revealed, plus the public descriptions of the secret families) and a matrix
``M``; it guesses whether ``M`` is a structured sample or uniform.

Work that depends only on the public view (e.g. building the coefficient map
``T_l`` of the generic rank attack) belongs in :meth:`Distinguisher.prepare`; the
experiment harness calls it once per public sample and reuses the state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..registry import Registry
from ..tdm.product import PublicView

DISTINGUISHERS: Registry[Distinguisher] = Registry("distinguisher")


@dataclass(frozen=True)
class Decision:
    structured: bool  # the guess: True means "M is a structured sample"
    score: float | None = None  # optional statistic (e.g. a rank) for later analysis
    info: dict = field(default_factory=dict)


class Distinguisher(ABC):
    name: ClassVar[str] = "distinguisher"

    def prepare(self, view: PublicView) -> Any:
        """Preprocess the public view; the return value is passed to :meth:`decide`."""
        return None

    @abstractmethod
    def decide(self, state: Any, matrix: np.ndarray, rng: np.random.Generator) -> Decision: ...

    def options(self) -> dict:
        return {}

    def describe(self) -> dict:
        return {"distinguisher": self.name, **self.options()}

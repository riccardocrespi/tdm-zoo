"""Peeling distinguisher in Macaulay form.

Output "structured" iff ``rank T(M) < rank T(U)`` for a uniformly random reference
``U``. For a structured ``M`` the witness monomial vector ``v*`` lies in the kernel
of ``T(M)``; the attack works in the regime where a uniform ``U`` has no such extra
kernel direction (gap ``delta(M) - delta(U) >= 1``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...tdm.product import PublicView
from ..base import DISTINGUISHERS, Decision, Distinguisher
from .system import PeelingSystem


@dataclass
class PeelingState:
    system: PeelingSystem
    reference_rank: int | None = None


@DISTINGUISHERS.register("peeling_macaulay")
class PeelingMacaulay(Distinguisher):
    """``d``: degree of the base relations; ``multipliers``: multiplier multidegree ``a``
    (an int for all blocks, or one per block in order ``W_l, ..., W_3``);
    ``reference="fresh"`` samples a new ``U`` per decision, ``"cached"`` one per public sample;
    ``rank_method``: ``"auto"`` (exact dense elimination up to ``dense_limit`` cells, Wiedemann on
    the implicit sparse operator beyond), ``"dense"`` or ``"wiedemann"``; ``repeats``: independent
    Wiedemann runs per rank (maximum taken; Wiedemann can only under-estimate); ``device``: torch
    device (e.g. ``"cuda"``) for the Wiedemann products.
    """

    def __init__(
        self,
        d: int = 3,
        multipliers: int | tuple[int, ...] = 0,
        reference: str = "fresh",
        seed: int | None = None,
        rank_method: str = "auto",
        dense_limit: int = 5 * 10**7,
        repeats: int = 1,
        device: str | None = None,
    ) -> None:
        if reference not in ("fresh", "cached"):
            raise ValueError("reference must be 'fresh' or 'cached'")
        self.d = d
        self.multipliers = multipliers
        self.reference = reference
        self.rank_method = rank_method
        self.dense_limit = dense_limit
        self.repeats = repeats
        self.device = device
        self._rng = np.random.default_rng(seed)

    def options(self) -> dict:
        multipliers = self.multipliers if np.isscalar(self.multipliers) else list(self.multipliers)
        return {
            "d": self.d,
            "multipliers": multipliers,
            "reference": self.reference,
            "rank_method": self.rank_method,
            "repeats": self.repeats,
            "device": self.device,
        }

    def prepare(self, view: PublicView) -> PeelingState:
        return PeelingState(PeelingSystem(view, self.d, self._rng))

    def rank(self, state: PeelingState, matrix: np.ndarray) -> int:
        system = state.system
        return system.macaulay_rank(
            matrix, self.multipliers, self.rank_method, self._rng, self.dense_limit, self.repeats, self.device
        )

    def decide(self, state: PeelingState, matrix: np.ndarray, rng: np.random.Generator) -> Decision:
        system = state.system
        if self.reference == "fresh" or state.reference_rank is None:
            uniform = system.field.random(rng, (system.n, system.n))
            state.reference_rank = self.rank(state, uniform)
        rank = self.rank(state, matrix)
        shape = system.shape(self.multipliers)
        return Decision(
            structured=rank < state.reference_rank,
            score=float(rank),
            info={
                "rank": rank,
                "reference_rank": state.reference_rank,
                **shape.describe(),
            },
        )

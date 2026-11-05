"""Membership test against the linear span of the structured distribution.

The public view fixes the distribution of ``M``, so anyone can sample it with secrets of their
own. If those samples span a proper subspace, the orthogonal complement is a set of public linear
relations that every structured ``M`` satisfies; a uniform matrix passes each with probability
``1 / |F|``. With a full span there are no relations and the test always answers "structured".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ops import linalg
from ..tdm.product import LayeredProduct, PublicView
from .base import DISTINGUISHERS, Decision, Distinguisher


@dataclass(frozen=True)
class LinearSpanState:
    field: object
    relations: np.ndarray  # (codim, rows * cols): r @ vec(M) = 0 for every structured M


@DISTINGUISHERS.register("linear_span")
class LinearSpan(Distinguisher):
    name = "linear_span"

    def __init__(self, seed: int = 0, margin: int = 20) -> None:
        self.seed = seed
        self.margin = margin
        self.relations: np.ndarray | None = None  # from the last prepare

    def options(self) -> dict:
        return {"seed": self.seed, "margin": self.margin}

    def prepare(self, view: PublicView) -> LinearSpanState:
        if any(public is None for public in view.publics):
            raise ValueError("the span test needs every public layer")
        rng = np.random.default_rng(self.seed)
        product = LayeredProduct(view.field, view.publics, view.secret_families)
        size = view.shape[0] * view.shape[1]
        stacked = np.stack([product.sample(rng).matrix.ravel() for _ in range(size + self.margin)])
        self.relations = linalg.nullspace(view.field, stacked).T
        return LinearSpanState(view.field, self.relations)

    def decide(self, state: LinearSpanState, matrix: np.ndarray, rng: np.random.Generator) -> Decision:
        if not state.relations.shape[0]:
            return Decision(True, 0.0)
        values = state.field.matmul(state.relations, matrix.reshape(-1, 1))
        violated = int(np.count_nonzero(values))
        return Decision(violated == 0, float(violated))

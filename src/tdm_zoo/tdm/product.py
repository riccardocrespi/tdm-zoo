"""Layered matrix products ``M = A_1 S_1 A_2 S_2 ... A_l S_l A_{l+1}``.

Conventions (shared by the whole package):

* layers are indexed left to right as in the paper: ``A_1`` is applied last;
* vectors are columns; the evaluation circuit computes ``v -> M v`` by applying
  ``A_{l+1}`` first;
* publics may be rectangular as long as shapes compose; ``M`` is ``rows(A_1) x
  cols(A_{l+1})``.

A :class:`LayeredProduct` fixes the public matrices (one sample of the public
distribution). Sampling it draws fresh secrets and returns a :class:`TDMSample`
with the matrix, the trapdoor and the fast evaluation circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from ..fields import Field
from ..ops.operators import ComposedOperator, LinearOperator
from .families.base import SecretFamily, SecretSample

Array = np.ndarray


@dataclass(frozen=True)
class PublicView:
    """What the distinguisher sees besides ``M``.

    ``publics[i]`` is ``None`` when ``A_{i+1}`` is hidden. The secret families are
    always visible: their symbolic parametrisations are public knowledge, only
    their samples are secret.
    """

    field: Field
    shape: tuple[int, int]
    publics: list[LinearOperator | None]
    secret_families: list[SecretFamily]
    metadata: dict

    @property
    def ell(self) -> int:
        return len(self.secret_families)

    def public_matrix(self, index: int) -> Array | None:
        """Dense ``A_{index+1}`` (0-based index), or ``None`` if hidden."""
        operator = self.publics[index]
        return None if operator is None else operator.to_dense()


@dataclass(eq=False)
class LayeredProduct:
    field: Field
    publics: list[LinearOperator]
    secrets: list[SecretFamily]
    revealed: list[bool] | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.publics) != len(self.secrets) + 1:
            raise ValueError("need exactly one more public layer than secret layers")
        if self.revealed is None:
            self.revealed = [True] * len(self.publics)
        if len(self.revealed) != len(self.publics):
            raise ValueError("revealed must have one flag per public layer")
        for index, secret in enumerate(self.secrets):
            left, right = self.publics[index], self.publics[index + 1]
            if left.cols != secret.shape[0] or secret.shape[1] != right.rows:
                raise ValueError(
                    f"layer {index + 1}: A{index + 1}{left.shape}, S{index + 1}{secret.shape}, "
                    f"A{index + 2}{right.shape} do not compose"
                )

    # -- structure ----------------------------------------------------------------------
    @property
    def ell(self) -> int:
        return len(self.secrets)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.publics[0].rows, self.publics[-1].cols)

    @property
    def layer_shapes(self) -> list[tuple[str, tuple[int, int]]]:
        shapes = []
        for index, secret in enumerate(self.secrets):
            shapes.append((f"A{index + 1}", self.publics[index].shape))
            shapes.append((f"S{index + 1}", secret.shape))
        shapes.append((f"A{self.ell + 1}", self.publics[-1].shape))
        return shapes

    @cached_property
    def _dense_publics(self) -> list[Array]:
        return [public.to_dense() for public in self.publics]

    # -- sampling -----------------------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> TDMSample:
        samples = [family.sample(rng) for family in self.secrets]
        return TDMSample(self, samples, self.matrix(samples))

    def sample_uniform(self, rng: np.random.Generator) -> Array:
        return self.field.random(rng, self.shape)

    def matrix(self, samples: list[SecretSample]) -> Array:
        """Dense product, accumulated right to left."""
        dense = self._dense_publics
        current = dense[-1]
        for index in reversed(range(self.ell)):
            current = self.field.matmul(samples[index].matrix(), current)
            current = self.field.matmul(dense[index], current)
        return current

    def evaluation_circuit(self, samples: list[SecretSample]) -> LinearOperator:
        factors: list[LinearOperator] = []
        for index, sample in enumerate(samples):
            factors += [self.publics[index], sample.operator()]
        factors.append(self.publics[-1])
        return ComposedOperator(factors)

    def public_view(self) -> PublicView:
        return PublicView(
            field=self.field,
            shape=self.shape,
            publics=[p if shown else None for p, shown in zip(self.publics, self.revealed, strict=True)],
            secret_families=list(self.secrets),
            metadata=dict(self.metadata),
        )


@dataclass(eq=False)
class TDMSample:
    product: LayeredProduct
    secrets: list[SecretSample]
    matrix: Array

    @property
    def trapdoor(self) -> list[Array]:
        return [sample.params for sample in self.secrets]

    @cached_property
    def evaluation_circuit(self) -> LinearOperator:
        return self.product.evaluation_circuit(self.secrets)

    def apply(self, vectors) -> Array:
        return self.evaluation_circuit.apply(vectors)

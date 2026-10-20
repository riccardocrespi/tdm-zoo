"""Secret matrix families.

A secret family exposes three views of the same object:

1. **sampling** - ``sample_params(rng)`` draws the secret randomness (part of the
   trapdoor);
2. **fast evaluation** - ``operator(params)`` is the matrix as a fast linear
   operator, i.e. this layer's piece of the evaluation circuit;
3. **symbolic description** - ``param_map()`` is a *linear* map ``f`` whose image
   contains every sampled matrix, and ``inverse_map()`` (when it exists) is a
   map ``g`` whose image contains every inverse. ``inverse_params(params)``
   returns a witness ``w*`` with ``g(w*) = S^{-1}``.

The symbolic view is what the attacker is assumed to know; samples are not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np

from ...fields import Field
from ...ops import linalg
from ...ops.matrix_maps import MatrixMap
from ...ops.operators import LinearOperator
from ...registry import Registry

Array = np.ndarray

SECRET_FAMILIES: Registry[SecretFamily] = Registry("secret family")

_MAX_REJECTIONS = 1000


class SecretFamily(ABC):
    name: ClassVar[str] = "secret"

    def __init__(self, field: Field, rows: int, cols: int) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("dimensions must be positive")
        self.field = field
        self.shape = (int(rows), int(cols))

    # -- required -----------------------------------------------------------------------
    @property
    @abstractmethod
    def nparams(self) -> int: ...

    @abstractmethod
    def sample_params(self, rng: np.random.Generator) -> Array: ...

    @abstractmethod
    def operator(self, params) -> LinearOperator: ...

    @abstractmethod
    def _param_map(self) -> MatrixMap: ...

    # -- optional -----------------------------------------------------------------------
    def _inverse_map(self) -> MatrixMap | None:
        return None

    def inverse_params(self, params) -> Array:
        raise NotImplementedError(f"{self.name} has no inverse parametrisation")

    def options(self) -> dict:
        """Constructor options beyond the field, for logging and reconstruction."""
        return {}

    # -- derived ------------------------------------------------------------------------
    @cached_property
    def param_map(self) -> MatrixMap:
        return self._param_map()

    @cached_property
    def inverse_map(self) -> MatrixMap | None:
        return self._inverse_map()

    @property
    def is_square(self) -> bool:
        return self.shape[0] == self.shape[1]

    def matrix(self, params) -> Array:
        return self.param_map.evaluate(params)

    def sample(self, rng: np.random.Generator) -> SecretSample:
        return SecretSample(self, self.sample_params(rng))

    def describe(self) -> dict:
        return {"family": self.name, "shape": self.shape, "nparams": self.nparams, **self.options()}

    def __repr__(self) -> str:
        options = ", ".join(f"{k}={v}" for k, v in self.options().items())
        return f"{type(self).__name__}({options})"

    # -- helpers for subclasses ---------------------------------------------------------
    def _rejection_sample(self, rng: np.random.Generator, draw, accept) -> Array:
        for _ in range(_MAX_REJECTIONS):
            params = draw(rng)
            if accept(params):
                return params
        raise RuntimeError(f"{self.name}: rejection sampling did not terminate")

    def _is_invertible(self, params) -> bool:
        return linalg.is_invertible(self.field, self.matrix(params))


@dataclass(frozen=True, eq=False)
class SecretSample:
    family: SecretFamily
    params: Array

    def operator(self) -> LinearOperator:
        return self.family.operator(self.params)

    def matrix(self) -> Array:
        return self.family.matrix(self.params)

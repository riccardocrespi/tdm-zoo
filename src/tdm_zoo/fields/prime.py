"""Prime fields F_p."""

from __future__ import annotations

import flint
import numpy as np

from ..arith import modp
from ..arith.backends import PrimeBackend, get_backend
from .base import Array, Field


class PrimeField(Field):
    """F_p for a prime ``p < 2**31``.

    ``backend`` selects the engine for dense linear algebra (matmul, rank,
    elimination); see :func:`tdm_zoo.arith.get_backend`.
    """

    def __init__(self, p: int, backend: str | PrimeBackend = "auto") -> None:
        p = modp.check_modulus(p)
        if not flint.fmpz(p).is_prime():
            raise ValueError(f"{p} is not prime")
        self.p = p
        self.m = 1
        self.q = p
        self.backend = get_backend(backend) if isinstance(backend, str) else backend

    def unpack(self, a: Array) -> Array:
        return np.asarray(a, dtype=np.int64)[..., None]

    def pack(self, digits: Array) -> Array:
        return np.asarray(digits, dtype=np.int64)[..., 0]

    def add(self, a: Array, b: Array) -> Array:
        return modp.add(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64), self.p)

    def sub(self, a: Array, b: Array) -> Array:
        return modp.sub(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64), self.p)

    def neg(self, a: Array) -> Array:
        return modp.neg(np.asarray(a, dtype=np.int64), self.p)

    def mul(self, a: Array, b: Array) -> Array:
        return modp.mul(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64), self.p)

    def inv(self, a: Array) -> Array:
        return modp.inv(a, self.p)

    def power(self, a: Array, exponent: int) -> Array:
        if exponent < 0:
            return modp.power(self.inv(a), -exponent, self.p)
        return modp.power(a, exponent, self.p)

    def sum(self, a: Array, axis=None) -> Array:
        return np.mod(np.sum(np.asarray(a, dtype=np.int64), axis=axis), self.p)

    def cumsum(self, a: Array, axis: int = 0) -> Array:
        return np.mod(np.cumsum(np.asarray(a, dtype=np.int64), axis=axis), self.p)

    def segment_sum(self, values: Array, segments: Array, count: int) -> Array:
        values = np.asarray(values, dtype=np.int64)
        out = np.zeros((count,) + values.shape[1:], dtype=np.int64)
        np.add.at(out, np.asarray(segments, dtype=np.int64), values)
        return np.mod(out, self.p)

    def matmul(self, left: Array, right: Array) -> Array:
        return self.backend.matmul(
            np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64), self.p
        )

    def __repr__(self) -> str:
        return f"GF({self.p})"

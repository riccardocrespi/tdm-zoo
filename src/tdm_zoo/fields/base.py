"""Field-agnostic interface.

Elements of every field are encoded as int64 integers in ``[0, q)``, so a matrix
over any field is a plain int64 numpy array of the same shape. For F_{p^m} an
element ``a_0 + a_1 x + ... + a_{m-1} x^{m-1}`` is encoded as the integer
``sum a_i p^i``; in particular the prime subfield is encoded identically in
F_p and F_{p^m}.

The operations mirror the black-box field oracle used by field-agnostic
protocols: addition, subtraction, multiplication, inversion, zero-test and
sampling. Everything above this layer should use only this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

Array = np.ndarray


class Field(ABC):
    p: int  # characteristic
    m: int  # extension degree over the prime field
    q: int  # order

    # -- encoding -----------------------------------------------------------------------
    def asarray(self, values) -> Array:
        """Validate and convert to canonical int64 element encodings."""
        array = np.asarray(values, dtype=np.int64)
        if array.size and (array.min() < 0 or array.max() >= self.q):
            raise ValueError(f"values are not canonical elements of {self}")
        return array

    def from_int(self, values) -> Array:
        """Embed integers into the prime subfield (reduction modulo p)."""
        return np.mod(np.asarray(values, dtype=np.int64), self.p)

    @abstractmethod
    def unpack(self, a: Array) -> Array:
        """Coordinates over F_p: shape ``a.shape + (m,)``."""

    @abstractmethod
    def pack(self, digits: Array) -> Array:
        """Inverse of :meth:`unpack`."""

    # -- constructors -------------------------------------------------------------------
    def zeros(self, shape) -> Array:
        return np.zeros(shape, dtype=np.int64)

    def ones(self, shape) -> Array:
        return np.ones(shape, dtype=np.int64)

    def eye(self, n: int, m: int | None = None) -> Array:
        return np.eye(n, m, dtype=np.int64)

    def random(self, rng: np.random.Generator, shape=()) -> Array:
        return rng.integers(0, self.q, size=shape, dtype=np.int64)

    def random_nonzero(self, rng: np.random.Generator, shape=()) -> Array:
        return rng.integers(1, self.q, size=shape, dtype=np.int64)

    # -- elementwise arithmetic ---------------------------------------------------------
    @abstractmethod
    def add(self, a: Array, b: Array) -> Array: ...

    @abstractmethod
    def neg(self, a: Array) -> Array: ...

    @abstractmethod
    def mul(self, a: Array, b: Array) -> Array: ...

    @abstractmethod
    def inv(self, a: Array) -> Array:
        """Elementwise inverse; raises ``ZeroDivisionError`` on zero."""

    def sub(self, a: Array, b: Array) -> Array:
        return self.add(a, self.neg(b))

    def div(self, a: Array, b: Array) -> Array:
        return self.mul(a, self.inv(b))

    def is_zero(self, a: Array) -> Array:
        return np.asarray(a) == 0

    def power(self, a: Array, exponent: int) -> Array:
        a = np.asarray(a, dtype=np.int64)
        if exponent < 0:
            return self.power(self.inv(a), -exponent)
        result = self.ones(a.shape)
        base = a
        while exponent:
            if exponent & 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            exponent >>= 1
        return result

    # -- reductions (additive structure is F_p^m) ---------------------------------------
    def sum(self, a: Array, axis=None) -> Array:
        digits = self.unpack(np.asarray(a, dtype=np.int64))
        if axis is None:
            total = digits.reshape(-1, self.m).sum(axis=0)
        else:
            axis = axis % np.ndim(a)
            total = digits.sum(axis=axis)
        return self.pack(np.mod(total, self.p))

    def cumsum(self, a: Array, axis: int = 0) -> Array:
        digits = self.unpack(np.asarray(a, dtype=np.int64))
        return self.pack(np.mod(np.cumsum(digits, axis=axis % np.ndim(a)), self.p))

    def segment_sum(self, values: Array, segments: Array, count: int) -> Array:
        """``out[s] = sum of values[i] with segments[i] == s`` along axis 0."""
        digits = self.unpack(np.asarray(values, dtype=np.int64))
        out = np.zeros((count,) + digits.shape[1:], dtype=np.int64)
        np.add.at(out, np.asarray(segments, dtype=np.int64), digits)
        return self.pack(np.mod(out, self.p))

    # -- matrices -----------------------------------------------------------------------
    @abstractmethod
    def matmul(self, left: Array, right: Array) -> Array: ...

    def scale(self, scalar, a: Array) -> Array:
        return self.mul(np.asarray(scalar, dtype=np.int64), a)

    # -- identity -----------------------------------------------------------------------
    @property
    def characteristic(self) -> int:
        return self.p

    @property
    def is_prime_field(self) -> bool:
        return self.m == 1

    def describe(self) -> dict:
        return {"p": self.p, "m": self.m, "q": self.q}

    def __eq__(self, other) -> bool:
        return isinstance(other, Field) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def _key(self) -> tuple:
        return (type(self).__name__, self.p, self.m)

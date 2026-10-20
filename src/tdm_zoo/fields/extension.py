"""Extension fields F_{p^m} = F_p[x] / (f) in the polynomial basis."""

from __future__ import annotations

import flint
import numpy as np

from ..arith import polymodp
from ..arith.backends import PrimeBackend
from .base import Array, Field
from .prime import PrimeField

MAX_ORDER = 2**62
TABLE_LIMIT = 2**20


class ExtensionField(Field):
    """F_{p^m} with integer-encoded elements (base-p digits = polynomial coefficients).

    Multiplication uses exp/log tables when ``q <= table_limit`` and polynomial
    multiplication modulo ``f`` otherwise. Matrix products are reduced to ``m**2``
    products over the prime subfield, so they inherit its backend.
    """

    def __init__(
        self,
        p: int,
        m: int,
        modulus: list[int] | None = None,
        backend: str | PrimeBackend = "auto",
        table_limit: int = TABLE_LIMIT,
    ) -> None:
        if m < 2:
            raise ValueError("use PrimeField for m = 1")
        self.subfield = PrimeField(p, backend)
        self.p, self.m, self.q = p, m, p**m
        if self.q >= MAX_ORDER:
            raise ValueError("field order must be below 2**62")
        if modulus is None:
            modulus = polymodp.first_irreducible(p, m)
        modulus = [int(c) % p for c in modulus]
        if len(modulus) != m + 1 or modulus[-1] != 1:
            raise ValueError("modulus must be monic of degree m")
        if not polymodp.is_irreducible(modulus, p):
            raise ValueError("modulus is not irreducible")
        self.modulus = modulus
        self._place = p ** np.arange(m, dtype=np.int64)
        self._reduction = polymodp.reduction_table(modulus, p, 2 * m - 1)
        self._exp: Array | None = None
        self._log: Array | None = None
        if self.q <= table_limit:
            self._build_tables()

    @property
    def backend(self) -> PrimeBackend:
        return self.subfield.backend

    # -- encoding -----------------------------------------------------------------------
    def unpack(self, a: Array) -> Array:
        a = np.asarray(a, dtype=np.int64)
        return (a[..., None] // self._place) % self.p

    def pack(self, digits: Array) -> Array:
        return np.asarray(digits, dtype=np.int64) @ self._place

    # -- arithmetic ---------------------------------------------------------------------
    def add(self, a: Array, b: Array) -> Array:
        a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
        if self.p == 2:
            return a ^ b
        return self.pack((self.unpack(a) + self.unpack(b)) % self.p)

    def neg(self, a: Array) -> Array:
        a = np.asarray(a, dtype=np.int64)
        if self.p == 2:
            return a.copy()
        return self.pack((-self.unpack(a)) % self.p)

    def mul(self, a: Array, b: Array) -> Array:
        a, b = np.broadcast_arrays(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
        if self._exp is not None:
            out = self._exp[self._log[a] + self._log[b]]
            return np.where((a == 0) | (b == 0), 0, out)
        return self.pack(self._mul_digits(self.unpack(a), self.unpack(b)))

    def inv(self, a: Array) -> Array:
        a = np.asarray(a, dtype=np.int64)
        if np.any(a == 0):
            raise ZeroDivisionError("inverse of zero")
        if self._exp is not None:
            return self._exp[(self.q - 1 - self._log[a]) % (self.q - 1)]
        return Field.power(self, a, self.q - 2)

    def matmul(self, left: Array, right: Array) -> Array:
        left, right = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
        if left.shape[1] != right.shape[0]:
            raise ValueError(f"incompatible shapes {left.shape} and {right.shape}")
        ld, rd = self.unpack(left), self.unpack(right)
        p, m = self.p, self.m
        products = np.zeros((2 * m - 1, left.shape[0], right.shape[1]), dtype=np.int64)
        for s in range(m):
            for t in range(m):
                products[s + t] += self.subfield.matmul(ld[..., s], rd[..., t])
        products %= p
        digits = np.einsum("tij,tu->iju", products, self._reduction) % p
        return self.pack(digits)

    # -- internals ----------------------------------------------------------------------
    def _mul_digits(self, a: Array, b: Array) -> Array:
        m, p = self.m, self.p
        product = np.zeros(a.shape[:-1] + (2 * m - 1,), dtype=np.int64)
        for s in range(m):
            product[..., s : s + m] += a[..., s : s + 1] * b
        product %= p
        return (product @ self._reduction) % p

    def _build_tables(self) -> None:
        generator = self._find_generator()
        order = self.q - 1
        # Blocked generation: the first `block` powers serially, then whole blocks at once.
        block = int(np.sqrt(order)) + 1
        powers = np.empty((block, self.m), dtype=np.int64)
        current = self.unpack(np.array(1))
        gen_digits = self.unpack(np.array(generator))
        for i in range(block):
            powers[i] = current
            current = self._mul_digits(current, gen_digits)
        stride = current  # generator**block
        blocks = [powers]
        while len(blocks) * block < order:
            blocks.append(self._mul_digits(blocks[-1], stride))
        exp = np.empty(2 * order, dtype=np.int64)
        exp[:order] = self.pack(np.concatenate(blocks)[:order])
        exp[order:] = exp[:order]
        log = np.zeros(self.q, dtype=np.int64)
        log[exp[:order]] = np.arange(order)
        self._exp, self._log = exp, log

    def _find_generator(self) -> int:
        order = self.q - 1
        prime_factors = [int(r) for r, _ in flint.fmpz(order).factor()]
        for candidate in range(2, self.q):
            digits = self.unpack(np.array(candidate))
            if all(self._pow_digits(digits, order // r) != 1 for r in prime_factors):
                return candidate
        raise RuntimeError("no generator found")  # unreachable

    def _pow_digits(self, digits: Array, exponent: int) -> int:
        result = self.unpack(np.array(1))
        base = digits
        while exponent:
            if exponent & 1:
                result = self._mul_digits(result, base)
            base = self._mul_digits(base, base)
            exponent >>= 1
        return int(self.pack(result))

    def describe(self) -> dict:
        return {**super().describe(), "modulus": self.modulus}

    def _key(self) -> tuple:
        return (*super()._key(), tuple(self.modulus))

    def __repr__(self) -> str:
        return f"GF({self.p}^{self.m})"

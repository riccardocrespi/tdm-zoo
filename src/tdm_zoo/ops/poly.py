"""Sparse multivariate polynomials over a field and monomial bookkeeping.

A monomial is a sorted tuple of variable indices with repetition, e.g.
``x_0^2 x_3 -> (0, 0, 3)``; its degree is its length. This is the format needed
for linearisation (coefficient extraction) in the attacks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations_with_replacement
from math import comb

import numpy as np

from ..fields import Field

Monomial = tuple[int, ...]


def num_monomials(nvars: int, degree: int) -> int:
    """``N(n, d) = binom(n + d - 1, d)``: dimension of degree-d homogeneous polynomials."""
    return comb(nvars + degree - 1, degree)


def homogeneous_monomials(nvars: int, degree: int) -> list[Monomial]:
    """All degree-``degree`` monomials in lexicographic order (the order of ``w^{(x)d}``)."""
    return list(combinations_with_replacement(range(nvars), degree))


def monomials_up_to(nvars: int, degree: int) -> list[Monomial]:
    """All monomials of degree ``<= degree``, including the empty monomial ``()``."""
    return [m for d in range(degree + 1) for m in homogeneous_monomials(nvars, d)]


def monomial_vector(field: Field, point, monomials: Sequence[Monomial]) -> np.ndarray:
    """Evaluate each monomial at ``point``."""
    point = np.asarray(point, dtype=np.int64)
    out = field.ones(len(monomials))
    for index, monomial in enumerate(monomials):
        for variable in monomial:
            out[index] = field.mul(out[index], point[variable])
    return out


class SparsePolynomial:
    """Polynomial with integer-encoded field coefficients, stored as ``{monomial: coefficient}``."""

    __slots__ = ("field", "nvars", "terms")

    def __init__(self, field: Field, nvars: int, terms: dict[Monomial, int] | None = None) -> None:
        self.field = field
        self.nvars = nvars
        self.terms = {tuple(sorted(m)): int(c) for m, c in (terms or {}).items() if int(c) != 0}

    @classmethod
    def variable(cls, field: Field, nvars: int, index: int) -> SparsePolynomial:
        return cls(field, nvars, {(index,): 1})

    @classmethod
    def constant(cls, field: Field, nvars: int, value: int) -> SparsePolynomial:
        return cls(field, nvars, {(): value})

    # -- arithmetic ---------------------------------------------------------------------
    def _f(self, function, a: int, b: int | None = None) -> int:
        args = (np.int64(a),) if b is None else (np.int64(a), np.int64(b))
        return int(function(*args))

    def __add__(self, other: SparsePolynomial) -> SparsePolynomial:
        terms = dict(self.terms)
        for monomial, coefficient in other.terms.items():
            terms[monomial] = self._f(self.field.add, terms.get(monomial, 0), coefficient)
        return SparsePolynomial(self.field, self.nvars, terms)

    def __neg__(self) -> SparsePolynomial:
        return SparsePolynomial(
            self.field, self.nvars, {m: self._f(self.field.neg, c) for m, c in self.terms.items()}
        )

    def __sub__(self, other: SparsePolynomial) -> SparsePolynomial:
        return self + (-other)

    def __mul__(self, other: SparsePolynomial) -> SparsePolynomial:
        terms: dict[Monomial, int] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                monomial = tuple(sorted(m1 + m2))
                product = self._f(self.field.mul, c1, c2)
                terms[monomial] = self._f(self.field.add, terms.get(monomial, 0), product)
        return SparsePolynomial(self.field, self.nvars, terms)

    def scale(self, scalar: int) -> SparsePolynomial:
        return SparsePolynomial(
            self.field, self.nvars, {m: self._f(self.field.mul, c, scalar) for m, c in self.terms.items()}
        )

    # -- queries ------------------------------------------------------------------------
    @property
    def degree(self) -> int:
        return max((len(m) for m in self.terms), default=-1)

    def is_homogeneous(self) -> bool:
        return len({len(m) for m in self.terms}) <= 1

    def evaluate(self, point) -> int:
        point = np.asarray(point, dtype=np.int64)
        total = 0
        for monomial, coefficient in self.terms.items():
            value = coefficient
            for variable in monomial:
                value = self._f(self.field.mul, value, int(point[variable]))
            total = self._f(self.field.add, total, value)
        return total

    def coefficients(self, monomials: Iterable[Monomial]) -> list[int]:
        return [self.terms.get(m, 0) for m in monomials]

    def __eq__(self, other) -> bool:
        return isinstance(other, SparsePolynomial) and self.field == other.field and self.terms == other.terms

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for monomial, coefficient in sorted(self.terms.items()):
            names = "*".join(f"x{v}" for v in monomial)
            parts.append(f"{coefficient}*{names}" if names else str(coefficient))
        return " + ".join(parts)

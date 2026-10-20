"""Univariate polynomials over F_p, used to build extension fields.

Polynomials are coefficient lists ordered from the constant term upwards.
"""

from __future__ import annotations

from itertools import product

import flint
import numpy as np


def is_irreducible(coefficients: list[int], p: int) -> bool:
    poly = flint.nmod_poly([int(c) % p for c in coefficients], p)
    degree = poly.degree()
    if degree < 1:
        return False
    _, factors = poly.factor()
    return len(factors) == 1 and factors[0][1] == 1 and factors[0][0].degree() == degree


def first_irreducible(p: int, m: int) -> list[int]:
    """Lexicographically first monic irreducible polynomial of degree ``m`` over F_p.

    The order is on ``(c_{m-1}, ..., c_0)`` read as base-p digits, so the result is
    deterministic and reproducible across runs.
    """
    if m < 2:
        raise ValueError("extension degree must be at least 2")
    for high_to_low in product(range(p), repeat=m):
        coefficients = list(reversed(high_to_low)) + [1]
        if coefficients[0] != 0 and is_irreducible(coefficients, p):
            return coefficients
    raise RuntimeError("no irreducible polynomial found")  # unreachable


def reduction_table(modulus: list[int], p: int, count: int) -> np.ndarray:
    """Rows ``t < count`` hold the coefficients of ``x**t mod modulus``."""
    m = len(modulus) - 1
    if modulus[-1] % p != 1:
        raise ValueError("modulus must be monic")
    table = np.zeros((count, m), dtype=np.int64)
    current = np.zeros(m, dtype=np.int64)
    current[0] = 1
    tail = np.array(modulus[:-1], dtype=np.int64) % p
    for t in range(count):
        table[t] = current
        carry = current[-1]
        current = np.concatenate(([0], current[:-1]))
        current = (current - carry * tail) % p
    return table

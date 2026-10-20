"""Vectorised arithmetic modulo a prime on int64 numpy arrays.

Inputs are canonical representatives in ``[0, p)`` and outputs are canonical as
well. The modulus must satisfy ``p < 2**31`` so that the product of two
representatives fits in int64.
"""

from __future__ import annotations

import numpy as np

MAX_MODULUS = 2**31
_FLOAT_EXACT = 2**53
_LIMB_BITS = 16


def check_modulus(p: int) -> int:
    p = int(p)
    if not 2 <= p < MAX_MODULUS:
        raise ValueError(f"modulus must lie in [2, 2**31), got {p}")
    return p


def reduce(a, p: int) -> np.ndarray:
    return np.mod(np.asarray(a, dtype=np.int64), p)


def add(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    return (a + b) % p


def sub(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    return (a - b) % p


def neg(a: np.ndarray, p: int) -> np.ndarray:
    return (-a) % p


def mul(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    return (a * b) % p


def power(a: np.ndarray, exponent: int, p: int) -> np.ndarray:
    """Elementwise ``a**exponent mod p`` by square-and-multiply."""
    if exponent < 0:
        raise ValueError("exponent must be nonnegative")
    base = np.asarray(a, dtype=np.int64) % p
    result = np.ones_like(base)
    while exponent:
        if exponent & 1:
            result = (result * base) % p
        base = (base * base) % p
        exponent >>= 1
    return result


def inv(a: np.ndarray, p: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64) % p
    if np.any(a == 0):
        raise ZeroDivisionError("inverse of zero modulo p")
    return power(a, p - 2, p)


def matmul(left: np.ndarray, right: np.ndarray, p: int) -> np.ndarray:
    """Exact ``left @ right mod p`` using float64 BLAS on exactly representable chunks."""
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError(f"incompatible shapes {left.shape} and {right.shape}")
    if left.shape[1] == 0:
        return np.zeros((left.shape[0], right.shape[1]), dtype=np.int64)
    if (p - 1) ** 2 < _FLOAT_EXACT:
        return _float_matmul_mod(left, right, p, (p - 1) ** 2)

    # Large moduli: split both operands into 16-bit limbs.
    mask = (1 << _LIMB_BITS) - 1
    l0, l1 = left & mask, left >> _LIMB_BITS
    r0, r1 = right & mask, right >> _LIMB_BITS
    bound = mask * mask
    low = _float_matmul_mod(l0, r0, p, bound)
    mid = (_float_matmul_mod(l0, r1, p, bound) + _float_matmul_mod(l1, r0, p, bound)) % p
    high = _float_matmul_mod(l1, r1, p, bound)
    shift = pow(2, _LIMB_BITS, p)
    return ((high * pow(2, 2 * _LIMB_BITS, p)) % p + (mid * shift) % p + low) % p


def _float_matmul_mod(left: np.ndarray, right: np.ndarray, p: int, entry_bound: int) -> np.ndarray:
    inner = left.shape[1]
    chunk = max(1, (_FLOAT_EXACT - 1) // max(entry_bound, 1))
    result = np.zeros((left.shape[0], right.shape[1]), dtype=np.int64)
    for start in range(0, inner, chunk):
        stop = min(start + chunk, inner)
        block = left[:, start:stop].astype(np.float64) @ right[start:stop].astype(np.float64)
        result = (result + np.mod(block.astype(np.int64), p)) % p
    return result

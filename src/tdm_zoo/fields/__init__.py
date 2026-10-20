"""Layer 1: finite fields with a field-agnostic interface."""

from __future__ import annotations

from ..arith.backends import PrimeBackend
from .base import Field
from .extension import ExtensionField
from .prime import PrimeField


def GF(
    p: int, m: int = 1, *, backend: str | PrimeBackend = "auto", modulus: list[int] | None = None
) -> Field:
    """Return F_p (``m == 1``) or F_{p^m}."""
    if m == 1:
        if modulus is not None:
            raise ValueError("modulus is only meaningful for extension fields")
        return PrimeField(p, backend)
    return ExtensionField(p, m, modulus=modulus, backend=backend)


__all__ = ["GF", "ExtensionField", "Field", "PrimeField"]

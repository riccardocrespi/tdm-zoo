"""Secret matrix families (one class per family, registered by name)."""

from __future__ import annotations

from ...fields import Field
from .base import SECRET_FAMILIES, SecretFamily, SecretSample
from .elementary import BlockDiagonalFamily, DenseFamily, DiagonalFamily, WeightedAccumulatorFamily
from .fft import (
    CirculantFamily,
    HankelFamily,
    QuasiCyclicFamily,
    ToeplitzFamily,
    gohberg_semencul_map,
    gohberg_semencul_witness,
)
from .sandwich import SandwichFamily


def make_family(name: str, field: Field, *args, **kwargs) -> SecretFamily:
    """Instantiate a registered family, e.g. ``make_family("toeplitz", F, 8)``."""
    return SECRET_FAMILIES.create(name, field, *args, **kwargs)


__all__ = [
    "SECRET_FAMILIES",
    "BlockDiagonalFamily",
    "CirculantFamily",
    "DenseFamily",
    "DiagonalFamily",
    "HankelFamily",
    "QuasiCyclicFamily",
    "SandwichFamily",
    "SecretFamily",
    "SecretSample",
    "ToeplitzFamily",
    "WeightedAccumulatorFamily",
    "gohberg_semencul_map",
    "gohberg_semencul_witness",
    "make_family",
]

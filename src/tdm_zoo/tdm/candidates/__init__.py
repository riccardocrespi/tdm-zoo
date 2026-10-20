"""Named TDM candidates (registered by name)."""

from __future__ import annotations

from ...fields import Field
from .base import CANDIDATES, Candidate
from .fft_style import FFTStyle
from .layered import Layered
from .luof import LUOF
from .raa import RAA


def make_candidate(name: str, field: Field, **kwargs) -> Candidate:
    """Instantiate a registered candidate, e.g. ``make_candidate("raa", F, k=16, c=3)``."""
    return CANDIDATES.create(name, field, **kwargs)


__all__ = ["CANDIDATES", "LUOF", "RAA", "Candidate", "FFTStyle", "Layered", "make_candidate"]

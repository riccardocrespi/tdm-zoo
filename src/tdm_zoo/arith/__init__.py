"""Layer 0: raw modular arithmetic on int64 arrays and prime-field linear-algebra backends.

Nothing in this layer knows about field objects; it works on canonical
representatives modulo a prime ``p`` (and univariate polynomials over F_p).
"""

from . import modp, polymodp
from .backends import BACKEND_NAMES, PrimeBackend, get_backend

__all__ = ["BACKEND_NAMES", "PrimeBackend", "get_backend", "modp", "polymodp"]

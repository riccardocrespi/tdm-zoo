"""Dense linear algebra modulo a prime, with interchangeable backends."""

from __future__ import annotations

from functools import cache

from .base import PrimeBackend
from .numpy_backend import NumpyBackend

BACKEND_NAMES = ("auto", "numpy", "flint", "torch", "cuda")


@cache
def get_backend(name: str = "auto") -> PrimeBackend:
    """Return a backend by name.

    ``auto`` prefers FLINT and falls back to numpy; ``torch`` uses CUDA when
    available and the CPU otherwise; ``cuda`` requires CUDA.
    """
    name = name.lower()
    if name == "auto":
        try:
            return get_backend("flint")
        except ImportError:  # pragma: no cover - flint is a declared dependency
            return NumpyBackend()
    if name == "numpy":
        return NumpyBackend()
    if name == "flint":
        from .flint_backend import FlintBackend

        return FlintBackend()
    if name in ("torch", "cuda"):
        from .torch_backend import TorchBackend

        return TorchBackend("cuda" if name == "cuda" else None)
    raise ValueError(f"unknown backend {name!r}; expected one of {BACKEND_NAMES}")


__all__ = ["BACKEND_NAMES", "PrimeBackend", "NumpyBackend", "get_backend"]

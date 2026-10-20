"""Record the software environment alongside experiment results."""

from __future__ import annotations

import platform
from functools import cache
from importlib import metadata


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


@cache
def environment() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tdm_zoo": _version("tdm-zoo"),
        "numpy": _version("numpy"),
        "python_flint": _version("python-flint"),
        "torch": _version("torch"),
    }
    try:
        import torch

        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["cuda_device"] = None
    return info

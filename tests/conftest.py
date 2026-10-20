from __future__ import annotations

import numpy as np
import pytest

from tdm_zoo import GF

FIELD_SPECS = [(101, 1), (2**31 - 1, 1), (2, 4), (3, 3), (2, 21)]


def _field_id(spec):
    p, m = spec
    return f"GF({p})" if m == 1 else f"GF({p}^{m})"


@pytest.fixture(params=FIELD_SPECS, ids=_field_id)
def field(request):
    p, m = request.param
    return GF(p, m)


@pytest.fixture(params=[(101, 1), (2, 4)], ids=_field_id)
def small_field(request):
    """Cheaper selection for heavier tests: one prime field, one extension field."""
    p, m = request.param
    return GF(p, m)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()

import numpy as np
import pytest

from tdm_zoo.ops import linalg
from tdm_zoo.tdm.families import (
    SECRET_FAMILIES,
    BlockDiagonalFamily,
    SandwichFamily,
    ToeplitzFamily,
    gohberg_semencul_map,
    gohberg_semencul_witness,
    make_family,
)
from tdm_zoo.tdm.publics import random_matrix

FAMILY_CASES = [
    ("diagonal", (6,), {}),
    ("diagonal", (6,), {"nonzero": False}),
    ("block_diagonal", (6, 3), {}),
    ("block_diagonal", (6, 2), {"invertible": False}),
    ("weighted_accumulator", (7,), {}),
    ("toeplitz", (6,), {}),
    ("toeplitz", (4, 7), {}),
    ("hankel", (6,), {}),
    ("hankel", (5, 3), {}),
    ("circulant", (6,), {}),
    ("quasi_cyclic", (4,), {}),
    ("quasi_cyclic", (4,), {"orientation": "vertical"}),
]


def test_registry_lists_all_families():
    assert {name for name, _, _ in FAMILY_CASES} <= set(SECRET_FAMILIES.names())


@pytest.mark.parametrize("name,args,kwargs", FAMILY_CASES, ids=lambda v: str(v))
def test_family_views_agree(small_field, rng, name, args, kwargs):
    family = make_family(name, small_field, *args, **kwargs)
    for _ in range(3):
        params = family.sample_params(rng)
        assert params.shape == (family.nparams,)
        dense = family.matrix(params)
        assert dense.shape == family.shape
        assert np.array_equal(family.operator(params).to_dense(), dense)
        assert family.param_map.degree == 1
        if getattr(family, "invertible", False) or getattr(family, "nonzero", False):
            assert linalg.is_invertible(small_field, dense)
        g = family.inverse_map
        if g is not None:
            assert np.array_equal(g.evaluate(family.inverse_params(params)), linalg.inv(small_field, dense))


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_gohberg_semencul_symbolic(small_field, rng, k):
    """The expanded degree-2 polynomials of g_T reproduce Toeplitz inverses."""
    family = ToeplitzFamily(small_field, k)
    gmap = gohberg_semencul_map(small_field, k)
    assert gmap.degree == 2 and gmap.nparams == 2 * k
    entries = gmap.entry_polynomials()
    params = family.sample_params(rng)
    witness = gohberg_semencul_witness(small_field, family.matrix(params))
    value = np.array([[poly.evaluate(witness) for poly in row] for row in entries], dtype=np.int64)
    assert np.array_equal(value, linalg.inv(small_field, family.matrix(params)))


def test_sandwich_family(small_field, rng):
    field = small_field
    inner = BlockDiagonalFamily(field, 8, 2, invertible=False)
    left, right = random_matrix(field, 3, 8, rng), random_matrix(field, 8, 3, rng)
    family = SandwichFamily(inner, left=left, right=right)
    params = family.sample_params(rng)
    expected = field.matmul(field.matmul(left.to_dense(), inner.matrix(params)), right.to_dense())
    assert family.shape == (3, 3)
    assert np.array_equal(family.matrix(params), expected)
    assert np.array_equal(family.operator(params).to_dense(), expected)


def test_invalid_shapes(small_field):
    with pytest.raises(ValueError):
        make_family("toeplitz", small_field, 3, 4, invertible=True)
    with pytest.raises(ValueError):
        make_family("block_diagonal", small_field, 5, 2)

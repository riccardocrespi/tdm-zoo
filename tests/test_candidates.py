import numpy as np
import pytest

from tdm_zoo import make_candidate
from tdm_zoo.ops import linalg
from tdm_zoo.tdm import CANDIDATES, LayeredProduct
from tdm_zoo.tdm.families import DiagonalFamily
from tdm_zoo.tdm.publics import random_matrix

CASES = [
    ("fft_style", {"k": 6}),
    ("fft_style", {"k": 6, "secret": "hankel", "outer_permutations": True}),
    ("fft_style", {"k": 6, "secret": "circulant", "ell": 4}),
    ("fft_style", {"k": 6, "secret": "quasi_cyclic"}),
    ("fft_style", {"k": 6, "secret": "quasi_cyclic", "ell": 2}),
    ("fft_style", {"k": 6, "secret": ["toeplitz", "hankel", "circulant"]}),
    ("raa", {"k": 5}),
    ("raa", {"k": 5, "view": "diagonal", "ell": 4}),
    ("raa", {"k": 5, "c": 2, "outer_permutations": False}),
    ("raa", {"k": 5, "secret_permutations": True}),
    ("luof", {"k": 6}),
    ("luof", {"k": 6, "view": "composite"}),
    ("luof", {"k": 6, "codes": "random", "ell": 2}),
    ("luof", {"k": 6, "shared_codes": False, "outer_permutations": True}),
    ("layered", {"k": 6, "families": ["toeplitz", ("block_diagonal", {"block": 3}), {"family": "hankel"}]}),
    (
        "layered",
        {"k": 6, "families": [("block_diagonal", {"block": 2})] * 3, "publics": "random", "outer": "random"},
    ),
]


def test_registry():
    assert set(CANDIDATES.names()) == {"fft_style", "raa", "luof", "layered", "layered_tdm"}


@pytest.mark.parametrize("name,kwargs", CASES, ids=lambda v: str(v))
def test_candidate_samples(small_field, rng, name, kwargs):
    candidate = make_candidate(name, small_field, **kwargs)
    product = candidate.sample_product(rng)
    assert product.ell == candidate.ell
    assert product.shape == (candidate.k, candidate.k)
    sample = product.sample(rng)
    circuit = sample.evaluation_circuit
    assert np.array_equal(circuit.to_dense(), sample.matrix)
    v = small_field.random(rng, candidate.k)
    assert np.array_equal(sample.apply(v), small_field.matmul(sample.matrix, v[:, None])[:, 0])
    assert len(sample.trapdoor) == candidate.ell
    assert product.sample_uniform(rng).shape == sample.matrix.shape
    assert candidate.describe()["candidate"] == name


def test_public_view_hides_only_what_it_should(small_field, rng):
    hidden = (
        make_candidate("raa", small_field, k=4, secret_permutations=True).sample_product(rng).public_view()
    )
    assert all(p is None for p in hidden.publics) and len(hidden.secret_families) == 3
    shown = make_candidate("raa", small_field, k=4).sample_product(rng).public_view()
    assert all(p is not None for p in shown.publics)
    assert shown.public_matrix(0).shape == (4, 12)


@pytest.mark.parametrize(
    "name,a,b",
    [
        ("raa", {"view": "accumulator"}, {"view": "diagonal"}),
        ("luof", {"view": "split"}, {"view": "composite"}),
    ],
)
def test_equivalent_views_give_identical_matrices(small_field, name, a, b):
    """The two views consume randomness identically, so equal seeds give equal M."""
    for seed in range(3):
        matrices = []
        for view in (a, b):
            rng = np.random.default_rng(seed)
            candidate = make_candidate(name, small_field, k=5, **view)
            matrices.append(candidate.sample_product(rng).sample(rng).matrix)
        assert np.array_equal(*matrices)


def test_structured_products_are_mostly_invertible(small_field, rng):
    candidate = make_candidate("fft_style", small_field, k=6)
    ranks = [linalg.rank(small_field, candidate.sample_product(rng).sample(rng).matrix) for _ in range(5)]
    assert all(rank == 6 for rank in ranks)


def test_layered_product_validates_shapes(small_field, rng):
    with pytest.raises(ValueError):
        LayeredProduct(
            small_field,
            [random_matrix(small_field, 3, 4, rng), random_matrix(small_field, 5, 3, rng)],
            [DiagonalFamily(small_field, 4)],
        )

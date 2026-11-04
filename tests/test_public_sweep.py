import numpy as np
import pytest

from tdm_zoo import GF, make_candidate
from tdm_zoo.experiments.public_sweep import composite_relations, public_point
from tdm_zoo.ops import linalg
from tdm_zoo.tdm.di14 import DI14Params, sample_luof_codes
from tdm_zoo.tdm.families import BlockDiagonalFamily
from tdm_zoo.tdm.publics import luof_matrix, luof_span_dimension

P = 2**31 - 1


def test_luof_matrix_is_extractor_h_encoder():
    field = GF(P)
    codes = sample_luof_codes(field, 4, np.random.default_rng(0))
    matrix = luof_matrix(field, codes, np.random.default_rng(1)).to_dense()
    family = BlockDiagonalFamily(field, codes.inner_dimension, codes.block_size, invertible=False)
    h = family.sample(np.random.default_rng(1)).matrix()
    expected = field.matmul(field.matmul(codes.extractor.to_dense(), h), codes.encoder.to_dense())
    assert np.array_equal(matrix, expected)


@pytest.mark.parametrize("seed", range(4))
def test_luof_span_matches_basis_of_h(seed):
    field = GF(P)
    codes = sample_luof_codes(field, 4, np.random.default_rng(seed), DI14Params(8, 2, 2))
    family = BlockDiagonalFamily(field, codes.inner_dimension, codes.block_size, invertible=False)
    left, right = codes.extractor.to_dense(), codes.encoder.to_dense()
    basis = []
    for index in range(family.nparams):
        params = np.zeros(family.nparams, dtype=np.int64)
        params[index] = 1
        basis.append(field.matmul(field.matmul(left, family.matrix(params)), right).ravel())
    assert luof_span_dimension(field, codes) == linalg.rank(field, np.array(basis))


def test_layered_tdm_luof_publics_are_invertible_and_shared():
    field = GF(P)
    candidate = make_candidate("layered_tdm", field, k=4, ell=3, publics="luof")
    product = candidate.sample_product(np.random.default_rng(2))
    dense = [layer.to_dense() for layer in product.publics]
    assert all(linalg.is_invertible(field, a) for a in dense)
    assert not np.array_equal(dense[0], dense[1])  # fresh H per layer
    assert product.metadata["luof_span"] <= 16 and product.metadata["shared_codes"]


def test_unknown_publics_rejected():
    with pytest.raises(ValueError):
        make_candidate("layered_tdm", GF(P), k=4, publics="identity")


@pytest.mark.parametrize("publics", ["random_invertible", "luof"])
def test_public_point_witness(publics):
    row = public_point("diagonal", 4, P, publics, ell=4, seed=1)
    assert row.status == "ok" and (row.rank_M, row.rank_U, row.gamma) == (367, 368, 1)
    assert row.witness_in_kernel and row.witness_not_in_uniform_kernel


def test_rank_reacts_to_non_generic_publics():
    assert public_point("diagonal", 4, P, "permutation", seed=0).rank_M == 0
    assert public_point("diagonal", 4, P, zeros=1, seed=0).rank_M != 367


def test_composite_family_with_full_image_has_no_relations():
    row = composite_relations(4, P, seed=1)
    assert row["family_image"] == 16 and row["K"] == 0

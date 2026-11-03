import numpy as np
import pytest

from tdm_zoo import GF, make_candidate
from tdm_zoo.attacks import DISTINGUISHERS, PeelingMacaulay
from tdm_zoo.attacks.peeling import PeelingSystem, core_map_values
from tdm_zoo.experiments import run_paired
from tdm_zoo.ops import linalg
from tdm_zoo.ops.multihomogeneous import BlockBasis, ProductBasis
from tdm_zoo.tdm.families import HankelFamily, ToeplitzFamily

from .conftest import torch_cuda_available

FIELD = GF(2**31 - 1)


def _setup(n, ell, peel="diagonal", seed=0, field=FIELD):
    rng = np.random.default_rng(seed)
    product = make_candidate("layered_tdm", field, k=n, ell=ell, peel_family=peel).sample_product(rng)
    system = PeelingSystem(product.public_view(), 3, rng)
    return rng, product, system


# --- candidate -------------------------------------------------------------------------------
def test_layered_tdm_candidate(rng):
    candidate = make_candidate("layered_tdm", FIELD, k=4, ell=5, peel_family="circulant")
    product = candidate.sample_product(rng)
    assert [s.name for s in product.secrets] == ["diagonal"] * 2 + ["circulant"] * 3
    assert all(linalg.is_invertible(FIELD, a.to_dense()) for a in product.publics)
    sample = product.sample(rng)
    assert np.array_equal(sample.evaluation_circuit.to_dense(), sample.matrix)
    with pytest.raises(ValueError):
        make_candidate("layered_tdm", FIELD, k=4, ell=2)
    bad = make_candidate(
        "layered_tdm", FIELD, k=4, ell=3, peel_family=("block_diagonal", {"block": 2, "invertible": False})
    )
    with pytest.raises(ValueError):
        bad.sample_product(rng)


@pytest.mark.parametrize("family_cls", [ToeplitzFamily, HankelFamily])
def test_homogenized_gohberg_semencul(rng, family_cls):
    family = family_cls(FIELD, 3)
    g = family.homogeneous_inverse_map
    assert g.degree == 2 and g.nparams == 7
    assert all(poly.is_homogeneous() and poly.degree == 2 for row in g.entry_polynomials() for poly in row)
    params = family.sample_params(rng)
    witness = family.homogeneous_inverse_params(params)
    assert witness[-1] == 1
    assert np.array_equal(g.evaluate(witness), linalg.inv(FIELD, family.matrix(params)))


# --- multi-homogeneous bases -------------------------------------------------------------------
def test_block_basis_multiplication_table():
    a, b = BlockBasis(3, 1), BlockBasis(3, 2)
    table = a.multiplication_table(b)
    target = BlockBasis(3, 3).monomials
    for i, ma in enumerate(a.monomials):
        for j, mb in enumerate(b.monomials):
            assert target[table[i, j]] == tuple(sorted(ma + mb))


def test_product_basis_evaluation_order(rng):
    basis = ProductBasis((2, 3), (1, 2))
    points = [FIELD.random(rng, 2), FIELD.random(rng, 3)]
    first = BlockBasis(2, 1).evaluate(FIELD, points[0])[0]
    second = BlockBasis(3, 2).evaluate(FIELD, points[1])[0]
    expected = FIELD.mul(first[:, None], second[None, :]).ravel()  # block 0 most significant
    assert np.array_equal(basis.evaluate(FIELD, points), expected)


# --- system ---------------------------------------------------------------------------------------
def test_base_relations_count_and_annihilation():
    rng, product, system = _setup(3, 3)
    assert system.relation_count == 65  # N(9,3) - N(3,3)^2 for diagonal base layers
    publics = [product.publics[i].to_dense() for i in range(3)]
    values = core_map_values(
        FIELD, publics, product.secrets[0].param_map, product.secrets[1].param_map, rng, 10
    )
    assert not np.any(system.relations.evaluate(FIELD, values))


CASES = [
    (3, 3, "diagonal"),
    (3, 4, "diagonal"),
    (3, 4, "circulant"),
    (3, 4, "weighted_accumulator"),
    (3, 3, "toeplitz"),
]


@pytest.mark.parametrize("n,ell,peel", CASES)
def test_composition_and_witness(n, ell, peel):
    rng, product, system = _setup(n, ell, peel)
    sample = product.sample(rng)
    coefficients = system.compose(sample.matrix)
    assert coefficients.shape == (system.relation_count, system.base.size)
    point = [FIELD.random(rng, block.nvars) for block in system.blocks]
    direct = system.relations.evaluate(FIELD, system.residual(sample.matrix, point)[None])[0]
    assert np.array_equal(
        FIELD.matmul(coefficients, system.base.evaluate(FIELD, point)[:, None])[:, 0], direct
    )
    witness = system.witness(sample)
    assert not np.any(system.relations.evaluate(FIELD, system.residual(sample.matrix, witness)[None]))
    for multipliers in (0, 1):
        if peel == "toeplitz" and multipliers:
            continue
        matrix = system.macaulay(sample.matrix, multipliers)
        vector = system.witness_vector(witness, multipliers)
        assert not np.any(FIELD.matmul(matrix, vector[:, None]))


def test_macaulay_rows_are_multiples(rng):
    rng, product, system = _setup(3, 4, seed=3)
    matrix_in = product.sample_uniform(rng)
    shape = system.shape((1, 0))
    assert shape.multipliers.degrees == (1, 0) and shape.columns.degrees == (4, 3)
    matrix = system.macaulay(matrix_in, (1, 0))
    assert matrix.shape == (shape.rows, shape.cols) == (65 * 3, 15 * 10)
    point = [FIELD.random(rng, block.nvars) for block in system.blocks]
    rows = FIELD.matmul(matrix, shape.columns.evaluate(FIELD, point)[:, None])[:, 0]
    p_values = FIELD.matmul(system.compose(matrix_in), system.base.evaluate(FIELD, point)[:, None])[:, 0]
    m_values = shape.multipliers.evaluate(FIELD, point)
    assert np.array_equal(
        rows.reshape(-1, system.relation_count), FIELD.mul(m_values[:, None], p_values[None, :])
    )


def test_macaulay_size_guard():
    rng, product, system = _setup(3, 4)
    with pytest.raises(MemoryError):
        system.macaulay(product.sample_uniform(rng), 2, limit=1000)


def test_extension_field_witness():
    field = GF(2, 20)
    rng, product, system = _setup(3, 3, field=field)
    sample = product.sample(rng)
    matrix = system.macaulay(sample.matrix)
    vector = system.witness_vector(system.witness(sample))
    assert not np.any(field.matmul(matrix, vector[:, None]))


# --- distinguisher --------------------------------------------------------------------------------
def test_distinguisher_registered_and_wins_small_case():
    assert "peeling_macaulay" in DISTINGUISHERS
    candidate = make_candidate("layered_tdm", FIELD, k=3, ell=3)
    result = run_paired(candidate, PeelingMacaulay(d=3, seed=0), trials=4, seed=0)
    assert result.advantage == 1.0
    info = result.records[0].info
    assert info["cols"] == 10 and info["rank"] < info["reference_rank"]


def test_distinguisher_four_layers_control():
    candidate = make_candidate("layered_tdm", FIELD, k=4, ell=4)
    result = run_paired(candidate, PeelingMacaulay(d=3, reference="cached", seed=1), trials=2, seed=2)
    assert result.advantage == 1.0


# --- sparse / implicit forms and black-box rank --------------------------------------------------
@pytest.mark.parametrize("n,ell,multipliers", [(3, 3, 1), (3, 4, (1, 0)), (3, 4, 1)])
def test_macaulay_forms_agree(n, ell, multipliers):
    rng, product, system = _setup(n, ell, seed=4)
    matrix_in = product.sample(rng).matrix
    dense = system.macaulay(matrix_in, multipliers, "dense")
    sparse = system.macaulay(matrix_in, multipliers, "sparse")
    implicit = system.macaulay(matrix_in, multipliers, "operator")
    assert np.array_equal(sparse.to_dense(), dense)
    assert sparse.nonzeros == implicit.nonzeros == np.count_nonzero(dense)
    x, y = FIELD.random(rng, (dense.shape[1], 2)), FIELD.random(rng, (dense.shape[0], 2))
    assert np.array_equal(implicit.apply(x), FIELD.matmul(dense, x))
    assert np.array_equal(implicit.apply_T(y), FIELD.matmul(dense.T, y))


def test_guards_count_nonzeros_for_sparse_and_cells_for_dense():
    rng, product, system = _setup(3, 4)
    matrix_in = product.sample_uniform(rng)
    count = system.macaulay(matrix_in, 1, "operator").nonzeros
    assert count < system.shape(1).entries
    assert system.macaulay(matrix_in, 1, "sparse", limit=count).nonzeros == count
    with pytest.raises(MemoryError):
        system.macaulay(matrix_in, 1, "sparse", limit=count - 1)
    with pytest.raises(MemoryError):
        system.macaulay(matrix_in, 1, "dense", limit=count)  # fits as sparse, not as dense


@pytest.mark.parametrize("n,ell,multipliers", [(3, 3, 0), (4, 4, 0), (3, 4, 1)])
def test_wiedemann_rank_matches_dense(n, ell, multipliers):
    rng, product, system = _setup(n, ell, seed=6)
    for matrix_in in (product.sample(rng).matrix, product.sample_uniform(rng)):
        exact = system.macaulay_rank(matrix_in, multipliers, "dense")
        assert system.macaulay_rank(matrix_in, multipliers, "wiedemann", rng) == exact


@pytest.mark.skipif(not torch_cuda_available(), reason="needs CUDA")
def test_cuda_operator_matches_cpu():
    rng, product, system = _setup(3, 4, seed=7)
    matrix_in = product.sample(rng).matrix
    cpu = system.macaulay(matrix_in, 1, "operator")
    gpu = system.macaulay(matrix_in, 1, "operator", device="cuda")
    x, y = FIELD.random(rng, (cpu.cols, 2)), FIELD.random(rng, (cpu.rows, 2))
    assert np.array_equal(cpu.apply(x), gpu.apply(x))
    assert np.array_equal(cpu.apply_T(y), gpu.apply_T(y))


def test_distinguisher_with_wiedemann():
    candidate = make_candidate("layered_tdm", FIELD, k=4, ell=4)
    result = run_paired(candidate, PeelingMacaulay(rank_method="wiedemann", seed=3), trials=2, seed=4)
    assert result.advantage == 1.0

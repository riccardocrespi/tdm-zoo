import numpy as np
import pytest

from tdm_zoo import GF, make_candidate
from tdm_zoo.experiments.luof_structure import (
    PARAMS,
    block_bound,
    block_pattern,
    direct_dimension,
    distinguisher_point,
    exact_span,
    fast_sample_count,
    fast_sampler,
    fast_span_point,
    gram_dimension,
    pattern_point,
    pattern_span,
    phase_check_point,
    phase_point,
    sample_count,
    span_point,
    structure_point,
    subspace_point,
)
from tdm_zoo.ops import linalg
from tdm_zoo.tdm.di14 import sample_luof_codes
from tdm_zoo.tdm.publics import luof_span_dimension

P = 2**31 - 1
Q = 10**9 + 7
SPARSE = {"block_size": 1, "base_length": 12, "expander_degree": 1}  # ell = 2, 3 proper at k = 32
HIGH_DEGREE = {"block_size": 1, "base_length": 12, "expander_degree": 12}  # saturated graphs
TINY_PROPER = {"block_size": 1, "base_length": 12, "expander_degree": 1}  # codim 5 at k = 16
SPARSER = {"block_size": 1, "base_length": 4, "expander_degree": 1}  # ell = 2 proper, ell = 3 full


def test_split_view_bottleneck_is_public():
    row = structure_point(16, "split", P)
    assert row.N == 256 and row.M_shape == (16, 16) and row.M_rank == 16
    assert row.publics == [
        ["A1", (16, 256), 16],
        ["A2", (256, 256), 16],
        ["A3", (256, 256), 16],
        ["A4", (256, 16), 16],
    ]
    assert all(s[1:] == ["block_diagonal", (256, 256), 1024] for s in row.secrets)


def test_composite_view_has_square_publics():
    row = structure_point(16, "composite", P)
    assert row.N == 256 and row.M_shape == (16, 16) and row.M_rank == 16
    assert all(p[1:] == [(16, 16), 16] for p in row.publics)
    assert all(s[1:] == ["sandwich", (16, 16), 1024] for s in row.secrets)


def test_span_dimension_is_the_explicit_outer_product_stack():
    field = GF(P)
    codes = sample_luof_codes(field, 16, np.random.default_rng(0), PARAMS, "di14")
    enc, ext = codes.encoder.to_dense(), codes.extractor.to_dense()
    columns = [
        field.mul(ext[:, 4 * b + i][:, None], enc[4 * b + j, :][None, :]).ravel()
        for b in range(enc.shape[0] // 4)
        for i in range(4)
        for j in range(4)
    ]
    assert linalg.rank(field, np.stack(columns, axis=1)) == luof_span_dimension(field, codes)


@pytest.mark.parametrize("k, dim_v, codim", [(16, 256, 0), (32, 891, 133)])
def test_subspace_dimension_at_seed_zero(k, dim_v, codim):
    row = subspace_point(k, P, seed=0)
    assert (row.dim_V, row.codim) == (dim_v, codim)


def test_samples_exceed_the_ambient_dimension():
    assert sample_count(16) == 340 and sample_count(64) == 5140
    assert all(sample_count(k) > k * k for k in (16, 32, 48, 64))


@pytest.mark.parametrize("prime", [P, Q])
@pytest.mark.parametrize("ell", [1, 2, 3])
def test_full_span_at_k16(ell, prime):
    row = span_point(16, ell, prime)
    assert row.span == 256 and not row.distinguisher and row.relations == []


@pytest.mark.slow
def test_single_layer_span_is_v_and_gives_working_relations():
    row = span_point(32, 1, P, seed=0)
    assert row.span == subspace_point(32, P, seed=0).dim_V == 891  # same seed, same codes
    assert row.distinguisher and len(row.relations) == row.codim > 0
    assert row.relations_hold_on_fresh and row.relations_reject_uniform


@pytest.mark.slow
def test_product_fills_the_space_at_k32():
    row = span_point(32, 2, P)
    assert row.span == 1024 and not row.distinguisher


def composite(k, ell, seed=0, field=None, **options):
    field = field or GF(P)
    rng = np.random.default_rng(seed)
    return make_candidate("luof", field, k=k, ell=ell, view="composite", **options).sample_product(rng), rng


@pytest.mark.parametrize("k", [16, 32])
def test_direct_and_gram_dimension_match_the_map_rank(k):
    field = GF(P)
    rng = np.random.default_rng(1)
    codes = sample_luof_codes(field, k, rng, PARAMS, "di14")
    exact = luof_span_dimension(field, codes)
    assert direct_dimension(field, codes) == exact
    assert gram_dimension(field, codes, rng) == exact
    assert exact <= block_bound(field, codes) <= codes.inner_dimension * codes.block_size


def test_subspace_row_carries_bounds_and_method():
    row = subspace_point(16, P, method="gram")
    assert row.method == "gram" and row.dim_V == 256 and row.params == 1024 and row.block_bound >= 256


def test_fast_layer_lies_in_v_and_covers_it():
    product, rng = composite(16, 1)
    field = product.field
    fast = fast_sampler(product)
    slow = np.stack([product.sample(rng).matrix.ravel() for _ in range(300)])
    joint = np.vstack([slow, np.stack([fast(rng).ravel() for _ in range(300)])])
    assert linalg.rank(field, slow) == linalg.rank(field, joint) == 256


def test_fast_sampler_uses_the_candidate_publics():
    product, rng = composite(16, 3)
    field = product.field
    secrets = [family.sample(rng) for family in product.secrets]
    dense = [public.to_dense() for public in product.publics]
    expected = dense[0]
    for index, secret in enumerate(secrets):
        expected = field.matmul(field.matmul(expected, secret.matrix()), dense[index + 1])
    assert np.array_equal(product.matrix(secrets), expected)
    assert fast_sampler(product)(rng).shape == (16, 16)


@pytest.mark.parametrize("ell", [1, 2, 3])
def test_fast_path_equals_slow_path_at_k16(ell):
    assert fast_span_point(16, ell, P).span == span_point(16, ell, P).span


@pytest.mark.slow
@pytest.mark.parametrize("ell, span", [(1, 891), (2, 1024)])
def test_fast_path_equals_slow_path_at_k32(ell, span):
    assert fast_span_point(32, ell, P).span == span_point(32, ell, P).span == span


def test_fast_span_detects_a_proper_ell_2_product_and_recovers_its_relations():
    row = fast_span_point(32, 2, P, **SPARSE)
    assert row.distinguisher and row.span == 922 and len(row.relations) == row.codim == 102
    assert row.relations_hold_on_fresh and row.relations_reject_uniform


def test_depth_restores_one_sparse_control_and_only_partly_the_other():
    assert fast_span_point(32, 2, P, **SPARSER).span == 958
    assert fast_span_point(32, 3, P, **SPARSER).span == 1024
    assert fast_span_point(32, 3, P, **SPARSE).span == 930


@pytest.mark.parametrize(
    "k, ell, options",
    [(16, 2, {}), (16, 3, {}), (32, 2, SPARSE), (32, 3, SPARSE), (32, 2, SPARSER), (32, 3, SPARSER)],
)
def test_pattern_formula_is_the_sampled_span(k, ell, options):
    product, _ = composite(k, ell, **options)
    assert pattern_span(product) == fast_span_point(k, ell, P, **options).span


@pytest.mark.parametrize(
    "k, ell, options",
    [(16, 2, {}), (16, 2, HIGH_DEGREE), (32, 2, SPARSE), (32, 3, SPARSE), (32, 2, SPARSER)],
)
def test_exact_span_is_the_pattern_formula(k, ell, options):
    product, _ = composite(k, ell, **options)
    assert exact_span(product, block_pattern(product)) == pattern_span(product)


def test_pattern_point_reports_full_and_proper():
    full, proper = pattern_point(32, 2, P), pattern_point(32, 2, P, **SPARSE)
    assert full.certified_full and full.codim == 0
    assert not proper.certified_full and (proper.span, proper.codim) == (922, 102)


def test_exact_span_matches_sampling_in_the_high_degree_pocket():
    row = phase_check_point(16, 2, P, **HIGH_DEGREE)
    assert row["agree"] and row["exact"] == 214


def test_phase_point_records_a_degenerate_cell():
    row = phase_point(16, 2, P, 0, 1, 1, 6)  # base_length must be a multiple of 4
    assert "degenerate" in row and row["k"] == 16


def test_span_test_breaks_a_proper_cell():
    row = distinguisher_point(16, 2, P, trials=40, **TINY_PROPER)
    assert row.relations_found == 256 - row.span == 5
    assert row.structured_accept == 1.0 and row.uniform_accept == 0.0 and row.advantage == 1.0


def test_span_test_has_no_advantage_on_a_full_cell():
    row = distinguisher_point(16, 2, P, trials=20)
    assert row.relations_found == 0 and row.advantage == 0.0


def test_dead_blocks_are_the_systematic_rows_beyond_k():
    row = pattern_point(64, 2, P)
    assert row.blocks - row.live_blocks == (row.N // 4 - 64) // 4


def test_fast_sample_count_always_exceeds_the_ambient_dimension():
    assert fast_sample_count(64) == sample_count(64) and fast_sample_count(128) == 128 * 128 + 64

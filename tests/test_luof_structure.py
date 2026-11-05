import numpy as np
import pytest

from tdm_zoo import GF
from tdm_zoo.experiments.luof_structure import (
    PARAMS,
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

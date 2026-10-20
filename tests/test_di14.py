import itertools

import numpy as np
import pytest

from tdm_zoo import GF
from tdm_zoo.ops import linalg
from tdm_zoo.ops.structured import BlockDiagonalOperator
from tdm_zoo.tdm.di14 import (
    DI14Params,
    SpielmanCode,
    di14_dimension,
    di14_encoder,
    di14_extractor,
    random_regular_bipartite,
    sample_luof_codes,
)


def _min_weight(field, generator):
    """Minimum Hamming weight of nonzero codewords (brute force, tiny codes only)."""
    best = generator.shape[0]
    for message in itertools.product(range(field.q), repeat=generator.shape[1]):
        if any(message):
            word = field.matmul(generator, np.array(message)[:, None])[:, 0]
            best = min(best, int(np.count_nonzero(word)))
    return best


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_spielman_code_shapes_rank_and_transpose(small_field, rng, level):
    code = SpielmanCode(small_field, level, DI14Params(), rng)
    dense = code.to_dense()
    assert dense.shape == (4 * code.cols, code.cols)
    assert linalg.rank(small_field, dense) == code.cols  # injective
    assert np.array_equal(code.T.to_dense(), dense.T)
    if level >= 1:
        assert np.array_equal(dense[: code.cols], small_field.eye(code.cols))  # systematic


def test_spielman_code_has_positive_distance(rng):
    field = GF(3)
    code = SpielmanCode(field, 1, DI14Params(base_length=8, expander_degree=3), rng)
    assert _min_weight(field, code.to_dense()) >= 2


def test_regular_bipartite_graph_is_regular(rng):
    graph = random_regular_bipartite(20, 4, rng)
    assert graph.shape == (20, 4)
    assert np.all(np.bincount(graph.ravel(), minlength=20) == 4)  # left degrees


def test_encoder_extractor_dimensions(small_field, rng):
    params = DI14Params(block_size=4)
    for k in (3, 8, 13):
        n = di14_dimension(k, params)
        enc = di14_encoder(small_field, k, params, rng)
        ext = di14_extractor(small_field, n, k, params, rng)
        assert enc.shape == (n, k) and ext.shape == (k, n) and n % params.block_size == 0
        assert linalg.rank(small_field, enc.to_dense()) == k
        assert linalg.rank(small_field, ext.to_dense()) == k
        assert np.array_equal(enc.T.to_dense(), enc.to_dense().T)


@pytest.mark.parametrize("construction", ["di14", "random"])
def test_luof_codes(small_field, rng, construction):
    codes = sample_luof_codes(small_field, 6, rng, construction=construction)
    assert codes.encoder.shape == (codes.inner_dimension, 6)
    assert codes.extractor.shape == (6, codes.inner_dimension)


def test_luof_uniform_output_statistics():
    """For a fixed nonzero input, A_r(x) should look uniform over the choice of H_r."""
    field = GF(3)
    rng = np.random.default_rng(7)
    codes = sample_luof_codes(field, 2, rng, DI14Params(base_length=8, expander_degree=4, block_size=2))
    x = np.array([1, 2])
    encoded = codes.encoder.apply(x)
    counts = np.zeros((field.q,) * 2, dtype=np.int64)
    trials = 2700
    for _ in range(trials):
        blocks = field.random(rng, (codes.inner_dimension // 2, 2, 2))
        y = codes.extractor.apply(BlockDiagonalOperator(field, blocks).apply(encoded))
        counts[tuple(y)] += 1
    expected = trials / counts.size
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 30  # 8 degrees of freedom; p ~ 2e-4 under uniformity

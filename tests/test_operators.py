import numpy as np
import pytest

from tdm_zoo.ops import linalg, structured
from tdm_zoo.ops.operators import (
    ComposedOperator,
    DenseOperator,
    DiagonalOperator,
    HStackOperator,
    IdentityOperator,
    PermutationOperator,
    SelectionOperator,
    SparseOperator,
    VStackOperator,
)


def _operators(field, rng):
    nz = field.random_nonzero
    yield DenseOperator(field, field.random(rng, (4, 6)))
    yield IdentityOperator(field, 5)
    yield PermutationOperator(field, rng.permutation(7))
    yield SelectionOperator(field, [0, 2, 2, 5, 1], 6)
    yield SparseOperator(field, [0, 0, 3, 2], [1, 4, 4, 0], field.random(rng, 4), (4, 5))
    yield DiagonalOperator(field, nz(rng, 5))
    yield HStackOperator(
        [DenseOperator(field, field.random(rng, (3, 2))), DiagonalOperator(field, nz(rng, 3))]
    )
    yield VStackOperator(
        [DenseOperator(field, field.random(rng, (2, 3))), DiagonalOperator(field, nz(rng, 3))]
    )
    yield structured.BlockDiagonalOperator(field, field.random(rng, (3, 2, 2)))
    yield structured.ToeplitzOperator(field, field.random(rng, 9), 4, 6)
    yield structured.ToeplitzOperator(field, field.random(rng, 150), 70, 81)  # flint convolution path
    yield structured.hankel_operator(field, field.random(rng, 8), 5, 4)
    yield structured.CirculantOperator(field, field.random(rng, 6))
    yield structured.AccumulatorOperator(field, field.random(rng, 6))
    yield structured.repetition(field, 3, 2)
    yield structured.block_sum(field, 3, 2)
    yield structured.difference(field, 5)


def test_apply_transpose_and_batches(field, rng):
    for op in _operators(field, rng):
        dense = op.to_dense()
        assert dense.shape == op.shape, op
        x = field.random(rng, (op.cols, 3))
        assert np.array_equal(op.apply(x), field.matmul(dense, x)), op
        assert np.array_equal(op.apply(x[:, 0]), field.matmul(dense, x[:, :1])[:, 0]), op
        assert np.array_equal(op.T.to_dense(), dense.T), op
        y = field.random(rng, op.rows)
        assert np.array_equal(op.apply_T(y), field.matmul(dense.T, y[:, None])[:, 0]), op


def test_composition(small_field, rng):
    field = small_field
    a = DenseOperator(field, field.random(rng, (3, 4)))
    b = structured.ToeplitzOperator(field, field.random(rng, 8), 4, 5)
    c = PermutationOperator(field, rng.permutation(5))
    composed = a @ b @ c
    assert isinstance(composed, ComposedOperator) and len(composed.factors) == 3
    expected = field.matmul(field.matmul(a.to_dense(), b.to_dense()), c.to_dense())
    assert np.array_equal(composed.to_dense(), expected)
    assert composed.cost == a.cost + b.cost + c.cost
    with pytest.raises(ValueError):
        c @ a


def test_inverses(small_field, rng):
    field = small_field
    ops = [
        PermutationOperator(field, rng.permutation(6)),
        DiagonalOperator(field, field.random_nonzero(rng, 6)),
        structured.AccumulatorOperator(field, field.random_nonzero(rng, 6)),
        structured.plain_accumulator(field, 6),
    ]
    circulant = structured.CirculantOperator(field, field.random(rng, 6))
    if linalg.is_invertible(field, circulant.to_dense()):
        ops.append(circulant)
    for op in ops:
        assert np.array_equal(field.matmul(op.inverse().to_dense(), op.to_dense()), field.eye(6)), op
        assert np.array_equal(field.matmul(op.T.inverse().to_dense(), op.T.to_dense()), field.eye(6)), op


def test_permutation_convention(small_field):
    perm = np.array([2, 0, 1])
    op = PermutationOperator(small_field, perm)
    for a in range(3):
        e = np.zeros(3, dtype=np.int64)
        e[a] = 1
        assert op.apply(e)[perm[a]] == 1  # P e_a = e_{perm[a]}


def test_repetition_and_accumulator_semantics(small_field):
    field = small_field
    assert structured.repetition(field, 2, 3).apply(np.array([4, 5])).tolist() == [4, 4, 4, 5, 5, 5]
    assert np.array_equal(
        structured.plain_accumulator(field, 4).to_dense(), np.tril(np.ones((4, 4), dtype=np.int64))
    )
    x = np.array([1, 2, 3, 4])
    expected = [field.add(x[0], x[1]), field.add(x[2], x[3])]
    assert structured.block_sum(field, 2, 2).apply(x).tolist() == expected

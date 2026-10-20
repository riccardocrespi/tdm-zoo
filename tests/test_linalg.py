import numpy as np
import pytest

from tdm_zoo.ops import linalg


def _random_rank(field, rng, rows, cols, rank):
    return field.matmul(field.random(rng, (rows, rank)), field.random(rng, (rank, cols)))


def test_rank_and_kernels(field, rng):
    a = _random_rank(field, rng, 7, 9, 4)
    assert linalg.rank(field, a) == 4
    right = linalg.nullspace(field, a)
    left = linalg.left_nullspace(field, a)
    assert right.shape == (9, 5) and not np.any(field.matmul(a, right))
    assert left.shape == (3, 7) and not np.any(field.matmul(left, a))
    assert linalg.rank(field, right) == 5


def test_inverse_and_solve(field, rng):
    a = field.random(rng, (6, 6))
    while not linalg.is_invertible(field, a):
        a = field.random(rng, (6, 6))
    assert np.array_equal(field.matmul(a, linalg.inv(field, a)), field.eye(6))
    rhs = field.random(rng, (6, 2))
    assert np.array_equal(field.matmul(a, linalg.solve(field, a, rhs)), rhs)


def test_singular(field, rng):
    a = _random_rank(field, rng, 5, 5, 3)
    assert not linalg.is_invertible(field, a)
    with pytest.raises(np.linalg.LinAlgError):
        linalg.inv(field, a)


def test_generalized_inverse(field, rng):
    for shape, rank in [((6, 4), 3), ((3, 8), 3), ((5, 5), 2), ((4, 4), 4)]:
        a = _random_rank(field, rng, *shape, rank)
        g = linalg.generalized_inverse(field, a)
        assert g.shape == shape[::-1]
        assert np.array_equal(field.matmul(field.matmul(a, g), a), a)

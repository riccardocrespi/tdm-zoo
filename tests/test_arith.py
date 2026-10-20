import numpy as np
import pytest

from tdm_zoo.arith import get_backend, modp, polymodp
from tdm_zoo.arith.backends import NumpyBackend

from .conftest import torch_cuda_available

PRIMES = [2, 3, 101, 65537, 2**31 - 1]


def _object_matmul(a, b, p):
    return (a.astype(object) @ b.astype(object)) % p


@pytest.mark.parametrize("p", PRIMES)
def test_elementwise_ops(p, rng):
    a = rng.integers(0, p, 500)
    b = rng.integers(0, p, 500)
    assert np.array_equal(modp.add(a, b, p), (a + b) % p)
    assert np.array_equal(modp.sub(a, b, p), (a - b) % p)
    assert np.array_equal(modp.mul(a, b, p), (a.astype(object) * b.astype(object)) % p)
    nz = a[a != 0]
    assert np.all(modp.mul(nz, modp.inv(nz, p), p) == 1)


@pytest.mark.parametrize("p", PRIMES)
def test_matmul_exact(p, rng):
    a = rng.integers(0, p, (7, 300))
    b = rng.integers(0, p, (300, 5))
    assert np.array_equal(modp.matmul(a, b, p), _object_matmul(a, b, p))


def test_inverse_of_zero_raises():
    with pytest.raises(ZeroDivisionError):
        modp.inv(np.array([1, 0]), 7)


def test_irreducible_and_reduction_table():
    f = polymodp.first_irreducible(2, 4)
    assert polymodp.is_irreducible(f, 2)
    assert f == [1, 1, 0, 0, 1]  # x^4 + x + 1
    table = polymodp.reduction_table(f, 2, 7)
    assert table[4].tolist() == [1, 1, 0, 0]  # x^4 = x + 1


BACKENDS = ["numpy", "flint"] + (["cuda"] if torch_cuda_available() else []) + ["torch"]


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("p", [3, 101, 2**31 - 1])
def test_backends_agree(name, p, rng):
    backend = get_backend(name)
    reference = NumpyBackend()
    a = rng.integers(0, p, (6, 9))
    a[3] = (a[0] + 2 * a[1]) % p  # force a dependency
    b = rng.integers(0, p, (9, 4))
    assert np.array_equal(backend.matmul(a, b, p), _object_matmul(a, b, p))
    assert backend.rank(a, p) == reference.rank(a, p) == 5
    reduced, pivots = backend.rref(a, p)
    ref_reduced, ref_pivots = reference.rref(a, p)
    assert pivots == ref_pivots and np.array_equal(reduced, ref_reduced)
    kernel = backend.nullspace(a, p)
    assert kernel.shape == (9, 4) and not np.any(_object_matmul(a, kernel, p))
    square = rng.integers(0, p, (5, 5))
    while reference.rank(square, p) < 5:
        square = rng.integers(0, p, (5, 5))
    inverse = backend.inv(square, p)
    assert np.array_equal(_object_matmul(square, inverse, p), np.eye(5, dtype=np.int64))
    rhs = rng.integers(0, p, 5)
    assert np.array_equal(_object_matmul(square, backend.solve(square, rhs, p)[:, None], p)[:, 0], rhs)


@pytest.mark.parametrize("name", BACKENDS)
def test_singular_inverse_raises(name):
    with pytest.raises(np.linalg.LinAlgError):
        get_backend(name).inv(np.array([[1, 2], [2, 4]]), 7)

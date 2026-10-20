import numpy as np
import pytest

from tdm_zoo import GF
from tdm_zoo.fields import ExtensionField


def test_axioms(field, rng):
    a, b, c = (field.random(rng, 300) for _ in range(3))
    zero, one = field.zeros(300), field.ones(300)
    assert np.array_equal(field.add(a, b), field.add(b, a))
    assert np.array_equal(field.mul(a, b), field.mul(b, a))
    assert np.array_equal(field.add(field.add(a, b), c), field.add(a, field.add(b, c)))
    assert np.array_equal(field.mul(field.mul(a, b), c), field.mul(a, field.mul(b, c)))
    assert np.array_equal(field.mul(a, field.add(b, c)), field.add(field.mul(a, b), field.mul(a, c)))
    assert np.array_equal(field.add(a, zero), a) and np.array_equal(field.mul(a, one), a)
    assert np.array_equal(field.add(a, field.neg(a)), zero)
    assert np.array_equal(field.sub(a, b), field.add(a, field.neg(b)))
    nz = field.random_nonzero(rng, 300)
    assert np.all(field.mul(nz, field.inv(nz)) == 1)
    assert np.all((a >= 0) & (a < field.q))


def test_power_and_frobenius(field, rng):
    a = field.random_nonzero(rng, 50)
    assert np.array_equal(field.power(a, 5), field.mul(field.power(a, 2), field.power(a, 3)))
    if field.q < 2**25:
        assert np.all(field.power(a, field.q - 1) == 1)


def test_sums(field, rng):
    a = field.random(rng, (4, 6))
    expected = a[0]
    for row in a[1:]:
        expected = field.add(expected, row)
    assert np.array_equal(field.sum(a, axis=0), expected)
    assert np.array_equal(field.cumsum(a, axis=0)[-1], expected)
    segments = np.array([0, 2, 0, 1])
    out = field.segment_sum(a, segments, 3)
    assert np.array_equal(out[0], field.add(a[0], a[2]))
    assert np.array_equal(out[1], a[3])


def test_matmul_matches_elementwise(field, rng):
    a, b = field.random(rng, (5, 7)), field.random(rng, (7, 3))
    expected = field.zeros((5, 3))
    for t in range(7):
        expected = field.add(expected, field.mul(a[:, t, None], b[None, t, :]))
    assert np.array_equal(field.matmul(a, b), expected)


def test_extension_table_and_polynomial_paths_agree(rng):
    with_tables = ExtensionField(3, 4)
    without = ExtensionField(3, 4, table_limit=0)
    a, b = with_tables.random(rng, 400), with_tables.random_nonzero(rng, 400)
    assert np.array_equal(with_tables.mul(a, b), without.mul(a, b))
    assert np.array_equal(with_tables.inv(b), without.inv(b))


def test_prime_subfield_embedding(rng):
    big, small = GF(5, 3), GF(5)
    a, b = small.random(rng, 100), small.random(rng, 100)
    assert np.array_equal(big.add(a, b), small.add(a, b))
    assert np.array_equal(big.mul(a, b), small.mul(a, b))


def test_validation():
    with pytest.raises(ValueError):
        GF(100)
    with pytest.raises(ValueError):
        GF(2, 4, modulus=[1, 0, 0, 0, 1])  # x^4 + 1 is reducible
    with pytest.raises(ValueError):
        GF(7).asarray([0, 7])
    assert GF(2, 4) == GF(2, 4) and GF(2, 4) != GF(2, 3)

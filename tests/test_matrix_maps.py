import numpy as np

from tdm_zoo.ops.matrix_maps import (
    ConstantMatrixMap,
    LinearMatrixMap,
    ProductSumMatrixMap,
    SandwichMatrixMap,
)
from tdm_zoo.ops.poly import (
    SparsePolynomial,
    homogeneous_monomials,
    monomial_vector,
    monomials_up_to,
    num_monomials,
)


def _evaluate_entries(entries, point):
    return np.array([[poly.evaluate(point) for poly in row] for row in entries], dtype=np.int64)


def test_monomial_counts():
    assert num_monomials(4, 3) == len(homogeneous_monomials(4, 3)) == 20
    assert len(monomials_up_to(3, 2)) == 1 + 3 + 6
    assert homogeneous_monomials(2, 2) == [(0, 0), (0, 1), (1, 1)]


def test_polynomial_arithmetic(small_field, rng):
    field = small_field
    x, y = (SparsePolynomial.variable(field, 2, i) for i in range(2))
    poly = (x + y) * (x - y) + SparsePolynomial.constant(field, 2, 3)
    assert poly.degree == 2 and not poly.is_homogeneous()
    point = field.random(rng, 2)
    xs, ys = int(point[0]), int(point[1])
    expected = field.add(field.sub(field.mul(xs, xs), field.mul(ys, ys)), 3)
    assert poly.evaluate(point) == int(expected)
    values = monomial_vector(field, point, homogeneous_monomials(2, 2))
    assert values.tolist() == [int(field.mul(xs, xs)), int(field.mul(xs, ys)), int(field.mul(ys, ys))]


def test_linear_map_views_agree(small_field, rng):
    field = small_field
    tensor = field.random(rng, (5, 3, 4))
    tensor[tensor % 3 == 0] = 0
    fmap = LinearMatrixMap.from_tensor(field, tensor)
    w = field.random(rng, 5)
    expected = field.sum(field.mul(w[:, None, None], tensor), axis=0)
    assert np.array_equal(fmap.evaluate(w), expected)
    assert np.array_equal(fmap.coefficient_tensor(), tensor)
    assert np.array_equal(_evaluate_entries(fmap.entry_polynomials(), w), expected)


def test_product_sum_and_sandwich(small_field, rng):
    field = small_field
    f1 = LinearMatrixMap.from_tensor(field, field.random(rng, (3, 2, 3)))
    f2 = LinearMatrixMap.from_tensor(field, field.random(rng, (3, 3, 2)))
    const = ConstantMatrixMap(field, field.random(rng, (2, 2)), 3)
    gmap = ProductSumMatrixMap(field, [(1, [f1, f2]), (-1, [const])])
    w = field.random(rng, 3)
    expected = field.sub(field.matmul(f1.evaluate(w), f2.evaluate(w)), const.matrix)
    assert gmap.degree == 2
    assert np.array_equal(gmap.evaluate(w), expected)
    assert np.array_equal(_evaluate_entries(gmap.entry_polynomials(), w), expected)

    left, right = field.random(rng, (4, 2)), field.random(rng, (3, 5))
    sandwich = SandwichMatrixMap(field, left, f1, right)
    expected = field.matmul(field.matmul(left, f1.evaluate(w)), right)
    assert np.array_equal(sandwich.evaluate(w), expected)
    assert np.array_equal(_evaluate_entries(sandwich.entry_polynomials(), w), expected)

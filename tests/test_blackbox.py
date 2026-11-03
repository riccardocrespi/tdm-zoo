import pytest

from tdm_zoo import GF
from tdm_zoo.ops import linalg
from tdm_zoo.ops.blackbox import wiedemann_rank
from tdm_zoo.ops.operators import DenseOperator, SparseOperator

FIELD = GF(2**31 - 1)


@pytest.mark.parametrize(
    "rows,cols,rank", [(50, 40, 40), (50, 40, 17), (40, 50, 23), (60, 60, 60), (60, 60, 59), (30, 30, 0)]
)
def test_wiedemann_matches_dense_rank(rng, rows, cols, rank):
    if rank:
        matrix = FIELD.matmul(FIELD.random(rng, (rows, rank)), FIELD.random(rng, (rank, cols)))
    else:
        matrix = FIELD.zeros((rows, cols))
    assert wiedemann_rank(FIELD, DenseOperator(FIELD, matrix), rng) == linalg.rank(FIELD, matrix) == rank


def test_wiedemann_on_sparse_operator(rng):
    for _ in range(3):
        nnz = 1200
        op = SparseOperator(
            FIELD,
            rng.integers(0, 300, nnz),
            rng.integers(0, 250, nnz),
            FIELD.random_nonzero(rng, nnz),
            (300, 250),
        )
        assert wiedemann_rank(FIELD, op, rng, repeats=2) == linalg.rank(FIELD, op.to_dense())


def test_wiedemann_requires_prime_field(rng):
    field = GF(2, 8)
    with pytest.raises(NotImplementedError):
        wiedemann_rank(field, DenseOperator(field, field.random(rng, (4, 4))), rng)

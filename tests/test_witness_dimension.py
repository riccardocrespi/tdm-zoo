import numpy as np

from tdm_zoo import GF
from tdm_zoo.experiments.witness_dimension import map_differential, witness_point
from tdm_zoo.tdm.families import ToeplitzFamily


def test_diagonal_witness_is_free_and_rigid():
    row = witness_point("diagonal", 3, 2**31 - 1, ell=4)
    assert row.relations_by_degree == {1: 0, 2: 0, 3: 0}
    assert row.effective_unknowns == 3
    assert row.floating == row.blocks and row.genuine_floating == 0


def test_homogenised_gohberg_semencul_differential_rank():
    field = GF(2**31 - 1)
    rng = np.random.default_rng(0)
    family = ToeplitzFamily(field, 4)
    g = family.homogeneous_inverse_map
    point = family.homogeneous_inverse_params(family.sample_params(rng))
    from tdm_zoo.ops import linalg

    assert linalg.rank(field, map_differential(field, g, point)) == 2 * 4 - 1  # 9 vars carry 7 dims

from tdm_zoo.experiments.family_sweep import sweep_point


def test_sweep_point_small_case():
    row = sweep_point("diagonal", 3, 2**31 - 1, ell=3, iterations=5)
    assert row.status == "ok" and row.gamma == 1
    assert row.witness_in_kernel and row.witness_not_in_uniform_kernel
    assert row.minimal and row.s <= row.C and row.notes == []


def test_sweep_point_reports_no_regime_without_building():
    row = sweep_point("toeplitz", 4, 2**31 - 1, ell=4)
    assert row.status.startswith("no regime") and row.C == 9_018_009 and row.rank_M is None

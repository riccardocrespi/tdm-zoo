import numpy as np

from tdm_zoo import GF, make_candidate
from tdm_zoo.attacks import Decision, Distinguisher
from tdm_zoo.experiments import append_jsonl, load_jsonl, run_paired


class AlwaysStructured(Distinguisher):
    name = "always_structured"

    def decide(self, state, matrix, rng):
        return Decision(True)


class CountsPrepare(Distinguisher):
    name = "counts_prepare"

    def __init__(self):
        self.prepared = 0

    def prepare(self, view):
        self.prepared += 1
        return view.shape

    def decide(self, state, matrix, rng):
        assert matrix.shape == state
        return Decision(bool(rng.integers(2)), score=float(matrix[0, 0]))


def test_constant_distinguisher_has_zero_advantage():
    candidate = make_candidate("fft_style", GF(101), k=4)
    result = run_paired(candidate, AlwaysStructured(), trials=5, seed=0)
    assert result.advantage == 0.0
    assert result.structured_accept_rate == result.uniform_accept_rate == 1.0
    assert len(result.records) == 10


def test_prepare_frequency_and_reproducibility():
    candidate = make_candidate("raa", GF(101), k=4)
    once, every = CountsPrepare(), CountsPrepare()
    first = run_paired(candidate, once, trials=4, seed=1)
    run_paired(candidate, every, trials=4, seed=1, fresh_publics=True)
    assert once.prepared == 1 and every.prepared == 4
    second = run_paired(candidate, CountsPrepare(), trials=4, seed=1)
    assert [r.score for r in first.records] == [r.score for r in second.records]


def test_jsonl_roundtrip(tmp_path):
    result = run_paired(make_candidate("fft_style", GF(2, 4), k=4), AlwaysStructured(), trials=2, seed=0)
    path = tmp_path / "out.jsonl"
    append_jsonl(path, result)
    append_jsonl(path, result)
    loaded = load_jsonl(path)
    assert len(loaded) == 2 and loaded[0]["candidate"]["field"] == {
        "p": 2,
        "m": 4,
        "q": 16,
        "modulus": [1, 1, 0, 0, 1],
    }
    assert np.isclose(loaded[0]["advantage"], 0.0)

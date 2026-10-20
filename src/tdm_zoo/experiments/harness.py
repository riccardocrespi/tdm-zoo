"""Paired structured-vs-uniform experiments.

In each trial the harness takes one public product, draws one structured sample
and one independent uniform matrix with the same public layers, and asks the
distinguisher about both. The advantage estimate is
``Pr[guess structured | structured] - Pr[guess structured | uniform]``.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..attacks.base import Distinguisher
from ..tdm.candidates.base import Candidate
from .environment import environment


@dataclass
class TrialRecord:
    trial: int
    sample: str  # "structured" or "uniform"
    guess: bool
    score: float | None
    seconds: float
    info: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    candidate: dict
    distinguisher: dict
    trials: int
    seed: int | None
    fresh_publics: bool
    structured_accept_rate: float
    uniform_accept_rate: float
    advantage: float
    advantage_ci95: tuple[float, float]
    prepare_seconds: float
    elapsed_seconds: float
    started_at: str
    environment: dict
    records: list[TrialRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        low, high = self.advantage_ci95
        return (
            f"{self.candidate.get('candidate')} vs {self.distinguisher.get('distinguisher')}: "
            f"advantage {self.advantage:+.4f} (95% CI [{low:+.4f}, {high:+.4f}]) over {self.trials} trials; "
            f"accept rates structured {self.structured_accept_rate:.3f}, "
            f"uniform {self.uniform_accept_rate:.3f}"
        )


def run_paired(
    candidate: Candidate,
    distinguisher: Distinguisher,
    trials: int,
    seed: int | None = None,
    fresh_publics: bool = False,
    keep_records: bool = True,
) -> ExperimentResult:
    """Run ``trials`` paired trials.

    ``fresh_publics=False`` samples the public layers once (and calls
    ``prepare`` once); ``True`` resamples them every trial.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    start = time.perf_counter()
    prepare_seconds = 0.0
    product = state = None
    records: list[TrialRecord] = []
    hits = {"structured": 0, "uniform": 0}

    for trial in range(trials):
        if product is None or fresh_publics:
            product = candidate.sample_product(rng)
            t0 = time.perf_counter()
            state = distinguisher.prepare(product.public_view())
            prepare_seconds += time.perf_counter() - t0
        samples = {"structured": product.sample(rng).matrix, "uniform": product.sample_uniform(rng)}
        for label in ("structured", "uniform"):
            t0 = time.perf_counter()
            decision = distinguisher.decide(state, samples[label], rng)
            seconds = time.perf_counter() - t0
            hits[label] += int(decision.structured)
            if keep_records:
                records.append(
                    TrialRecord(
                        trial, label, bool(decision.structured), decision.score, seconds, dict(decision.info)
                    )
                )

    p_structured = hits["structured"] / trials
    p_uniform = hits["uniform"] / trials
    advantage = p_structured - p_uniform
    stderr = math.sqrt((p_structured * (1 - p_structured) + p_uniform * (1 - p_uniform)) / trials)
    return ExperimentResult(
        candidate=candidate.describe(),
        distinguisher=distinguisher.describe(),
        trials=trials,
        seed=seed,
        fresh_publics=fresh_publics,
        structured_accept_rate=p_structured,
        uniform_accept_rate=p_uniform,
        advantage=advantage,
        advantage_ci95=(advantage - 1.96 * stderr, advantage + 1.96 * stderr),
        prepare_seconds=prepare_seconds,
        elapsed_seconds=time.perf_counter() - start,
        started_at=started,
        environment=environment(),
        records=records,
    )

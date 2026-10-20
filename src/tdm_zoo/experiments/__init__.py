"""Experiment harness: paired structured/uniform trials, advantage estimates, JSONL output."""

from .environment import environment
from .harness import ExperimentResult, TrialRecord, run_paired
from .io import append_jsonl, load_jsonl

__all__ = ["ExperimentResult", "TrialRecord", "append_jsonl", "environment", "load_jsonl", "run_paired"]

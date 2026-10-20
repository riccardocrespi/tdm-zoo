"""JSON-lines persistence of experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .harness import ExperimentResult


def _default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def append_jsonl(path: str | Path, result: ExperimentResult | dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict() if isinstance(result, ExperimentResult) else result
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_default) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

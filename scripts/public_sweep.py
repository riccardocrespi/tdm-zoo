"""Public-layer sweep: random A_i against Druk-Ishai A_i = G H_i P E (does the rank change?).

Example:
    python scripts/public_sweep.py                        # full grid
    python scripts/public_sweep.py --families diagonal --n 4 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from tdm_zoo.experiments.io import append_jsonl
from tdm_zoo.experiments.public_sweep import composite_relations, public_point

FAMILIES = ["diagonal", "circulant", "weighted_accumulator"]
PRIME = 2**31 - 1

# label -> (publics, zeros, LUOF options)
CONFIGS = {
    "random": ("random_invertible", 0, {}),
    "ghpe D=4 deg=6": ("luof", 0, {}),
    "ghpe D=4 deg=6 unshared": ("luof", 0, {"shared_codes": False}),
    "ghpe D=2 deg=2": ("luof", 0, {"block_size": 2, "base_length": 12, "expander_degree": 2}),
    "ghpe D=1 deg=2": ("luof", 0, {"block_size": 1, "base_length": 12, "expander_degree": 2}),
    "permutation": ("permutation", 0, {}),
    "random, 1 zero": ("random_invertible", 1, {}),
    "random, 2 zeros": ("random_invertible", 2, {}),
    "random, n zeros": ("random_invertible", -1, {}),
}


def run(job: tuple) -> tuple[str, dict]:
    label, family, n, seed, ell = job
    publics, zeros, options = CONFIGS[label]
    row = public_point(
        family, n, PRIME, publics, ell=ell, seed=seed, zeros=n if zeros < 0 else zeros, **options
    )
    return label, row.to_dict()


def spread(values) -> str:
    values = sorted({v for v in values if v is not None})
    if not values:
        return "–"
    return str(values[0]) if len(values) == 1 else f"{values[0]}..{values[-1]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--families", nargs="+", default=FAMILIES)
    parser.add_argument("--n", nargs="+", type=int, default=[4, 5])
    parser.add_argument("--ell", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out", default="results/public_sweep")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (label, family, n, seed, args.ell)
        for n in args.n
        for family in args.families
        for label in args.configs
        for seed in range(args.seeds)
    ]
    groups = defaultdict(list)
    with ProcessPoolExecutor(args.jobs) as pool:
        for label, row in pool.map(run, jobs):
            append_jsonl(out / "results.jsonl", {"config": label, **row})
            groups[row["family"], row["n"], label].append(row)

    header = (
        "| family | n | publics | measured / seeds | span | zeros in A_2 | K | C | rank M | rank U "
        "| δ(M) | δ(U) | γ | witness checks |"
    )
    lines = [header, "|" + "---|" * 14]
    for (family, n, label), rows in groups.items():
        ok = [r for r in rows if r["status"] == "ok"]
        checks = all(r["witness_in_kernel"] and r["witness_not_in_uniform_kernel"] for r in ok)
        cells = [family, n, label, f"{len(ok)} / {len(rows)}"]
        cells += [spread(r[key] for r in ok) for key in ("span", "zero_entries", "K", "C")]
        cells += [spread(r[key] for r in ok) for key in ("rank_M", "rank_U", "delta_M", "delta_U", "gamma")]
        lines.append("| " + " | ".join(str(c) for c in [*cells, "ok" if checks else "FAILED"]) + " |")
    table = "\n".join(lines)
    (out / "table.md").write_text(table + "\n")
    print(table)

    composite = [composite_relations(n, PRIME, seed) for n in args.n for seed in range(3)]
    (out / "composite.json").write_text(json.dumps(composite, indent=1))
    print("\nsecret S = G H P E:", composite)


if __name__ == "__main__":
    main()

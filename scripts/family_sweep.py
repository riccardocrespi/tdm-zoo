"""Secret-family sweep at ell = 4 (does the secret's shape create a sparse relation?).

Example:
    python scripts/family_sweep.py                       # full grid, both primes
    python scripts/family_sweep.py --families diagonal --n 4 --primes 2147483647
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdm_zoo.experiments.family_sweep import sweep_point
from tdm_zoo.experiments.io import append_jsonl

FAMILIES = ["diagonal", "circulant", "weighted_accumulator", "toeplitz"]
PRIMES = [2**31 - 1, 1_000_000_007]


def fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--families", nargs="+", default=FAMILIES)
    parser.add_argument("--n", nargs="+", type=int, default=[4, 5])
    parser.add_argument("--primes", nargs="+", type=int, default=PRIMES)
    parser.add_argument("--ell", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--max-tests", type=int, default=20000)
    parser.add_argument("--out", default="results/family_sweep")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for prime in args.primes:
        for n in args.n:
            for family in args.families:
                row = sweep_point(
                    family,
                    n,
                    prime,
                    ell=args.ell,
                    seed=args.seed,
                    iterations=args.iterations,
                    max_tests=args.max_tests,
                )
                rows.append(row)
                append_jsonl(out / "results.jsonl", row.to_dict())
                print(
                    f"{family:22s} n={n} p={prime}: status={row.status} C={row.C} "
                    f"ranks M/U={row.rank_M}/{row.rank_U} gamma={row.gamma} s_witness={row.s_witness} "
                    f"s={row.s} ({fmt(row.s_over_C)}) tests={row.candidates_tested} [{fmt(row.seconds)}s]",
                    flush=True,
                )

    header = (
        "| family | n | prime | C | rank M | rank U | δ(M) | δ(U) | γ | dim(ker M ∩ ker U) "
        "| s_witness | lightest ker M | s | s/C | s − rank U | minimal | T(M)v*=0 | T(U)v*≠0 | status |"
    )
    lines = [header, "|" + "---|" * 19]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                fmt(v)
                for v in (
                    r.family,
                    r.n,
                    r.prime,
                    r.C,
                    r.rank_M,
                    r.rank_U,
                    r.delta_M,
                    r.delta_U,
                    r.gamma,
                    r.shared_kernel_dim,
                    r.s_witness,
                    r.lightest_kernel_vector,
                    r.s,
                    r.s_over_C,
                    r.s_minus_rank_U,
                    r.minimal,
                    r.witness_in_kernel,
                    r.witness_not_in_uniform_kernel,
                    r.status,
                )
            )
            + " |"
        )
    table = "\n".join(lines)
    (out / "table.md").write_text(table + "\n")
    sparse = {f"{r.family}-n{r.n}-p{r.prime}": r.sparse_vector for r in rows if r.sparse_vector}
    (out / "sparse_vectors.json").write_text(json.dumps(sparse, indent=1))
    print("\n" + table)


if __name__ == "__main__":
    main()

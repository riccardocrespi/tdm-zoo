"""Does secret structure give fewer witness unknowns? (Tests A and B at ell = 4.)

Example:
    python scripts/witness_dimension.py
    python scripts/witness_dimension.py --families diagonal --n 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from tdm_zoo.experiments.io import append_jsonl
from tdm_zoo.experiments.witness_dimension import witness_point

FAMILIES = ["diagonal", "circulant", "weighted_accumulator", "toeplitz", "block_diagonal:2"]


def parse(spec: str) -> tuple[str, dict]:
    if spec.startswith("block_diagonal"):
        _, _, block = spec.partition(":")
        return "block_diagonal", {"block": int(block or 2)}
    return spec, {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--families", nargs="+", default=FAMILIES)
    parser.add_argument("--n", nargs="+", type=int, default=[4, 5])
    parser.add_argument("--prime", type=int, default=2**31 - 1)
    parser.add_argument("--ell", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/witness_dimension")
    parser.add_argument("--table-only", action="store_true", help="rebuild table.md from results.jsonl")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.table_only:
        for line in (out / "results.jsonl").read_text().splitlines():
            data = json.loads(line)
            data["relations_by_degree"] = {int(k): v for k, v in data["relations_by_degree"].items()}
            rows.append(SimpleNamespace(**data))
        write_table(rows, out)
        return
    for n in args.n:
        for spec in args.families:
            family, kwargs = parse(spec)
            if "block" in kwargs and n % kwargs["block"]:
                print(f"skip {spec} at n={n}: block size does not divide n", flush=True)
                continue
            row = witness_point(family, n, args.prime, ell=args.ell, seed=args.seed, family_kwargs=kwargs)
            rows.append(row)
            append_jsonl(out / "results.jsonl", row.to_dict())
            print(
                f"{row.family:28s} n={n}: v={row.vars_per_block} params={row.family_params} "
                f"relations={row.relations_by_degree} effective={row.effective_unknowns} "
                f"floating={row.floating} (scaling {row.blocks}, param-redundancy "
                f"{row.parametrisation_redundancy}, genuine {row.genuine_floating}) [{row.seconds:.1f}s]",
                flush=True,
            )

    write_table(rows, out)


def write_table(rows, out: Path) -> None:
    lines = [
        "| family | n | unknowns/block v | family params | new relations (deg 1/2/3) "
        "| effective unknowns/block "
        "| Jacobian rank / total | floating | = scaling + param. redundancy + genuine "
        "| bar ℓ=5 (n^(2/3)) | clears bar |",
        "|" + "---|" * 11,
    ]
    for r in rows:
        rel = "/".join(str(r.relations_by_degree[k]) for k in sorted(r.relations_by_degree))
        lines.append(
            f"| {r.family} | {r.n} | {r.vars_per_block} | {r.family_params} | {rel} | {r.effective_unknowns} "
            f"| {r.jacobian_rank} / {r.total_unknowns} | {r.floating} "
            f"| {r.blocks} + {r.parametrisation_redundancy} + {r.genuine_floating} "
            f"| {r.bar_l5:.2f} | {'yes' if r.clears_bar else 'no'} |"
        )
    relations = [f"- {r.family}, n={r.n}: {rel}" for r in rows for rel in r.relations_written]
    text = "\n".join(lines) + "\n\nRelations found (Test A):\n" + ("\n".join(relations) or "none") + "\n"
    (out / "table.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()

"""Rank bottleneck of LUOF: public structure, per-layer subspace, linear span of achievable M.

Results accumulate in results.jsonl; table.md keeps the latest row of every point, so a
``--big`` run or a second prime adds to the table rather than replacing it.

Example:
    python scripts/luof_structure.py                       # k <= 48, a few minutes
    python scripts/luof_structure.py --big                 # adds k = 64 (~1.5 GB peak, slow)
    python scripts/luof_structure.py --prime 1000000007 --seeds 4
    python scripts/luof_structure.py --only subspace       # seconds
    python scripts/luof_structure.py --table-only

Sampling M dominates the span runs: the composite map multiplies through a dense N x N ``H``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from tdm_zoo.experiments.io import append_jsonl, load_jsonl
from tdm_zoo.experiments.luof_structure import span_point, structure_point, subspace_point

EXPERIMENTS = ("structure", "subspace", "span")
STRUCTURE_POINTS = [(k, view) for k in (16, 32) for view in ("split", "composite")]
SUBSPACE_K = [16, 32, 48]
SPAN_POINTS = [(k, ell) for k in (16, 32) for ell in (1, 2, 3)]
BIG_SUBSPACE_K = [64]
BIG_SPAN_POINTS = [(64, 2)]


def key(row: dict) -> tuple:
    return (row["experiment"], row["k"], row.get("view"), row.get("ell"), row["prime"], row["seed"])


def save(row, out: Path) -> dict:
    payload = row.to_dict()
    relations = payload.pop("relations", None)
    append_jsonl(out / "results.jsonl", payload)
    if relations:
        path = out / "relations.json"
        stored = json.loads(path.read_text()) if path.exists() else {}
        stored[f"k={row.k} ell={row.ell} prime={row.prime} seed={row.seed}"] = relations
        path.write_text(json.dumps(stored))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--prime", nargs="+", type=int, default=[2**31 - 1])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--big", action="store_true", help="include k = 64")
    parser.add_argument("--out", default="results/luof_structure")
    parser.add_argument("--table-only", action="store_true", help="rebuild table.md from results.jsonl")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.table_only:
        write_table(out)
        return

    subspace_k = SUBSPACE_K + (BIG_SUBSPACE_K if args.big else [])
    span_points = SPAN_POINTS + (BIG_SPAN_POINTS if args.big else [])
    for prime in args.prime:
        for seed in range(args.seeds):
            if "structure" in args.only:
                for k, view in STRUCTURE_POINTS:
                    row = save(structure_point(k, view, prime, seed=seed), out)
                    print(f"structure k={k} {view} p={prime} seed={seed}: {row['publics']}", flush=True)
            if "subspace" in args.only:
                for k in subspace_k:
                    row = save(subspace_point(k, prime, seed=seed), out)
                    print(
                        f"subspace k={k} p={prime} seed={seed}: dim V={row['dim_V']} "
                        f"codim={row['codim']} [{row['seconds']:.1f}s]",
                        flush=True,
                    )
            if "span" in args.only:
                for k, ell in span_points:
                    row = save(span_point(k, ell, prime, seed=seed), out)
                    print(
                        f"span k={k} ell={ell} p={prime} seed={seed}: {row['span']}/{row['k2']} "
                        f"[{row['seconds']:.1f}s]",
                        flush=True,
                    )
    write_table(out)


def write_table(out: Path) -> None:
    latest = {key(row): row for row in load_jsonl(out / "results.jsonl")}
    rows = sorted(latest.values(), key=lambda r: (r["k"], r.get("ell") or 0, r["prime"], r["seed"]))

    def section(name: str) -> list[dict]:
        return [r for r in rows if r["experiment"] == name]

    lines = [
        "## Structure and rank bottleneck",
        "",
        "| k | view | prime | seed | N | M shape | rank M | publics (shape, rank) "
        "| secrets (shape, params) |",
        "|" + "---|" * 9,
    ]
    for r in section("structure"):
        publics = ", ".join(f"{name} {s[0]}x{s[1]} r{rank}" for name, s, rank in r["publics"])
        secrets = ", ".join(f"{name} {fam} {s[0]}x{s[1]} p{n}" for name, fam, s, n in r["secrets"])
        shape = r["M_shape"]
        lines.append(
            f"| {r['k']} | {r['view']} | {r['prime']} | {r['seed']} | {r['N']} | {shape[0]}x{shape[1]} "
            f"| {r['M_rank']} | {publics} | {secrets} |"
        )
    lines += [
        "",
        "## Per-layer secret subspace V",
        "",
        "| k | prime | seed | N | k^2 | dim V | codim | fill % |",
        "|" + "---|" * 8,
    ]
    for r in section("subspace"):
        lines.append(
            f"| {r['k']} | {r['prime']} | {r['seed']} | {r['N']} | {r['k2']} | {r['dim_V']} "
            f"| {r['codim']} | {r['fill_percent']:.1f} |"
        )
    lines += [
        "",
        "## Linear span of achievable M (publics fixed)",
        "",
        "| k | ell | prime | seed | samples | span | k^2 | codim | verdict | relations hold / reject U |",
        "|" + "---|" * 10,
    ]
    for r in section("span"):
        verdict = f"distinguisher, codim {r['codim']}" if r["distinguisher"] else "no linear distinguisher"
        check = (
            f"{r['relations_hold_on_fresh']} / {r['relations_reject_uniform']}" if r["distinguisher"] else "–"
        )
        lines.append(
            f"| {r['k']} | {r['ell']} | {r['prime']} | {r['seed']} | {r['samples']} | {r['span']} "
            f"| {r['k2']} | {r['codim']} | {verdict} | {check} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "table.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()

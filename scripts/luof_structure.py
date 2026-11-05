"""Rank bottleneck of LUOF: public structure, per-layer subspace, linear span of achievable M.

Results accumulate in results.jsonl; table.md keeps the latest row of every point, so a
``--big`` run or a second prime adds to the table rather than replacing it.

Example:
    python scripts/luof_structure.py                       # k <= 48, a few minutes
    python scripts/luof_structure.py --big --seeds 3       # the scaling grids, k up to 256; hours
    python scripts/luof_structure.py --big --only pattern  # exact span of M at every k, minutes
    python scripts/luof_structure.py --big --only span_fast --k 96 --seeds 3
    python scripts/luof_structure.py --prime 1000000007 --only subspace
    python scripts/luof_structure.py --table-only

``span`` samples real secrets, through the dense N x N H; ``span_fast`` does not, and is checked
against it in the tests. At k = 128 a span point peaks near 4.5 GB.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import flint

from tdm_zoo.experiments.io import append_jsonl, load_jsonl
from tdm_zoo.experiments.luof_structure import (
    fast_span_point,
    pattern_point,
    span_point,
    structure_point,
    subspace_point,
)

EXPERIMENTS = ("structure", "subspace", "span", "span_fast", "pattern")
LAYERS = (2, 3, 4, 5)
STRUCTURE_POINTS = [(k, view) for k in (16, 32) for view in ("split", "composite")]
SUBSPACE_K = [16, 32, 48]
SPAN_POINTS = [(k, ell) for k in (16, 32) for ell in (1, 2, 3)]
FAST_K = [16, 32]
PATTERN_K = [16, 32, 64]
BIG_SUBSPACE_K = [64, 96, 128, 192, 256]
BIG_FAST_K = [64, 96, 128]
BIG_PATTERN_K = [96, 128, 192, 256]


def key(row: dict) -> tuple:
    method = row.get("method", "direct")
    return (row["experiment"], row["k"], row.get("view"), row.get("ell"), row["prime"], row["seed"], method)


def save(row, out: Path) -> dict:
    payload = row.to_dict()
    relations = payload.pop("relations", None)
    append_jsonl(out / "results.jsonl", payload)
    if relations:
        path = out / "relations.json"
        stored = json.loads(path.read_text()) if path.exists() else {}
        stored[f"{row.experiment} k={row.k} ell={row.ell} prime={row.prime} seed={row.seed}"] = relations
        path.write_text(json.dumps(stored))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--prime", nargs="+", type=int, default=[2**31 - 1])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--big", action="store_true", help="include the scaling grids, k >= 64")
    parser.add_argument("--k", nargs="+", type=int, help="restrict every grid to these k")
    parser.add_argument("--layers", nargs="+", type=int, default=list(LAYERS), help="ell for the fast grids")
    parser.add_argument("--threads", type=int, default=4, help="FLINT threads for the big eliminations")
    parser.add_argument("--method", choices=("direct", "gram"), help="force the dim V method")
    parser.add_argument("--out", default="results/luof_structure")
    parser.add_argument("--table-only", action="store_true", help="rebuild table.md from results.jsonl")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.table_only:
        write_table(out)
        return

    flint.ctx.threads = args.threads

    def grid(small: list, big: list) -> list:
        ks = sorted({*small, *(big if args.big else [])})
        return [k for k in ks if args.k is None or k in args.k]

    subspace_k = grid(SUBSPACE_K, BIG_SUBSPACE_K)
    span_points = [(k, ell) for k, ell in SPAN_POINTS if args.k is None or k in args.k]
    fast_points = [(k, ell) for k in grid(FAST_K, BIG_FAST_K) for ell in args.layers]
    pattern_points = [(k, ell) for k in grid(PATTERN_K, BIG_PATTERN_K) for ell in args.layers]
    for prime in args.prime:
        for seed in range(args.seeds):
            if "structure" in args.only:
                for k, view in STRUCTURE_POINTS:
                    row = save(structure_point(k, view, prime, seed=seed), out)
                    print(f"structure k={k} {view} p={prime} seed={seed}: {row['publics']}", flush=True)
            if "subspace" in args.only:
                for k in subspace_k:
                    row = save(subspace_point(k, prime, seed=seed, method=args.method), out)
                    print(
                        f"subspace k={k} p={prime} seed={seed} {row['method']}: dim V={row['dim_V']} "
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
            if "pattern" in args.only:
                for k, ell in pattern_points:
                    row = save(pattern_point(k, ell, prime, seed=seed), out)
                    print(
                        f"pattern k={k} ell={ell} p={prime} seed={seed}: dead={row['dead_pairs']} "
                        f"certified={row['certified_full']} [{row['seconds']:.1f}s]",
                        flush=True,
                    )
            if "span_fast" in args.only:
                for k, ell in fast_points:
                    row = save(fast_span_point(k, ell, prime, seed=seed), out)
                    print(
                        f"span_fast k={k} ell={ell} p={prime} seed={seed}: {row['span']}/{row['k2']} "
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
        "| k | prime | seed | method | N | k^2 | dim V | codim | dim V / k^2 | dim V / N "
        "| bound: params | bound: blocks |",
        "|" + "---|" * 12,
    ]
    for r in section("subspace"):
        lines.append(
            f"| {r['k']} | {r['prime']} | {r['seed']} | {r.get('method', 'direct')} | {r['N']} | {r['k2']} "
            f"| {r['dim_V']} | {r['codim']} | {r['dim_V'] / r['k2']:.3f} | {r['dim_V'] / r['N']:.2f} "
            f"| {r.get('params', '–')} | {r.get('block_bound', '–')} |"
        )
    for name, title in (
        ("span", "Linear span of achievable M, real secrets (publics fixed)"),
        ("span_fast", "Linear span of achievable M, fast path (publics fixed)"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "| k | ell | prime | seed | samples | span | k^2 | codim | verdict | relations hold / reject U "
            "| seconds |",
            "|" + "---|" * 11,
        ]
        for r in section(name):
            verdict = (
                f"distinguisher, codim {r['codim']}" if r["distinguisher"] else "no linear distinguisher"
            )
            hold, reject = r["relations_hold_on_fresh"], r["relations_reject_uniform"]
            check = f"{hold} / {reject}" if r["distinguisher"] else "–"
            lines.append(
                f"| {r['k']} | {r['ell']} | {r['prime']} | {r['seed']} | {r['samples']} | {r['span']} "
                f"| {r['k2']} | {r['codim']} | {verdict} | {check} | {r['seconds']:.0f} |"
            )
    lines += [
        "",
        "## Block pattern (exact, no sampling)",
        "",
        "| k | ell | prime | seed | N | blocks | live | dead live pairs | span = k^2 proved | seconds |",
        "|" + "---|" * 10,
    ]
    for r in section("pattern"):
        lines.append(
            f"| {r['k']} | {r['ell']} | {r['prime']} | {r['seed']} | {r['N']} | {r['blocks']} "
            f"| {r['live_blocks']} | {r['dead_pairs']} | {'yes' if r['certified_full'] else 'no'} "
            f"| {r['seconds']:.0f} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "table.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()

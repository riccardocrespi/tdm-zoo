"""Rank bottleneck of LUOF: public structure, per-layer subspace, linear span of achievable M.

Results accumulate in results.jsonl; table.md keeps the latest row of every point, so a
``--big`` run or a second prime adds to the table rather than replacing it.

Example:
    python scripts/luof_structure.py                       # k <= 48, a few minutes
    python scripts/luof_structure.py --big --seeds 3       # the scaling grids, k up to 256; hours
    python scripts/luof_structure.py --big --only pattern  # exact span of M at every k, minutes
    python scripts/luof_structure.py --big --only span_fast --k 96 --seeds 3
    python scripts/luof_structure.py --prime 1000000007 --only subspace
    python scripts/luof_structure.py --only phase --seeds 3 --workers 8   # (Delta, degree) grid
    python scripts/luof_structure.py --only phase_check distinguisher --seeds 3
    python scripts/luof_structure.py --table-only

``phase``, ``phase_check`` and ``distinguisher`` run only when named in ``--only``.
``span`` samples real secrets, through the dense N x N H; ``span_fast`` does not, and is checked
against it in the tests. At k = 128 a span point peaks near 4.5 GB.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product as grid_product
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import flint

from tdm_zoo.experiments.io import append_jsonl, load_jsonl
from tdm_zoo.experiments.luof_structure import (
    distinguisher_point,
    fast_span_point,
    pattern_point,
    phase_check_point,
    phase_point,
    span_point,
    structure_point,
    subspace_point,
)

DEFAULT_EXPERIMENTS = ("structure", "subspace", "span", "span_fast", "pattern")
EXPERIMENTS = (*DEFAULT_EXPERIMENTS, "phase", "phase_check", "distinguisher")
LAYERS = (2, 3, 4, 5)
STRUCTURE_POINTS = [(k, view) for k in (16, 32) for view in ("split", "composite")]
SUBSPACE_K = [16, 32, 48]
SPAN_POINTS = [(k, ell) for k in (16, 32) for ell in (1, 2, 3)]
FAST_K = [16, 32]
PATTERN_K = [16, 32, 64]
BIG_SUBSPACE_K = [64, 96, 128, 192, 256]
BIG_FAST_K = [64, 96, 128]
BIG_PATTERN_K = [96, 128, 192, 256]

PRIME = 2**31 - 1
DEFAULT_CELL = (4, 6, 8)  # (Delta, degree, base_length)
PHASE_K = [16, 32, 64]
PHASE_DELTA = [1, 2, 4, 8]
PHASE_DEGREE = [1, 2, 3, 4, 6, 8, 12]
PHASE_BASE = [4, 8, 12]


def cell(delta: int, degree: int, base: int) -> dict:
    return {"block_size": delta, "expander_degree": degree, "base_length": base}


# Spot checks: the default, deep in the proper region, on the boundary, and the high-degree pocket.
PHASE_CHECKS = [
    (32, 2, cell(*DEFAULT_CELL)),
    (32, 2, cell(1, 1, 12)),
    (32, 2, cell(2, 1, 12)),
    (32, 2, cell(1, 2, 4)),
    (64, 2, cell(1, 2, 8)),
    (64, 2, cell(2, 1, 8)),
    (64, 3, cell(1, 1, 8)),
    (16, 2, cell(1, 12, 12)),
]
DISTINGUISHER_CELLS = [
    (32, 2, cell(1, 1, 12)),
    (32, 3, cell(1, 1, 12)),
    (32, 2, cell(1, 1, 8)),
    (32, 2, cell(2, 1, 12)),
    (64, 2, cell(2, 1, 8)),
    (64, 3, cell(1, 1, 8)),
    (32, 2, cell(*DEFAULT_CELL)),  # control: full span, advantage 0
]


def run_phase(job: tuple) -> dict:
    return phase_point(*job)


def key(row: dict) -> tuple:
    method = row.get("method", "direct")
    options = json.dumps(row.get("options", {}), sort_keys=True)
    experiment, k, view, ell = row["experiment"], row["k"], row.get("view"), row.get("ell")
    return (experiment, k, view, ell, row["prime"], row["seed"], method, options)


def save(row, out: Path) -> dict:
    payload = row if isinstance(row, dict) else row.to_dict()
    relations = payload.pop("relations", None)
    append_jsonl(out / "results.jsonl", payload)
    if relations:
        path = out / "relations.json"
        stored = json.loads(path.read_text()) if path.exists() else {}
        options = json.dumps(payload.get("options", {}), sort_keys=True)
        name = f"{row.experiment} k={row.k} ell={row.ell} prime={row.prime} seed={row.seed} {options}"
        stored[name] = relations
        path.write_text(json.dumps(stored))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--prime", nargs="+", type=int, default=[2**31 - 1])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--big", action="store_true", help="include the scaling grids, k >= 64")
    parser.add_argument("--k", nargs="+", type=int, help="restrict every grid to these k")
    parser.add_argument("--layers", nargs="+", type=int, default=list(LAYERS), help="ell for the fast grids")
    parser.add_argument("--threads", type=int, default=4, help="FLINT threads for the big eliminations")
    parser.add_argument("--workers", type=int, default=8, help="processes for the phase grid")
    parser.add_argument("--trials", type=int, default=200, help="paired trials per distinguisher point")
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
            if "phase_check" in args.only:
                for k, ell, options in PHASE_CHECKS:
                    row = save(phase_check_point(k, ell, prime, seed=seed, **options), out)
                    print(
                        f"phase_check k={k} ell={ell} {options} p={prime} seed={seed}: "
                        f"sampled={row['sampled']} exact={row['exact']} [{row['seconds']:.1f}s]",
                        flush=True,
                    )
            if "distinguisher" in args.only:
                for k, ell, options in DISTINGUISHER_CELLS:
                    point = distinguisher_point(k, ell, prime, seed=seed, trials=args.trials, **options)
                    row = save(point, out)
                    print(
                        f"distinguisher k={k} ell={ell} {options} p={prime} seed={seed}: "
                        f"span={row['span']} relations={row['relations_found']} rank M={row['rank_M']} "
                        f"advantage={row['advantage']:.3f} [{row['seconds']:.1f}s]",
                        flush=True,
                    )
        if "phase" in args.only:
            phase_k = [k for k in PHASE_K if args.k is None or k in args.k]
            axes = (phase_k, args.layers, [prime], range(args.seeds), PHASE_DELTA, PHASE_DEGREE, PHASE_BASE)
            jobs = sorted(grid_product(*axes), key=lambda job: job[0])
            # spawn, not fork: FLINT's thread pool does not survive a fork, and a forked worker
            # hangs on its first threaded elimination.
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
                futures = [pool.submit(run_phase, job) for job in jobs]
                for done, future in enumerate(as_completed(futures), 1):
                    save(future.result(), out)
                    if done % 100 == 0:
                        print(f"phase: {done}/{len(jobs)}", flush=True)
            print(f"phase: {len(jobs)} points at p={prime}", flush=True)
    write_table(out)


def spread(values) -> str:
    values = sorted(values)
    return str(values[0]) if values[0] == values[-1] else f"{values[0]}..{values[-1]}"


def phase_tables(rows: list[dict]) -> list[str]:
    """One table per ``(k, ell, base_length)``; a cell is ``codim / rank deficit`` over seeds."""
    if not rows:
        return []
    cells: dict[tuple, list] = {}
    for r in rows:
        o = r["options"]
        where = (r["k"], r["ell"], o["base_length"], o["block_size"], o["expander_degree"])
        deficit = None if "degenerate" in r or r.get("rank_M") is None else r["k"] - r["rank_M"]
        cells.setdefault(where, []).append((r.get("codim"), deficit))
    lines = [
        "",
        "## Phase diagram: codim of span{M} / k - rank M",
        "",
        "Exact codim, generic rank; a range is over seeds; bold is the default cell.",
    ]
    for k, ell, base in sorted({where[:3] for where in cells}):
        lines += ["", f"k={k}, ell={ell}, base_length={base}", ""]
        lines += ["| Delta \\ degree | " + " | ".join(map(str, PHASE_DEGREE)) + " |"]
        lines += ["|" + "---|" * (len(PHASE_DEGREE) + 1)]
        for delta in PHASE_DELTA:
            entries = []
            for degree in PHASE_DEGREE:
                values = cells.get((k, ell, base, delta, degree), [])
                known = [v for v in values if v[0] is not None]
                text = "–" if not known else f"{spread(v[0] for v in known)} / {spread(v[1] for v in known)}"
                if len(known) < len(values):
                    text += " (degenerate)"
                entries.append(f"**{text}**" if (delta, degree, base) == DEFAULT_CELL else text)
            lines.append(f"| {delta} | " + " | ".join(entries) + " |")
    return lines


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
    lines += phase_tables([r for r in section("phase") if r["prime"] == PRIME])
    lines += [
        "",
        "## Phase spot checks: sampled span against the exact span",
        "",
        "| k | ell | Delta | degree | base | prime | seed | sampled | exact | agree |",
        "|" + "---|" * 10,
    ]
    for r in section("phase_check"):
        o = r["options"]
        lines.append(
            f"| {r['k']} | {r['ell']} | {o['block_size']} | {o['expander_degree']} | {o['base_length']} "
            f"| {r['prime']} | {r['seed']} | {r['sampled']} | {r['exact']} | {r['agree']} |"
        )
    lines += [
        "",
        "## Distinguisher on proper cells (public-view span test, paired trials)",
        "",
        "| k | ell | Delta | degree | base | prime | seed | span | relations | rank M | accept M | accept U "
        "| advantage | trials |",
        "|" + "---|" * 14,
    ]
    for r in section("distinguisher"):
        o = r["options"]
        lines.append(
            f"| {r['k']} | {r['ell']} | {o['block_size']} | {o['expander_degree']} | {o['base_length']} "
            f"| {r['prime']} | {r['seed']} | {r['span']} | {r['relations_found']} | {r['rank_M']} "
            f"| {r['structured_accept']:.3f} | {r['uniform_accept']:.3f} | {r['advantage']:.3f} "
            f"| {r['trials']} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "table.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()

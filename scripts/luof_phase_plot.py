"""Phase diagram of LUOF over (Delta, degree): self-contained HTML from the ``phase`` rows.

Small multiples: one section per base_length, rows ell, columns k. A cell shows the worst seed:
shade is codim(span M) / k^2, hatching marks singular M, the outline is the default cell.

Example:
    python scripts/luof_phase_plot.py
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from tdm_zoo.experiments.io import load_jsonl

PRIME = 2**31 - 1
DEFAULT_CELL = (4, 6, 8)  # (Delta, degree, base_length)
DELTAS = [1, 2, 4, 8]
DEGREES = [1, 2, 3, 4, 6, 8, 12]
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STOPS = [0.002, 0.01, 0.03, 0.06, 0.1, 0.15]  # codim / k^2 between ramp steps
CELL_W, CELL_H, LEFT, TOP = 34, 26, 30, 30

STYLE = """
:root { --bg: #ffffff; --ink: #1f1f1d; --muted: #6b6a66; --full: #f0efec; --line: #1f1f1d; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg: #1b1b19; --ink: #ecebe7; --muted: #a3a29c;
    --full: #383835; --line: #ecebe7; }
}
:root[data-theme="dark"] { --bg: #1b1b19; --ink: #ecebe7; --muted: #a3a29c; --full: #383835;
  --line: #ecebe7; }
body { background: var(--bg); color: var(--ink); margin: 24px 16px; max-width: 1040px;
  font: 14px/1.45 system-ui, sans-serif; }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 28px 0 8px; }
p { color: var(--muted); margin: 0 0 12px; max-width: 72ch; }
.grid { display: grid; gap: 14px 18px; grid-template-columns: repeat(3, max-content); overflow-x: auto; }
.panel-title { font-size: 12px; color: var(--muted); margin-bottom: 2px; }
svg text { fill: var(--muted); font-size: 10px; }
svg text.value { fill: var(--ink); } svg text.value.on-dark { fill: #ffffff; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: center; font-size: 12px;
  color: var(--muted); margin-bottom: 8px; }
.swatch { display: inline-block; width: 14px; height: 14px; vertical-align: -2px; margin-right: 4px; }
"""


def shade(fraction: float) -> tuple[str, bool]:
    """Fill and whether the cell needs light text."""
    if fraction == 0:
        return "var(--full)", False
    step = sum(fraction > stop for stop in STOPS)
    return RAMP[step], step >= 3


def panel(cells: dict, k: int, ell: int, base: int) -> str:
    width = LEFT + CELL_W * len(DEGREES) + 2
    height = TOP + CELL_H * len(DELTAS) + 2
    parts = [f'<svg width="{width}" height="{height}" role="img" aria-label="k={k}, ell={ell}">']
    for j, degree in enumerate(DEGREES):
        x = LEFT + CELL_W * j + CELL_W / 2
        parts.append(f'<text x="{x}" y="{TOP - 6}" text-anchor="middle">{degree}</text>')
    for i, delta in enumerate(DELTAS):
        y = TOP + CELL_H * i + CELL_H / 2 + 3
        parts.append(f'<text x="{LEFT - 6}" y="{y}" text-anchor="end">{delta}</text>')
        for j, degree in enumerate(DEGREES):
            x, top = LEFT + CELL_W * j, TOP + CELL_H * i
            rows = cells.get((k, ell, base, delta, degree), [])
            measured = [r for r in rows if "codim" in r]
            if not measured:
                continue
            codim = max(r["codim"] for r in measured)
            deficit = max(k - r["rank_M"] for r in measured)
            fill, dark = shade(codim / k**2)
            title = (
                f"k={k}, ell={ell}, base={base}, Delta={delta}, degree={degree}: codim "
                f"{sorted(r['codim'] for r in measured)}, k - rank M "
                f"{sorted(k - r['rank_M'] for r in measured)} over seeds"
            )
            parts.append(f"<g><title>{escape(title)}</title>")
            parts.append(
                f'<rect x="{x + 1}" y="{top + 1}" width="{CELL_W - 2}" height="{CELL_H - 2}" '
                f'rx="3" fill="{fill}"/>'
            )
            if deficit:
                parts.append(
                    f'<rect x="{x + 1}" y="{top + 1}" width="{CELL_W - 2}" height="{CELL_H - 2}" '
                    'rx="3" fill="url(#singular)"/>'
                )
            if codim:
                css = "value on-dark" if dark else "value"
                parts.append(
                    f'<text class="{css}" x="{x + CELL_W / 2}" y="{top + CELL_H / 2 + 3}" '
                    f'text-anchor="middle">{codim}</text>'
                )
            if (delta, degree, base) == DEFAULT_CELL:
                parts.append(
                    f'<rect x="{x + 1}" y="{top + 1}" width="{CELL_W - 2}" height="{CELL_H - 2}" '
                    'rx="3" fill="none" stroke="var(--line)" stroke-width="2"/>'
                )
            parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def build(rows: list[dict]) -> str:
    cells: dict[tuple, list] = {}
    for r in rows:
        o = r["options"]
        where = (r["k"], r["ell"], o["base_length"], o["block_size"], o["expander_degree"])
        cells.setdefault(where, []).append(r)
    ks = sorted({where[0] for where in cells})
    ells = sorted({where[1] for where in cells})
    bases = sorted({where[2] for where in cells})
    legend = ['<span><i class="swatch" style="background:var(--full)"></i>full span</span>']
    bounds = [0, *STOPS]
    for index, color in enumerate(RAMP):
        label = f"&gt; {100 * bounds[index]:g}%" if index else f"≤ {100 * STOPS[0]:g}%"
        legend.append(f'<span><i class="swatch" style="background:{color}"></i>{label}</span>')
    legend.append(
        '<span><svg width="14" height="14" style="vertical-align:-2px;margin-right:4px">'
        '<rect width="14" height="14" fill="var(--full)"/><rect width="14" height="14" '
        'fill="url(#singular)"/></svg>M singular</span>'
    )
    legend.append(
        '<span><i class="swatch" style="outline:2px solid var(--line);outline-offset:-2px"></i>'
        "default cell</span>"
    )
    body = [
        "<h1>LUOF phase diagram</h1>",
        "<p>Rows are block size Δ, columns are expander degree. Each cell is the worst of the seeds: "
        "shade is codim(span M) / k², the number is the codim, hatching means M is singular. "
        "Hover a cell for every seed. Exact values, from the block-pattern formula.</p>",
        '<svg width="0" height="0" style="position:absolute"><defs><pattern id="singular" width="6" '
        'height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<line x1="0" y1="0" x2="0" y2="6" stroke="var(--line)" stroke-width="1.2"/></pattern></defs></svg>',
        f'<div class="legend">{"".join(legend)}</div>',
    ]
    for base in bases:
        body.append(f"<h2>base_length = {base}</h2>")
        body.append(f'<div class="grid" style="grid-template-columns:repeat({len(ks)},max-content)">')
        for ell in ells:
            for k in ks:
                body.append(f'<div><div class="panel-title">k = {k}, ℓ = {ell}</div>')
                body.append(panel(cells, k, ell, base) + "</div>")
        body.append("</div>")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>LUOF Phase Diagram</title><style>{STYLE}</style></head><body>"
        + "".join(body)
        + "</body></html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="results/luof_structure")
    args = parser.parse_args()
    out = Path(args.out)
    latest = {}
    for r in load_jsonl(out / "results.jsonl"):
        if r["experiment"] == "phase" and r["prime"] == PRIME:
            o = r["options"]
            latest[(r["k"], r["ell"], r["seed"], *sorted(o.items()))] = r
    (out / "phase_diagram.html").write_text(build(list(latest.values())))
    print(out / "phase_diagram.html")


if __name__ == "__main__":
    main()

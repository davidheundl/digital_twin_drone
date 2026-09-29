"""Render a compiled scenario to a top-down SVG.

Draws the *compiled* geometry, not the ASCII source, so it doubles as a check
on the meshing: if a wall is missing here it is missing from the building the
drone will fly in. No dependencies — an AABB is a rectangle.

    python tools/plot_floorplan.py scenarios/office_small.yaml -o plan.svg
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.config import load_config
from sim.floorplan import load_scenario

PX_PER_M = 44

STYLE = {
    "wall": ("#2f3542", "#1e2430"),
    "debris": ("#c98a2e", "#8a5d16"),
}

FEATURE_STYLE = {
    # kind:      (fill,      radius_m, label)
    "victim":    ("#e0484d", 0.34, "V"),
    "decoy":     ("#8d6fc4", 0.30, "?"),
    "anchor":    ("#2f9e8f", 0.26, "A"),
    "spawn":     ("#3b82c4", 0.36, "E"),
    "stair":     ("#6b7280", 0.30, "S"),
    "door":      ("#9aa5b1", 0.16, ""),
}


def render(building, title: str) -> str:
    w_m, h_m, _ = building.size
    w, h = w_m * PX_PER_M, h_m * PX_PER_M
    pad = 28
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w + 2 * pad:.0f}" '
        f'height="{h + 2 * pad + 34:.0f}" viewBox="0 0 {w + 2 * pad:.0f} '
        f'{h + 2 * pad + 34:.0f}">',
        '<rect width="100%" height="100%" fill="#f6f7f9"/>',
        f'<g transform="translate({pad},{pad})">',
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" fill="#ffffff" '
        f'stroke="#c3c9d2" stroke-width="1.5"/>',
    ]

    # y is up in the world and down in SVG, so every rectangle flips.
    def sy(y_m: float) -> float:
        return (h_m - y_m) * PX_PER_M

    order = sorted(range(len(building.solids)),
                   key=lambda i: building.solids.tags[i] != "wall")
    for i in order:
        lo, hi = building.solids.lower[i], building.solids.upper[i]
        fill, stroke = STYLE.get(building.solids.tags[i], ("#888", "#555"))
        out.append(
            f'<rect x="{lo[0] * PX_PER_M:.1f}" y="{sy(hi[1]):.1f}" '
            f'width="{(hi[0] - lo[0]) * PX_PER_M:.1f}" '
            f'height="{(hi[1] - lo[1]) * PX_PER_M:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.75"/>')

    for f in building.features:
        style = FEATURE_STYLE.get(f.kind)
        if style is None:
            continue
        fill, r_m, label = style
        cx, cy = f.position[0] * PX_PER_M, sy(f.position[1])
        r = r_m * PX_PER_M
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
                   f'stroke="#ffffff" stroke-width="1.5"/>')
        if label:
            out.append(
                f'<text x="{cx:.1f}" y="{cy + r * 0.36:.1f}" font-size="{r * 1.0:.1f}" '
                f'font-family="ui-sans-serif,system-ui,sans-serif" font-weight="600" '
                f'fill="#ffffff" text-anchor="middle">{label}</text>')

    counts = {}
    for f in building.features:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    summary = "   ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    out += [
        "</g>",
        f'<text x="{pad}" y="{h + 2 * pad + 16:.0f}" font-size="13" '
        f'font-family="ui-sans-serif,system-ui,sans-serif" fill="#3d4550">'
        f'{title} — {w_m:g}×{h_m:g} m, {len(building.solids)} boxes   {summary}</text>',
        "</svg>",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", help="path to a scenario YAML")
    ap.add_argument("-o", "--out", default=None, help="output SVG (default: alongside)")
    args = ap.parse_args()

    cfg = load_config()
    building = load_scenario(args.scenario, cfg["room"])
    out = Path(args.out) if args.out else Path(args.scenario).with_suffix(".svg")
    out.write_text(render(building, Path(args.scenario).stem))
    print(f"{out}  —  {len(building.solids)} boxes, {len(building.features)} features")


if __name__ == "__main__":
    main()

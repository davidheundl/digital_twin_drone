"""Compiling an authored floorplan into building geometry.

An ASCII grid per storey is the authoring format: one character per cell, with
row 0 read as *maximum* y so the text reads like a map with north at the top.
Solid characters are merged into as few AABBs as possible by greedy meshing;
every other marked character becomes a `Feature` and never becomes geometry,
because a victim is something to find, not something to bump into.

Author at 0.5 m cells with two-cell doorways. A one-cell gap is nominally wide
enough for a 0.14 m collision sphere, but leaves no room for the position
loop's error, so doors that narrow are effectively closed.

Stairs are markers, not geometry: Phase 1 is one storey, and a stairwell is a
place a later phase will spawn a drone, not a ramp to fly up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .building import Building, Feature, FloorSpec
from .geometry import BoxSet

FREE = "."

#: Character -> material class. These are merged into AABBs.
SOLIDS = {"#": "wall", "%": "debris"}

#: Character -> (feature kind, id prefix). These sit in free space.
MARKERS = {
    "+": ("door", "d"),
    "v": ("victim", "v"),
    "a": ("anchor", "a"),
    "h": ("decoy", "h"),
    "e": ("spawn", "e"),
    "/": ("stair", "s"),
}

#: Default z extent per material, relative to the storey floor.
#: An upper of None means "up to the ceiling".
DEFAULT_MATERIAL_Z = {"wall": (0.0, None), "debris": (0.0, 0.9)}

#: Default height above the storey floor for each marker kind.
#: None means "just under the ceiling".
DEFAULT_MARKER_Z = {
    "victim": 0.4,
    "decoy": 0.4,
    "door": 1.0,
    "spawn": 0.3,
    "stair": 0.0,
    "anchor": None,
}

ANCHOR_DROP = 0.2  # m below the ceiling that anchors are mounted


class FloorplanError(ValueError):
    """An authored floorplan the compiler refuses to guess about."""


# ----------------------------------------------------------------- parsing

def _parse_grid(text: str, floor_index: int) -> list[str]:
    """Split a grid block into rows, rejecting anything ambiguous."""
    if not isinstance(text, str):
        raise FloorplanError(f"floor {floor_index}: 'grid' must be a text block")

    lines = text.splitlines()
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        raise FloorplanError(f"floor {floor_index}: grid is empty")

    width = len(lines[0])
    known = set(SOLIDS) | set(MARKERS) | {FREE}
    for r, line in enumerate(lines):
        if len(line) != width:
            raise FloorplanError(
                f"floor {floor_index}, row {r}: ragged grid — {len(line)} characters, "
                f"expected {width}. Pad with {FREE!r}; a space is not free space.")
        for c, ch in enumerate(line):
            if ch not in known:
                raise FloorplanError(
                    f"floor {floor_index}, row {r}, column {c}: unknown character "
                    f"{ch!r}. Known characters: {' '.join(sorted(known))}")
    return lines


def _scan(lines: list[str]):
    """Rows of text -> per-material masks indexed [x, y], plus marker components.

    Row 0 is maximum y, so a grid `height` rows tall puts row r at y index
    `height - 1 - r`.

    Markers are grouped into 4-connected runs of the same character, one
    feature per run: a two-cell doorway is one door, not two. Runs come back
    in reading order of their first cell — top row first, left to right —
    which is what makes the generated ids stable across runs.
    """
    height, width = len(lines), len(lines[0])
    masks = {material: np.zeros((width, height), dtype=bool)
             for material in sorted(set(SOLIDS.values()))}
    for r, line in enumerate(lines):
        iy = height - 1 - r
        for ix, ch in enumerate(line):
            if ch in SOLIDS:
                masks[SOLIDS[ch]][ix, iy] = True

    seen = np.zeros((height, width), dtype=bool)
    markers: list[tuple[str, str, float, float]] = []
    for r in range(height):
        for c in range(width):
            ch = lines[r][c]
            if ch not in MARKERS or seen[r, c]:
                continue
            kind, prefix = MARKERS[ch]
            seen[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if (0 <= nr < height and 0 <= nc < width
                            and not seen[nr, nc] and lines[nr][nc] == ch):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            ix = sum(cc for _, cc in cells) / len(cells) + 0.5
            iy = sum(height - 1 - rr for rr, _ in cells) / len(cells) + 0.5
            markers.append((kind, prefix, ix, iy))
    return masks, markers, width, height


# ------------------------------------------------------------ greedy meshing

def _greedy_rects(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cover a bool mask indexed [x, y] with maximal rectangles.

    Grown in x first, then in y while the whole x-run stays available. Emitted
    in (y0, x0) order because the scan is y-outer, which is what makes the box
    list byte-stable across runs.
    """
    width, height = mask.shape
    claimed = np.zeros_like(mask)
    rects: list[tuple[int, int, int, int]] = []

    for y in range(height):
        for x in range(width):
            if not mask[x, y] or claimed[x, y]:
                continue
            x1 = x
            while x1 + 1 < width and mask[x1 + 1, y] and not claimed[x1 + 1, y]:
                x1 += 1
            y1 = y
            while (y1 + 1 < height
                   and mask[x:x1 + 1, y1 + 1].all()
                   and not claimed[x:x1 + 1, y1 + 1].any()):
                y1 += 1
            claimed[x:x1 + 1, y:y1 + 1] = True
            rects.append((x, y, x1 + 1, y1 + 1))
    return rects


# ---------------------------------------------------------------- compiling

def build_from_spec(spec: dict, room_cfg: dict | None = None) -> Building:
    """Compile a parsed scenario mapping into a `Building`.

    `room_cfg` supplies the contact constants (restitution, friction, crash
    speed) so a scenario only has to describe shape; any of them can be
    overridden at the top level of the scenario.
    """
    if not isinstance(spec, dict):
        raise FloorplanError("a scenario must be a mapping")

    room_cfg = room_cfg or {}
    cell = float(spec.get("cell", 0.5))
    storey_height = float(spec.get("height", room_cfg.get("size", [0, 0, 3.0])[2]))
    if cell <= 0.0 or storey_height <= 0.0:
        raise FloorplanError("'cell' and 'height' must both be positive")

    material_z = dict(DEFAULT_MATERIAL_Z)
    for material, override in (spec.get("materials") or {}).items():
        if material not in material_z:
            raise FloorplanError(
                f"unknown material {material!r}; known: {' '.join(sorted(material_z))}")
        z_lo, z_hi = override["z"]
        material_z[material] = (float(z_lo), None if z_hi is None else float(z_hi))

    floors_spec = spec.get("floors")
    if floors_spec is None:
        if "grid" not in spec:
            raise FloorplanError("a scenario needs either 'grid' or 'floors'")
        floors_spec = [{"grid": spec["grid"]}]
    if not isinstance(floors_spec, list) or not floors_spec:
        raise FloorplanError("'floors' must be a non-empty list")

    lower: list[list[float]] = []
    upper: list[list[float]] = []
    tags: list[str] = []
    box_floor: list[int] = []
    features: list[Feature] = []
    floors: list[FloorSpec] = []
    counters: dict[str, int] = {}
    shape: tuple[int, int] | None = None

    for index, floor_spec in enumerate(floors_spec):
        lines = _parse_grid(floor_spec.get("grid"), index)
        masks, markers, width, height = _scan(lines)
        if shape is None:
            shape = (width, height)
        elif (width, height) != shape:
            raise FloorplanError(
                f"floor {index} is {width}x{height} cells but floor 0 is "
                f"{shape[0]}x{shape[1]}; every storey must share a footprint")

        z_lo = index * storey_height
        z_hi = z_lo + storey_height
        floors.append(FloorSpec(index, z_lo, z_hi))

        for material in sorted(masks):
            m_lo, m_hi = material_z[material]
            top = z_hi if m_hi is None else min(z_lo + m_hi, z_hi)
            base = z_lo + m_lo
            for x0, y0, x1, y1 in _greedy_rects(masks[material]):
                lower.append([x0 * cell, y0 * cell, base])
                upper.append([x1 * cell, y1 * cell, top])
                tags.append(material)
                box_floor.append(index)

        for kind, prefix, ix, iy in markers:
            counters[prefix] = counters.get(prefix, 0) + 1
            offset = DEFAULT_MARKER_Z[kind]
            z = z_hi - ANCHOR_DROP if offset is None else z_lo + offset
            features.append(Feature(
                kind=kind,
                id=f"{prefix}{counters[prefix]}",
                position=np.array([ix * cell, iy * cell, z]),
                floor=index,
            ))

    features = _apply_entities(features, spec.get("entities") or {})

    width, height = shape
    return Building(
        shell_size=[width * cell, height * cell, len(floors) * storey_height],
        restitution=float(spec.get("wall_restitution", room_cfg.get("wall_restitution", 0.25))),
        friction=float(spec.get("wall_friction", room_cfg.get("wall_friction", 0.4))),
        crash_speed=float(spec.get("crash_speed", room_cfg.get("crash_speed", 3.0))),
        solids=BoxSet(lower or np.zeros((0, 3)), upper or np.zeros((0, 3)),
                      tags=tags, floor=np.asarray(box_floor, dtype=int)),
        features=features,
        floors=floors,
    )


def _apply_entities(features: list[Feature], entities: dict) -> list[Feature]:
    """Merge per-instance overrides keyed by generated id.

    An id that matches nothing is an error rather than a no-op: it is almost
    always a typo, and silently dropping it loses a victim's detectability.
    """
    known = {f.id for f in features}
    unknown = sorted(set(entities) - known)
    if unknown:
        raise FloorplanError(
            f"'entities' refers to ids that the grid does not contain: "
            f"{', '.join(unknown)}. Present: {', '.join(sorted(known)) or '(none)'}")

    out = []
    for f in features:
        attrs = entities.get(f.id)
        out.append(f if not attrs else Feature(
            kind=f.kind, id=f.id, position=f.position, floor=f.floor,
            attrs={**f.attrs, **attrs}))
    return out


def load_scenario(path: str | Path, room_cfg: dict | None = None) -> Building:
    """Read a scenario YAML file and compile it."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    with open(path, "r") as fh:
        spec = yaml.safe_load(fh)
    try:
        return build_from_spec(spec, room_cfg)
    except FloorplanError as exc:
        raise FloorplanError(f"{path}: {exc}") from None

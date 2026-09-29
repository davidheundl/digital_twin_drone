"""The world's geometry: an exterior shell plus whatever solids stand inside it.

An empty building is exactly the old single-room box, which is why the default
config still describes a plain 10x8x3 m room and the protocol still reports
`room.size`. A building adds interior walls, debris and multiple storeys
without changing either.

Features (victims, UWB anchors, decoys, doors, spawn points) ride along here
because they are authored in the same floorplan, but they are *not* geometry:
nothing collides with them and nothing raycasts against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import EPS, BoxSet


@dataclass(frozen=True)
class Feature:
    kind: str                      # victim | anchor | decoy | door | spawn | stair
    id: str
    position: np.ndarray
    floor: int = 0
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FloorSpec:
    index: int
    z_lo: float
    z_hi: float


class Building:
    def __init__(self, shell_size, restitution, friction, crash_speed,
                 solids=None, features=None, floors=None, broadphase_cell=2.0):
        self.size = np.asarray(shell_size, dtype=float)
        self.lower = np.zeros(3)
        self.upper = self.size.copy()
        self.restitution = float(restitution)
        self.friction = float(friction)
        self.crash_speed = float(crash_speed)

        self.solids = solids if solids is not None else BoxSet.empty()
        self.features = list(features) if features else []
        self.floors = list(floors) if floors else [
            FloorSpec(0, float(self.lower[2]), float(self.upper[2]))
        ]
        self.broadphase_cell = float(broadphase_cell)
        if len(self.solids):
            self.solids.build_grid(self.broadphase_cell)

    @classmethod
    def empty_room(cls, cfg: dict) -> "Building":
        """The original bare box, straight from the `room:` config block."""
        return cls(shell_size=cfg["size"],
                   restitution=cfg["wall_restitution"],
                   friction=cfg["wall_friction"],
                   crash_speed=cfg["crash_speed"])

    def features_of(self, kind: str) -> list[Feature]:
        return [f for f in self.features if f.kind == kind]

    # ------------------------------------------------------------ containment

    def contains(self, p: np.ndarray, radius: float = 0.0) -> bool:
        """True when a sphere at `p` fits inside the shell and clear of solids."""
        if not (np.all(p >= self.lower + radius) and np.all(p <= self.upper - radius)):
            return False
        return not self.solids.penetrations(p, radius)

    def contacts(self, p: np.ndarray, radius: float):
        """Every surface a sphere at `p` is interpenetrating.

        Returns (normal, depth) pairs with the normal pointing the way the
        sphere must move. Shell faces and interior solids are reported the
        same way, so callers do not care which is which.

        All of them, not just the deepest: in a concave corner, correcting one
        surface alone pushes the sphere straight into its neighbour.
        """
        hits = []
        low_gap = p - (self.lower + radius)      # negative => through a low face
        high_gap = (self.upper - radius) - p     # negative => through a high face
        for axis in range(3):
            if low_gap[axis] < -EPS:
                normal = np.zeros(3)
                normal[axis] = 1.0
                hits.append((normal, float(-low_gap[axis])))
            if high_gap[axis] < -EPS:
                normal = np.zeros(3)
                normal[axis] = -1.0
                hits.append((normal, float(-high_gap[axis])))

        hits.extend((n, float(d)) for n, d, _ in self.solids.penetrations(p, radius))
        return hits

    def ground_z(self, xy) -> float:
        """Height of the surface directly beneath `xy` — a solid top, or the shell floor."""
        probe = np.array([float(xy[0]), float(xy[1]), float(self.upper[2])])
        top = self.solids.support_z(probe, 0.0)
        return float(self.lower[2]) if not np.isfinite(top) else top

    def support_z(self, p: np.ndarray, radius: float) -> float:
        """Surface a sphere at `p` would settle onto."""
        top = self.solids.support_z(p, radius)
        return float(self.lower[2]) if not np.isfinite(top) else max(top, float(self.lower[2]))

    # ---------------------------------------------------------------- raycast

    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_range: float) -> float:
        """Distance to the first surface along `direction` — shell or solid."""
        return float(self.raycast_batch(origin, np.atleast_2d(direction), max_range)[0])

    def raycast_batch(self, origin, dirs, max_range: float, idx=None) -> np.ndarray:
        """Vectorised raycast for (R,3) directions. Returns (R,).

        Pass `idx` from `solids.candidates(...)` to narrow the solid test. The
        caller owns that cache — the building is shared and stateless so a
        swarm does not thrash a single cursor.
        """
        dirs = np.atleast_2d(np.asarray(dirs, dtype=float))
        origin = np.asarray(origin, dtype=float)

        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        unit = np.divide(dirs, norms, out=np.zeros_like(dirs), where=norms > EPS)

        signed = np.where(unit < 0.0, -1.0, 1.0)
        inv = 1.0 / np.where(np.abs(unit) < EPS, signed * EPS, unit)
        # Ray starts inside the shell, so the exit is the nearest far-side slab.
        exit_t = np.maximum((self.lower - origin) * inv,
                            (self.upper - origin) * inv).min(axis=1)
        shell = np.clip(exit_t, 0.0, max_range)
        shell[norms[:, 0] <= EPS] = max_range

        if len(self.solids) == 0:
            return shell
        return np.minimum(shell, self.solids.raycast_batch(origin, dirs, max_range, idx))

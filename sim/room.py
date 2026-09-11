"""The world: one axis-aligned empty box.

Kept deliberately behind a small interface (`contains`, `penetration`, `raycast`)
so obstacles can be added later without touching the drone or the protocol.
"""

from __future__ import annotations

import numpy as np


class Room:
    def __init__(self, cfg: dict):
        self.size = np.array(cfg["size"], dtype=float)
        self.restitution = float(cfg["wall_restitution"])
        self.friction = float(cfg["wall_friction"])
        self.crash_speed = float(cfg["crash_speed"])
        self.lower = np.zeros(3)
        self.upper = self.size.copy()

    def contains(self, p: np.ndarray, radius: float = 0.0) -> bool:
        return bool(np.all(p >= self.lower + radius) and np.all(p <= self.upper - radius))

    def penetration(self, p: np.ndarray, radius: float):
        """Deepest wall violation for a sphere at p.

        Returns (normal, depth) with the normal pointing back into the room,
        or (None, 0.0) when the sphere is fully inside.
        """
        low_gap = p - (self.lower + radius)     # negative => through a low wall / floor
        high_gap = (self.upper - radius) - p    # negative => through a high wall / ceiling

        normal = np.zeros(3)
        depth = 0.0
        for axis in range(3):
            if low_gap[axis] < -1e-12 and -low_gap[axis] > depth:
                depth = -low_gap[axis]
                normal = np.zeros(3)
                normal[axis] = 1.0
            if high_gap[axis] < -1e-12 and -high_gap[axis] > depth:
                depth = -high_gap[axis]
                normal = np.zeros(3)
                normal[axis] = -1.0
        if depth == 0.0:
            return None, 0.0
        return normal, depth

    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_range: float) -> float:
        """Distance from origin to the first wall along direction (unit vector).

        Slab method against the box interior. Returns max_range if nothing is hit
        within range, and 0.0 if the origin is already outside.
        """
        d = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            return max_range
        d = d / norm

        best = np.inf
        for axis in range(3):
            if abs(d[axis]) < 1e-12:
                continue
            for bound in (self.lower[axis], self.upper[axis]):
                t = (bound - origin[axis]) / d[axis]
                if t < 0:
                    continue
                hit = origin + t * d
                ok = True
                for other in range(3):
                    if other == axis:
                        continue
                    if not (self.lower[other] - 1e-9 <= hit[other] <= self.upper[other] + 1e-9):
                        ok = False
                        break
                if ok:
                    best = min(best, t)
        if not np.isfinite(best):
            return max_range
        return float(min(best, max_range))

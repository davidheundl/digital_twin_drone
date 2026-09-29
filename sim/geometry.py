"""Axis-aligned box geometry: the raw spatial primitives, and nothing else.

No config, no drone, no simulation concepts — just a set of AABBs you can
raycast against, test a sphere for overlap with, and rasterise to a grid.
`sim/building.py` wraps this into something the world can use.

The whole set is immutable once constructed. That is what lets the broadphase
grid be built once at load and shared freely.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


class BoxSet:
    """N axis-aligned boxes, stored column-wise for vectorised queries.

    `floor` carries the storey index so multi-floor buildings are a data
    change rather than a code change.
    """

    def __init__(self, lower, upper, tags=None, floor=None):
        self.lower = np.atleast_2d(np.asarray(lower, dtype=float))
        self.upper = np.atleast_2d(np.asarray(upper, dtype=float))
        if self.lower.shape != self.upper.shape or self.lower.shape[1:] != (3,):
            raise ValueError(f"lower/upper must both be (N,3), got {self.lower.shape} "
                             f"and {self.upper.shape}")
        if np.any(self.upper < self.lower):
            bad = int(np.argmax(np.any(self.upper < self.lower, axis=1)))
            raise ValueError(f"box {bad} has upper < lower: "
                             f"{self.lower[bad]} .. {self.upper[bad]}")

        n = len(self.lower)
        self.tags = list(tags) if tags is not None else ["solid"] * n
        self.floor = (np.asarray(floor, dtype=int) if floor is not None
                      else np.zeros(n, dtype=int))
        if len(self.tags) != n or len(self.floor) != n:
            raise ValueError("tags/floor length must match the box count")

        self._cell = None
        self._grid_lower = None
        self._grid_shape = None
        self._offsets = None
        self._items = None

    @classmethod
    def empty(cls) -> "BoxSet":
        return cls(np.zeros((0, 3)), np.zeros((0, 3)), [], np.zeros(0, dtype=int))

    def __len__(self) -> int:
        return len(self.lower)

    @property
    def bounds(self):
        """(lower, upper) of the whole set, or None when empty."""
        if len(self) == 0:
            return None
        return self.lower.min(axis=0), self.upper.max(axis=0)

    # ------------------------------------------------------------- broadphase

    def build_grid(self, cell: float) -> None:
        """Bucket boxes into a uniform XY grid, stored CSR-style.

        XY only: buildings are wide and thin, and a z dimension would mostly
        add empty cells. Built once — the set is immutable.
        """
        if len(self) == 0:
            self._cell = float(cell)
            return

        lo, hi = self.bounds
        self._cell = float(cell)
        self._grid_lower = lo[:2].copy()
        span = np.maximum(hi[:2] - lo[:2], self._cell)
        self._grid_shape = tuple(int(np.floor(s / self._cell)) + 1 for s in span)

        ix0, iy0 = self._cell_index(self.lower[:, :2])
        ix1, iy1 = self._cell_index(self.upper[:, :2])

        ncells = self._grid_shape[0] * self._grid_shape[1]
        counts = np.zeros(ncells, dtype=np.int64)
        spans = []
        for b in range(len(self)):
            xs = np.arange(ix0[b], ix1[b] + 1)
            ys = np.arange(iy0[b], iy1[b] + 1)
            flat = (np.repeat(xs, len(ys)) * self._grid_shape[1]
                    + np.tile(ys, len(xs)))
            spans.append(flat)
            counts[flat] += 1

        self._offsets = np.zeros(ncells + 1, dtype=np.int64)
        np.cumsum(counts, out=self._offsets[1:])
        self._items = np.empty(self._offsets[-1], dtype=np.int32)
        cursor = self._offsets[:-1].copy()
        for b, flat in enumerate(spans):
            self._items[cursor[flat]] = b
            cursor[flat] += 1

    def _cell_index(self, xy: np.ndarray):
        rel = (np.asarray(xy, dtype=float) - self._grid_lower) / self._cell
        idx = np.floor(rel).astype(int)
        idx[..., 0] = np.clip(idx[..., 0], 0, self._grid_shape[0] - 1)
        idx[..., 1] = np.clip(idx[..., 1], 0, self._grid_shape[1] - 1)
        return idx[..., 0], idx[..., 1]

    def candidates(self, centre, radius: float) -> np.ndarray:
        """Box indices whose XY extent may lie within `radius` of `centre`.

        A superset of the true hit set — never a filter, only a narrowing.
        Returns every index when no grid has been built.
        """
        if len(self) == 0:
            return np.zeros(0, dtype=np.int32)
        if self._offsets is None:
            return np.arange(len(self), dtype=np.int32)

        centre = np.asarray(centre, dtype=float)
        query = np.stack([centre[:2] - radius, centre[:2] + radius])
        ix, iy = self._cell_index(query)
        ix0, ix1 = int(ix.min()), int(ix.max())
        iy0, iy1 = int(iy.min()), int(iy.max())

        flat = (np.repeat(np.arange(ix0, ix1 + 1), iy1 - iy0 + 1) * self._grid_shape[1]
                + np.tile(np.arange(iy0, iy1 + 1), ix1 - ix0 + 1))
        starts, ends = self._offsets[flat], self._offsets[flat + 1]
        if not np.any(ends > starts):
            return np.zeros(0, dtype=np.int32)
        picked = np.concatenate([self._items[s:e] for s, e in zip(starts, ends) if e > s])
        return np.unique(picked)

    # ---------------------------------------------------------------- queries

    def raycast_batch(self, origin, dirs, max_range: float, idx=None) -> np.ndarray:
        """Distance from `origin` to the first box entered, per direction.

        `dirs` is (R,3); returns (R,). Directions need not be normalised.
        Solid-entry semantics: a ray starting inside a box returns 0.0.
        Returns `max_range` where nothing is hit.

        One batched slab test over all candidate boxes. Do not "optimise" this
        into a per-ray grid walk — Python call overhead dominates, and a
        per-ray loop measures slower than this does over the whole set.
        """
        dirs = np.atleast_2d(np.asarray(dirs, dtype=float))
        out = np.full(len(dirs), float(max_range))
        if len(self) == 0:
            return out

        boxes = np.arange(len(self)) if idx is None else np.asarray(idx, dtype=int)
        if len(boxes) == 0:
            return out

        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        live = norms[:, 0] > EPS
        if not np.any(live):
            return out
        d = np.divide(dirs, norms, out=np.zeros_like(dirs), where=norms > EPS)[live]

        # Preserve sign on near-zero components: a ray parallel to a slab must
        # produce ±inf bounds, not a spurious finite crossing.
        signed = np.where(d < 0.0, -1.0, 1.0)
        inv = 1.0 / np.where(np.abs(d) < EPS, signed * EPS, d)

        origin = np.asarray(origin, dtype=float)
        lo = self.lower[boxes] - origin          # (B,3)
        hi = self.upper[boxes] - origin

        t1 = lo[None, :, :] * inv[:, None, :]    # (R,B,3)
        t2 = hi[None, :, :] * inv[:, None, :]
        tmin = np.minimum(t1, t2).max(axis=2)
        tmax = np.maximum(t1, t2).min(axis=2)

        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmax >= 0.0) & (tmin <= max_range)
        entry = np.where(tmin < 0.0, 0.0, tmin)
        dist = np.where(hit, entry, np.inf).min(axis=1)

        out[live] = np.minimum(dist, max_range)
        return out

    def penetrations(self, p, radius: float, idx=None):
        """Every box a sphere at `p` overlaps.

        Returns a list of (normal, depth, box_index), the normal pointing out
        of the solid — i.e. the direction to push the sphere. Callers must
        resolve *all* of these, not just the deepest: in a concave corner,
        correcting one box alone shoves the sphere into its neighbour.
        """
        if len(self) == 0:
            return []
        boxes = self.candidates(p, radius) if idx is None else np.asarray(idx, dtype=int)
        if len(boxes) == 0:
            return []

        p = np.asarray(p, dtype=float)
        lo, hi = self.lower[boxes], self.upper[boxes]
        closest = np.clip(p, lo, hi)
        delta = p - closest
        dist2 = np.einsum("ij,ij->i", delta, delta)

        # A centre strictly inside a box always counts, however small the
        # sphere — otherwise a zero-radius point test never reports anything.
        overlapping = (dist2 <= EPS) | (dist2 < radius * radius)

        hits = []
        for k in np.nonzero(overlapping)[0]:
            b = int(boxes[k])
            if dist2[k] > EPS:
                dist = float(np.sqrt(dist2[k]))
                hits.append((delta[k] / dist, radius - dist, b))
            else:
                # Centre is inside the box: push out through the nearest face.
                gaps = np.stack([p - lo[k], hi[k] - p])      # (2,3)
                axis = int(np.argmin(gaps) % 3)
                side = int(np.argmin(gaps) // 3)
                normal = np.zeros(3)
                normal[axis] = -1.0 if side == 0 else 1.0
                hits.append((normal, float(gaps[side, axis]) + radius, b))
        return hits

    def support_z(self, p, radius: float, tol: float = 1e-6) -> float:
        """Top of the highest box the sphere at `p` could be resting on.

        XY footprint expanded by `radius` so a sphere overhanging an edge is
        still supported. Returns -inf when nothing is underneath.
        """
        if len(self) == 0:
            return -np.inf
        boxes = self.candidates(p, radius)
        if len(boxes) == 0:
            return -np.inf

        p = np.asarray(p, dtype=float)
        lo, hi = self.lower[boxes], self.upper[boxes]
        under = (
            (p[0] >= lo[:, 0] - radius) & (p[0] <= hi[:, 0] + radius)
            & (p[1] >= lo[:, 1] - radius) & (p[1] <= hi[:, 1] + radius)
            & (hi[:, 2] <= p[2] + tol)
        )
        return float(hi[under, 2].max()) if np.any(under) else -np.inf

    def rasterize(self, res: float, z_lo: float, z_hi: float, bounds) -> np.ndarray:
        """Occupancy mask over an XY grid: True where a box overlaps the cell.

        `bounds` is (lower_xy, upper_xy). Only boxes whose z range intersects
        [z_lo, z_hi] contribute, which is what makes it per-floor.
        """
        lower_xy, upper_xy = (np.asarray(b, dtype=float)[:2] for b in bounds)
        shape = tuple(int(np.ceil((upper_xy[i] - lower_xy[i]) / res)) for i in range(2))
        mask = np.zeros(shape, dtype=bool)
        if len(self) == 0:
            return mask

        on_floor = (self.upper[:, 2] > z_lo) & (self.lower[:, 2] < z_hi)
        for b in np.nonzero(on_floor)[0]:
            i0, j0 = np.floor((self.lower[b, :2] - lower_xy) / res).astype(int)
            i1, j1 = np.ceil((self.upper[b, :2] - lower_xy) / res).astype(int)
            i0, j0 = max(i0, 0), max(j0, 0)
            i1, j1 = min(i1, shape[0]), min(j1, shape[1])
            if i1 > i0 and j1 > j0:
                mask[i0:i1, j0:j1] = True
        return mask

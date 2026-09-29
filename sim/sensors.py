"""Simulated sensor suite.

Every reading is derived from ground truth plus configurable noise and bias.
Biases are drawn once per reset from the seeded RNG, so a given seed always
produces the same sensor behaviour - this is what makes runs reproducible.

Beams are arranged in rings by elevation rather than a single horizontal fan:
a flat fan cannot see a stairwell, and a drone that pitches to translate ends
up sweeping its "horizontal" beams across the floor and ceiling.
"""

from __future__ import annotations

import numpy as np

from .drone import GRAVITY, DroneParams
from .vecmath import quat_to_matrix


class SensorSuite:
    def __init__(self, cfg: dict, params: DroneParams, room):
        self.cfg = cfg
        self.params = params
        self.room = room
        self.noise_enabled = bool(cfg.get("noise_enabled", True))
        self.seed = int(cfg.get("seed", 0))

        dirs, azimuth, elevation = _beam_layout(cfg["beams"]["rings"])
        self.beam_dirs_body = dirs
        self.beam_azimuth_deg = azimuth
        self.beam_elevation_deg = elevation
        self.beam_count = len(dirs)

        # Rays are cast in one batch: every beam plus the downward rangefinder.
        self._ray_dirs_body = np.vstack([dirs, [[0.0, 0.0, -1.0]]])
        self._reach = max(float(cfg["beams"]["max_range"]),
                          float(cfg["rangefinder"]["max_range"]))
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        imu = self.cfg["imu"]
        if self.noise_enabled:
            self.accel_bias = self.rng.normal(0.0, float(imu["accel_bias"]), 3)
            self.gyro_bias = self.rng.normal(0.0, float(imu["gyro_bias"]), 3)
        else:
            self.accel_bias = np.zeros(3)
            self.gyro_bias = np.zeros(3)
        self._candidate_idx = None
        self._candidate_centre = None

    def _noise(self, sigma: float, size=None):
        if not self.noise_enabled or sigma <= 0:
            return np.zeros(size) if size else 0.0
        return self.rng.normal(0.0, sigma, size)

    def _candidates(self, position: np.ndarray):
        """Broadphase box indices, cached until the drone leaves the margin.

        Rebuilding this every frame costs more than the raycast it accelerates,
        so the query is padded by half a cell and reused until the drone has
        moved that far.
        """
        margin = 0.5 * self.room.broadphase_cell
        if (self._candidate_idx is None
                or np.linalg.norm(position[:2] - self._candidate_centre[:2]) > margin):
            self._candidate_idx = self.room.solids.candidates(position, self._reach + margin)
            self._candidate_centre = position.copy()
        return self._candidate_idx

    def read(self, state, accel_world: np.ndarray, powered: bool) -> dict:
        """Produce one sensor frame. accel_world excludes gravity's free-fall effect."""
        if not powered:
            return {"valid": False}

        R = quat_to_matrix(state.orientation)
        imu = self.cfg["imu"]

        # Accelerometer measures specific force: (a - g) expressed in the body frame.
        specific_force_world = accel_world - GRAVITY
        accel_body = R.T @ specific_force_world
        accel = accel_body + self.accel_bias + self._noise(float(imu["accel_noise"]), 3)

        gyro = state.omega + self.gyro_bias + self._noise(float(imu["gyro_noise"]), 3)

        baro_cfg = self.cfg["barometer"]
        altitude = float(state.position[2]) + float(self._noise(float(baro_cfg["noise"])))

        dirs_world = self._ray_dirs_body @ R.T
        hits = self.room.raycast_batch(state.position, dirs_world, self._reach,
                                       idx=self._candidates(state.position))
        hits = np.maximum(0.0, hits - self.params.body_radius)

        rf = self.cfg["rangefinder"]
        raw_down = min(float(hits[-1]), float(rf["max_range"]))
        if raw_down < float(rf["min_range"]):
            down_reading = float(rf["min_range"])
        else:
            down_reading = raw_down + float(self._noise(float(rf["noise"])))

        beams_cfg = self.cfg["beams"]
        beams = np.minimum(hits[:-1], float(beams_cfg["max_range"]))
        beams = beams + self._noise(float(beams_cfg["noise"]), self.beam_count)

        return {
            "valid": True,
            "accel": [round(float(v), 5) for v in accel],
            "gyro": [round(float(v), 6) for v in gyro],
            "altitude": round(altitude, 4),
            "range_down": round(down_reading, 4),
            "beams": [round(float(b), 4) for b in beams],
        }


def _beam_layout(rings):
    """Build unit beam directions in the body frame from a ring spec.

    Each ring is `{elevation_deg, count}`: `count` beams evenly spaced in
    azimuth at a fixed elevation. Elevation 90 with count 1 is the zenith beam.
    """
    dirs, azimuth, elevation = [], [], []
    for ring in rings:
        el_deg = float(ring["elevation_deg"])
        count = int(ring["count"])
        if count < 1:
            raise ValueError(f"beam ring at {el_deg} deg needs at least one beam")
        el = np.radians(el_deg)
        for az in np.linspace(0.0, 2 * np.pi, count, endpoint=False):
            dirs.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            azimuth.append(float(np.degrees(az)))
            elevation.append(el_deg)
    return np.array(dirs, dtype=float), azimuth, elevation

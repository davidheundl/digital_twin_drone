"""Simulated sensor suite.

Every reading is derived from ground truth plus configurable noise and bias.
Biases are drawn once per reset from the seeded RNG, so a given seed always
produces the same sensor behaviour - this is what makes runs reproducible.
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
        self.beam_count = int(cfg["beams"]["count"])
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

    def _noise(self, sigma: float, size=None):
        if not self.noise_enabled or sigma <= 0:
            return np.zeros(size) if size else 0.0
        return self.rng.normal(0.0, sigma, size)

    def beam_directions_body(self) -> np.ndarray:
        angles = np.linspace(0.0, 2 * np.pi, self.beam_count, endpoint=False)
        return np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)

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

        rf = self.cfg["rangefinder"]
        down_body = np.array([0.0, 0.0, -1.0])
        down_world = R @ down_body
        raw_down = self.room.raycast(state.position, down_world, float(rf["max_range"]))
        raw_down = max(0.0, raw_down - self.params.body_radius)
        if raw_down < float(rf["min_range"]):
            down_reading = float(rf["min_range"])
        else:
            down_reading = raw_down + float(self._noise(float(rf["noise"])))

        beams_cfg = self.cfg["beams"]
        beams = []
        for direction_body in self.beam_directions_body():
            direction_world = R @ direction_body
            dist = self.room.raycast(state.position, direction_world,
                                     float(beams_cfg["max_range"]))
            dist = max(0.0, dist - self.params.body_radius)
            beams.append(dist + float(self._noise(float(beams_cfg["noise"]))))

        return {
            "valid": True,
            "accel": [round(float(v), 5) for v in accel],
            "gyro": [round(float(v), 6) for v in gyro],
            "altitude": round(altitude, 4),
            "range_down": round(down_reading, 4),
            "beams": [round(float(b), 4) for b in beams],
        }

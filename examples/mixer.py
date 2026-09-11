"""Motor mixer for the twin's X layout.

Converts a desired total thrust (N) and body torques (N m) into the four
per-motor thrusts, then normalises them to [0, 1] commands.

Derived from the layout in sim/drone.py:
    tx =  a(-T0 + T1 + T2 - T3)
    ty =  a(-T0 + T1 - T2 + T3)
    tz =  c( T0 + T1 - T2 - T3)
    T  =    T0 + T1 + T2 + T3
with a = arm_length / sqrt(2) and c = torque_coefficient.

This lives in examples/ rather than sim/ on purpose: the twin itself must not
contain any part of your control stack.
"""

import numpy as np


class Mixer:
    def __init__(self, arm_length: float, torque_coefficient: float, max_thrust: float):
        self.a = arm_length / np.sqrt(2.0)
        self.c = torque_coefficient
        self.max_thrust = max_thrust

    def mix(self, total_thrust: float, tx: float, ty: float, tz: float) -> np.ndarray:
        x = tx / (4.0 * self.a)
        y = ty / (4.0 * self.a)
        z = tz / (4.0 * self.c)
        q = total_thrust / 4.0
        thrusts = np.array([
            q - x - y + z,
            q + x + y + z,
            q + x - y - z,
            q - x + y - z,
        ])
        return np.clip(thrusts / self.max_thrust, 0.0, 1.0)

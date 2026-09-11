"""Rigid-body quadcopter dynamics.

There is no autopilot here, by design. The only input is four normalised motor
commands in [0, 1]; everything the drone does follows from those plus physics.

Motor layout (X configuration, body FLU frame, viewed from above):

        x (forward)
            ^
     m2     |     m0
       \\    |    /
        \\   |   /
   y <----- + -----      m0 front-right (CW)   m1 back-left  (CW)
        /   |   \\        m2 front-left  (CCW)  m3 back-right (CCW)
       /    |    \\
     m1     |     m3

Diagonal pairs spin the same way, so yaw comes from the imbalance between the
two pairs and a symmetric throttle produces no net torque.
"""

from __future__ import annotations

import numpy as np

from .vecmath import quat_derivative, quat_from_euler, quat_normalize, quat_to_matrix

GRAVITY = np.array([0.0, 0.0, -9.80665])

# Motor index -> (unit offset direction in body XY, spin sign).
# Spin sign +1 means the rotor's angular velocity is along body +z (CCW seen
# from above), which produces a reaction torque on the airframe along -z.
_MOTOR_LAYOUT = [
    (np.array([+1.0, -1.0]) / np.sqrt(2), -1.0),   # m0 front-right, CW
    (np.array([-1.0, +1.0]) / np.sqrt(2), -1.0),   # m1 back-left,   CW
    (np.array([+1.0, +1.0]) / np.sqrt(2), +1.0),   # m2 front-left,  CCW
    (np.array([-1.0, -1.0]) / np.sqrt(2), +1.0),   # m3 back-right,  CCW
]


class DroneParams:
    def __init__(self, cfg: dict):
        self.mass = float(cfg["mass"])
        self.inertia = np.array(cfg["inertia"], dtype=float)
        self.inertia_inv = 1.0 / self.inertia
        self.arm_length = float(cfg["arm_length"])
        self.body_radius = float(cfg["body_radius"])
        self.motor_max_thrust = float(cfg["motor_max_thrust"])
        self.motor_tau = float(cfg["motor_tau"])
        self.torque_coefficient = float(cfg["torque_coefficient"])
        self.linear_drag = np.array(cfg["linear_drag"], dtype=float)
        self.angular_drag = float(cfg["angular_drag"])
        self.initial_position = np.array(cfg["initial_position"], dtype=float)
        self.initial_yaw = float(cfg["initial_yaw"])

        self.motor_positions = np.array([
            np.array([offset[0], offset[1], 0.0]) * self.arm_length
            for offset, _ in _MOTOR_LAYOUT
        ])
        self.motor_spin = np.array([spin for _, spin in _MOTOR_LAYOUT])

    @property
    def weight(self) -> float:
        return self.mass * abs(GRAVITY[2])

    @property
    def hover_throttle(self) -> float:
        """Normalised per-motor command that exactly cancels gravity."""
        return self.weight / (4.0 * self.motor_max_thrust)


class DroneState:
    """Full physical state. `thrust` is per-motor actual thrust in newtons."""

    __slots__ = ("position", "velocity", "orientation", "omega", "thrust")

    def __init__(self, position, velocity, orientation, omega, thrust):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.orientation = np.asarray(orientation, dtype=float)
        self.omega = np.asarray(omega, dtype=float)
        self.thrust = np.asarray(thrust, dtype=float)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity, self.orientation,
                               self.omega, self.thrust])

    @staticmethod
    def from_vector(v: np.ndarray) -> "DroneState":
        return DroneState(v[0:3], v[3:6], quat_normalize(v[6:10]), v[10:13], v[13:17])

    def copy(self) -> "DroneState":
        return DroneState(self.position.copy(), self.velocity.copy(),
                          self.orientation.copy(), self.omega.copy(), self.thrust.copy())


def initial_state(params: DroneParams, room_floor_z: float = 0.0) -> DroneState:
    """Drone at rest on the floor, level, motors stopped."""
    pos = params.initial_position.copy()
    pos[2] = room_floor_z + params.body_radius
    return DroneState(
        position=pos,
        velocity=np.zeros(3),
        orientation=quat_from_euler(0.0, 0.0, params.initial_yaw),
        omega=np.zeros(3),
        thrust=np.zeros(4),
    )


def body_forces_torques(params: DroneParams, thrust: np.ndarray):
    """Net force (body frame) and torque (body frame) from the four rotors."""
    total_thrust = float(np.sum(thrust))
    force_body = np.array([0.0, 0.0, total_thrust])

    # Torque from thrust offsets: sum over rotors of p_i x (T_i * z_hat)
    torque = np.zeros(3)
    torque[0] = float(np.sum(thrust * params.motor_positions[:, 1]))
    torque[1] = float(-np.sum(thrust * params.motor_positions[:, 0]))
    # Reaction torque about body z, opposite to each rotor's spin direction.
    torque[2] = float(-np.sum(params.motor_spin * thrust) * params.torque_coefficient)
    return force_body, torque


def derivative(params: DroneParams, state: DroneState, motor_cmd: np.ndarray) -> np.ndarray:
    """State derivative for the free-flight (no contact) case."""
    R = quat_to_matrix(state.orientation)
    force_body, torque_body = body_forces_torques(params, state.thrust)

    force_world = R @ force_body
    force_world = force_world + params.mass * GRAVITY
    force_world = force_world - params.linear_drag * state.velocity

    accel = force_world / params.mass

    torque_body = torque_body - params.angular_drag * state.omega
    gyroscopic = np.cross(state.omega, params.inertia * state.omega)
    omega_dot = params.inertia_inv * (torque_body - gyroscopic)

    q_dot = quat_derivative(state.orientation, state.omega)

    thrust_target = np.clip(motor_cmd, 0.0, 1.0) * params.motor_max_thrust
    thrust_dot = (thrust_target - state.thrust) / params.motor_tau

    return np.concatenate([state.velocity, accel, q_dot, omega_dot, thrust_dot])


def integrate_rk4(params: DroneParams, state: DroneState, motor_cmd: np.ndarray,
                  dt: float) -> DroneState:
    y0 = state.to_vector()

    k1 = derivative(params, DroneState.from_vector(y0), motor_cmd)
    k2 = derivative(params, DroneState.from_vector(y0 + 0.5 * dt * k1), motor_cmd)
    k3 = derivative(params, DroneState.from_vector(y0 + 0.5 * dt * k2), motor_cmd)
    k4 = derivative(params, DroneState.from_vector(y0 + dt * k3), motor_cmd)

    y = y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    out = DroneState.from_vector(y)
    out.orientation = quat_normalize(out.orientation)
    out.thrust = np.clip(out.thrust, 0.0, params.motor_max_thrust)
    return out

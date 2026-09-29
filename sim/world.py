"""The simulated world: room + one drone + power state machine + contacts.

This module is pure simulation. It knows nothing about sockets, JSON or wall
clocks - it exposes `step(dt)` and a state snapshot, and the server layer drives
it. That separation is what lets the same code run in real time, in lockstep,
or as fast as possible inside a unit test.
"""

from __future__ import annotations

import numpy as np

from .battery import Battery
from .drone import DroneParams, DroneState, GRAVITY, initial_state, integrate_rk4
from .building import Building
from .floorplan import load_scenario
from .sensors import SensorSuite
from .vecmath import quat_from_euler, quat_normalize, quat_to_euler, quat_to_matrix

# Power / flight states. There is no FLYING or LANDING state on purpose: the
# twin has no notion of intent, only of physics. "airborne" is derived.
STATE_OFF = "OFF"
STATE_BOOTING = "BOOTING"
STATE_IDLE = "IDLE"
STATE_ARMED = "ARMED"
STATE_CRASHED = "CRASHED"

BOOT_DURATION = 0.5  # s
CONTACT_PASSES = 4   # deepest-first contact resolution sweeps per step


class World:
    def __init__(self, cfg):
        self.cfg = cfg
        # No `scenario:` means the original bare box, which is what keeps the
        # default config — and every test written against it — unchanged.
        scenario = cfg.get("scenario")
        self.building = (load_scenario(scenario, cfg["room"]) if scenario
                         else Building.empty_room(cfg["room"]))
        self.params = DroneParams(cfg["drone"])
        # A floorplan that names an entry point owns where the drone starts;
        # `drone.initial_position` is a coordinate in an empty box and means
        # nothing once there are walls to start inside of.
        spawn = self.building.features_of("spawn")
        if spawn:
            self.params.initial_position[:2] = spawn[0].position[:2]
        self.battery = Battery(cfg["battery"])
        self.sensors = SensorSuite(cfg["sensors"], self.params, self.building)
        self.reset()

    @property
    def room(self) -> Building:
        """The building, under the name the protocol and viewer still use."""
        return self.building

    @room.setter
    def room(self, building: Building) -> None:
        self.building = building

    # ---------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        self.t = 0.0
        self.step_count = 0
        self.state = initial_state(
            self.params, self.room.ground_z(self.params.initial_position[:2]))
        self.mode = STATE_OFF
        self.motor_cmd = np.zeros(4)
        self.last_accel_world = np.zeros(3)
        self.battery.reset()
        self.sensors.reset()
        self._boot_timer = 0.0
        self._landed = True
        self._support_z = self.room.ground_z(self.state.position[:2])
        self._pending_events: list[dict] = []

    def _emit(self, name: str, **data) -> None:
        self._pending_events.append({"event": name, "t": round(self.t, 6), **data})

    def drain_events(self) -> list[dict]:
        events, self._pending_events = self._pending_events, []
        return events

    # ----------------------------------------------------------------- commands

    def power_on(self) -> tuple[bool, str]:
        if self.mode == STATE_CRASHED:
            return False, "drone is crashed; call reset first"
        if self.mode != STATE_OFF:
            return False, f"already powered (state={self.mode})"
        if self.battery.empty:
            return False, "battery empty"
        self.mode = STATE_BOOTING
        self._boot_timer = 0.0
        self._emit("state_changed", state=self.mode)
        return True, "booting"

    def power_off(self) -> tuple[bool, str]:
        if self.mode == STATE_OFF:
            return False, "already off"
        self.mode = STATE_OFF
        self.motor_cmd = np.zeros(4)
        self.state.thrust = np.zeros(4)
        self._emit("state_changed", state=self.mode)
        return True, "powered off"

    def arm(self) -> tuple[bool, str]:
        if self.mode == STATE_CRASHED:
            return False, "drone is crashed; call reset first"
        if self.mode == STATE_OFF:
            return False, "not powered on"
        if self.mode == STATE_BOOTING:
            return False, "still booting"
        if self.mode == STATE_ARMED:
            return False, "already armed"
        if float(np.max(self.motor_cmd)) > 0.05:
            return False, "refusing to arm with motor command above idle"
        if self.battery.empty:
            return False, "battery empty"
        self.mode = STATE_ARMED
        self._emit("state_changed", state=self.mode)
        return True, "armed"

    def disarm(self) -> tuple[bool, str]:
        if self.mode != STATE_ARMED:
            return False, f"not armed (state={self.mode})"
        self.mode = STATE_IDLE
        self.motor_cmd = np.zeros(4)
        self._emit("state_changed", state=self.mode)
        return True, "disarmed"

    def emergency_stop(self) -> tuple[bool, str]:
        self.motor_cmd = np.zeros(4)
        self.state.thrust = np.zeros(4)
        if self.mode == STATE_ARMED:
            self.mode = STATE_IDLE
            self._emit("state_changed", state=self.mode)
        self._emit("emergency_stop")
        return True, "motors cut"

    def set_motors(self, values) -> tuple[bool, str]:
        arr = np.asarray(values, dtype=float)
        if arr.shape != (4,):
            return False, "expected exactly 4 motor values"
        if not np.all(np.isfinite(arr)):
            return False, "motor values must be finite"
        self.motor_cmd = np.clip(arr, 0.0, 1.0)
        return True, "ok"

    def teleport(self, position=None, yaw=None) -> tuple[bool, str]:
        if position is not None:
            p = np.asarray(position, dtype=float)
            if p.shape != (3,) or not np.all(np.isfinite(p)):
                return False, "position must be 3 finite numbers"
            if not self.room.contains(p, self.params.body_radius):
                return False, "position is outside the room"
            self.state.position = p
        if yaw is not None:
            self.state.orientation = quat_from_euler(0.0, 0.0, float(yaw))
        self.state.velocity = np.zeros(3)
        self.state.omega = np.zeros(3)
        self._emit("teleported")
        return True, "ok"

    # -------------------------------------------------------------------- step

    def step(self, dt: float) -> None:
        """Advance the world by one fixed physics step."""
        if self.mode == STATE_BOOTING:
            self._boot_timer += dt
            if self._boot_timer >= BOOT_DURATION:
                self.mode = STATE_IDLE
                self._emit("state_changed", state=self.mode)

        powered = self.mode != STATE_OFF
        # Motors only respond when armed. Anything else means zero thrust demand.
        effective_cmd = self.motor_cmd if self.mode == STATE_ARMED else np.zeros(4)

        velocity_before = self.state.velocity.copy()
        self.state = integrate_rk4(self.params, self.state, effective_cmd, dt)
        self._resolve_contacts(dt)

        self.last_accel_world = (self.state.velocity - velocity_before) / dt

        for name in self.battery.update(dt, powered, float(np.sum(self.state.thrust))):
            self._emit(name, percent=round(self.battery.percent, 2))
            if name == "battery_empty" and self.mode != STATE_OFF:
                self.power_off()

        self.t += dt
        self.step_count += 1

    def _resolve_contacts(self, dt: float) -> None:
        radius = self.params.body_radius
        applied: list[np.ndarray] = []
        on_floor = False

        # Resolve the deepest contact, then look again. A sphere wedged in a
        # concave corner touches several surfaces at once, and correcting only
        # one of them drives it straight into the next.
        for _ in range(CONTACT_PASSES):
            hits = self.room.contacts(self.state.position, radius)
            if not hits:
                break
            normal, depth = max(hits, key=lambda hit: hit[1])
            self.state.position = self.state.position + normal * depth
            applied.append(normal)
            if normal[2] > 0.5:
                on_floor = True
                self._support_z = float(self.state.position[2]) - radius

        impact_speed, impact_normal = 0.0, None
        for normal in applied:
            v_normal = float(np.dot(self.state.velocity, normal))
            if v_normal >= 0.0:
                continue
            if -v_normal > impact_speed:
                impact_speed, impact_normal = -v_normal, normal
            v_tangential = self.state.velocity - v_normal * normal
            self.state.velocity = (
                -self.room.restitution * v_normal * normal
                + (1.0 - self.room.friction) * v_tangential
            )

        if impact_normal is not None:
            self.state.omega = self.state.omega * (1.0 - self.room.friction)
            if impact_speed > self.room.crash_speed and self.mode == STATE_ARMED:
                self._crash(impact_speed)
                return
            if impact_speed > 0.05:
                self._emit("collision",
                           speed=round(impact_speed, 3),
                           normal=[float(n) for n in impact_normal])

        # Settle on the floor when the rotors cannot lift the airframe. Without
        # this the drone micro-bounces forever instead of just sitting there.
        thrust_total = float(np.sum(self.state.thrust))
        if on_floor and thrust_total < self.params.weight * 0.98:
            if not self._landed:
                self._landed = True
                self._emit("landed", speed=round(float(np.linalg.norm(self.state.velocity)), 3))
            self.state.position[2] = self._support_z + self.params.body_radius
            self.state.velocity = np.zeros(3)
            self.state.omega = np.zeros(3)
            yaw = float(quat_to_euler(self.state.orientation)[2])
            self.state.orientation = quat_from_euler(0.0, 0.0, yaw)
        elif self._landed and thrust_total >= self.params.weight * 0.98:
            self._landed = False
            self._emit("takeoff")

    def _crash(self, impact_speed: float) -> None:
        self.mode = STATE_CRASHED
        self.motor_cmd = np.zeros(4)
        self.state.thrust = np.zeros(4)
        self.state.velocity = np.zeros(3)
        self.state.omega = np.zeros(3)
        self._emit("crashed", speed=round(impact_speed, 3))
        self._emit("state_changed", state=self.mode)

    # ------------------------------------------------------------------ output

    @property
    def airborne(self) -> bool:
        return not self._landed

    def snapshot(self) -> dict:
        euler = quat_to_euler(self.state.orientation)
        sensors = self.sensors.read(self.state, self.last_accel_world,
                                    powered=self.mode != STATE_OFF)
        return {
            "t": round(self.t, 6),
            "step": self.step_count,
            "mode": self.mode,
            "airborne": self.airborne,
            "motors": {
                "command": [round(float(c), 5) for c in self.motor_cmd],
                "thrust_n": [round(float(t), 5) for t in self.state.thrust],
            },
            "battery": {
                "percent": round(self.battery.percent, 3),
                "voltage": round(self.battery.voltage, 3),
                "current": round(self.battery.current, 3),
            },
            "sensors": sensors,
            "truth": {
                "position": [round(float(v), 5) for v in self.state.position],
                "velocity": [round(float(v), 5) for v in self.state.velocity],
                "quaternion": [round(float(v), 6) for v in self.state.orientation],
                "euler": [round(float(v), 6) for v in euler],
                "omega": [round(float(v), 6) for v in self.state.omega],
            },
        }

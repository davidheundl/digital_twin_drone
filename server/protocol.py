"""Message parsing, validation and dispatch.

Every inbound message is validated here before it reaches the simulation, so a
buggy or hostile controller cannot put the world into an invalid state. Every
command produces exactly one ack - success or failure with a reason.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SCHEMA_VERSION = 1

CONTROLLER_COMMANDS = {
    "power_on", "power_off", "arm", "disarm", "emergency_stop", "set_motors",
    "reset", "teleport", "pause", "resume", "step", "set_time_scale",
}
OBSERVER_COMMANDS = {"get_state", "ping"}
ALL_COMMANDS = CONTROLLER_COMMANDS | OBSERVER_COMMANDS


class CommandError(Exception):
    pass


def parse_message(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise CommandError("message must be a JSON object")
    msg_type = raw.get("type")
    if not isinstance(msg_type, str):
        raise CommandError("missing string field 'type'")
    if msg_type not in ALL_COMMANDS and msg_type != "hello":
        raise CommandError(f"unknown command '{msg_type}'")
    return raw


def _numbers(value, count: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise CommandError(f"'{name}' must be a list of {count} numbers")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CommandError(f"'{name}' must contain only numbers")
        value_f = float(item)
        if not np.isfinite(value_f):
            raise CommandError(f"'{name}' must contain only finite numbers")
        out.append(value_f)
    return out


def dispatch(msg: dict, world, runtime) -> tuple[bool, str, dict]:
    """Apply a validated command. Returns (ok, reason, extra_payload)."""
    kind = msg["type"]

    if kind == "ping":
        return True, "pong", {}

    if kind == "get_state":
        return True, "ok", {"state": world.snapshot()}

    if kind == "power_on":
        runtime.note_command()
        return (*world.power_on(), {})
    if kind == "power_off":
        return (*world.power_off(), {})
    if kind == "arm":
        # Refresh the command clock so the failsafe does not fire in the gap
        # between arming and the controller's first motor command.
        runtime.note_command()
        return (*world.arm(), {})
    if kind == "disarm":
        return (*world.disarm(), {})
    if kind == "emergency_stop":
        return (*world.emergency_stop(), {})

    if kind == "set_motors":
        values = _numbers(msg.get("motors"), 4, "motors")
        ok, reason = world.set_motors(values)
        if ok:
            runtime.note_command()
        return ok, reason, {}

    if kind == "reset":
        world.reset()
        runtime.note_command()
        return True, "world reset", {}

    if kind == "teleport":
        position = msg.get("position")
        if position is not None:
            position = _numbers(position, 3, "position")
        yaw = msg.get("yaw")
        if yaw is not None:
            if isinstance(yaw, bool) or not isinstance(yaw, (int, float)):
                raise CommandError("'yaw' must be a number")
            yaw = float(yaw)
        return (*world.teleport(position, yaw), {})

    if kind == "pause":
        runtime.paused = True
        return True, "paused", {}

    if kind == "resume":
        runtime.paused = False
        runtime.note_command()
        return True, "resumed", {}

    if kind == "step":
        count = msg.get("steps", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count > 100000:
            raise CommandError("'steps' must be an integer in [1, 100000]")
        runtime.request_steps(count)
        return True, f"stepping {count}", {}

    if kind == "set_time_scale":
        scale = msg.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise CommandError("'scale' must be a number")
        scale = float(scale)
        if not (0.0 < scale <= 1000.0):
            raise CommandError("'scale' must be in (0, 1000]")
        runtime.time_scale = scale
        return True, f"time scale {scale}", {}

    raise CommandError(f"command '{kind}' is not implemented")


def hello_message(world, runtime) -> dict:
    """Sent to every client on connect: the contract plus the world's shape."""
    params = world.params
    return {
        "type": "hello",
        "schema_version": SCHEMA_VERSION,
        "sim": {
            "mode": runtime.mode,
            "physics_hz": runtime.physics_hz,
            "telemetry_hz": runtime.telemetry_hz,
            "time_scale": runtime.time_scale,
            "command_timeout_ms": runtime.command_timeout_ms,
        },
        "room": {"size": [float(v) for v in world.room.size]},
        "drone": {
            "mass": params.mass,
            "arm_length": params.arm_length,
            "body_radius": params.body_radius,
            "motor_max_thrust": params.motor_max_thrust,
            "motor_tau": params.motor_tau,
            "torque_coefficient": params.torque_coefficient,
            "hover_throttle": round(params.hover_throttle, 6),
            "inertia": [float(v) for v in params.inertia],
            "motor_layout": "X: m0 front-right CW, m1 back-left CW, "
                            "m2 front-left CCW, m3 back-right CCW",
        },
        "sensors": {
            "beam_count": world.sensors.beam_count,
            # Parallel arrays, all of length beam_count and in the same order
            # as telemetry's `beams`. A client that zips them against a
            # different length silently reads undefined.
            "beam_angles_deg": [round(a, 2) for a in world.sensors.beam_azimuth_deg],
            "beam_elevations_deg": [round(e, 2) for e in world.sensors.beam_elevation_deg],
            "beam_max_range": float(world.sensors.cfg["beams"]["max_range"]),
            "noise_enabled": world.sensors.noise_enabled,
        },
    }

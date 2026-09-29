"""Driving the world without a server.

Tests and the episode runner both need to step a `World` and dispatch real
protocol commands at it without an event loop or a socket. `StubRuntime`
supplies the handful of attributes `server.protocol` reads off the real
`Runtime`, and records what it was asked to do.
"""

from __future__ import annotations


class StubRuntime:
    """Stand-in for `server.runtime.Runtime` — records, never steps."""

    def __init__(self, cfg: dict):
        sim = cfg["sim"]
        self.mode = sim["mode"]
        self.physics_hz = int(sim["physics_hz"])
        self.telemetry_hz = int(sim["telemetry_hz"])
        self.time_scale = float(sim["time_scale"])
        self.command_timeout_ms = int(sim["command_timeout_ms"])
        self.paused = False
        self.commands = 0
        self.steps_requested = 0

    def note_command(self) -> None:
        self.commands += 1

    def request_steps(self, n: int) -> None:
        self.steps_requested += n

"""Drives World forward in time and owns the timing policy.

Two modes:
  realtime - the physics loop chases the wall clock (scaled by time_scale).
             Your controller races the sim, exactly like real hardware.
  lockstep - the sim advances only when the controller asks it to. Fully
             deterministic and safe to debug with breakpoints.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np


class Runtime:
    def __init__(self, world, cfg):
        sim_cfg = cfg["sim"]
        self.world = world
        self.physics_hz = int(sim_cfg["physics_hz"])
        self.telemetry_hz = int(sim_cfg["telemetry_hz"])
        self.viewer_hz = int(sim_cfg["viewer_hz"])
        self.time_scale = float(sim_cfg["time_scale"])
        self.mode = str(sim_cfg["mode"])
        self.command_timeout_ms = float(sim_cfg["command_timeout_ms"])
        self.failsafe_rampdown_ms = float(sim_cfg["failsafe_rampdown_ms"])

        if self.mode not in ("realtime", "lockstep"):
            raise ValueError(f"sim.mode must be 'realtime' or 'lockstep', got {self.mode!r}")

        self.dt = 1.0 / self.physics_hz
        self.steps_per_telemetry = max(1, round(self.physics_hz / self.telemetry_hz))
        self.paused = False
        self.running = False

        self._last_command_time = time.monotonic()
        self._failsafe_active = False
        self._pending_steps = 0
        self._step_event = asyncio.Event()
        self._subscribers: list = []
        self._seq = 0

    # ---------------------------------------------------------- subscriptions

    def subscribe(self, queue, every_n: int = 1) -> None:
        self._subscribers.append((queue, max(1, every_n), [0]))

    def unsubscribe(self, queue) -> None:
        self._subscribers = [s for s in self._subscribers if s[0] is not queue]

    def _publish(self, message: dict) -> None:
        for queue, every_n, counter in self._subscribers:
            counter[0] += 1
            if counter[0] % every_n != 0:
                continue
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # Slow consumer: drop the oldest frame rather than block physics.
                try:
                    queue.get_nowait()
                    queue.put_nowait(message)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    def _publish_events(self, events: list[dict]) -> None:
        for event in events:
            for queue, _, _ in self._subscribers:
                try:
                    queue.put_nowait({"type": "event", **event})
                except asyncio.QueueFull:
                    pass

    # --------------------------------------------------------------- stepping

    def note_command(self) -> None:
        self._last_command_time = time.monotonic()
        if self._failsafe_active:
            self._failsafe_active = False
            self.world._emit("failsafe_cleared")
        if self.mode == "lockstep":
            self._pending_steps = max(self._pending_steps, self.steps_per_telemetry)
            self._step_event.set()

    def request_steps(self, count: int) -> None:
        self._pending_steps += count
        self._step_event.set()

    def _apply_failsafe(self) -> None:
        """Hold the last command, then ramp motors to zero. No hidden autopilot."""
        if self.mode != "realtime":
            return
        # Only meaningful while the motors can actually do something.
        if self.world.mode != "ARMED":
            self._failsafe_active = False
            return
        silent_ms = (time.monotonic() - self._last_command_time) * 1000.0
        if silent_ms <= self.command_timeout_ms:
            return
        if not self._failsafe_active:
            self._failsafe_active = True
            self.world._emit("failsafe_engaged", silent_ms=round(silent_ms, 1))
        over = silent_ms - self.command_timeout_ms
        factor = max(0.0, 1.0 - over / max(1e-6, self.failsafe_rampdown_ms))
        self.world.motor_cmd = self.world.motor_cmd * factor

    def _emit_telemetry(self) -> None:
        self._seq += 1
        self._publish({
            "type": "telemetry",
            "seq": self._seq,
            "t_wall": round(time.time(), 6),
            **self.world.snapshot(),
        })

    async def run(self) -> None:
        self.running = True
        if self.mode == "realtime":
            await self._run_realtime()
        else:
            await self._run_lockstep()

    async def _run_realtime(self) -> None:
        next_deadline = time.monotonic()
        steps_since_telemetry = 0
        while self.running:
            if self.paused and self._pending_steps <= 0:
                await asyncio.sleep(0.005)
                next_deadline = time.monotonic()
                continue

            self._apply_failsafe()
            self.world.step(self.dt)
            if self._pending_steps > 0:
                self._pending_steps -= 1
            self._publish_events(self.world.drain_events())

            steps_since_telemetry += 1
            if steps_since_telemetry >= self.steps_per_telemetry:
                steps_since_telemetry = 0
                self._emit_telemetry()

            next_deadline += self.dt / self.time_scale
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            elif sleep_for < -0.25:
                # Fell badly behind; resynchronise instead of spiralling.
                next_deadline = time.monotonic()
                self.world._emit("sim_overrun")
            else:
                await asyncio.sleep(0)

    async def _run_lockstep(self) -> None:
        steps_since_telemetry = 0
        while self.running:
            if self._pending_steps <= 0:
                self._step_event.clear()
                try:
                    await asyncio.wait_for(self._step_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
            while self._pending_steps > 0 and self.running:
                self.world.step(self.dt)
                self._pending_steps -= 1
                self._publish_events(self.world.drain_events())
                steps_since_telemetry += 1
                if steps_since_telemetry >= self.steps_per_telemetry:
                    steps_since_telemetry = 0
                    self._emit_telemetry()
                await asyncio.sleep(0)

    def stop(self) -> None:
        self.running = False
        self._step_event.set()

# Drone Digital Twin

A physics-accurate digital twin of a single quadcopter sitting, powered off,
on the floor of an empty room. It exists to be flown *by your own program* —
this repository contains the world and the drone, nothing that flies it.

- **What it is**: 6-DOF rigid-body quadcopter physics in a closed box room,
  streamed over WebSocket in real time, with noisy sensors, a battery model,
  collisions, and a deterministic lockstep mode for reproducible testing.
- **What it is not**: an autopilot. There is no `takeoff()`, no `fly_to()`, no
  stabilization inside the twin. You send four raw motor commands; the twin
  tells you the truth about what happened. Building the flight stack —
  attitude control up through search algorithms — is the point of the
  project, and it happens entirely in your own process.

See [`PROTOCOL.md`](PROTOCOL.md) for the full wire contract.

## Quick start

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# terminal 1: the twin
./.venv/bin/python -m server.app

# terminal 2: prove the loop works end to end
./.venv/bin/python examples/hover_client.py --altitude 1.5 --duration 10
```

Open http://127.0.0.1:8080/ for a read-only 3D view (room, drone, trajectory
trail, live rangefinder beams). It's just another subscriber to the same
telemetry stream — it can never affect the simulation.

## Writing your own controller

Start with [`examples/minimal_client.py`](examples/minimal_client.py) — the
entire protocol surface in ~50 lines, in Python. The protocol is plain JSON
over a WebSocket, so any language with a WebSocket client works equally well.

[`examples/hover_client.py`](examples/hover_client.py) is a **reference
fixture**, not part of the twin: a conventional PID cascade (altitude → thrust,
attitude → rate → torque → motor mix) that proves the physics are flyable.
Read it if you're stuck on the inner loop; otherwise ignore it and build your
own — that's the whole point.

```
your program                    this repository
─────────────                   ───────────────
state estimator      <── noisy ── sensors (IMU, baro, rangefinder, 8 beams)
rate/attitude control
position control                     rigid-body physics
search algorithm      ── motors ──>  (gravity, drag, motors, battery, walls)
```

## Configuration

Everything about the world — room size, airframe mass/inertia, motor limits,
sensor noise, rates, ports — lives in [`config.yaml`](config.yaml). Nothing
is hardcoded elsewhere.

```bash
./.venv/bin/python -m server.app --mode lockstep   # deterministic stepping
./.venv/bin/python -m server.app --no-noise        # perfect sensors, for debugging
./.venv/bin/python -m server.app --time-scale 5    # 5x real time
```

## Project layout

```
sim/        physics: rigid body, room/collisions, battery, sensors — no I/O
server/     WebSocket server, protocol validation, real-time/lockstep runtime
viewer/     static read-only 3D viewer (three.js, no build step)
examples/   minimal client, mixer, reference hover controller — not the twin
tests/      physics invariants, protocol validation, determinism
config.yaml single source of truth for room/drone/sensor/rate parameters
PROTOCOL.md the wire contract — read this before writing a controller
```

## Testing

```bash
./.venv/bin/python -m pytest tests/ -v
```

36 tests cover: hover equilibrium, free-fall matching `g`, motors inert unless
armed, differential thrust producing the correct roll/yaw direction, the
drone never escaping the room under any impact, crash vs. safe-landing impact
thresholds, battery drainage, NaN/blowup resistance under adversarial input,
full protocol command validation and rejection reasons, and bit-identical
determinism given the same seed and command sequence.

## Design choices worth knowing

- **No autopilot in the twin.** `set_motors` is the only flight command. This
  is deliberate per the project scope — you write and own the entire control
  stack, from rate loop to search algorithm.
- **Ground truth is a cheat channel.** Telemetry carries both noisy `sensors`
  and exact `truth`. Use `truth` for debugging/scoring, not as your primary
  control input, or you're not testing anything realistic.
- **Failsafe, not autopilot.** Losing the link holds the last command briefly
  then ramps motors to zero — the drone falls, like real hardware on a lost
  link. It does not auto-hover.
- **Determinism requires lockstep.** `realtime` mode races the wall clock and
  cannot be bit-identical between runs by definition; use `lockstep` (driven
  by `step` or by every `set_motors` call) when you need to replay a bug.

# Drone Twin Protocol (schema_version 1)

WebSocket, `ws://127.0.0.1:8765` by default. One JSON object per message, both
directions. This file is the entire contract between the twin and your
controller — the twin implements nothing beyond what's written here.

## Connecting

The server sends a `hello` message the instant you connect:

```json
{
  "type": "hello",
  "schema_version": 1,
  "sim": {"mode": "realtime", "physics_hz": 1000, "telemetry_hz": 200,
          "time_scale": 1.0, "command_timeout_ms": 200},
  "room": {"size": [10.0, 8.0, 3.0]},
  "drone": {"mass": 0.25, "arm_length": 0.125, "body_radius": 0.14,
            "motor_max_thrust": 1.55, "motor_tau": 0.03,
            "hover_throttle": 0.3954, "inertia": [0.0013, 0.0013, 0.0024],
            "torque_coefficient": 0.016, "motor_layout": "X: ..."},
  "sensors": {"beam_count": 8, "beam_angles_deg": [0, 45, ...], "noise_enabled": true}
}
```

Everything you need to write a controller without reading the sim's source is
in this message: airframe parameters, room size, and rates.

By default your connection is a **controller** — it may send any command. To
open a read-only connection (e.g. a second dashboard), send
`{"type": "hello", "role": "observer"}` first; observers get telemetry at
`viewer_hz` instead of `telemetry_hz` and any control command they send is
rejected.

## Commands you send

Every command gets exactly one `ack` back:
`{"type": "ack", "id": <echoed if you sent one>, "ok": true|false, "reason": "..."}`.
`ok: false` always comes with a human-readable `reason` — nothing fails silently.

| Command | Payload | Notes |
|---|---|---|
| `power_on` | — | Rejected if crashed or battery empty. Takes ~0.5 s to boot. |
| `power_off` | — | Instant. Cuts motors. |
| `arm` | — | Only from `IDLE`. Rejected if any motor command is already above idle. |
| `disarm` | — | Only from `ARMED`. Zeroes motor commands. |
| `emergency_stop` | — | Cuts motors immediately from any state. Always succeeds. |
| `set_motors` | `motors: [m0,m1,m2,m3]` | Normalised 0.0–1.0 per rotor. Out-of-range values are clamped, not rejected. Ignored unless `ARMED`. This is the hot path — see rate note below. |
| `teleport` | `position: [x,y,z]`, `yaw: rad` (either optional) | Test/debug only. Rejected if the position is outside the room. Zeroes velocity and angular rate. |
| `reset` | — | Full world reset: `OFF`, on the floor, full battery. |
| `pause` / `resume` | — | Freezes/unfreezes physics. Realtime mode only in practice. |
| `step` | `steps: int` (1–100000) | Advance exactly N physics steps. Required to drive `lockstep` mode. |
| `set_time_scale` | `scale: float` (0, 1000] | Realtime mode only; 1.0 = wall clock. |
| `get_state` | — | Returns the current snapshot in the ack's `state` field, without waiting for the next telemetry tick. |
| `ping` | — | Ack reason is `"pong"`. Liveness check. |

### Motor layout (X configuration, body FLU frame)

```
        x (forward)
            ^
     m2     |     m0
       \    |    /
        \   |   /
   y <----- + -----
        /   |   \
       /    |    \
     m1     |     m3
```

`m0` front-right, `m1` back-left — both spin **CW**.
`m2` front-left, `m3` back-right — both spin **CCW**.
Equal thrust on all four produces zero net torque. Reaction torque and the
resulting yaw follow the standard quadcopter convention: increasing the CW
pair relative to the CCW pair yaws one way, decreasing yaws the other.

### Command rate

The twin holds your last `set_motors` value between messages — like real
hardware. There is no minimum rate enforced by the protocol, but a rate/attitude
controller needs roughly 200–500 Hz to keep the drone upright. If no command
arrives for `command_timeout_ms` (default 200 ms) while `ARMED`, the twin
engages a **failsafe**: it holds the last command briefly, then linearly ramps
every motor to zero over `failsafe_rampdown_ms`. This is not a hidden hover —
the drone falls, exactly as a real quad would on a lost link.

## What you receive

**`telemetry`** — at `telemetry_hz` (200 Hz default), monotonic `seq`:

```json
{
  "type": "telemetry", "seq": 4821, "t": 12.345, "t_wall": 1780000000.123,
  "step": 12345, "mode": "ARMED", "airborne": true,
  "motors": {"command": [0.4,0.4,0.4,0.4], "thrust_n": [0.61,0.61,0.61,0.61]},
  "battery": {"percent": 91.2, "voltage": 10.8, "current": 4.1},
  "sensors": {
    "valid": true,
    "accel": [0.01, -0.02, 9.79], "gyro": [0.001, -0.0004, 0.0],
    "altitude": 1.503, "range_down": 1.499,
    "beams": [4.87, 3.10, 4.87, 3.10, 4.87, 3.10, 4.87, 3.10]
  },
  "truth": {
    "position": [5.0, 4.0, 1.5], "velocity": [0,0,0],
    "quaternion": [1,0,0,0], "euler": [0,0,0], "omega": [0,0,0]
  }
}
```

- **`sensors`** is what a real flight controller would have: noisy, biased,
  `valid: false` while powered off. Beam order matches `beam_angles_deg` from
  `hello`, measured clockwise from the body +x (forward) axis.
- **`truth`** is ground truth — exact position/attitude. It exists for
  debugging and for scoring your algorithms against reality; a controller that
  only reads `truth` isn't testing anything realistic, so treat it as a cheat
  channel, not your primary feedback.
- `motors.thrust_n` is the *actual* per-motor thrust after the spin-up lag —
  useful for noticing your commands aren't being tracked yet.

**`event`** — pushed as they happen, not polled:

```json
{"type": "event", "event": "collision", "t": 8.201, "speed": 1.2, "normal": [0,0,1]}
```

Event names: `state_changed` (carries `state`), `takeoff`, `landed`,
`collision`, `crashed`, `emergency_stop`, `teleported`, `battery_low`,
`battery_critical`, `battery_empty`, `failsafe_engaged`, `failsafe_cleared`,
`sim_overrun`.

## State machine

```
OFF --power_on--> BOOTING --(~0.5s)--> IDLE --arm--> ARMED --disarm--> IDLE
 ^                                                     |
 +---------------------power_off----------------------+
                                                        |
                                              hard impact -> CRASHED
                                              (reset required to recover)
```

There is no `FLYING`/`LANDING` state — the twin has no notion of intent, only
physics. `airborne` in telemetry is derived from contact with the floor, not
from mode.

## Timing modes

- **`realtime`** (default): the sim runs on the wall clock, scaled by
  `time_scale`. Your controller races it, including any latency and jitter —
  this is the honest mode for testing against something like real hardware.
- **`lockstep`**: the sim advances only in response to `set_motors` or `step`.
  Fully deterministic given a fixed seed — use this to reproduce a bug or to
  run many test episodes as fast as your controller can push commands.

## Versioning

`schema_version` will only ever grow for a breaking change. Check it on
connect; a client built against version 1 should refuse to fly against a
server that reports anything else.

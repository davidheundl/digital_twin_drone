"""Your flight controller. Everything below the connection boilerplate is yours
to write - state estimation, rate/attitude control, search algorithms, all of it.

The twin is passive: it sends you telemetry and reacts to whatever you send it.
It has no autopilot and does nothing on its own. See PROTOCOL.md at the repo
root for the full command/telemetry reference.

Run it against a running server:
    ./.venv/bin/python -m server.app        # in one terminal
    ./.venv/bin/python controller/my_controller.py --x 8 --y 6 --altitude 1.5
"""

import asyncio
import json
import argparse

import websockets

URI = "ws://127.0.0.1:8765"

# Motor layout (X config, body FLU frame), from PROTOCOL.md:
#   m0 front-right (CW)   m1 back-left  (CW)
#   m2 front-left  (CCW)  m3 back-right (CCW)
ARM_LENGTH = 0.125
TORQUE_COEFFICIENT = 0.016
MOTOR_MAX_THRUST = 1.55

# Position control tilts the drone to accelerate horizontally - too much tilt
# and it loses too much vertical thrust (or flips). Clamp desired roll/pitch
# setpoints coming out of the position loop to this many radians (~14 deg).
MAX_TILT = 0.25


class PID:
    """A single PID axis with its own memory (intsum, last_error).

    Each instance keeps its own state, so you can create one per axis
    (altitude, roll, pitch, x, y, ...) without them interfering with
    each other.
    """

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.intsum = 0.0
        self.last_error = None

    def update(self, error, dt):
        self.intsum += error * dt

        if self.last_error is None:
            self.last_error = error  # first call: no rate of change yet

        derivative = (error - self.last_error) / dt
        self.last_error = error

        return error * self.kp + self.intsum * self.ki + derivative * self.kd


def mix(total_thrust, tx, ty, tz):
    """Turn a desired total thrust (N) and body torques (N*m) into 4 motor
    commands in [0, 1]. Inverse of the physics in sim/drone.py - see
    controller/tune_attitude.py for the offline version of this same math.

    Roll/pitch/yaw torques depend only on the DIFFERENCES between the 4
    thrusts, not their absolute values. So when a command is out of range,
    we shift all four by the same amount instead of clamping each one
    independently - that keeps the differences (and therefore the torque
    balance) intact and only sacrifices a bit of total thrust. Clamping
    each motor separately breaks that balance and can inject a fake yaw
    torque out of nowhere (confirmed with controller/tune_attitude.py-style
    offline tests - this was the cause of the crash during aggressive
    position + altitude changes).
    """
    a = ARM_LENGTH / 2 ** 0.5
    x = tx / (4 * a)
    y = ty / (4 * a)
    z = tz / (4 * TORQUE_COEFFICIENT)
    q = total_thrust / 4
    thrusts = [q - x - y + z, q + x + y + z, q + x - y - z, q - x + y - z]

    overflow = max(thrusts) - MOTOR_MAX_THRUST
    if overflow > 0:
        thrusts = [t - overflow for t in thrusts]
    underflow = -min(thrusts)
    if underflow > 0:
        thrusts = [t + underflow for t in thrusts]

    return [max(0.0, min(1.0, t / MOTOR_MAX_THRUST)) for t in thrusts]


async def send(ws, kind: str, **payload):
    """Send one command. The twin replies on the same connection with an
    {"type": "ack", "ok": true/false, "reason": "..."} message - read it from
    the main receive loop below, it does not come back from this call."""
    await ws.send(json.dumps({"type": kind, **payload}))


async def main(target_x, target_y, target_altitude):
    # 200 Hz telemetry by default (see hello["sim"]["telemetry_hz"]) -> 0.005s
    # per tick. Fixed dt, matching the rest of your control loop so far.
    dt = 0.005

    # Cascade: position error -> desired tilt angle -> attitude PID -> torque.
    # Empirically confirmed direction (controller/tune_position.py-style test):
    #   positive pitch  -> moves in +x   =>  x_pid feeds pitch directly
    #   positive roll   -> moves in -y   =>  y_pid feeds roll with a minus sign
    altitude_pid = PID(kp=0.8, ki=0.6, kd=0.3)
    roll_pid = PID(kp=0.3, ki=0.0, kd=0.05)
    pitch_pid = PID(kp=0.3, ki=0.0, kd=0.05)
    x_pid = PID(kp=0.08, ki=0.0, kd=0.15)
    y_pid = PID(kp=0.08, ki=0.0, kd=0.15)

    async with websockets.connect(URI) as ws:
        # First message on any connection is always "hello": room size, drone
        # parameters (mass, motor limits, hover throttle...), sensor layout.
        hello = json.loads(await ws.recv())
        print("hello:", json.dumps(hello, indent=2))
        print(f"target position: x={target_x} y={target_y} altitude={target_altitude}")

        await send(ws, "power_on")

        async for raw in ws:
            msg = json.loads(raw)

            if msg["type"] == "ack":
                if not msg["ok"]:
                    print("REJECTED:", msg["reason"])
                continue

            if msg["type"] == "event":
                print("event:", msg["event"], msg)
                continue

            if msg["type"] != "telemetry":
                continue

            # One of these arrives every physics tick (200 Hz by default).
            # msg["sensors"]  -> noisy IMU/baro/rangefinder/beams (what a real
            #                    controller would actually have)
            # msg["truth"]    -> exact position/velocity/attitude (ground
            #                    truth - fine for debugging, don't fly on it)
            # msg["mode"]     -> OFF / BOOTING / IDLE / ARMED / CRASHED
            # msg["battery"]  -> percent / voltage / current

            if msg["mode"] == "CRASHED":
                await send(ws, "reset")

            if msg["mode"] == "OFF":
                await send(ws, "power_on")

            if msg["mode"] == "IDLE":
                await send(ws, "arm")

            if msg["mode"] == "ARMED":
                x, y, altitude = msg["truth"]["position"]
                roll, pitch, _ = msg["truth"]["euler"]

                # Outer loop: position error -> desired tilt angle
                desired_pitch = x_pid.update(target_x - x, dt)
                desired_pitch = max(-MAX_TILT, min(MAX_TILT, desired_pitch))

                desired_roll = -y_pid.update(target_y - y, dt)
                desired_roll = max(-MAX_TILT, min(MAX_TILT, desired_roll))

                # Inner loop: attitude error (against the desired tilt, not 0!)
                tx = roll_pid.update(desired_roll - roll, dt)
                ty = pitch_pid.update(desired_pitch - pitch, dt)

                throttle = altitude_pid.update(target_altitude - altitude, dt)
                throttle = max(0.0, min(1.0, throttle))
                total_thrust = throttle * 4 * MOTOR_MAX_THRUST

                motors = mix(total_thrust, tx, ty, 0.0)

                print(f"x={x:.2f} y={y:.2f} alt={altitude:.2f}  "
                      f"tilt_target=({desired_roll:+.3f},{desired_pitch:+.3f})  "
                      f"actual=({roll:+.3f},{pitch:+.3f})")

                await send(ws, "set_motors", motors=motors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=8.0, help="target x position (m)")
    parser.add_argument("--y", type=float, default=6.0, help="target y position (m)")
    parser.add_argument("--altitude", type=float, default=1.5)
    args = parser.parse_args()

    asyncio.run(main(args.x, args.y, args.altitude))

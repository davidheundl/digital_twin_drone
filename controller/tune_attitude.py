"""Offline test harness for tuning roll/pitch (attitude) stabilization -
no server, no WebSocket, just the physics directly. Runs in milliseconds
instead of seconds, so you can try many gain values quickly.

Why this exists: starting the drone perfectly level (as it always does) means
symmetric thrust never tilts it, so a normal flight test never actually
exercises your attitude PID - there's simply no error for it to correct. This
script creates a *deliberate* disturbance (a starting tilt) so you can watch
whether your roll/pitch PIDs bring it back to level, and tune Kp/Kd/Ki against
real numbers instead of guessing.

Reuses the exact same PID class from my_controller.py, so whatever you tune
here is the same code that flies for real.

Run it:
    ./.venv/bin/python controller/tune_attitude.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # for "sim"
sys.path.insert(0, _HERE)                      # for "my_controller"

from sim import World, load_config
from sim.vecmath import quat_from_euler, quat_to_euler
from my_controller import PID

TARGET_ALTITUDE = 1.5
DT = 0.005  # matches telemetry_hz=200 in config.yaml


def mix(total_thrust, tx, ty, tz, arm_length, torque_coefficient, max_thrust):
    """Turn a desired total thrust (N) and body torques (N*m) into 4 motor
    commands in [0, 1]. Same X-layout and sign convention as sim/drone.py:
    m0 front-right (CW), m1 back-left (CW), m2 front-left (CCW), m3 back-right (CCW).
    This is the inverse of sim/drone.py's body_forces_torques().
    """
    a = arm_length / 2 ** 0.5
    x = tx / (4 * a)
    y = ty / (4 * a)
    z = tz / (4 * torque_coefficient)
    q = total_thrust / 4
    thrusts = [q - x - y + z, q + x + y + z, q + x - y - z, q - x + y - z]
    return [max(0.0, min(1.0, t / max_thrust)) for t in thrusts]


def run_test(roll_disturbance, pitch_disturbance, roll_gains, pitch_gains,
             altitude_gains=(0.8, 0.6, 0.3), seconds=8.0):
    """roll_gains/pitch_gains/altitude_gains are (kp, ki, kd) tuples."""
    cfg = load_config()
    world = World(cfg)
    world.power_on()
    for _ in range(600):
        world.step(0.001)
    world.arm()

    # The actual disturbance - without this, the test proves nothing.
    world.state.orientation = quat_from_euler(roll_disturbance, pitch_disturbance, 0.0)

    altitude_pid = PID(*altitude_gains)
    roll_pid = PID(*roll_gains)
    pitch_pid = PID(*pitch_gains)
    params = world.params

    print(f"{'t':>6} {'roll':>9} {'pitch':>9} {'alt':>8}   mode")
    for step in range(int(seconds / DT)):
        roll, pitch, _ = quat_to_euler(world.state.orientation)
        altitude = world.state.position[2]

        throttle = max(0.0, min(1.0, altitude_pid.update(TARGET_ALTITUDE - altitude, DT)))
        tx = roll_pid.update(0.0 - roll, DT)
        ty = pitch_pid.update(0.0 - pitch, DT)

        total_thrust = throttle * 4 * params.motor_max_thrust
        motors = mix(total_thrust, tx, ty, 0.0,
                     params.arm_length, params.torque_coefficient, params.motor_max_thrust)
        world.set_motors(motors)

        for _ in range(5):  # 5 physics steps per control tick, same ratio as the live server
            world.step(0.001)

        if step % 200 == 0:
            print(f"{world.t:6.2f} {roll:+9.4f} {pitch:+9.4f} {altitude:8.3f}   {world.mode}")

        if world.mode == "CRASHED":
            print("CRASHED - Regelung war nicht schnell/stark genug.")
            return False

    roll, pitch, _ = quat_to_euler(world.state.orientation)
    print(f"\nEnde: roll={roll:+.5f}  pitch={pitch:+.5f}  altitude={world.state.position[2]:.3f}")
    return True


if __name__ == "__main__":
    # Startet ~17 Grad nach rechts und ~14 Grad nach vorne gekippt.
    # Passe roll_gains/pitch_gains (kp, ki, kd) hier an und beobachte oben,
    # ob roll/pitch sauber gegen 0 gehen oder schwingen/crashen.
    run_test(
        roll_disturbance=0.3,
        pitch_disturbance=-0.25,
        roll_gains=(0.3, 0.0, 0.05),
        pitch_gains=(0.3, 0.0, 0.05),
    )

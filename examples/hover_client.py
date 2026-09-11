"""Reference controller - a test fixture, not part of the digital twin.

It exists to prove the twin is flyable end to end: connect, power on, arm,
climb to a target altitude, hold it, then descend and disarm. The stack is a
conventional cascade, all of it on the client side:

    altitude P -> climb-rate PI            -> total thrust
    attitude P -> body-rate PD             -> body torques
    mixer                                  -> four motor commands

It reads ground-truth attitude instead of integrating the gyro, so treat it as
a baseline to start from, not a finished flight controller. Replacing it - with
a real state estimator and your own control laws - is the point of the project.

    python examples/hover_client.py --altitude 1.5 --duration 10
"""

import argparse
import asyncio
import json
import math
import os
import sys

import numpy as np
import websockets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mixer import Mixer  # noqa: E402

GRAVITY = 9.80665

# Gains tuned against the default config.yaml airframe.
KP_Z, KP_VZ, KI_VZ = 1.6, 4.0, 2.0
KP_ATTITUDE, KP_RATE, KD_RATE = 9.0, 0.015, 0.0008
KP_YAW, KP_YAW_RATE = 3.0, 0.005


class Controller:
    def __init__(self, drone, dt):
        self.mixer = Mixer(drone["arm_length"], drone["torque_coefficient"],
                           drone["motor_max_thrust"])
        self.mass = drone["mass"]
        self.max_total_thrust = 4.0 * drone["motor_max_thrust"]
        self.dt = dt
        self.climb_integral = 0.0

    def update(self, target_altitude, target_yaw, position, velocity, euler, omega):
        roll, pitch, yaw = euler

        # Altitude: position error -> climb-rate setpoint -> thrust.
        climb_setpoint = float(np.clip(KP_Z * (target_altitude - position[2]), -1.5, 2.0))
        climb_error = climb_setpoint - velocity[2]
        self.climb_integral = float(np.clip(self.climb_integral + climb_error * self.dt, -5.0, 5.0))
        vertical_accel = KP_VZ * climb_error + KI_VZ * self.climb_integral

        # Tilting costs vertical thrust; compensate so altitude survives banking.
        tilt = max(0.4, math.cos(roll) * math.cos(pitch))
        total_thrust = float(np.clip(
            self.mass * (GRAVITY + vertical_accel) / tilt, 0.0, self.max_total_thrust))

        # Attitude: angle error -> body-rate setpoint -> torque.
        yaw_error = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
        rate_setpoint = np.clip(np.array([
            KP_ATTITUDE * (0.0 - roll),
            KP_ATTITUDE * (0.0 - pitch),
            KP_YAW * yaw_error,
        ]), -8.0, 8.0)
        rate_error = rate_setpoint - omega

        tx = KP_RATE * rate_error[0] - KD_RATE * omega[0]
        ty = KP_RATE * rate_error[1] - KD_RATE * omega[1]
        tz = KP_YAW_RATE * rate_error[2]

        return self.mixer.mix(total_thrust, tx, ty, tz)


async def run(args):
    uri = f"ws://{args.host}:{args.port}"
    async with websockets.connect(uri, max_queue=64) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello", hello
        drone = hello["drone"]
        dt = 1.0 / hello["sim"]["telemetry_hz"]
        controller = Controller(drone, dt)
        print(f"connected  room={hello['room']['size']}  "
              f"hover_throttle={drone['hover_throttle']:.4f}  dt={dt*1000:.1f} ms")

        async def command(kind, **payload):
            await ws.send(json.dumps({"type": kind, **payload}))

        phase = "boot"
        armed = False
        elapsed = 0.0
        target_altitude = 0.0
        target_yaw = None
        report_every = max(1, int(1.0 / dt))
        tick = 0

        await command("power_on")

        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "event":
                print(f"  [event] t={msg.get('t')} {msg.get('event')}")
                continue
            if msg["type"] == "ack":
                if not msg.get("ok", True):
                    print(f"  [rejected] {msg.get('reason')}")
                continue
            if msg["type"] != "telemetry":
                continue

            truth = msg["truth"]
            position = np.array(truth["position"])
            velocity = np.array(truth["velocity"])
            euler = truth["euler"]
            omega = np.array(truth["omega"])
            mode = msg["mode"]
            elapsed += dt
            tick += 1

            if target_yaw is None:
                target_yaw = euler[2]

            if phase == "boot":
                if mode == "IDLE" and not armed:
                    await command("arm")
                    armed = True
                elif mode == "ARMED":
                    phase = "climb"
                    target_altitude = args.altitude
                    elapsed = 0.0
                    print(f"armed; climbing to {target_altitude:.2f} m")
                await command("set_motors", motors=[0.0, 0.0, 0.0, 0.0])
                continue

            if mode != "ARMED":
                print(f"stopping: mode={mode}")
                break

            if phase == "climb" and abs(position[2] - target_altitude) < 0.05:
                phase, elapsed = "hold", 0.0
                print(f"holding at {position[2]:.3f} m for {args.duration:.0f} s")
            elif phase == "hold" and elapsed >= args.duration:
                phase, target_altitude, elapsed = "descend", 0.05, 0.0
                print("descending")
            elif phase == "descend" and not msg["airborne"]:
                print(f"landed at t={msg['t']:.2f}; disarming")
                await command("set_motors", motors=[0.0, 0.0, 0.0, 0.0])
                await command("disarm")
                await command("power_off")
                break

            motors = controller.update(target_altitude, target_yaw,
                                       position, velocity, euler, omega)
            await command("set_motors", motors=[float(m) for m in motors])

            if tick % report_every == 0:
                print(f"  t={msg['t']:6.2f}  z={position[2]:5.2f}  vz={velocity[2]:+5.2f}  "
                      f"rp=({euler[0]:+.3f},{euler[1]:+.3f})  "
                      f"batt={msg['battery']['percent']:5.1f}%  phase={phase}")


def main():
    parser = argparse.ArgumentParser(description="Reference hover controller")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--altitude", type=float, default=1.5)
    parser.add_argument("--duration", type=float, default=10.0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

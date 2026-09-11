"""The smallest useful client: connect, look around, spin the motors, stop.

Start here when writing your own controller in another language - this is the
entire protocol surface you need.

    python examples/minimal_client.py
"""

import asyncio
import json

import websockets

URI = "ws://127.0.0.1:8765"


async def main():
    async with websockets.connect(URI) as ws:
        hello = json.loads(await ws.recv())
        print("room size:", hello["room"]["size"])
        print("hover throttle per motor:", hello["drone"]["hover_throttle"])
        print("telemetry rate:", hello["sim"]["telemetry_hz"], "Hz")

        async def send(kind, **payload):
            await ws.send(json.dumps({"type": kind, **payload}))

        await send("power_on")
        await send("arm")

        frames = 0
        async for raw in ws:
            msg = json.loads(raw)

            if msg["type"] == "ack" and not msg["ok"]:
                print("rejected:", msg["reason"])
                continue
            if msg["type"] == "event":
                print("event:", msg["event"])
                continue
            if msg["type"] != "telemetry":
                continue

            frames += 1
            if frames == 1:
                print("\nfirst telemetry frame:")
                print(json.dumps(msg, indent=2)[:900], "...\n")

            # Open-loop: a gentle, symmetric throttle. This will NOT hold a
            # stable attitude - that is your job. It just proves the motors bite.
            if msg["mode"] == "ARMED" and frames < 400:
                throttle = 0.30
                await send("set_motors", motors=[throttle] * 4)
            else:
                await send("set_motors", motors=[0.0, 0.0, 0.0, 0.0])
                await send("disarm")
                await send("power_off")
                print(f"done at t={msg['t']:.2f}s, z={msg['truth']['position'][2]:.3f} m")
                break


if __name__ == "__main__":
    asyncio.run(main())

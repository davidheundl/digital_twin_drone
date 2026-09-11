"""WebSocket front end for the digital twin.

    python -m server.app                 # realtime, viewer served on :8080
    python -m server.app --mode lockstep # deterministic stepping
    python -m server.app --no-noise      # perfect sensors
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import socketserver
import threading
from datetime import datetime, timezone
from pathlib import Path

import websockets

from sim import World, load_config

from .protocol import CommandError, dispatch, hello_message, parse_message, CONTROLLER_COMMANDS

ROOT = Path(__file__).resolve().parent.parent


class SessionLog:
    """Append-only JSONL of everything that happened. Replayable, diffable."""

    def __init__(self, directory: Path | None):
        self.fh = None
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = directory / f"session-{stamp}.jsonl"
        self.fh = open(self.path, "a", buffering=1)

    def write(self, kind: str, payload: dict) -> None:
        if self.fh is None:
            return
        self.fh.write(json.dumps({"kind": kind, **payload}) + "\n")

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()


async def handle_client(websocket, runtime, world, log: SessionLog):
    role = "controller"
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    every_n = 1
    subscribed = False

    async def pump():
        while True:
            message = await queue.get()
            await websocket.send(json.dumps(message))

    pump_task = None
    try:
        await websocket.send(json.dumps(hello_message(world, runtime)))

        async for raw in websocket:
            try:
                msg = parse_message(json.loads(raw))
            except (json.JSONDecodeError, CommandError) as exc:
                await websocket.send(json.dumps({
                    "type": "ack", "ok": False, "reason": str(exc)}))
                continue

            if msg["type"] == "hello":
                requested = msg.get("role", "controller")
                if requested not in ("controller", "observer"):
                    await websocket.send(json.dumps({
                        "type": "ack", "id": msg.get("id"), "ok": False,
                        "reason": "role must be 'controller' or 'observer'"}))
                    continue
                role = requested
                every_n = max(1, round(runtime.telemetry_hz / runtime.viewer_hz)) \
                    if role == "observer" else 1
                if not subscribed:
                    runtime.subscribe(queue, every_n)
                    subscribed = True
                    pump_task = asyncio.create_task(pump())
                await websocket.send(json.dumps({
                    "type": "ack", "id": msg.get("id"), "ok": True,
                    "reason": f"role={role}"}))
                continue

            if not subscribed:
                runtime.subscribe(queue, every_n)
                subscribed = True
                pump_task = asyncio.create_task(pump())

            if role == "observer" and msg["type"] in CONTROLLER_COMMANDS:
                await websocket.send(json.dumps({
                    "type": "ack", "id": msg.get("id"), "ok": False,
                    "reason": "observers may not send control commands"}))
                continue

            try:
                ok, reason, extra = dispatch(msg, world, runtime)
            except CommandError as exc:
                ok, reason, extra = False, str(exc), {}

            log.write("command", {"t": world.t, "msg": msg, "ok": ok, "reason": reason})
            await websocket.send(json.dumps({
                "type": "ack", "id": msg.get("id"), "ok": ok, "reason": reason, **extra}))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        if pump_task is not None:
            pump_task.cancel()
        if subscribed:
            runtime.unsubscribe(queue)


def serve_viewer(port: int) -> threading.Thread:
    directory = str(ROOT / "viewer")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)

    class QuietServer(socketserver.TCPServer):
        allow_reuse_address = True

        def handle_error(self, request, client_address):
            pass

    def run():
        with QuietServer(("127.0.0.1", port), handler) as httpd:
            httpd.serve_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


async def main_async(args) -> None:
    overrides: dict = {"sim": {}, "sensors": {}}
    if args.mode:
        overrides["sim"]["mode"] = args.mode
    if args.time_scale is not None:
        overrides["sim"]["time_scale"] = args.time_scale
    if args.no_noise:
        overrides["sensors"]["noise_enabled"] = False
    if args.seed is not None:
        overrides["sensors"]["seed"] = args.seed

    cfg = load_config(args.config, overrides)
    world = World(cfg)

    from .runtime import Runtime
    runtime = Runtime(world, cfg)

    log = SessionLog(None if args.no_log else ROOT / cfg["server"]["log_dir"])
    host = args.host or cfg["server"]["host"]
    port = args.port or int(cfg["server"]["port"])

    if not args.no_viewer:
        viewer_port = int(cfg["server"]["viewer_port"])
        serve_viewer(viewer_port)
        print(f"viewer    http://127.0.0.1:{viewer_port}/")

    handler = functools.partial(handle_client, runtime=runtime, world=world, log=log)
    async with websockets.serve(handler, host, port, max_queue=64):
        print(f"telemetry ws://{host}:{port}   mode={runtime.mode} "
              f"physics={runtime.physics_hz}Hz telemetry={runtime.telemetry_hz}Hz")
        print(f"hover throttle = {world.params.hover_throttle:.4f} per motor")
        try:
            await runtime.run()
        except asyncio.CancelledError:
            pass
        finally:
            runtime.stop()
            log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Drone digital twin server")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--mode", choices=["realtime", "lockstep"], default=None)
    parser.add_argument("--time-scale", type=float, default=None)
    parser.add_argument("--no-noise", action="store_true", help="perfect sensors")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()

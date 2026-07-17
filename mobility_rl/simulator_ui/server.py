# Usage: python simulator_ui/server.py
# HTTP: http://localhost:8080
# WebSocket: ws://localhost:8765

import asyncio
import json
import os
import sys
import threading
import warnings
from http.server import HTTPServer, SimpleHTTPRequestHandler

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets

from simulator_ui.engine import BASE_PARAM_KEYS, ENV_OPTION_KEYS, SimulationEngine


HTTP_PORT = 8080
WS_PORT = 8765
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def log_message(self, format, *args):
        return None


def start_http_server():
    server = HTTPServer(("0.0.0.0", HTTP_PORT), StaticHandler)
    print(f"  HTTP server: http://localhost:{HTTP_PORT}")
    server.serve_forever()


engine = SimulationEngine()
sim_running = False
sim_speed = 1.0
connected_clients = set()


def _json_default(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _serialize(msg_dict):
    return json.dumps(msg_dict, default=_json_default)


async def _broadcast(msg_dict):
    data = _serialize(msg_dict)
    stale = []
    for ws in list(connected_clients):
        try:
            await ws.send(data)
        except Exception:
            stale.append(ws)
    for ws in stale:
        connected_clients.discard(ws)


async def _send_config(websocket):
    await websocket.send(_serialize({"type": "CONFIG", "data": engine.get_config()}))


async def _send_state(websocket):
    snapshot = engine.last_snapshot if engine.last_snapshot is not None else engine.reset()
    await websocket.send(_serialize({"type": "STATE", "data": snapshot}))


async def _broadcast_config_and_state():
    await _broadcast({"type": "CONFIG", "data": engine.get_config()})
    snapshot = engine.last_snapshot if engine.last_snapshot is not None else engine.reset()
    await _broadcast({"type": "STATE", "data": snapshot})


async def _broadcast_tick_state(snapshot, *, previous_policy_id=None, previous_warning=None):
    runtime = snapshot.get("runtime", {})
    if (
        previous_policy_id is not None
        and (
            runtime.get("policy_id") != previous_policy_id
            or runtime.get("policy_warning") != previous_warning
        )
    ):
        await _broadcast({"type": "CONFIG", "data": engine.get_config()})
    await _broadcast({"type": "STATE", "data": snapshot})


async def ws_handler(websocket):
    global sim_running, sim_speed
    connected_clients.add(websocket)
    print(f"  Client connected ({len(connected_clients)} total)")

    try:
        await _send_config(websocket)
        await _send_state(websocket)

        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = str(msg.get("type", "")).upper()

            if mtype == "RUN":
                sim_running = True
                await websocket.send(_serialize({"type": "ACK", "action": "RUN"}))
                continue

            if mtype == "PAUSE":
                sim_running = False
                await websocket.send(_serialize({"type": "ACK", "action": "PAUSE"}))
                continue

            if mtype == "STEP":
                sim_running = False
                previous_policy_id = engine.runtime.get("policy_id")
                previous_warning = engine.runtime_status.get("policy_warning")
                snapshot = engine.tick()
                await _broadcast_tick_state(
                    snapshot,
                    previous_policy_id=previous_policy_id,
                    previous_warning=previous_warning,
                )
                continue

            if mtype == "RESET":
                sim_running = False
                snapshot = engine.reset()
                await _broadcast({"type": "STATE", "data": snapshot})
                await websocket.send(_serialize({"type": "ACK", "action": "RESET"}))
                continue

            if mtype == "SET_SPEED":
                sim_speed = max(0.1, min(float(msg.get("factor", 1.0)), 20.0))
                engine.set_runtime_value("speed_factor", sim_speed)
                await websocket.send(_serialize({"type": "ACK", "action": "SET_SPEED", "factor": sim_speed}))
                continue

            if mtype == "GET_CONFIG":
                await _send_config(websocket)
                continue

            if mtype == "EXPORT":
                sim_running = False
                run_dir = engine.export()
                await websocket.send(_serialize({"type": "EXPORT_DONE", "path": run_dir}))
                continue

            if mtype == "SET_PRESET":
                entry = engine.select_preset(str(msg.get("preset_id")))
                await _broadcast_config_and_state()
                await websocket.send(_serialize({"type": "ACK_CONFIG", "key": "selected_preset_id", "old": entry["old_value"], "new": entry["new_value"], "mode": entry["apply_mode"]}))
                continue

            if mtype == "SET_POLICY":
                entry = engine.select_policy(str(msg.get("policy_id")))
                await _broadcast_config_and_state()
                await websocket.send(_serialize({"type": "ACK_CONFIG", "key": "policy_id", "old": entry["old_value"], "new": entry["new_value"], "mode": entry["apply_mode"]}))
                continue

            if mtype == "SET_RUNTIME":
                key = str(msg.get("key"))
                value = msg.get("value")
                entry = engine.set_runtime_value(key, value)
                await _broadcast_config_and_state()
                await websocket.send(_serialize({"type": "ACK_CONFIG", "key": key, "old": entry["old_value"], "new": entry["new_value"], "mode": entry["apply_mode"]}))
                continue

            if mtype == "SET_BASE_PARAM":
                key = str(msg.get("key"))
                value = msg.get("value")
                entry = engine.set_base_param(key, value)
                await _broadcast_config_and_state()
                await websocket.send(_serialize({"type": "ACK_CONFIG", "key": key, "old": entry["old_value"], "new": entry["new_value"], "mode": entry["apply_mode"]}))
                continue

            if mtype == "SET_ENV_OPTION":
                key = str(msg.get("key"))
                value = msg.get("value")
                entry = engine.set_env_option(key, value)
                await _broadcast_config_and_state()
                await websocket.send(_serialize({"type": "ACK_CONFIG", "key": key, "old": entry["old_value"], "new": entry["new_value"], "mode": entry["apply_mode"]}))
                continue

            if mtype == "SET_CONFIG":
                key = str(msg.get("key"))
                value = msg.get("value")
                if key in BASE_PARAM_KEYS:
                    entry = engine.set_base_param(key, value)
                elif key in ENV_OPTION_KEYS:
                    entry = engine.set_env_option(key, value)
                else:
                    continue
                await _broadcast_config_and_state()
                await websocket.send(_serialize({"type": "ACK_CONFIG", "key": key, "old": entry["old_value"], "new": entry["new_value"], "mode": entry["apply_mode"]}))
                continue

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"  Client disconnected ({len(connected_clients)} total)")


async def sim_loop():
    global sim_running
    base_interval = 0.20
    while True:
        if sim_running and connected_clients:
            previous_policy_id = engine.runtime.get("policy_id")
            previous_warning = engine.runtime_status.get("policy_warning")
            snapshot = engine.tick()
            await _broadcast_tick_state(
                snapshot,
                previous_policy_id=previous_policy_id,
                previous_warning=previous_warning,
            )
            await asyncio.sleep(base_interval / max(sim_speed, 0.1))
        else:
            await asyncio.sleep(0.05)


async def main():
    print("=" * 55)
    print("  Decentralized FANET Live Visualizer")
    print("=" * 55)

    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    print(f"  WebSocket:   ws://localhost:{WS_PORT}")
    print(f"  Open browser: http://localhost:{HTTP_PORT}")
    print("=" * 55)

    async with websockets.serve(ws_handler, "0.0.0.0", WS_PORT):
        await sim_loop()


if __name__ == "__main__":
    asyncio.run(main())

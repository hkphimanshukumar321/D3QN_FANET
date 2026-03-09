# server.py — WebSocket + HTTP Server for Real-Time 3D Simulator
# Usage: python simulator_ui/server.py
# HTTP: http://localhost:8080  (serves static frontend)
# WebSocket: ws://localhost:8765  (sim control channel)

import os
import sys
import json
import asyncio
import threading
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

# Ensure project imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets

from simulator_ui.engine import SimulationEngine

# ======================================================================
# Configuration
# ======================================================================
HTTP_PORT = 8080
WS_PORT = 8765
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ======================================================================
# HTTP Server for static files
# ======================================================================
class StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


def start_http_server():
    server = HTTPServer(("0.0.0.0", HTTP_PORT), StaticHandler)
    print(f"  HTTP server: http://localhost:{HTTP_PORT}")
    server.serve_forever()


# ======================================================================
# Simulation runner state
# ======================================================================
engine = SimulationEngine()
sim_running = False
sim_speed = 1.0       # multiplier: 1.0 = real-time
connected_clients = set()


# ======================================================================
# WebSocket handler
# ======================================================================
async def ws_handler(websocket):
    global sim_running, sim_speed, engine
    connected_clients.add(websocket)
    print(f"  Client connected ({len(connected_clients)} total)")

    try:
        # Send initial config
        await websocket.send(json.dumps({
            "type": "CONFIG", "data": engine.get_config(),
        }))
        # Send initial state
        snap = engine._build_snapshot()
        await websocket.send(json.dumps({
            "type": "STATE", "data": snap,
        }, default=_json_default))

        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type", "").upper()

            if mtype == "RUN":
                sim_running = True
                await websocket.send(json.dumps({"type": "ACK", "action": "RUN"}))

            elif mtype == "PAUSE":
                sim_running = False
                await websocket.send(json.dumps({"type": "ACK", "action": "PAUSE"}))

            elif mtype == "STEP":
                sim_running = False
                snap = engine.tick()
                await _broadcast({"type": "STATE", "data": snap})

            elif mtype == "RESET":
                sim_running = False
                if engine._pending_changes:
                    engine.apply_pending_changes()
                else:
                    engine.reset()
                snap = engine._build_snapshot()
                await _broadcast({"type": "STATE", "data": snap})
                await websocket.send(json.dumps({"type": "ACK", "action": "RESET"}))

            elif mtype == "SET_CONFIG":
                key = msg.get("key")
                value = msg.get("value")
                mode = msg.get("mode", "live")
                if key is not None and value is not None:
                    # Type coercion
                    value = _coerce_config_value(key, value)
                    entry = engine.update_config(key, value, mode)
                    await websocket.send(json.dumps({
                        "type": "ACK_CONFIG",
                        "key": key, "old": entry["old_value"],
                        "new": entry["new_value"], "mode": mode,
                    }, default=_json_default))

            elif mtype == "GET_CONFIG":
                await websocket.send(json.dumps({
                    "type": "CONFIG", "data": engine.get_config(),
                }, default=_json_default))

            elif mtype == "SET_SPEED":
                sim_speed = max(0.1, min(float(msg.get("factor", 1.0)), 20.0))
                await websocket.send(json.dumps({
                    "type": "ACK", "action": "SET_SPEED", "factor": sim_speed,
                }))

            elif mtype == "EXPORT":
                sim_running = False
                run_dir = engine.export()
                await websocket.send(json.dumps({
                    "type": "EXPORT_DONE", "path": run_dir,
                }))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"  Client disconnected ({len(connected_clients)} total)")


async def _broadcast(msg_dict):
    data = json.dumps(msg_dict, default=_json_default)
    for ws in list(connected_clients):
        try:
            await ws.send(data)
        except Exception:
            connected_clients.discard(ws)


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _coerce_config_value(key, value):
    """Coerce incoming config value to the correct Python type."""
    int_keys = {"N", "SEED", "QMAX", "CW_MIN", "CW_MAX", "DIFS_SLOTS", "SIFS_SLOTS",
                "ACK_SLOTS", "ACK_TIMEOUT_SLOTS", "MAX_RETRY", "RTS_SLOTS", "CTS_SLOTS",
                "RL_TRAFFIC_BINS", "OFFERED_PPS", "PAYLOAD_BYTES"}
    float_keys = {"AREA_X", "AREA_Y", "AREA_Z", "SINK_X", "SINK_Y", "SINK_Z",
                  "COMM_RANGE_R", "V_MIN", "V_MAX", "V_MEAN", "V_STD",
                  "SPEED_UPDATE_INTERVAL", "GM_ALPHA", "RWP_PAUSE_TIME",
                  "CIRC_RADIUS", "CIRC_OMEGA_MEAN", "CIRC_OMEGA_STD", "CIRC_CLIMB_RATE",
                  "MOBILITY_DT", "SLOT_TIME_S", "PHY_RATE_BPS", "TDMA_GUARD_TIME_S",
                  "RL_ALPHA", "RL_EPSILON", "RL_WT", "RL_WD",
                  "PATHLOSS_K", "PATHLOSS_ETA"}
    bool_keys = {"RTS_CTS_ENABLED", "ACK_ENABLED", "ENABLE_RL_SELECTOR",
                 "ENABLE_PATHLOSS", "ENABLE_PROP_DELAY"}

    if key in int_keys:
        return int(float(value))
    elif key in float_keys:
        return float(value)
    elif key in bool_keys:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    return value


# ======================================================================
# Simulation loop — runs ticks and broadcasts state
# ======================================================================
async def sim_loop():
    global sim_running
    base_interval = 0.033  # ~30 Hz target
    while True:
        if sim_running and connected_clients:
            snap = engine.tick()
            await _broadcast({"type": "STATE", "data": snap})
            await asyncio.sleep(base_interval / sim_speed)
        else:
            await asyncio.sleep(0.05)


# ======================================================================
# Main
# ======================================================================
async def main():
    print("=" * 55)
    print("  FANET 3D Simulator — Real-Time UI Server")
    print("=" * 55)

    # Start HTTP in background thread
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    # Start WebSocket server
    print(f"  WebSocket:   ws://localhost:{WS_PORT}")
    print(f"  Open browser: http://localhost:{HTTP_PORT}")
    print("=" * 55)

    async with websockets.serve(ws_handler, "0.0.0.0", WS_PORT):
        await sim_loop()


if __name__ == "__main__":
    asyncio.run(main())

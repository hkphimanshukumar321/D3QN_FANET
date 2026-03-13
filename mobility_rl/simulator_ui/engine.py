# engine.py — Tick-based Simulation Engine for Real-Time 3D Simulator
# Wraps mobility + MAC into a step-able interface for WebSocket streaming.

import os
import sys
import json
import math
import datetime
import numpy as np
import pandas as pd

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from algorithms.mobility.speed import SpeedEngine
from algorithms.mobility.models import create_mobility_model
from algorithms.mobility.link import compute_distances, compute_link_up
from algorithms.rl.qlearning_selector import QLearningAgent


# ======================================================================
# Default config (mirrors config.py defaults)
# ======================================================================
DEFAULT_CONFIG = {
    # World
    "N": 70, "AREA_X": 1000.0, "AREA_Y": 1000.0, "AREA_Z": 300.0,
    "SINK_X": 500.0, "SINK_Y": 500.0, "SINK_Z": 0.0,
    "COMM_RANGE_R": 500.0, "SEED": 42,
    # Mobility
    "MOBILITY_MODEL": "gauss_markov", "SPEED_MODE": "uniform",
    "V_MIN": 5.0, "V_MAX": 30.0, "V_MEAN": 15.0, "V_STD": 5.0,
    "SPEED_UPDATE_INTERVAL": 5.0,
    "GM_ALPHA": 0.5, "RWP_PAUSE_TIME": 1.0,
    "CIRC_RADIUS": 100.0, "CIRC_OMEGA_MEAN": 0.1,
    "CIRC_OMEGA_STD": 0.02, "CIRC_CLIMB_RATE": 0.5,
    "MOBILITY_DT": 0.1,
    # MAC
    "MAC_PROTOCOL": "CSMA_CA",
    "SLOT_TIME_S": 9e-6, "PHY_RATE_BPS": 5e6,
    "PAYLOAD_BYTES": 1500, "QMAX": 100,
    "CW_MIN": 15, "CW_MAX": 1023, "DIFS_SLOTS": 4, "SIFS_SLOTS": 2,
    "ACK_SLOTS": 2, "ACK_TIMEOUT_SLOTS": 10, "MAX_RETRY": 10,
    "RTS_CTS_ENABLED": True, "ACK_ENABLED": True,
    "RTS_SLOTS": 3, "CTS_SLOTS": 3,
    "TDMA_GUARD_TIME_S": 1e-6,
    # Traffic
    "OFFERED_PPS": 200,
    # RL
    "ENABLE_RL_SELECTOR": True,
    "RL_ALPHA": 0.1, "RL_GAMMA": 0.0, "RL_EPSILON": 0.0,
    "RL_WT": 0.5, "RL_WD": 0.5, "RL_TRAFFIC_BINS": 14,
    "RL_DECISION_INTERVAL": 10,  # ticks between RL decisions
    # Pathloss
    "ENABLE_PATHLOSS": True, "PATHLOSS_K": 0.001, "PATHLOSS_ETA": 2.0,
    "ENABLE_PROP_DELAY": True,
    # MARL
    "ENABLE_MARL": True,
    # Render
    "RENDER_INTERPOLATION": True, "SNAPSHOT_BUFFER_SIZE": 2,
}


class SimulationEngine:
    """
    Tick-based simulation engine combining 3D mobility + MAC protocol.
    Each tick = MOBILITY_DT seconds.
    Within each tick, MAC runs slots_per_tick MAC slots internally.
    """

    def __init__(self, config=None):
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.config_change_log = []
        self._pending_changes = {}
        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        """Re-initialize all state from current config."""
        c = self.config
        self.sim_time = 0.0
        self.tick_count = 0

        N = int(c["N"])
        self.N = N
        dt = float(c["MOBILITY_DT"])
        self.dt = dt
        seed = int(c["SEED"])
        self.rng = np.random.default_rng(seed)

        # --- Mobility ---
        bounds = (c["AREA_X"], c["AREA_Y"], c["AREA_Z"])
        self.bounds = bounds
        self.sink_pos = np.array([c["SINK_X"], c["SINK_Y"], c["SINK_Z"]], dtype=float)
        self.comm_range = float(c["COMM_RANGE_R"])

        self.speed_engine = SpeedEngine(
            n_nodes=N, v_min=c["V_MIN"], v_max=c["V_MAX"],
            mode=c["SPEED_MODE"], v_mean=c["V_MEAN"], v_std=c["V_STD"],
            update_interval=c["SPEED_UPDATE_INTERVAL"],
            rng=self.rng,
        )
        self.mobility_model = create_mobility_model(
            name=c["MOBILITY_MODEL"], n_nodes=N, bounds=bounds,
            speed_engine=self.speed_engine, rng=self.rng,
            gm_alpha=c["GM_ALPHA"], rwp_pause_time=c["RWP_PAUSE_TIME"],
            circ_radius=c["CIRC_RADIUS"], circ_omega_mean=c["CIRC_OMEGA_MEAN"],
            circ_omega_std=c["CIRC_OMEGA_STD"], circ_climb_rate=c["CIRC_CLIMB_RATE"],
        )

        # Current state
        self.positions = self.mobility_model.positions.copy()
        self.velocities = self.mobility_model.velocities.copy()
        
        from configs import config as params
        if getattr(params, 'DECENTRALIZED_COMM', False):
            diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)
            self.distances = np.sqrt(np.min(dist_sq, axis=1))
        else:
            self.distances = compute_distances(self.positions, self.sink_pos)
        self.link_up = compute_link_up(self.distances, self.comm_range)

        # --- MAC state ---
        slot_time_s = float(c["SLOT_TIME_S"])
        phy_rate_bps = float(c["PHY_RATE_BPS"])
        payload_bytes = int(c["PAYLOAD_BYTES"])
        payload_bits = payload_bytes * 8
        tx_time_s = payload_bits / phy_rate_bps
        TX_slots = int(math.ceil(tx_time_s / slot_time_s))

        self.slot_time_s = slot_time_s
        self.payload_bits = payload_bits
        self.TX_slots = TX_slots
        self.QMAX = int(c["QMAX"])

        # Slots per mobility tick
        self.slots_per_tick = max(1, int(round(dt / slot_time_s)))

        # TDMA
        guard_slots = int(math.ceil(float(c["TDMA_GUARD_TIME_S"]) / slot_time_s))
        self.tdma_slot_total = TX_slots + guard_slots

        # CSMA/CA durations
        rts_cts = bool(c["RTS_CTS_ENABLED"])
        ack_en = bool(c["ACK_ENABLED"])
        sifs = int(c["SIFS_SLOTS"])
        if rts_cts and ack_en:
            self.busy_success = int(c["RTS_SLOTS"]) + sifs + int(c["CTS_SLOTS"]) + sifs + TX_slots + sifs + int(c["ACK_SLOTS"])
            self.busy_collision = int(c["RTS_SLOTS"]) + int(c["ACK_TIMEOUT_SLOTS"])
        elif ack_en:
            self.busy_success = TX_slots + sifs + int(c["ACK_SLOTS"])
            self.busy_collision = TX_slots + int(c["ACK_TIMEOUT_SLOTS"])
        else:
            self.busy_success = TX_slots
            self.busy_collision = TX_slots

        # Per-node state
        self.queues = [[] for _ in range(N)]  # arrival timestamps
        self.q_lens = np.zeros(N, dtype=int)

        # CSMA state
        self.cw = np.full(N, int(c["CW_MIN"]), dtype=int)
        self.backoff = np.full(N, -1, dtype=int)
        self.retry = np.zeros(N, dtype=int)
        self.channel_busy = 0
        self.idle_since = 0
        self.global_slot = 0

        # Traffic arrivals
        lam_node = float(c["OFFERED_PPS"]) / max(N, 1)
        self.lam_node = max(lam_node, 1e-12)
        self.next_arrival = self.rng.exponential(1.0 / self.lam_node, size=N)

        # --- Metrics (cumulative & windowed) ---
        self._reset_metrics()

        # --- RL agent ---
        self.rl_agent = None
        self.rl_selections = []  # [{tick, traffic_pps, mac_selected, q_tdma, q_csma, reward}]
        self._rl_window_metrics = {"TDMA": {"bits": 0, "delay": 0.0, "count": 0},
                                   "CSMA_CA": {"bits": 0, "delay": 0.0, "count": 0}}
        if bool(c.get("ENABLE_RL_SELECTOR", False)):
            self.rl_agent = QLearningAgent(
                alpha=float(c.get("RL_ALPHA", 0.1)),
                gamma=float(c.get("RL_GAMMA", 0.0)),
                epsilon=float(c.get("RL_EPSILON", 0.0)),
                n_bins=int(c.get("RL_TRAFFIC_BINS", 14)),
            )

        # --- Config snapshot ---
        self._write_config_snapshot()

        # --- History for export ---
        self.history = []
        
        # --- MARL state ---
        self.marl_model = None
        self.marl_decisions = {}  # per-tick: {agent_id: action}
        if bool(c.get("ENABLE_MARL", False)):
            self._load_marl_model()

    def _reset_metrics(self):
        self.total_pkts_success = 0
        self.total_bits_tx = 0
        self.total_pkts_dropped = 0
        self.total_collisions = 0
        self.total_delay_sum = 0.0
        self.total_delay_count = 0
        self.total_channel_busy_slots = 0
        self.total_slots_run = 0
        # Per-tick window
        self.tick_pkts_success = 0
        self.tick_bits_tx = 0
        self.tick_pkts_dropped = 0
        self.tick_collisions = 0
        self.tick_delay_sum = 0.0
        self.tick_delay_count = 0

    def _reset_tick_metrics(self):
        self.tick_pkts_success = 0
        self.tick_bits_tx = 0
        self.tick_pkts_dropped = 0
        self.tick_collisions = 0
        self.tick_delay_sum = 0.0
        self.tick_delay_count = 0

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------
    def update_config(self, key, value, mode="live"):
        """Update a config parameter. mode='live' or 'restart'."""
        old = self.config.get(key)
        self.config[key] = value
        entry = {
            "timestamp": self.sim_time, "tick": self.tick_count,
            "param": key, "old_value": old, "new_value": value,
            "apply_mode": mode,
        }
        self.config_change_log.append(entry)

        if mode == "live":
            # Apply immediately for parameters that can change live
            self._apply_live_change(key, value)
        else:
            self._pending_changes[key] = value
        return entry

    def _apply_live_change(self, key, value):
        """Apply a parameter change at the next tick boundary."""
        if key == "OFFERED_PPS":
            self.lam_node = max(float(value) / max(self.N, 1), 1e-12)
        elif key == "COMM_RANGE_R":
            self.comm_range = float(value)
        elif key in ("SINK_X", "SINK_Y", "SINK_Z"):
            idx = {"SINK_X": 0, "SINK_Y": 1, "SINK_Z": 2}[key]
            self.sink_pos[idx] = float(value)
        elif key == "QMAX":
            self.QMAX = int(value)
        elif key in ("CW_MIN", "CW_MAX", "MAX_RETRY"):
            pass  # These are read each tick from self.config

    def apply_pending_changes(self):
        """Apply all queued restart-mode changes and reset."""
        if self._pending_changes:
            self._pending_changes.clear()
            self.reset()

    def get_config(self):
        return dict(self.config)

    # ------------------------------------------------------------------
    # Tick — one mobility step + MAC slots
    # ------------------------------------------------------------------
    def tick(self):
        """Advance simulation by one MOBILITY_DT step. Returns snapshot dict."""
        self._reset_tick_metrics()
        c = self.config

        # 1) Mobility update
        if self.tick_count == 0:
            self.positions = self.mobility_model.positions.copy()
            self.velocities = self.mobility_model.velocities.copy()
        else:
            self.positions, self.velocities = self.mobility_model.update(self.dt)

        from configs import config as params
        if getattr(params, 'DECENTRALIZED_COMM', False):
            diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)
            self.distances = np.sqrt(np.min(dist_sq, axis=1))
        else:
            self.distances = compute_distances(self.positions, self.sink_pos)
        self.link_up = compute_link_up(self.distances, self.comm_range)

        # 2) RL MAC selection (if enabled)
        protocol = str(c["MAC_PROTOCOL"]).upper()
        if self.rl_agent is not None:
            interval = int(c.get("RL_DECISION_INTERVAL", 10))
            if self.tick_count % interval == 0 and self.tick_count > 0:
                protocol = self._rl_select_mac()
                self.config["MAC_PROTOCOL"] = protocol

        # 3) Run MAC slots
        if protocol == "TDMA":
            self._run_tdma_slots()
        else:
            self._run_csma_slots()

        # 4) Compute snapshot
        snapshot = self._build_snapshot()

        # 5) Record history
        self.history.append({
            "t": self.sim_time,
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "distances": self.distances.copy(),
            "link_up": self.link_up.copy(),
            "q_lens": self.q_lens.copy(),
            "metrics": snapshot["metrics"],
        })

        self.sim_time += self.dt
        self.tick_count += 1
        return snapshot

    def _load_marl_model(self):
        """Try to load a MARL GNN model for per-agent MAC decisions."""
        try:
            import torch
            from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork
            cp_path = os.path.join(project_root, "results", "checkpoints", "gnn_marl_model.pth")
            if os.path.exists(cp_path):
                device = torch.device("cpu")
                self.marl_model = MAGAT_D3QN_QNetwork(node_in_dim=8, hidden_dim=64, num_actions=2, heads=4).to(device)
                self.marl_model.load_state_dict(torch.load(cp_path, map_location=device))
                self.marl_model.eval()
        except Exception as e:
            print(f"Warning: MARL model load failed: {e}")
            self.marl_model = None

    # ------------------------------------------------------------------
    # RL MAC selection
    # ------------------------------------------------------------------
    def _rl_select_mac(self):
        """Use RL agent to select TDMA or CSMA_CA based on recent metrics."""
        c = self.config
        wT = float(c.get("RL_WT", 0.5))
        wD = float(c.get("RL_WD", 0.5))
        traffic_pps = float(c.get("OFFERED_PPS", 200))
        state = self.rl_agent.get_state(traffic_pps, 10, 1000)

        # Compute reward from current tick metrics
        elapsed = max(self.dt, 1e-12)
        thr_mbps = self.tick_bits_tx / elapsed / 1e6
        delay_ms = (self.tick_delay_sum / max(self.tick_delay_count, 1)) * 1000

        # Normalize
        max_thr = float(c.get("PHY_RATE_BPS", 5e6)) / 1e6
        max_delay = 100.0  # ms cap for normalization
        r_thr = min(thr_mbps / max(max_thr, 1e-9), 1.0)
        r_delay = max(1.0 - delay_ms / max_delay, 0.0)
        reward = wT * r_thr + wD * r_delay

        # Current protocol's action index
        current_mac = str(c.get("MAC_PROTOCOL", "CSMA_CA")).upper()
        current_action = 0 if current_mac == "TDMA" else 1

        # Update Q for current action
        self.rl_agent.update(state, current_action, reward, state)

        # Select next action
        action = self.rl_agent.select_action(state)
        selected = "TDMA" if action == 0 else "CSMA_CA"

        q_vals = self.rl_agent.get_q_values(state)
        self.rl_selections.append({
            "tick": self.tick_count,
            "sim_time": round(self.sim_time, 4),
            "traffic_pps": traffic_pps,
            "state": state,
            "mac_selected": selected,
            "q_tdma": round(q_vals[0], 6),
            "q_csma": round(q_vals[1], 6),
            "reward": round(reward, 6),
            "throughput_mbps": round(thr_mbps, 4),
            "delay_ms": round(delay_ms, 4),
        })
        return selected

    def _write_config_snapshot(self):
        """Write immutable config snapshot at run start."""
        try:
            snap_dir = os.path.join(project_root, "results", "_config_snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(snap_dir, f"config_snapshot_{ts}.json")
            with open(snap_path, "w") as f:
                json.dump(self.config, f, indent=2, default=str)
        except Exception:
            pass  # non-critical

    # ------------------------------------------------------------------
    # MAC: TDMA (tick-scoped)
    # ------------------------------------------------------------------
    def _run_tdma_slots(self):
        N = self.N
        for s_offset in range(self.slots_per_tick):
            s = self.global_slot
            t = s * self.slot_time_s

            # Arrivals
            arriving = np.where(t >= self.next_arrival)[0]
            for i in arriving:
                if self.link_up[i]:
                    if len(self.queues[i]) < self.QMAX:
                        self.queues[i].append(t)
                        self.q_lens[i] += 1
                    else:
                        self.tick_pkts_dropped += 1
                        self.total_pkts_dropped += 1
                self.next_arrival[i] += self.rng.exponential(1.0 / self.lam_node)

            # Channel
            if self.channel_busy > 0:
                self.channel_busy -= 1
                self.total_channel_busy_slots += 1
                self.global_slot += 1
                self.total_slots_run += 1
                continue

            # TDMA slot assignment
            if s % self.tdma_slot_total == 0:
                owner = (s // self.tdma_slot_total) % N
                if len(self.queues[owner]) > 0 and self.link_up[owner]:
                    arrival_t = self.queues[owner].pop(0)
                    self.q_lens[owner] -= 1
                    delay = (s + self.TX_slots) * self.slot_time_s - arrival_t
                    self.tick_delay_sum += delay
                    self.tick_delay_count += 1
                    self.total_delay_sum += delay
                    self.total_delay_count += 1
                    self.tick_pkts_success += 1
                    self.total_pkts_success += 1
                    self.tick_bits_tx += self.payload_bits
                    self.total_bits_tx += self.payload_bits
                    self.channel_busy = self.tdma_slot_total - 1
                    self.total_channel_busy_slots += self.TX_slots

            self.global_slot += 1
            self.total_slots_run += 1

    # ------------------------------------------------------------------
    # MAC: CSMA/CA (tick-scoped)
    # ------------------------------------------------------------------
    def _run_csma_slots(self):
        N = self.N
        c = self.config
        cw_min = int(c["CW_MIN"])
        cw_max = int(c["CW_MAX"])
        max_retry = int(c["MAX_RETRY"])
        difs_slots = int(c["DIFS_SLOTS"])

        for s_offset in range(self.slots_per_tick):
            s = self.global_slot
            t = s * self.slot_time_s

            # Arrivals
            arriving = np.where(t >= self.next_arrival)[0]
            for i in arriving:
                if self.link_up[i]:
                    if len(self.queues[i]) < self.QMAX:
                        self.queues[i].append(t)
                        self.q_lens[i] += 1
                    else:
                        self.tick_pkts_dropped += 1
                        self.total_pkts_dropped += 1
                self.next_arrival[i] += self.rng.exponential(1.0 / self.lam_node)

            q_sum = int(self.q_lens.sum())

            if self.channel_busy > 0:
                self.channel_busy -= 1
                self.total_channel_busy_slots += 1
                if self.channel_busy == 0:
                    self.idle_since = 0
                self.global_slot += 1
                self.total_slots_run += 1
                continue

            self.idle_since += 1

            if q_sum > 0:
                if self.idle_since <= difs_slots:
                    self.global_slot += 1
                    self.total_slots_run += 1
                    continue

                has_pkt = np.array([len(self.queues[i]) > 0 and self.link_up[i] for i in range(N)])
                need_bo = has_pkt & (self.backoff < 0)
                if need_bo.any():
                    for i in np.where(need_bo)[0]:
                        self.backoff[i] = self.rng.integers(0, self.cw[i] + 1)

                tx_nodes = np.where(has_pkt & (self.backoff == 0))[0]
                if tx_nodes.size == 0:
                    counting = has_pkt & (self.backoff > 0)
                    self.backoff[counting] -= 1
                    self.global_slot += 1
                    self.total_slots_run += 1
                    continue

                self.idle_since = 0
                if tx_nodes.size == 1:
                    i = tx_nodes[0]
                    arrival_t = self.queues[i].pop(0)
                    self.q_lens[i] -= 1
                    delay = (s + self.busy_success) * self.slot_time_s - arrival_t
                    self.tick_delay_sum += delay
                    self.tick_delay_count += 1
                    self.total_delay_sum += delay
                    self.total_delay_count += 1
                    self.tick_pkts_success += 1
                    self.total_pkts_success += 1
                    self.tick_bits_tx += self.payload_bits
                    self.total_bits_tx += self.payload_bits
                    self.cw[i] = cw_min
                    self.retry[i] = 0
                    self.backoff[i] = -1
                    self.channel_busy = self.busy_success - 1
                    self.total_channel_busy_slots += 1
                else:
                    # Collision
                    self.tick_collisions += 1
                    self.total_collisions += 1
                    for i in tx_nodes:
                        self.retry[i] += 1
                        if self.retry[i] > max_retry:
                            if len(self.queues[i]) > 0:
                                self.queues[i].pop(0)
                                self.q_lens[i] -= 1
                                self.tick_pkts_dropped += 1
                                self.total_pkts_dropped += 1
                            self.retry[i] = 0
                            self.cw[i] = cw_min
                        else:
                            self.cw[i] = min(2 * self.cw[i] + 1, cw_max)
                        self.backoff[i] = -1
                    self.channel_busy = self.busy_collision - 1
                    self.total_channel_busy_slots += 1

            self.global_slot += 1
            self.total_slots_run += 1

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def _build_snapshot(self):
        elapsed = max(self.sim_time + self.dt, 1e-12)
        avg_delay = self.total_delay_sum / max(self.total_delay_count, 1)
        tick_delay = self.tick_delay_sum / max(self.tick_delay_count, 1)
        utilization = self.total_channel_busy_slots / max(self.total_slots_run, 1)
        link_ratio = float(self.link_up.sum()) / max(self.N, 1)
        avg_q = float(self.q_lens.sum()) / max(self.N, 1)

        return {
            "tick": self.tick_count,
            "sim_time": round(self.sim_time, 4),
            "N": self.N,
            "protocol": str(self.config["MAC_PROTOCOL"]),
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "distances": self.distances.tolist(),
            "link_up": self.link_up.tolist(),
            "q_lens": self.q_lens.tolist(),
            "sink_pos": self.sink_pos.tolist(),
            "bounds": list(self.bounds),
            "comm_range": self.comm_range,
            "metrics": {
                "throughput_mbps": round(self.total_bits_tx / elapsed / 1e6, 4),
                "tick_throughput_mbps": round(self.tick_bits_tx / max(self.dt, 1e-12) / 1e6, 4),
                "avg_delay_ms": round(avg_delay * 1000, 4),
                "tick_delay_ms": round(tick_delay * 1000, 4),
                "total_drops": self.total_pkts_dropped,
                "total_collisions": self.total_collisions,
                "total_success": self.total_pkts_success,
                "utilization": round(utilization, 4),
                "link_up_ratio": round(link_ratio, 4),
                "avg_queue_len": round(avg_q, 2),
            },
            "marl_decisions": self.marl_decisions,
        }

    # ------------------------------------------------------------------
    # Export (standard results folder)
    # ------------------------------------------------------------------
    def export(self, base_results_dir=None):
        """Write results in the standard project folder format."""
        if base_results_dir is None:
            base_results_dir = os.path.join(project_root, "results")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        protocol = self.config["MAC_PROTOCOL"]
        phy_mbps = int(float(self.config["PHY_RATE_BPS"]) / 1e6)
        run_dir = os.path.join(
            base_results_dir,
            f"UI_N{self.N}_PHY{phy_mbps}M_{protocol}",
            f"trial_{ts}",
        )
        csv_dir = os.path.join(run_dir, "csv")
        img_dir = os.path.join(run_dir, "images")
        log_dir = os.path.join(run_dir, "logs")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # metadata.json
        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "simulator_ui",
            "seed": self.config["SEED"],
            "config": {k: v for k, v in self.config.items()},
            "ticks_run": self.tick_count,
            "sim_time_s": self.sim_time,
        }
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # mobility_positions.csv
        rows_pos = []
        for h in self.history:
            pos = h["positions"]
            vel = h["velocities"]
            for i in range(self.N):
                sp = float(np.linalg.norm(vel[i]))
                rows_pos.append({
                    "timestamp": round(h["t"], 6), "uav_id": i,
                    "x": round(pos[i][0], 4), "y": round(pos[i][1], 4), "z": round(pos[i][2], 4),
                    "vx": round(vel[i][0], 4), "vy": round(vel[i][1], 4), "vz": round(vel[i][2], 4),
                    "speed": round(sp, 4),
                })
        pd.DataFrame(rows_pos).to_csv(os.path.join(csv_dir, "mobility_positions.csv"), index=False)

        # uav_sink_distance.csv
        rows_d = []
        for h in self.history:
            for i in range(self.N):
                rows_d.append({
                    "timestamp": round(h["t"], 6), "uav_id": i,
                    "d_to_sink": round(h["distances"][i], 4),
                    "link_up": int(h["link_up"][i]),
                })
        pd.DataFrame(rows_d).to_csv(os.path.join(csv_dir, "uav_sink_distance.csv"), index=False)

        # config_changes.csv
        if self.config_change_log:
            pd.DataFrame(self.config_change_log).to_csv(
                os.path.join(csv_dir, "config_changes.csv"), index=False)

        # Generate mobility plots
        try:
            from algorithms.mobility.plotting import (
                plot_trajectories_3d, plot_distance_vs_time, plot_link_up_ratio,
            )
            hist_pos = [h["positions"] for h in self.history]
            hist_dist = [h["distances"] for h in self.history]
            hist_lu = [h["link_up"] for h in self.history]
            hist_times = [h["t"] for h in self.history]

            plot_trajectories_3d(hist_pos, hist_times, self.N, self.bounds,
                                self.sink_pos, img_dir, top_k=min(5, self.N))
            plot_distance_vs_time(hist_dist, hist_times, self.N,
                                 self.comm_range, img_dir, top_k=min(5, self.N))
            plot_link_up_ratio(hist_lu, hist_times, self.N, img_dir)
        except Exception as e:
            print(f"Warning: plot generation failed: {e}")

        # RL selection CSV + plot
        if self.rl_selections:
            rl_df = pd.DataFrame(self.rl_selections)
            rl_df.to_csv(os.path.join(csv_dir, "qlearning_mac_selection.csv"), index=False)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = rl_df["mac_selected"].map({"TDMA": "#2196F3", "CSMA_CA": "#FF5722"})
                ax.scatter(rl_df["sim_time"], rl_df["mac_selected"], c=colors, s=12, alpha=0.7)
                ax.set_xlabel("Simulation Time (s)")
                ax.set_ylabel("Selected MAC")
                ax.set_title("RL MAC Selection Over Time")
                fig.tight_layout()
                fig.savefig(os.path.join(img_dir, "qlearning_selected_mac_vs_traffic_rate.png"), dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"Warning: RL plot failed: {e}")

        return run_dir

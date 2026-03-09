# run_validation_experiments.py — Controlled A/B Experiments
# Validates impact of 3D mobility + path loss on throughput & delay.
#
# Cases: A(static,no-PL), B(mobile,no-PL), C(static,PL), D(mobile,PL)
# Each case: traffic sweep × multi-seed → aggregate mean±std
#
# Usage: python work/run_validation_experiments.py

import os
import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from algorithms.mac.baseline import Config, Logger
from algorithms.mac.channel_aware_mac import simulate_tdma_aware, simulate_csma_aware
from algorithms.mobility.speed import SpeedEngine
from algorithms.mobility.models import create_mobility_model
from algorithms.mobility.link import (
    compute_distances, compute_link_up, compute_pathloss_success_prob, compute_pathloss_db,
)

import config as CFG

# ======================================================================
# Experiment Parameters
# ======================================================================
CASES = {
    "A_static_noPL":  {"mobility": "static",       "pathloss": False},
    "B_mobile_noPL":  {"mobility": "gauss_markov",  "pathloss": False},
    "C_static_PL":    {"mobility": "static",       "pathloss": True},
    "D_mobile_PL":    {"mobility": "gauss_markov",  "pathloss": True},
}

SEEDS = [1, 2, 3]
MAC_PROTOCOLS = ["CSMA_CA", "TDMA"]

# Use faster settings for practical execution
# (N=20, 5s sim → ~555K slots per run instead of 1.67M)
VAL_N = 20          # Smaller N for fast validation
VAL_SIM_TIME = 5.0  # Shorter sim time

SWEEP_LOADS = np.array([10, 50, 100, 200, 400, 600, 800, 1000], dtype=int)
SIM_TIME_S = VAL_SIM_TIME
MOBILITY_DT = CFG.MOBILITY_DT
N = VAL_N

# Path-loss params
PL_K = CFG.PATHLOSS_K
PL_ETA = CFG.PATHLOSS_ETA


# ======================================================================
# Helpers
# ======================================================================
def precompute_mobility_trace(model_name, seed):
    """Run a full mobility trace and return link schedule arrays."""
    n_steps = int(np.ceil(SIM_TIME_S / MOBILITY_DT))
    rng = np.random.default_rng(seed)
    se = SpeedEngine(
        n_nodes=N, v_min=CFG.V_MIN, v_max=CFG.V_MAX,
        mode=CFG.SPEED_MODE, v_mean=CFG.V_MEAN, v_std=CFG.V_STD,
        update_interval=CFG.SPEED_UPDATE_INTERVAL, rng=rng,
    )
    model = create_mobility_model(
        name=model_name, n_nodes=N, bounds=(CFG.AREA_X, CFG.AREA_Y, CFG.AREA_Z),
        speed_engine=se, rng=rng,
        gm_alpha=CFG.GM_ALPHA, rwp_pause_time=CFG.RWP_PAUSE_TIME,
        circ_radius=CFG.CIRC_RADIUS, circ_omega_mean=CFG.CIRC_OMEGA_MEAN,
        circ_omega_std=CFG.CIRC_OMEGA_STD, circ_climb_rate=CFG.CIRC_CLIMB_RATE,
    )
    sink_pos = np.array([CFG.SINK_X, CFG.SINK_Y, CFG.SINK_Z])

    # Storage
    link_up_schedule = np.zeros((N, n_steps), dtype=int)
    success_prob_schedule = np.ones((N, n_steps), dtype=float)
    all_distances = np.zeros((N, n_steps))
    all_speeds = np.zeros((N, n_steps))
    all_positions = np.zeros((n_steps, N, 3))

    for step_idx in range(n_steps):
        if step_idx == 0:
            pos = model.positions.copy()
            vel = model.velocities.copy()
        else:
            pos, vel = model.update(MOBILITY_DT)

        distances = compute_distances(pos, sink_pos)
        lu = compute_link_up(distances, CFG.COMM_RANGE_R)
        sp = compute_pathloss_success_prob(distances, k=PL_K, eta=PL_ETA)

        link_up_schedule[:, step_idx] = lu
        success_prob_schedule[:, step_idx] = sp
        all_distances[:, step_idx] = distances
        all_speeds[:, step_idx] = np.linalg.norm(vel, axis=1)
        all_positions[step_idx] = pos

    return {
        "link_up": link_up_schedule,
        "success_prob": success_prob_schedule,
        "distances": all_distances,
        "speeds": all_speeds,
        "positions": all_positions,
    }


def run_single(mac, offered_pps, seed, link_up, success_prob, use_pathloss):
    """Run a single (protocol, load, seed) point with channel awareness."""
    cfg = Config(
        N=N, sim_time_s=SIM_TIME_S, slot_time_s=CFG.SLOT_TIME_S,
        phy_rate_bps=CFG.PHY_RATE_BPS, payload_bytes=CFG.PAYLOAD_BYTES,
        QMAX=CFG.QMAX, seed=seed,
        cw_min=CFG.CW_MIN, cw_max=CFG.CW_MAX,
        difs_slots=CFG.DIFS_SLOTS, sifs_slots=CFG.SIFS_SLOTS,
        ack_slots=CFG.ACK_SLOTS, ack_timeout_slots=CFG.ACK_TIMEOUT_SLOTS,
        max_retry=CFG.MAX_RETRY, log_interval_slots=CFG.LOG_INTERVAL_SLOTS,
        rts_cts_enabled=CFG.RTS_CTS_ENABLED, ack_enabled=CFG.ACK_ENABLED,
        rts_slots=CFG.RTS_SLOTS, cts_slots=CFG.CTS_SLOTS,
        tdma_guard_time_s=CFG.TDMA_GUARD_TIME_S,
    )
    log = Logger(load_pps=offered_pps, protocol_name=mac)

    sp_sched = success_prob if use_pathloss else None

    if mac == "TDMA":
        simulate_tdma_aware(cfg, offered_pps, log, link_up, sp_sched, MOBILITY_DT)
    else:
        simulate_csma_aware(cfg, offered_pps, log, link_up, sp_sched, MOBILITY_DT)

    throughput_mbps = log.get_throughput_bps(SIM_TIME_S) / 1e6
    delay_ms = log.get_avg_end_to_end_delay_s() * 1000
    delivery_ratio = log.pkts_success / max(log.pkts_generated, 1)

    link_stats = getattr(log, '_link_stats', {})

    return {
        "throughput_mbps": throughput_mbps,
        "delay_ms": delay_ms,
        "delivery_ratio": delivery_ratio,
        "pkts_generated": log.pkts_generated,
        "pkts_success": log.pkts_success,
        "pkts_dropped": log.pkts_dropped_qfull + log.pkts_dropped_mac,
        "collisions": log.collision_events,
        "link_blocked": link_stats.get("link_blocked", 0),
        "pathloss_failed": link_stats.get("pathloss_failed", 0),
    }


# ======================================================================
# Main Experiment Loop
# ======================================================================
def run_experiments():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", f"validation_AB_{ts}")
    csv_dir = os.path.join(output_dir, "csv")
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    print("=" * 65)
    print("  Controlled A/B Validation Experiments")
    print(f"  Cases: {list(CASES.keys())}")
    print(f"  MACs: {MAC_PROTOCOLS}")
    print(f"  Seeds: {SEEDS} | Traffic: {SWEEP_LOADS[0]}–{SWEEP_LOADS[-1]} pps ({len(SWEEP_LOADS)} pts)")
    print(f"  N={N} | SIM={SIM_TIME_S}s | Area={CFG.AREA_X}×{CFG.AREA_Y}×{CFG.AREA_Z}m")
    print(f"  Output: {output_dir}")
    print("=" * 65)

    # ---- Precompute mobility traces (cached per case × seed) ----
    traces = {}
    for case_name, case_cfg in CASES.items():
        for seed in SEEDS:
            key = (case_name, seed)
            print(f"  Precomputing trace: {case_name}, seed={seed}...", end="", flush=True)
            traces[key] = precompute_mobility_trace(case_cfg["mobility"], seed)
            print(" done")

    # ---- Run experiments ----
    all_results = []
    total_runs = len(CASES) * len(MAC_PROTOCOLS) * len(SWEEP_LOADS) * len(SEEDS)
    run_count = 0

    for case_name, case_cfg in CASES.items():
        for mac in MAC_PROTOCOLS:
            for load in SWEEP_LOADS:
                for seed in SEEDS:
                    run_count += 1
                    trace = traces[(case_name, seed)]
                    result = run_single(
                        mac, int(load), seed,
                        trace["link_up"], trace["success_prob"],
                        use_pathloss=case_cfg["pathloss"],
                    )
                    result["case"] = case_name
                    result["mac"] = mac
                    result["load_pps"] = int(load)
                    result["seed"] = seed
                    all_results.append(result)

                    if run_count % 50 == 0 or run_count == total_runs:
                        print(f"  [{run_count}/{total_runs}] {case_name} {mac} load={load} seed={seed} "
                              f"→ Thr={result['throughput_mbps']:.3f} Mbps, Del={result['delay_ms']:.2f} ms, "
                              f"DR={result['delivery_ratio']:.3f}")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(csv_dir, "all_results.csv"), index=False)
    print(f"\n  Saved {len(df)} rows to all_results.csv")

    # ---- Aggregate (mean ± std across seeds) ----
    agg = df.groupby(["case", "mac", "load_pps"]).agg(
        throughput_mean=("throughput_mbps", "mean"),
        throughput_std=("throughput_mbps", "std"),
        delay_mean=("delay_ms", "mean"),
        delay_std=("delay_ms", "std"),
        delivery_mean=("delivery_ratio", "mean"),
        delivery_std=("delivery_ratio", "std"),
        pkts_gen_mean=("pkts_generated", "mean"),
        pkts_succ_mean=("pkts_success", "mean"),
        drops_mean=("pkts_dropped", "mean"),
        collisions_mean=("collisions", "mean"),
        link_blocked_mean=("link_blocked", "mean"),
        pl_failed_mean=("pathloss_failed", "mean"),
    ).reset_index()
    agg.to_csv(os.path.join(csv_dir, "aggregate_summary.csv"), index=False)
    print(f"  Saved aggregate_summary.csv ({len(agg)} rows)")

    # ---- Mobility & link stats (per seed, from traces) ----
    mob_rows = []
    link_rows = []
    for case_name, case_cfg in CASES.items():
        for seed in SEEDS:
            tr = traces[(case_name, seed)]
            dists = tr["distances"]  # (N, T)
            speeds = tr["speeds"]    # (N, T)
            lu = tr["link_up"]       # (N, T)

            mob_rows.append({
                "case": case_name, "seed": seed,
                "mean_speed": float(speeds.mean()),
                "p95_speed": float(np.percentile(speeds, 95)),
                "max_speed": float(speeds.max()),
                "mean_distance": float(dists.mean()),
                "p95_distance": float(np.percentile(dists, 95)),
                "link_up_ratio": float(lu.mean()),
            })

            # Per-load link success stats
            for load in SWEEP_LOADS:
                link_rows.append({
                    "case": case_name, "seed": seed, "load_pps": int(load),
                    "avg_distance": float(dists.mean()),
                    "p95_distance": float(np.percentile(dists, 95)),
                    "link_up_ratio": float(lu.mean()),
                    "mean_success_prob": float(tr["success_prob"].mean()) if case_cfg["pathloss"] else 1.0,
                })

    pd.DataFrame(mob_rows).to_csv(os.path.join(csv_dir, "mobility_stats.csv"), index=False)
    pd.DataFrame(link_rows).to_csv(os.path.join(csv_dir, "link_success_stats.csv"), index=False)

    # ---- Distance / link-up time series CSV (seed=1 only) ----
    for case_name in CASES:
        tr = traces[(case_name, 1)]
        dist_rows = []
        n_steps = tr["distances"].shape[1]
        for s_idx in range(n_steps):
            t = s_idx * MOBILITY_DT
            for i in range(N):
                dist_rows.append({
                    "timestamp": round(t, 4), "uav_id": i,
                    "d_to_sink_m": round(float(tr["distances"][i, s_idx]), 2),
                    "link_up": int(tr["link_up"][i, s_idx]),
                    "pathloss_db": round(float(compute_pathloss_db(
                        np.array([tr["distances"][i, s_idx]]), eta=PL_ETA)[0]), 2),
                })
        pd.DataFrame(dist_rows).to_csv(
            os.path.join(csv_dir, f"uav_sink_distance_{case_name}.csv"), index=False)

    print(f"  Saved mobility_stats.csv, link_success_stats.csv, uav_sink_distance CSVs")

    # ---- Plots ----
    print("\n  Generating comparison plots...")
    generate_plots(agg, traces, csv_dir, img_dir)

    # ---- Metadata ----
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "N": N, "sim_time_s": SIM_TIME_S,
        "area": [CFG.AREA_X, CFG.AREA_Y, CFG.AREA_Z],
        "sink": [CFG.SINK_X, CFG.SINK_Y, CFG.SINK_Z],
        "comm_range": CFG.COMM_RANGE_R,
        "pathloss_k": PL_K, "pathloss_eta": PL_ETA,
        "mobility_model": "gauss_markov", "gm_alpha": CFG.GM_ALPHA,
        "speed_mode": CFG.SPEED_MODE, "v_min": CFG.V_MIN, "v_max": CFG.V_MAX,
        "seeds": SEEDS, "sweep_loads": SWEEP_LOADS.tolist(),
        "mac_protocols": MAC_PROTOCOLS,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Interpretation ----
    write_interpretation(agg, mob_rows, output_dir)

    print(f"\n  ALL DONE. Results: {output_dir}")
    return output_dir


# ======================================================================
# Plot Generation
# ======================================================================
def generate_plots(agg, traces, csv_dir, img_dir):
    """Generate all comparison plots."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#fafafa",
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    })

    for mac in MAC_PROTOCOLS:
        _plot_throughput_static_vs_mobile(agg, mac, img_dir)
        _plot_delay_static_vs_mobile(agg, mac, img_dir)
        _plot_delivery_ratio_pathloss(agg, mac, img_dir)

    _plot_distance_time_series(traces, img_dir)
    _plot_link_up_ratio_time(traces, img_dir)


def _plot_throughput_static_vs_mobile(agg, mac, img_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"A_static_noPL": "#2196F3", "B_mobile_noPL": "#FF5722",
              "C_static_PL": "#4CAF50", "D_mobile_PL": "#9C27B0"}
    labels = {"A_static_noPL": "Static, No PL", "B_mobile_noPL": "Mobile, No PL",
              "C_static_PL": "Static + PL", "D_mobile_PL": "Mobile + PL"}

    for case in CASES:
        sub = agg[(agg["case"] == case) & (agg["mac"] == mac)]
        if sub.empty:
            continue
        ax.plot(sub["load_pps"], sub["throughput_mean"], "-o", ms=3,
                color=colors[case], label=labels[case])
        ax.fill_between(sub["load_pps"],
                        sub["throughput_mean"] - sub["throughput_std"],
                        sub["throughput_mean"] + sub["throughput_std"],
                        color=colors[case], alpha=0.15)

    ax.set_xlabel("Offered Load (pps)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title(f"Throughput: Static vs Mobile — {mac}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, f"throughput_static_vs_mobile_{mac}.png"), dpi=150)
    plt.close(fig)


def _plot_delay_static_vs_mobile(agg, mac, img_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"A_static_noPL": "#2196F3", "B_mobile_noPL": "#FF5722",
              "C_static_PL": "#4CAF50", "D_mobile_PL": "#9C27B0"}
    labels = {"A_static_noPL": "Static, No PL", "B_mobile_noPL": "Mobile, No PL",
              "C_static_PL": "Static + PL", "D_mobile_PL": "Mobile + PL"}

    for case in CASES:
        sub = agg[(agg["case"] == case) & (agg["mac"] == mac)]
        if sub.empty:
            continue
        ax.plot(sub["load_pps"], sub["delay_mean"], "-o", ms=3,
                color=colors[case], label=labels[case])
        ax.fill_between(sub["load_pps"],
                        sub["delay_mean"] - sub["delay_std"],
                        sub["delay_mean"] + sub["delay_std"],
                        color=colors[case], alpha=0.15)

    ax.set_xlabel("Offered Load (pps)")
    ax.set_ylabel("End-to-End Delay (ms)")
    ax.set_title(f"Delay: Static vs Mobile — {mac}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, f"delay_static_vs_mobile_{mac}.png"), dpi=150)
    plt.close(fig)


def _plot_delivery_ratio_pathloss(agg, mac, img_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"A_static_noPL": "#2196F3", "B_mobile_noPL": "#FF5722",
              "C_static_PL": "#4CAF50", "D_mobile_PL": "#9C27B0"}
    labels = {"A_static_noPL": "Static, No PL", "B_mobile_noPL": "Mobile, No PL",
              "C_static_PL": "Static + PL", "D_mobile_PL": "Mobile + PL"}

    for case in CASES:
        sub = agg[(agg["case"] == case) & (agg["mac"] == mac)]
        if sub.empty:
            continue
        ax.plot(sub["load_pps"], sub["delivery_mean"], "-o", ms=3,
                color=colors[case], label=labels[case])
        ax.fill_between(sub["load_pps"],
                        sub["delivery_mean"] - sub["delivery_std"],
                        sub["delivery_mean"] + sub["delivery_std"],
                        color=colors[case], alpha=0.15)

    ax.set_xlabel("Offered Load (pps)")
    ax.set_ylabel("Delivery Ratio")
    ax.set_title(f"Delivery Ratio: Path Loss ON vs OFF — {mac}")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, f"delivery_ratio_pathloss_on_off_{mac}.png"), dpi=150)
    plt.close(fig)


def _plot_distance_time_series(traces, img_dir):
    """Distance-to-sink vs time for representative UAVs (seed=1)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    case_list = list(CASES.keys())
    top_k = min(5, N)

    for idx, case_name in enumerate(case_list):
        ax = axes[idx // 2][idx % 2]
        tr = traces[(case_name, 1)]
        dists = tr["distances"]  # (N, T)
        n_steps = dists.shape[1]
        times = np.arange(n_steps) * MOBILITY_DT

        for uav_id in range(top_k):
            ax.plot(times, dists[uav_id], linewidth=0.8, alpha=0.8, label=f"UAV {uav_id}")

        ax.axhline(y=CFG.COMM_RANGE_R, color="red", linestyle="--", alpha=0.5, label="Comm Range")
        ax.set_ylabel("Distance (m)")
        ax.set_title(case_name.replace("_", " "))
        if idx >= 2:
            ax.set_xlabel("Time (s)")
        ax.legend(fontsize=7, ncol=3)

    fig.suptitle("Distance to Sink vs Time (Top-5 UAVs, Seed=1)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "distance_to_sink_time_uavK.png"), dpi=150)
    plt.close(fig)


def _plot_link_up_ratio_time(traces, img_dir):
    """Link-up ratio (fraction connected) vs time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"A_static_noPL": "#2196F3", "B_mobile_noPL": "#FF5722",
              "C_static_PL": "#4CAF50", "D_mobile_PL": "#9C27B0"}

    for case_name in CASES:
        tr = traces[(case_name, 1)]
        lu = tr["link_up"]  # (N, T)
        ratio = lu.mean(axis=0)  # mean across UAVs per step
        n_steps = lu.shape[1]
        times = np.arange(n_steps) * MOBILITY_DT
        ax.plot(times, ratio, linewidth=1, color=colors[case_name],
                label=case_name.replace("_", " "))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Link-Up Ratio")
    ax.set_title("Fraction of Connected UAVs Over Time (Seed=1)")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "link_up_ratio_time.png"), dpi=150)
    plt.close(fig)


# ======================================================================
# Interpretation Write-up
# ======================================================================
def write_interpretation(agg, mob_rows, output_dir):
    """Auto-generate a structured interpretation of results."""
    lines = []
    lines.append("=" * 70)
    lines.append("  VALIDATION RESULTS — Mobility & Path Loss Impact")
    lines.append("=" * 70)

    mob_df = pd.DataFrame(mob_rows)

    lines.append("\n1. MOBILITY STATISTICS (averaged across seeds)")
    lines.append("-" * 50)
    for case in CASES:
        sub = mob_df[mob_df["case"] == case]
        lines.append(f"  {case}:")
        lines.append(f"    Mean speed:       {sub['mean_speed'].mean():.2f} m/s")
        lines.append(f"    P95 speed:        {sub['p95_speed'].mean():.2f} m/s")
        lines.append(f"    Mean distance:    {sub['mean_distance'].mean():.1f} m")
        lines.append(f"    P95 distance:     {sub['p95_distance'].mean():.1f} m")
        lines.append(f"    Link-up ratio:    {sub['link_up_ratio'].mean():.3f}")

    lines.append("\n2. THROUGHPUT IMPACT")
    lines.append("-" * 50)
    for mac in MAC_PROTOCOLS:
        lines.append(f"\n  [{mac}]")
        # Compare A vs B (mobility only)
        a = agg[(agg["case"] == "A_static_noPL") & (agg["mac"] == mac)]
        b = agg[(agg["case"] == "B_mobile_noPL") & (agg["mac"] == mac)]
        if not a.empty and not b.empty:
            a_peak = a["throughput_mean"].max()
            b_peak = b["throughput_mean"].max()
            delta = (b_peak - a_peak) / max(a_peak, 1e-9) * 100
            lines.append(f"    Static peak throughput:  {a_peak:.4f} Mbps")
            lines.append(f"    Mobile peak throughput:  {b_peak:.4f} Mbps")
            lines.append(f"    Δ (mobile vs static):   {delta:+.1f}%")

        # Compare A vs C (path loss only)
        c = agg[(agg["case"] == "C_static_PL") & (agg["mac"] == mac)]
        if not a.empty and not c.empty:
            c_peak = c["throughput_mean"].max()
            delta_pl = (c_peak - a_peak) / max(a_peak, 1e-9) * 100
            lines.append(f"    Static+PL peak:         {c_peak:.4f} Mbps")
            lines.append(f"    Δ (PL vs no-PL):        {delta_pl:+.1f}%")

    lines.append("\n3. DELAY IMPACT")
    lines.append("-" * 50)
    for mac in MAC_PROTOCOLS:
        lines.append(f"\n  [{mac}]")
        a = agg[(agg["case"] == "A_static_noPL") & (agg["mac"] == mac)]
        b = agg[(agg["case"] == "B_mobile_noPL") & (agg["mac"] == mac)]
        if not a.empty and not b.empty:
            # At highest shared load
            max_load = min(a["load_pps"].max(), b["load_pps"].max())
            da = a[a["load_pps"] == max_load]["delay_mean"].values
            db = b[b["load_pps"] == max_load]["delay_mean"].values
            if len(da) > 0 and len(db) > 0:
                lines.append(f"    Delay at {max_load} pps (static): {da[0]:.2f} ms")
                lines.append(f"    Delay at {max_load} pps (mobile): {db[0]:.2f} ms")
                delta_d = (db[0] - da[0]) / max(da[0], 1e-9) * 100
                lines.append(f"    Δ delay (mobile):       {delta_d:+.1f}%")

    lines.append("\n4. DELIVERY RATIO IMPACT")
    lines.append("-" * 50)
    for mac in MAC_PROTOCOLS:
        lines.append(f"\n  [{mac}]")
        for case in CASES:
            sub = agg[(agg["case"] == case) & (agg["mac"] == mac)]
            if not sub.empty:
                avg_dr = sub["delivery_mean"].mean()
                lines.append(f"    {case}: avg delivery ratio = {avg_dr:.4f}")

    lines.append("\n5. CONCLUSIONS")
    lines.append("-" * 50)
    lines.append("  (Auto-generated — verify against plots)")

    # Mobility conclusion
    for mac in MAC_PROTOCOLS:
        a = agg[(agg["case"] == "A_static_noPL") & (agg["mac"] == mac)]
        b = agg[(agg["case"] == "B_mobile_noPL") & (agg["mac"] == mac)]
        if not a.empty and not b.empty:
            a_dr = a["delivery_mean"].mean()
            b_dr = b["delivery_mean"].mean()
            if b_dr < a_dr - 0.01:
                lines.append(f"  ✓ [{mac}] Mobility REDUCES delivery ratio ({a_dr:.3f} → {b_dr:.3f})")
                lines.append(f"    → Link outages from UAV movement cause packet loss")
            else:
                lines.append(f"  ≈ [{mac}] Mobility has MINIMAL impact on delivery ratio ({a_dr:.3f} → {b_dr:.3f})")
                lines.append(f"    → Most UAVs remain within comm range during simulation")

    # Path-loss conclusion
    for mac in MAC_PROTOCOLS:
        a = agg[(agg["case"] == "A_static_noPL") & (agg["mac"] == mac)]
        c = agg[(agg["case"] == "C_static_PL") & (agg["mac"] == mac)]
        if not a.empty and not c.empty:
            a_dr = a["delivery_mean"].mean()
            c_dr = c["delivery_mean"].mean()
            if c_dr < a_dr - 0.01:
                lines.append(f"  ✓ [{mac}] Path loss REDUCES delivery ratio ({a_dr:.3f} → {c_dr:.3f})")
                lines.append(f"    → Distance-dependent success probability drops packets")
            else:
                lines.append(f"  ≈ [{mac}] Path loss has MINIMAL impact ({a_dr:.3f} → {c_dr:.3f})")
                lines.append(f"    → Distances may be within reliable range for current k/eta")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    with open(os.path.join(output_dir, "analysis_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    # Print with safe encoding for Windows console
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))


# ======================================================================
if __name__ == "__main__":
    t0 = time.time()
    run_experiments()
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

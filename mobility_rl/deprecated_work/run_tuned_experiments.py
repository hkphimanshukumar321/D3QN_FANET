# run_tuned_experiments.py — Tuned 3-Phase Experiment Runner
# Phase 1: Mobility Sensitivity (Geometry & Speed)
# Phase 2: Path Loss Sensitivity (K value tuning)
# Phase 3: Congestion Stress Test (MAC comparison under load)
#
# Usage: python work/run_tuned_experiments.py

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
    compute_distances, compute_link_up, compute_pathloss_success_prob
)

import config as CFG

# ======================================================================
# Phase Definitions
# ======================================================================

PHASE1_GEOMETRIES = {
    "G1_Baseline": {"area": (1000, 1000, 300), "R": 500, "model": "gauss_markov", "v": (5, 30)},
    "G2_Stretch":  {"area": (1500, 1500, 400), "R": 400, "model": "gauss_markov", "v": (15, 35)},
    "G3_Wide":     {"area": (2000, 2000, 500), "R": 300, "model": "gauss_markov", "v": (15, 35)},
    "G4_Extreme":  {"area": (2000, 2000, 500), "R": 300, "model": "random_waypoint", "v": (20, 45)},
}

PHASE2_PATHLOSS = {
    "PL_mild":     1e-6,
    "PL_moderate": 5e-6,
    "PL_balanced": 1e-5,
}

SEEDS = [1, 2]

# ======================================================================
# Core Runner Helpers
# ======================================================================

def precompute_mobility_trace(N, sim_time_s, dt, area, R, mob_model, v_min, v_max, seed):
    n_steps = int(np.ceil(sim_time_s / dt))
    rng = np.random.default_rng(seed)
    
    se = SpeedEngine(
        n_nodes=N, v_min=v_min, v_max=v_max,
        mode="uniform", rng=rng,
        update_interval=CFG.SPEED_UPDATE_INTERVAL
    )
    
    # Static model handles zero velocity internally
    model = create_mobility_model(
        name=mob_model, n_nodes=N, bounds=area,
        speed_engine=se, rng=rng, gm_alpha=0.5, rwp_pause_time=1.0
    )
    
    # Sink at bottom center
    sink_pos = np.array([area[0]/2, area[1]/2, 0.0])

    link_up_schedule = np.zeros((N, n_steps), dtype=int)
    all_distances = np.zeros((N, n_steps))
    all_speeds = np.zeros((N, n_steps))

    for step_idx in range(n_steps):
        if step_idx == 0:
            pos = model.positions.copy()
            vel = model.velocities.copy()
        else:
            pos, vel = model.update(dt)

        distances = compute_distances(pos, sink_pos)
        lu = compute_link_up(distances, R)

        link_up_schedule[:, step_idx] = lu
        all_distances[:, step_idx] = distances
        all_speeds[:, step_idx] = np.linalg.norm(vel, axis=1)

    return {
        "link_up": link_up_schedule,
        "distances": all_distances,
        "speeds": all_speeds,
    }

def run_single(N, sim_time_s, mac, offered_pps, seed, link_up, success_prob, use_pathloss):
    cfg = Config(
        N=N, sim_time_s=sim_time_s, slot_time_s=CFG.SLOT_TIME_S,
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
        simulate_tdma_aware(cfg, offered_pps, log, link_up, sp_sched, CFG.MOBILITY_DT)
    else:
        simulate_csma_aware(cfg, offered_pps, log, link_up, sp_sched, CFG.MOBILITY_DT)

    throughput_mbps = log.get_throughput_bps(sim_time_s) / 1e6
    delay_ms = log.get_avg_end_to_end_delay_s() * 1000
    delivery_ratio = log.pkts_success / max(log.pkts_generated, 1)

    return {
        "throughput_mbps": throughput_mbps,
        "delay_ms": delay_ms,
        "delivery_ratio": delivery_ratio,
        "pkts_generated": log.pkts_generated,
        "pkts_success": log.pkts_success,
        "pkts_dropped": log.pkts_dropped_qfull + log.pkts_dropped_mac,
        "collisions": log.collision_events,
    }

def aggregate_and_save(df, csv_dir, filename="all_results.csv", agg_filename="aggregate_summary.csv"):
    df.to_csv(os.path.join(csv_dir, filename), index=False)
    
    # Grouping columns: case, mac, load_pps (and geom/pl_k if present)
    group_cols = [c for c in ["case", "geom_name", "pl_k", "mac", "load_pps"] if c in df.columns]
    
    agg = df.groupby(group_cols).agg(
        throughput_mean=("throughput_mbps", "mean"),
        delay_mean=("delay_ms", "mean"),
        delivery_mean=("delivery_ratio", "mean"),
        drops_mean=("pkts_dropped", "mean"),
        collisions_mean=("collisions", "mean"),
    ).reset_index()
    agg.to_csv(os.path.join(csv_dir, agg_filename), index=False)
    return agg

# ======================================================================
# Phase 1: Mobility Sensitivity
# ======================================================================
def run_phase1(out_dir):
    print("\n--- PHASE 1: Mobility Sensitivity ---")
    N = 20
    sim_time_s = 1.0
    dt = CFG.MOBILITY_DT
    loads = [400]  # Single moderate load to test geometry impact
    mac = "CSMA_CA"
    
    csv_dir = os.path.join(out_dir, "phase1_csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    all_results = []
    
    for geom_name, g in PHASE1_GEOMETRIES.items():
        for is_mobile in [False, True]:
            case_name = f"{geom_name}_{'mobile' if is_mobile else 'static'}"
            mob_model = g["model"] if is_mobile else "static"
            
            for seed in SEEDS:
                trace = precompute_mobility_trace(
                    N, sim_time_s, dt, g["area"], g["R"], mob_model, g["v"][0], g["v"][1], seed
                )
                
                # Plot distance variation (quick diagnostic)
                if seed == SEEDS[0]:
                    plt.figure(figsize=(6,3))
                    for i in range(min(5, N)):
                        plt.plot(np.arange(trace["distances"].shape[1])*dt, trace["distances"][i])
                    plt.axhline(g["R"], color='r', linestyle='--', label=f'R={g["R"]}')
                    plt.title(f"Distance to Sink - {case_name}")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Distance (m)")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"phase1_dist_{case_name}.png"))
                    plt.close()
                
                # Metric: average link-up ratio across all nodes + time
                avg_link_up = float(trace["link_up"].mean())
                
                for load in loads:
                    # Path loss OFF (all 1.0)
                    sp = np.ones_like(trace["link_up"], dtype=float)
                    res = run_single(N, sim_time_s, mac, load, seed, trace["link_up"], sp, use_pathloss=False)
                    res.update({
                        "phase": 1, "geom_name": geom_name, "case": case_name, 
                        "mac": mac, "load_pps": load, "seed": seed,
                        "avg_link_up_ratio": avg_link_up
                    })
                    all_results.append(res)
                    print(f"  P1 {case_name} sd={seed} → LU: {avg_link_up:.2%} | Thr: {res['throughput_mbps']:.2f} Mbps, DR: {res['delivery_ratio']:.2%}")

    df = pd.DataFrame(all_results)
    aggregate_and_save(df, csv_dir, "phase1_results.csv", "phase1_agg.csv")
    return df

# ======================================================================
# Phase 2: Path Loss Sensitivity
# ======================================================================
def run_phase2(out_dir):
    print("\n--- PHASE 2: Path Loss Sensitivity ---")
    N = 20
    sim_time_s = 1.0
    dt = CFG.MOBILITY_DT
    loads = [400]
    mac = "CSMA_CA"
    
    # Use G3_Wide geometry
    g = PHASE1_GEOMETRIES["G3_Wide"]
    
    csv_dir = os.path.join(out_dir, "phase2_csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    all_results = []
    
    for pl_name, k_val in PHASE2_PATHLOSS.items():
        for is_mobile in [False, True]:
            case_name = f"G3_{'mobile' if is_mobile else 'static'}_{pl_name}"
            mob_model = g["model"] if is_mobile else "static"
            
            for seed in SEEDS:
                trace = precompute_mobility_trace(
                    N, sim_time_s, dt, g["area"], g["R"], mob_model, g["v"][0], g["v"][1], seed
                )
                
                # Compute actual path loss array
                sp_sched = compute_pathloss_success_prob(trace["distances"], k=k_val, eta=2.0)
                avg_sp = float(sp_sched.mean())
                
                for load in loads:
                    res = run_single(N, sim_time_s, mac, load, seed, trace["link_up"], sp_sched, use_pathloss=True)
                    res.update({
                        "phase": 2, "pl_k_name": pl_name, "pl_k": k_val, "case": case_name,
                        "mac": mac, "load_pps": load, "seed": seed,
                        "avg_succ_prob": avg_sp
                    })
                    all_results.append(res)
                    print(f"  P2 {case_name} sd={seed} → P_succ: {avg_sp:.2%} | Thr: {res['throughput_mbps']:.2f} Mbps, DR: {res['delivery_ratio']:.2%}")

    df = pd.DataFrame(all_results)
    aggregate_and_save(df, csv_dir, "phase2_results.csv", "phase2_agg.csv")
    return df

# ======================================================================
# Phase 3: Congestion Stress Test
# ======================================================================
def run_phase3(out_dir):
    print("\n--- PHASE 3: Congestion Stress Test ---")
    N = 30
    sim_time_s = 1.0
    dt = CFG.MOBILITY_DT
    sweep_loads = [100, 300, 500, 800, 1000]
    macs = ["CSMA_CA", "TDMA"]
    
    # Fixed config: G3 geometry + PL_balanced
    g = PHASE1_GEOMETRIES["G3_Wide"]
    k_val = PHASE2_PATHLOSS["PL_balanced"]
    
    csv_dir = os.path.join(out_dir, "phase3_csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    cases = {
        "A_static_noPL":  {"mobile": False, "pl": False},
        "B_mobile_noPL":  {"mobile": True,  "pl": False},
        "C_static_PL":    {"mobile": False, "pl": True},
        "D_mobile_PL":    {"mobile": True,  "pl": True},
    }
    
    all_results = []
    
    for case_name, c in cases.items():
        mob_model = g["model"] if c["mobile"] else "static"
        
        for seed in SEEDS:
            trace = precompute_mobility_trace(
                N, sim_time_s, dt, g["area"], g["R"], mob_model, g["v"][0], g["v"][1], seed
            )
            
            if c["pl"]:
                sp_sched = compute_pathloss_success_prob(trace["distances"], k=k_val, eta=2.0)
            else:
                sp_sched = np.ones_like(trace["link_up"], dtype=float)
            
            for mac in macs:
                for load in sweep_loads:
                    res = run_single(N, sim_time_s, mac, load, seed, trace["link_up"], sp_sched, use_pathloss=c["pl"])
                    res.update({
                        "phase": 3, "case": case_name, "mac": mac, "load_pps": load, "seed": seed
                    })
                    all_results.append(res)
                    print(f"  P3 {case_name} {mac} L={load} sd={seed} → Thr: {res['throughput_mbps']:.2f} Mbps, Del: {res['delay_ms']:.1f} ms, DR: {res['delivery_ratio']:.2%}")

    df = pd.DataFrame(all_results)
    agg = aggregate_and_save(df, csv_dir, "phase3_results.csv", "phase3_agg.csv")
    
    # Generate Phase 3 plots
    plot_dir = os.path.join(out_dir, "phase3_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    for case_name in cases.keys():
        sub = agg[agg["case"] == case_name]
        if sub.empty: continue
        
        # Throughput
        plt.figure(figsize=(8, 5))
        for mac in macs:
            s_mac = sub[sub["mac"] == mac].sort_values("load_pps")
            plt.plot(s_mac["load_pps"], s_mac["throughput_mean"], marker="o", label=mac)
            plt.fill_between(s_mac["load_pps"], 
                             s_mac["throughput_mean"] - s_mac["throughput_mean"].std(), # Note std from original agg was not stored, simple line plot
                             s_mac["throughput_mean"] + s_mac["throughput_mean"].std(), alpha=0.2)
        plt.title(f"Throughput vs Load ({case_name})")
        plt.xlabel("Offered Load (pps)")
        plt.ylabel("Throughput (Mbps)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f"throughput_{case_name}.png"))
        plt.close()
        
        # Delay
        plt.figure(figsize=(8, 5))
        for mac in macs:
            s_mac = sub[sub["mac"] == mac].sort_values("load_pps")
            plt.plot(s_mac["load_pps"], s_mac["delay_mean"], marker="s", label=mac)
        plt.title(f"Delay vs Load ({case_name})")
        plt.xlabel("Offered Load (pps)")
        plt.ylabel("Delay (ms)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f"delay_{case_name}.png"))
        plt.close()

    return df

# ======================================================================
# Main Execution
# ======================================================================
if __name__ == "__main__":
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, "results", f"tuned_experiment_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("  Tuned Configurations Experiment Runner")
    print(f"  Output directory: {out_dir}")
    print("=" * 60)
    
    t0 = time.time()
    
    run_phase1(out_dir)
    run_phase2(out_dir)
    run_phase3(out_dir)
    
    t1 = time.time()
    print("\n" + "=" * 60)
    print(f"  All phases completed in {t1-t0:.1f}s")
    print(f"  Results saved into: {out_dir}")
    print("=" * 60)

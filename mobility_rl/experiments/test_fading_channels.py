"""
Fading Channel Performance Experiment — Updated for Current FANET Scenario.

Matches config.py: N=50, SIM_TIME=15s, QPSK, Nakagami default, PHY=5Mbps,
PAYLOAD=1000B, COMM_RANGE=200m, TX=20dBm, NOISE=-80dBm.

Generates:
  1. BER vs SNR curves (Monte Carlo, all 4 channels)
  2. TDMA throughput under each fading model
  3. CSMA/CA throughput under each fading model
  4. Combined delay comparison
  5. Packet success probability vs distance heatmap
  6. Summary CSV
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from algorithms.mac.baseline import Config, Logger
from algorithms.mac.channel_aware_mac import simulate_tdma_aware, simulate_csma_aware
from algorithms.mobility.link import compute_fading_success_prob
from algorithms.channel.fading import (
    BERCalculator, AWGNChannel, RayleighChannel, RicianChannel, NakagamiChannel
)
from configs import config as global_cfg

# ── Journal-quality matplotlib style ─────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Current scenario parameters (synced from config.py) ──────────────
N               = global_cfg.N                    # 50
SIM_TIME_S      = global_cfg.SIM_TIME_S           # 15
PHY_RATE_BPS    = global_cfg.PHY_RATE_BPS         # 5e6
PAYLOAD_BYTES   = global_cfg.PAYLOAD_BYTES        # 1000
PAYLOAD_BITS    = PAYLOAD_BYTES * 8               # 8000
TX_POWER_DBM    = global_cfg.TX_POWER_DBM         # 20
NOISE_POWER_DBM = global_cfg.NOISE_POWER_DBM      # -80
MODULATION      = global_cfg.MODULATION           # QPSK
NAKAGAMI_M      = global_cfg.NAKAGAMI_M           # 2.0
RICIAN_K        = global_cfg.RICIAN_K             # 3.0
COMM_RANGE      = global_cfg.COMM_RANGE_R         # 200
QMAX            = global_cfg.QMAX                 # 100
SEED            = global_cfg.SEED                 # 42

# Channel palette shared by all figures
CHANNELS = [
    AWGNChannel(),
    RayleighChannel(),
    RicianChannel(K=RICIAN_K),
    NakagamiChannel(m=NAKAGAMI_M),
]
CH_MARKERS = ['o', 's', '^', 'D']
CH_COLORS  = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
CH_NAMES   = [ch.name for ch in CHANNELS]


def _output_dir():
    d = os.path.join(project_root, "results", "fading_test")
    os.makedirs(d, exist_ok=True)
    return d


# =====================================================================
# FIGURE 1: BER vs SNR — Monte Carlo (100k samples per point)
# =====================================================================
def plot_ber_vs_snr(output_dir):
    print("\n[1/5] Generating BER vs SNR curves ...")
    snr_db  = np.linspace(0, 30, 120)
    snr_lin = 10.0 ** (snr_db / 10.0)
    calc    = BERCalculator(MODULATION)
    rng     = np.random.default_rng(SEED)
    n_mc    = 100_000

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for idx, ch in enumerate(CHANNELS):
        avg_ber = np.zeros_like(snr_lin)
        for i, s in enumerate(snr_lin):
            gains     = ch.sample_gain(n_mc, rng)
            inst_snr  = s * gains
            inst_ber  = calc.compute_ber(inst_snr)
            avg_ber[i] = np.mean(inst_ber)

        ax.semilogy(snr_db, avg_ber, label=ch.name, linewidth=2.2,
                    marker=CH_MARKERS[idx], color=CH_COLORS[idx],
                    markersize=5, markevery=8)

    ax.set_ylim([1e-6, 1])
    ax.set_xlim([0, 30])
    ax.set_xlabel("Average SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(f"BER vs SNR under Fading Channels ({MODULATION})")
    ax.legend(loc="lower left", frameon=True, shadow=True, borderpad=0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="both", ls="--", alpha=0.4)

    path = os.path.join(output_dir, "ber_vs_snr_fading.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")
    return path


# =====================================================================
# FIGURE 2: Packet Success Probability vs Distance
# =====================================================================
def plot_psucc_vs_distance(output_dir):
    print("[2/5] Generating P_succ vs Distance heatmap ...")
    distances = np.linspace(10, 400, 200)
    calc = BERCalculator(MODULATION)
    rng  = np.random.default_rng(SEED)
    n_mc = 10_000

    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, ch in enumerate(CHANNELS):
        p_avg = np.zeros_like(distances)
        for i, d in enumerate(distances):
            d_arr   = np.full(n_mc, d)
            p_succ  = compute_fading_success_prob(
                d_arr, ch, calc, TX_POWER_DBM, NOISE_POWER_DBM, PAYLOAD_BITS, rng)
            p_avg[i] = np.mean(p_succ)

        ax.plot(distances, p_avg, label=ch.name, linewidth=2.2,
                color=CH_COLORS[idx], linestyle="-", marker=CH_MARKERS[idx],
                markersize=4, markevery=15)

    ax.axvline(x=COMM_RANGE, color='gray', linestyle=':', linewidth=1.5,
               label=f"Comm Range ({COMM_RANGE} m)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Average Packet Success Probability")
    ax.set_title(f"Packet Success vs Distance — {MODULATION}, Tx={TX_POWER_DBM} dBm")
    ax.legend(loc="upper right", frameon=True, shadow=True)
    ax.set_xlim([10, 400])
    ax.set_ylim([-0.02, 1.05])

    path = os.path.join(output_dir, "psucc_vs_distance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")
    return path


# =====================================================================
# FIGURES 3-5: MAC Throughput & Delay under Fading
# =====================================================================
def run_mac_fading_sweep(output_dir):
    print("[3/5] Running MAC fading sweep ...")

    traffic_pps_list = np.linspace(
        global_cfg.SWEEP_MIN_PPS, global_cfg.SWEEP_MAX_PPS, global_cfg.SWEEP_STEPS
    ).astype(int)

    # Fixed distance to isolate fading effect.
    # Use 100m (well within AWGN link budget) so all channels have
    # meaningful throughput and fading degradation is clearly visible.
    TEST_DISTANCE_M = 100.0
    distances        = np.full(N, TEST_DISTANCE_M)
    link_up_schedule = np.ones((N, 1), dtype=int)

    calc = BERCalculator(MODULATION)
    rng  = np.random.default_rng(SEED)

    results = []

    for ch_idx, ch in enumerate(CHANNELS):
        print(f"  Channel: {ch.name}")

        p_succ_sample = compute_fading_success_prob(
            distances, ch, calc, TX_POWER_DBM, NOISE_POWER_DBM, PAYLOAD_BITS, rng)
        avg_succ = float(np.mean(p_succ_sample))
        print(f"    Avg Link P_succ = {avg_succ:.4f}")

        sp_sched = np.tile(p_succ_sample[:, np.newaxis], (1, 1))

        for pps in traffic_pps_list:
            cfg = Config(
                N=N,
                sim_time_s=SIM_TIME_S,
                phy_rate_bps=PHY_RATE_BPS,
                payload_bytes=PAYLOAD_BYTES,
                QMAX=QMAX,
                cw_min=global_cfg.CW_MIN,
                cw_max=global_cfg.CW_MAX,
                difs_slots=global_cfg.DIFS_SLOTS,
                sifs_slots=global_cfg.SIFS_SLOTS,
                ack_slots=global_cfg.ACK_SLOTS,
                ack_timeout_slots=global_cfg.ACK_TIMEOUT_SLOTS,
                max_retry=global_cfg.MAX_RETRY,
                rts_cts_enabled=global_cfg.RTS_CTS_ENABLED,
                ack_enabled=global_cfg.ACK_ENABLED,
                rts_slots=global_cfg.RTS_SLOTS,
                cts_slots=global_cfg.CTS_SLOTS,
                tdma_guard_time_s=global_cfg.TDMA_GUARD_TIME_S,
            )

            # --- TDMA ---
            cfg.seed = SEED + pps
            log_tdma = Logger(load_pps=pps, protocol_name="TDMA")
            simulate_tdma_aware(cfg, pps, log_tdma, link_up_schedule, sp_sched,
                                mobility_dt=global_cfg.MOBILITY_DT)

            # --- CSMA/CA ---
            cfg.seed = SEED + pps
            log_csma = Logger(load_pps=pps, protocol_name="CSMA/CA")
            simulate_csma_aware(cfg, pps, log_csma, link_up_schedule, sp_sched,
                                mobility_dt=global_cfg.MOBILITY_DT)

            results.append({
                "Channel":             ch.name,
                "Offered_Load_pps":    int(pps),
                "Avg_Succ_Prob":       avg_succ,
                "TDMA_Throughput_Mbps": log_tdma.get_throughput_bps(SIM_TIME_S) / 1e6,
                "CSMA_Throughput_Mbps": log_csma.get_throughput_bps(SIM_TIME_S) / 1e6,
                "TDMA_Delay_ms":       log_tdma.get_avg_end_to_end_delay_s() * 1000,
                "CSMA_Delay_ms":       log_csma.get_avg_end_to_end_delay_s() * 1000,
                "TDMA_Drops":          log_tdma.pkts_dropped_qfull + log_tdma.pkts_dropped_mac,
                "CSMA_Drops":          log_csma.pkts_dropped_qfull + log_csma.pkts_dropped_mac,
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "fading_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV -> {csv_path}")

    # ── Figure 3: TDMA throughput ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ch in enumerate(CHANNELS):
        sub = df[df["Channel"] == ch.name]
        ax.plot(sub["Offered_Load_pps"], sub["TDMA_Throughput_Mbps"],
                label=ch.name, marker=CH_MARKERS[i], color=CH_COLORS[i],
                linewidth=2, markersize=5)
    ax.set_xlabel("Offered Traffic Rate (PPS)")
    ax.set_ylabel("TDMA Throughput (Mbps)")
    ax.set_title(f"TDMA Throughput vs Load — N={N}, d={TEST_DISTANCE_M:.0f}m, {MODULATION}, PHY={PHY_RATE_BPS/1e6:.0f} Mbps")
    ax.legend(frameon=True, shadow=True)
    p = os.path.join(output_dir, "tdma_fading_throughput.png")
    fig.savefig(p); plt.close(fig)
    print(f"[4/5]  -> {p}")

    # ── Figure 4: CSMA throughput ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ch in enumerate(CHANNELS):
        sub = df[df["Channel"] == ch.name]
        ax.plot(sub["Offered_Load_pps"], sub["CSMA_Throughput_Mbps"],
                label=ch.name, marker=CH_MARKERS[i], color=CH_COLORS[i],
                linewidth=2, linestyle="--", markersize=5)
    ax.set_xlabel("Offered Traffic Rate (PPS)")
    ax.set_ylabel("CSMA/CA Throughput (Mbps)")
    ax.set_title(f"CSMA/CA Throughput vs Load — N={N}, d={TEST_DISTANCE_M:.0f}m, {MODULATION}, PHY={PHY_RATE_BPS/1e6:.0f} Mbps")
    ax.legend(frameon=True, shadow=True)
    p = os.path.join(output_dir, "csma_fading_throughput.png")
    fig.savefig(p); plt.close(fig)
    print(f"       -> {p}")

    # ── Figure 5: Combined delay ──
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for i, ch in enumerate(CHANNELS):
        sub = df[df["Channel"] == ch.name]
        ax.plot(sub["Offered_Load_pps"], sub["TDMA_Delay_ms"],
                label=f"TDMA ({ch.name})", marker=CH_MARKERS[i],
                color=CH_COLORS[i], linewidth=2, linestyle="-")
        ax.plot(sub["Offered_Load_pps"], sub["CSMA_Delay_ms"],
                label=f"CSMA ({ch.name})", marker=CH_MARKERS[i],
                color=CH_COLORS[i], linewidth=2, linestyle="--")
    ax.set_xlabel("Offered Traffic Rate (PPS)")
    ax.set_ylabel("End-to-End Delay (ms)")
    ax.set_title(f"E2E Delay vs Load — N={N}, d={TEST_DISTANCE_M:.0f}m, {MODULATION}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, shadow=True)
    p = os.path.join(output_dir, "fading_delay_combined.png")
    fig.savefig(p); plt.close(fig)
    print(f"[5/5]  -> {p}")

    return df


# =====================================================================
# Verdict
# =====================================================================
def verdict(df):
    print("\n" + "=" * 64)
    print(" VERDICT — Fading Channel Impact ".center(64, "="))
    print("=" * 64)

    awgn_max = df[df["Channel"] == "AWGN"]["TDMA_Throughput_Mbps"].max()
    ray_max  = df[df["Channel"] == "Rayleigh"]["TDMA_Throughput_Mbps"].max()
    ric_max  = df[df["Channel"] == CH_NAMES[2]]["TDMA_Throughput_Mbps"].max()
    nak_max  = df[df["Channel"] == CH_NAMES[3]]["TDMA_Throughput_Mbps"].max()

    print(f"  AWGN peak TDMA throughput:     {awgn_max:.3f} Mbps")
    print(f"  Rayleigh peak TDMA throughput:  {ray_max:.3f} Mbps")
    print(f"  {CH_NAMES[2]} peak TDMA:        {ric_max:.3f} Mbps")
    print(f"  {CH_NAMES[3]} peak TDMA:        {nak_max:.3f} Mbps")

    # At a moderate distance AWGN should be best, fading channels degrade.
    if ray_max < awgn_max:
        print("  [PASS] Rayleigh fading correctly degrades throughput vs AWGN.")
    else:
        print("  [FAIL] Rayleigh should degrade throughput relative to AWGN.")

    if nak_max > ray_max:
        print(f"  [PASS] {CH_NAMES[3]} mitigates fading better than Rayleigh.")
    else:
        print(f"  [NOTE] {CH_NAMES[3]} did not outperform Rayleigh -- check m value.")

    print("=" * 64)


# =====================================================================
# Main
# =====================================================================
def run_fading_experiment():
    print("=" * 64)
    print(" FADING CHANNEL PERFORMANCE EXPERIMENT ".center(64, "="))
    print("=" * 64)
    print(f"  Scenario: N={N}, T={SIM_TIME_S}s, PHY={PHY_RATE_BPS/1e6:.0f} Mbps")
    print(f"  Modulation: {MODULATION}, Payload: {PAYLOAD_BYTES}B")
    print(f"  Tx={TX_POWER_DBM} dBm, Noise={NOISE_POWER_DBM} dBm")
    print(f"  Nakagami-m={NAKAGAMI_M}, Rician K={RICIAN_K} dB")
    print(f"  Comm Range={COMM_RANGE} m, QMAX={QMAX}")

    out = _output_dir()

    plot_ber_vs_snr(out)
    plot_psucc_vs_distance(out)
    df = run_mac_fading_sweep(out)
    verdict(df)

    print(f"\nAll outputs saved to {out}")


if __name__ == "__main__":
    run_fading_experiment()

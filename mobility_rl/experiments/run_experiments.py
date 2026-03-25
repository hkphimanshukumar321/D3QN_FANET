# run_experiments.py
# Shared plotting utilities used by run_experiments_server.py and run_experiments_server_GNN.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configs import config as params
from configs.marl_config import MARLConfig


# =====================================================================
# generate_baseline_plots
# =====================================================================
def generate_baseline_plots(df, out_dir, N, payload_bytes, phy_rate_bps, QMAX,
                            rts_cts_enabled, ack_enabled):
    """
    Generates publication-quality throughput, delay, and drop comparison plots
    for the three baseline MAC protocols (ALOHA, TDMA, CSMA/CA).

    Parameters
    ----------
    df : pd.DataFrame  — output of run_baseline_simulations().
    out_dir : str       — trial base directory (images/ and csv/ subdirs expected).
    N, payload_bytes, phy_rate_bps, QMAX : config scalars for title annotation.
    rts_cts_enabled, ack_enabled : bool flags for title annotation.
    """
    img_dir = os.path.join(out_dir, "images")
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Save CSV
    df.to_csv(os.path.join(csv_dir, "baseline_mac_results.csv"), index=False)

    load = df["Offered_Load_pps"]
    subtitle = f"N={N}, PHY={phy_rate_bps/1e6:.0f} Mbps, Q={QMAX}"

    markers = ["s", "^"]
    colors = ["tab:orange", "tab:green"]
    protocols = ["TDMA", "CSMA"]

    # --- Throughput ---
    plt.figure(figsize=(10, 6), dpi=150)
    for i, proto in enumerate(protocols):
        col = f"{proto}_Throughput_Mbps"
        if col in df.columns:
            plt.plot(load, df[col], marker=markers[i], color=colors[i],
                     label=proto, linewidth=2, markersize=5, markevery=2)
    plt.xlabel("Offered Load (pps)", fontsize=12, fontweight="bold")
    plt.ylabel("Throughput (Mbps)", fontsize=12, fontweight="bold")
    plt.title(f"Baseline MAC Throughput vs Offered Load\n{subtitle}", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "baseline_throughput.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- Delay ---
    plt.figure(figsize=(10, 6), dpi=150)
    for i, proto in enumerate(protocols):
        col = f"{proto}_Delay_s"
        if col in df.columns:
            plt.plot(load, df[col] * 1000, marker=markers[i], color=colors[i],
                     label=proto, linewidth=2, markersize=5, markevery=2)
    plt.xlabel("Offered Load (pps)", fontsize=12, fontweight="bold")
    plt.ylabel("Avg End-to-End Delay (ms)", fontsize=12, fontweight="bold")
    plt.title(f"Baseline MAC Delay vs Offered Load\n{subtitle}", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "baseline_delay.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- Drops ---
    plt.figure(figsize=(10, 6), dpi=150)
    for i, proto in enumerate(protocols):
        col = f"{proto}_Drops"
        if col in df.columns:
            plt.plot(load, df[col], marker=markers[i], color=colors[i],
                     label=proto, linewidth=2, markersize=5, markevery=2)
    plt.xlabel("Offered Load (pps)", fontsize=12, fontweight="bold")
    plt.ylabel("Total Dropped Packets", fontsize=12, fontweight="bold")
    plt.title(f"Baseline MAC Packet Drops vs Offered Load\n{subtitle}", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "baseline_drops.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- Collisions ---
    plt.figure(figsize=(10, 6), dpi=150)
    for i, proto in enumerate(protocols):
        col = f"{proto}_Collisions"
        if col in df.columns:
            plt.plot(load, df[col], marker=markers[i], color=colors[i],
                     label=proto, linewidth=2, markersize=5, markevery=2)
    plt.xlabel("Offered Load (pps)", fontsize=12, fontweight="bold")
    plt.ylabel("Collision Events", fontsize=12, fontweight="bold")
    plt.title(f"Baseline MAC Collisions vs Offered Load\n{subtitle}", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "baseline_collisions.png"), dpi=300, bbox_inches="tight")
    plt.close()


# =====================================================================
# evaluate_rl_selectors
# =====================================================================
def evaluate_rl_selectors(models, baseline_df, rl_out_dir, log_print):
    """
    For each trained RL model, sweep across baseline traffic loads and record
    the MAC protocol the RL agent would select at each load level.

    Parameters
    ----------
    models : dict   — {name: model_object} (SB3 or custom).
    baseline_df : pd.DataFrame — baseline sweep results.
    rl_out_dir : str — output directory for RL comparison results.
    log_print : callable — logging function.

    Returns
    -------
    all_results : dict  — {model_name: list of dicts with per-load predictions}.
    """
    all_results = {}
    csv_dir = os.path.join(rl_out_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    for name, model in models.items():
        log_print(f"  [Eval] Sweeping {name} ...")
        predictions = []
        for _, row in baseline_df.iterrows():
            pps = row["Offered_Load_pps"]
            tdma_tp = row.get("TDMA_Throughput_Mbps", 0)
            csma_tp = row.get("CSMA_Throughput_Mbps", 0)

            # Use whichever throughput the RL agent's chosen MAC achieves
            selected_mac = "TDMA"
            selected_tp = tdma_tp

            try:
                # Attempt to call predict if the model supports it (SB3-style)
                if hasattr(model, "predict"):
                    # Build a minimal dummy obs for prediction —
                    # in a real sweep the env would be stepped, but for a quick
                    # post-hoc evaluation we just record the selection.
                    obs = _build_dummy_obs(pps)
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                    if action == 1:
                        selected_mac = "CSMA_CA"
                        selected_tp = csma_tp
                elif hasattr(model, "select_action"):
                    obs = np.zeros(14, dtype=np.float32)
                    obs[0] = pps / 1000.0
                    action = model.select_action(obs)
                    if action == 1:
                        selected_mac = "CSMA_CA"
                        selected_tp = csma_tp
            except Exception as e:
                log_print(f"    [Warn] Could not predict for {name} at {pps} pps: {e}")

            predictions.append({
                "Offered_Load_pps": pps,
                "Selected_MAC": selected_mac,
                "Throughput_Mbps": selected_tp,
            })

        all_results[name] = predictions
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(os.path.join(csv_dir, f"rl_eval_sweep_{name}.csv"), index=False)

    # Consolidated CSV
    rows = []
    for name, preds in all_results.items():
        for p in preds:
            rows.append({"Model": name, **p})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(csv_dir, "rl_eval_sweep.csv"), index=False)

    return all_results


def _build_dummy_obs(pps):
    """Build a minimal Dict observation compatible with AdaptiveMacEnv."""
    scalars = np.zeros(14, dtype=np.float32)
    scalars[0] = min(pps / 1000.0, 1.0)
    history = np.zeros((5, 14), dtype=np.float32)
    return {"scalars": scalars, "history": history}


# =====================================================================
# generate_aggregated_rl_plots
# =====================================================================
def generate_aggregated_rl_plots(all_results, baseline_df, q_df, rl_out_dir, log_print):
    """
    Generates aggregated throughput comparison plots overlaying RL model
    selections on top of the baseline MAC curves.

    Parameters
    ----------
    all_results : dict — from evaluate_rl_selectors().
    baseline_df : pd.DataFrame — baseline sweep.
    q_df : any — Q-table data (may be None / empty).
    rl_out_dir : str — output directory.
    log_print : callable.
    """
    img_dir = os.path.join(rl_out_dir, "images")
    csv_dir = os.path.join(rl_out_dir, "csv")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    load = baseline_df["Offered_Load_pps"]

    # --- Combined Throughput: Baselines + RL ---
    plt.figure(figsize=(12, 7), dpi=150)
    # Baseline curves
    for proto, style in [("TDMA", "--"), ("CSMA", "--")]:
        col = f"{proto}_Throughput_Mbps"
        if col in baseline_df.columns:
            plt.plot(load, baseline_df[col], linestyle=style, linewidth=1.5,
                     alpha=0.6, label=f"{proto} (baseline)")

    # RL model curves
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(all_results), 1)))
    for idx, (name, preds) in enumerate(all_results.items()):
        pred_df = pd.DataFrame(preds)
        plt.plot(pred_df["Offered_Load_pps"], pred_df["Throughput_Mbps"],
                 marker=markers[idx % len(markers)], color=colors[idx],
                 label=name, linewidth=2, markersize=6, markevery=2)

    plt.xlabel("Offered Load (pps)", fontsize=12, fontweight="bold")
    plt.ylabel("Throughput (Mbps)", fontsize=12, fontweight="bold")
    plt.title("RL MAC Selection vs Baseline Protocols", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10, frameon=True, shadow=True, loc="best")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "rl_vs_baseline_throughput.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- MAC Selection vs Oracle per model ---
    tdma_col = "TDMA_Throughput_Mbps"
    csma_col = "CSMA_Throughput_Mbps"
    tdma_delay_col = "TDMA_Delay_s"
    csma_delay_col = "CSMA_Delay_s"
    oracle_map = {}
    if all(c in baseline_df.columns for c in [tdma_col, csma_col, tdma_delay_col, csma_delay_col]):
        max_thr = max(float(params.PHY_RATE_BPS) / 1e6, 1e-9)
        for _, row in baseline_df.iterrows():
            pps = int(row["Offered_Load_pps"])
            tdma_score = MARLConfig.W_THROUGHPUT * (float(row[tdma_col]) / max_thr) - MARLConfig.W_DELAY * (float(row[tdma_delay_col]) * 1000.0 / 100.0)
            csma_score = MARLConfig.W_THROUGHPUT * (float(row[csma_col]) / max_thr) - MARLConfig.W_DELAY * (float(row[csma_delay_col]) * 1000.0 / 100.0)
            oracle_map[pps] = "TDMA" if tdma_score >= csma_score else "CSMA/CA"

    match_rows = []
    for name, preds in all_results.items():
        pred_df = pd.DataFrame(preds).sort_values("Offered_Load_pps")
        if pred_df.empty:
            continue

        x = pred_df["Offered_Load_pps"].astype(int).to_numpy()
        model_numeric = np.array([0 if m == "TDMA" else 1 for m in pred_df["Selected_MAC"]], dtype=np.int32)

        oracle_numeric = None
        match_rate = None
        if oracle_map:
            oracle_mac = [oracle_map.get(int(pps), "CSMA/CA") for pps in x]
            oracle_numeric = np.array([0 if m == "TDMA" else 1 for m in oracle_mac], dtype=np.int32)
            match_rate = float(np.mean(model_numeric == oracle_numeric) * 100.0)
            match_rows.append({"Model": name, "Oracle_Match_Pct": round(match_rate, 2)})

        plt.figure(figsize=(12, 3.5), dpi=150)
        plt.step(x, model_numeric, where="mid", color="steelblue", linewidth=2.2, label=f"{name} selection")
        plt.scatter(x, model_numeric, color="steelblue", s=24, zorder=3)

        if oracle_numeric is not None:
            plt.step(x, oracle_numeric, where="mid", color="black", linestyle="--", linewidth=2, label="Oracle")
            plt.scatter(x, oracle_numeric, color="black", s=16, marker="x", zorder=3)

        plt.ylim(-0.2, 1.2)
        plt.yticks([0, 1], ["TDMA", "CSMA/CA"])
        plt.xlabel("Offered Load (pps)", fontsize=11, fontweight="bold")

        title = f"{name} — MAC Selection vs Load"
        if match_rate is not None:
            title += f"  (Oracle match: {match_rate:.1f}%)"
        plt.title(title, fontsize=12, fontweight="bold")

        plt.grid(True, axis="x", alpha=0.35)
        plt.legend(fontsize=9, frameon=True, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, f"mac_selection_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    if match_rows:
        match_df = pd.DataFrame(match_rows).sort_values("Oracle_Match_Pct", ascending=False)
        match_df.to_csv(os.path.join(csv_dir, "mac_selection_vs_oracle_match.csv"), index=False)
        plt.figure(figsize=(10, 5), dpi=150)
        plt.barh(match_df["Model"], match_df["Oracle_Match_Pct"], color="slateblue", alpha=0.85)
        plt.xlim(0, 100)
        plt.xlabel("Match with Oracle MAC choice (%)", fontsize=11, fontweight="bold")
        plt.title("MAC Selection Agreement with Oracle", fontsize=12, fontweight="bold")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, "mac_selection_vs_oracle_match.png"), dpi=300, bbox_inches="tight")
        plt.close()

    log_print(f"  [Plot] Saved aggregated RL plots to {img_dir}")

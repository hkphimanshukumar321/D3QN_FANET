#!/usr/bin/env python3
"""
IEEE-Quality Pareto, Weight Sensitivity & Hyperparameter Importance Figures
===========================================================================
Reads all_trials_summary.csv and all_pareto_fronts.csv from results_combined/csv/
and generates publication-quality figures for the paper.

Output: alternate_tj_latex_template_ap/figures/
"""
import os, sys, shutil
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.75",
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_DIR = os.path.join(ROOT, "results", "results_combined", "csv")
PLOT_DIR = os.path.join(ROOT, "results", "results_combined", "plots")
FIG_DIR = os.path.join(ROOT, "alternate_tj_latex_template_ap", "figures")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

ALGO_STYLE = {
    "MAGAT-D3QN": {"marker": "D", "color": "#D62728", "ms": 18},
    "QMIX":       {"marker": "s", "color": "#1F77B4", "ms": 14},
    "VDN":        {"marker": "^", "color": "#2CA02C", "ms": 16},
    "IQL":        {"marker": "v", "color": "#9467BD", "ms": 16},
    "MAPPO":      {"marker": "o", "color": "#FF7F0E", "ms": 14},
}
ORDER = ["MAGAT-D3QN", "QMIX", "VDN", "IQL", "MAPPO"]

DCOL = 7.16  # IEEE double-column width in inches


def save(fig, name):
    for d in [PLOT_DIR, FIG_DIR]:
        fig.savefig(os.path.join(d, name + ".pdf"))
        fig.savefig(os.path.join(d, name + ".png"))
    plt.close(fig)
    print(f"  Saved {name}.pdf/png")


def legend_handles():
    return [
        Line2D([0], [0], marker=ALGO_STYLE[a]["marker"], color="w",
               markerfacecolor=ALGO_STYLE[a]["color"],
               markersize=6, markeredgecolor="black", markeredgewidth=0.3,
               label=a)
        for a in ORDER
    ]


# ═══════════════════════════════════════════════════════════════════
# Fig A: Reward Weight Sensitivity (4×4 grid)
# ═══════════════════════════════════════════════════════════════════
def fig_reward_weight_sensitivity(df):
    w_cols = ["wT", "wD", "wDrop", "wCol"]
    w_labels = [r"$w_T$", r"$w_D$", r"$w_{\mathrm{drop}}$", r"$w_{\mathrm{col}}$"]
    obj_cols = ["obj_throughput", "obj_delay", "obj_drops", "obj_collisions"]
    obj_labels = ["Throughput (Mbps)", "Delay (ms)", "Packet Drops", "Collisions"]

    fig, axes = plt.subplots(4, 4, figsize=(DCOL, DCOL * 0.85))

    for i, (wc, wl) in enumerate(zip(w_cols, w_labels)):
        for j, (oc, ol) in enumerate(zip(obj_cols, obj_labels)):
            ax = axes[i][j]
            for algo in ORDER:
                s = ALGO_STYLE[algo]
                adf = df[(df["Algorithm"] == algo) & df[wc].notna() & df[oc].notna()]
                if adf.empty:
                    continue
                ax.scatter(adf[wc], adf[oc],
                           marker=s["marker"], color=s["color"],
                           s=s["ms"], alpha=0.55, edgecolors="none", zorder=3)
            ax.set_xlabel(wl, fontsize=7)
            if j == 0:
                ax.set_ylabel(ol, fontsize=6.5)
            else:
                ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=5.5)

            # Top row: add objective name as column header
            if i == 0:
                ax.set_title(ol.split("(")[0].strip(), fontsize=7, fontweight="bold")

    fig.legend(handles=legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 1.01), ncol=5, fontsize=6.5,
               columnspacing=0.8, handletextpad=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.4, w_pad=0.3)
    save(fig, "optuna_reward_weight_sensitivity")


# ═══════════════════════════════════════════════════════════════════
# Fig B: Hyperparameter Importance (2×2)
# ═══════════════════════════════════════════════════════════════════
def fig_hyperparameter_importance(df):
    hp_cols = ["param_lr", "param_batch_size", "param_hidden_dim", "param_epsilon_decay"]
    hp_labels = ["Learning Rate", "Batch Size", "Hidden Dim", r"$\epsilon$-Decay Steps"]
    log_x = [True, False, False, False]

    fig, axes = plt.subplots(2, 2, figsize=(DCOL, DCOL * 0.55))

    for ax, col, label, logx in zip(axes.flatten(), hp_cols, hp_labels, log_x):
        for algo in ORDER:
            s = ALGO_STYLE[algo]
            adf = df[(df["Algorithm"] == algo) & df[col].notna() & df["obj_throughput"].notna()]
            if adf.empty:
                continue
            ax.scatter(adf[col], adf["obj_throughput"],
                       marker=s["marker"], color=s["color"],
                       s=s["ms"], alpha=0.55, edgecolors="none", zorder=3)
        ax.set_xlabel(label, fontsize=7.5)
        ax.set_ylabel("Throughput (Mbps)", fontsize=7.5)
        if logx:
            ax.set_xscale("log")
        ax.tick_params(axis="both", labelsize=6)

    fig.legend(handles=legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=6.5,
               columnspacing=0.8, handletextpad=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.94], h_pad=0.5, w_pad=0.5)
    save(fig, "optuna_hyperparameter_importance")


# ═══════════════════════════════════════════════════════════════════
# Fig C: Pareto Front Overlay (1×3 panel — for the paper)
# ═══════════════════════════════════════════════════════════════════
def fig_pareto_overlay(pareto_df):
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.2))
    pairs = [
        ("obj_delay", "obj_throughput", "Delay (ms)", "Throughput (Mbps)"),
        ("obj_drops", "obj_throughput", "Packet Drops", "Throughput (Mbps)"),
        ("obj_collisions", "obj_throughput", "Collisions", "Throughput (Mbps)"),
    ]

    for ax, (xc, yc, xl, yl) in zip(axes, pairs):
        for algo in ORDER:
            s = ALGO_STYLE[algo]
            pdf = pareto_df[pareto_df["Algorithm"] == algo].dropna(subset=[xc, yc])
            if pdf.empty:
                continue
            pdf_s = pdf.sort_values(xc)
            ax.scatter(pdf_s[xc], pdf_s[yc],
                       marker=s["marker"], color=s["color"],
                       s=28, edgecolors="black", linewidth=0.3,
                       alpha=0.85, zorder=5)
            if len(pdf_s) > 1:
                ax.plot(pdf_s[xc], pdf_s[yc],
                        "--", color=s["color"], alpha=0.45, linewidth=0.8, zorder=4)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    axes[0].legend(handles=legend_handles(), fontsize=5.5, loc="lower left",
                   handletextpad=0.15, borderpad=0.3)
    plt.tight_layout(w_pad=0.6)
    save(fig, "results_pareto_front")


# ═══════════════════════════════════════════════════════════════════
# Fig D: Combined 2×3 Pareto Analysis (full-page)
# ═══════════════════════════════════════════════════════════════════
def fig_combined_analysis(df, pareto_df, rep_df):
    fig, axes = plt.subplots(2, 3, figsize=(DCOL, 4.8))

    balanced = rep_df[rep_df["representative_label"] == "balanced_min_L2_to_ideal"]

    # (a) Throughput vs Delay with ★ balanced
    ax = axes[0, 0]
    for algo in ORDER:
        s = ALGO_STYLE[algo]
        pdf = pareto_df[pareto_df["Algorithm"] == algo].dropna(subset=["obj_delay", "obj_throughput"])
        if pdf.empty:
            continue
        pdf_s = pdf.sort_values("obj_delay")
        ax.scatter(pdf_s["obj_delay"], pdf_s["obj_throughput"],
                   marker=s["marker"], color=s["color"],
                   s=22, edgecolors="black", linewidth=0.25, alpha=0.8, zorder=5)
        if len(pdf_s) > 1:
            ax.plot(pdf_s["obj_delay"], pdf_s["obj_throughput"],
                    "--", color=s["color"], alpha=0.4, linewidth=0.7)
    for _, r in balanced.iterrows():
        if r["Algorithm"] in ALGO_STYLE:
            ax.scatter(r["obj_delay"], r["obj_throughput"], marker="*",
                       color=ALGO_STYLE[r["Algorithm"]]["color"],
                       s=110, edgecolors="black", linewidth=0.6, zorder=10)
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("(a) Throughput vs. Delay", fontsize=8, fontweight="bold")

    # (b) Throughput vs Drops
    ax = axes[0, 1]
    for algo in ORDER:
        s = ALGO_STYLE[algo]
        pdf = pareto_df[pareto_df["Algorithm"] == algo].dropna(subset=["obj_drops", "obj_throughput"])
        if pdf.empty:
            continue
        ax.scatter(pdf["obj_drops"], pdf["obj_throughput"],
                   marker=s["marker"], color=s["color"],
                   s=22, edgecolors="black", linewidth=0.25, alpha=0.8, zorder=5)
    ax.set_xlabel("Packet Drops")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("(b) Throughput vs. Drops", fontsize=8, fontweight="bold")

    # (c) Throughput vs Collisions
    ax = axes[0, 2]
    for algo in ORDER:
        s = ALGO_STYLE[algo]
        pdf = pareto_df[pareto_df["Algorithm"] == algo].dropna(subset=["obj_collisions", "obj_throughput"])
        if pdf.empty:
            continue
        ax.scatter(pdf["obj_collisions"], pdf["obj_throughput"],
                   marker=s["marker"], color=s["color"],
                   s=22, edgecolors="black", linewidth=0.25, alpha=0.8, zorder=5)
    ax.set_xlabel("Collisions")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("(c) Throughput vs. Collisions", fontsize=8, fontweight="bold")

    # (d) wT vs wD scatter (each dot = one trial, shows weight diversity)
    ax = axes[1, 0]
    for algo in ORDER:
        s = ALGO_STYLE[algo]
        adf = df[(df["Algorithm"] == algo) & df["wT"].notna() & df["wD"].notna()]
        if adf.empty:
            continue
        ax.scatter(adf["wT"], adf["wD"],
                   marker=s["marker"], color=s["color"],
                   s=s["ms"], alpha=0.5, edgecolors="none", zorder=3)
    # Mark balanced
    for _, r in balanced.iterrows():
        if r["Algorithm"] in ALGO_STYLE:
            ax.scatter(r["wT"], r["wD"], marker="*",
                       color=ALGO_STYLE[r["Algorithm"]]["color"],
                       s=110, edgecolors="black", linewidth=0.6, zorder=10)
    ax.set_xlabel(r"$w_T$ (throughput weight)")
    ax.set_ylabel(r"$w_D$ (delay weight)")
    ax.set_title(r"(d) Weight space: $w_T$ vs $w_D$", fontsize=8, fontweight="bold")

    # (e) Balanced-best throughput bars
    ax = axes[1, 1]
    b_algos, b_thr, b_colors = [], [], []
    for algo in ORDER:
        row = balanced[balanced["Algorithm"] == algo]
        if row.empty:
            continue
        b_algos.append(algo)
        b_thr.append(float(row.iloc[0]["obj_throughput"]))
        b_colors.append(ALGO_STYLE[algo]["color"])
    if b_algos:
        bars = ax.bar(range(len(b_algos)), b_thr, color=b_colors,
                      edgecolor="black", linewidth=0.3, width=0.65)
        ax.set_xticks(range(len(b_algos)))
        ax.set_xticklabels(b_algos, rotation=25, ha="right", fontsize=6)
        ax.set_ylabel("Throughput (Mbps)")
        ax.set_ylim(0, max(b_thr) * 1.2)
        for bar, v in zip(bars, b_thr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax.set_title("(e) Best selected throughput", fontsize=8, fontweight="bold")

    # (f) Balanced-best delay bars
    ax = axes[1, 2]
    b_del = []
    for algo in ORDER:
        row = balanced[balanced["Algorithm"] == algo]
        if row.empty:
            continue
        b_del.append(float(row.iloc[0]["obj_delay"]))
    if b_algos and b_del:
        bars = ax.bar(range(len(b_algos)), b_del, color=b_colors,
                      edgecolor="black", linewidth=0.3, width=0.65)
        ax.set_xticks(range(len(b_algos)))
        ax.set_xticklabels(b_algos, rotation=25, ha="right", fontsize=6)
        ax.set_ylabel("Delay (ms)")
        ax.set_ylim(0, max(b_del) * 1.2)
        for bar, v in zip(bars, b_del):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax.set_title("(f) Best selected delay", fontsize=8, fontweight="bold")

    # Legend
    handles = legend_handles()
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                          markersize=9, markeredgecolor="black", label="Best selected"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=6, fontsize=6.5,
               columnspacing=0.7, handletextpad=0.15)
    plt.tight_layout(h_pad=0.7, w_pad=0.5)
    save(fig, "combined_pareto_analysis")


# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  IEEE Pareto + Sensitivity + Importance Figures")
    print("=" * 60)

    trials_path = os.path.join(CSV_DIR, "all_trials_summary.csv")
    pareto_path = os.path.join(CSV_DIR, "all_pareto_fronts.csv")
    rep_path    = os.path.join(CSV_DIR, "all_representative_pareto.csv")

    df = pd.read_csv(trials_path)
    df = df[df["state"] == "COMPLETE"].copy()
    pareto_df = pd.read_csv(pareto_path)
    rep_df = pd.read_csv(rep_path)

    print(f"  {len(df)} complete trials, {len(pareto_df)} Pareto, {len(rep_df)} representative")
    print(f"  Algorithms: {sorted(df['Algorithm'].unique())}")

    print("\nGenerating figures...")
    fig_reward_weight_sensitivity(df)
    fig_hyperparameter_importance(df)
    fig_pareto_overlay(pareto_df)
    fig_combined_analysis(df, pareto_df, rep_df)

    print(f"\nDone. Figures in:\n  {PLOT_DIR}\n  {FIG_DIR}")


if __name__ == "__main__":
    main()

"""
Combine multi-server MARL results into a unified journal package.
Merges Optuna studies, training rewards, journal evidence, and generates plots.
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"

# ── Source directories ──────────────────────────────────────────────────────
SOURCES = {
    "MAGAT-D3QN": {
        "optuna": RESULTS / "results_MAGAT" / "optuna" / "journal_tune_magat_d3qn",
        "evidence": sorted(glob.glob(str(RESULTS / "results_MAGAT" / "journal_parallel" / "*")))[-1]
                    if glob.glob(str(RESULTS / "results_MAGAT" / "journal_parallel" / "*")) else None,
        "checkpoint": RESULTS / "results_MAGAT" / "N50_PHY5M_Q100_RTSCTS_ACK_enabled" /
                      "trial_UNIFIED_20260421_135739_pid606" / "checkpoints",
        "reward_file": "magat_d3qn_training_rewards.csv",
        "model_file": "unified_gnn_marl_model.pth",
    },
    "MAPPO": {
        "optuna": RESULTS / "results_MAPPO_VDN" / "optuna" / "journal_tune_mappo",
        "evidence": RESULTS / "results_MAPPO_VDN" / "journal_evidence_20260422_135559",
        "checkpoint": RESULTS / "results_MAPPO_VDN" / "journal_best_checkpoints_20260422_135559",
        "reward_file": "mappo_training_rewards.csv",
        "model_file": "unified_mappo_model.pth",
    },
    "VDN": {
        "optuna": RESULTS / "results_MAPPO_VDN" / "optuna" / "journal_tune_vdn",
        "evidence": RESULTS / "results_MAPPO_VDN" / "journal_evidence_20260422_135559",
        "checkpoint": RESULTS / "results_MAPPO_VDN" / "journal_best_checkpoints_20260422_135559",
        "reward_file": "vdn_training_rewards.csv",
        "model_file": "unified_vdn_model.pth",
    },
    "IQL": {
        "optuna": RESULTS / "results_iql_qmix" / "optuna" / "journal_tune_iql",
        "evidence": sorted(glob.glob(str(RESULTS / "results_iql_qmix" / "journal_parallel" / "*")))[-1]
                    if glob.glob(str(RESULTS / "results_iql_qmix" / "journal_parallel" / "*")) else None,
        "checkpoint": RESULTS / "results_iql_qmix" / "optuna_then_pipeline" /
                      "20260418_140806" / "final_combined" / "checkpoints",
        "reward_file": "iql_training_rewards.csv",
        "model_file": "unified_iql_model.pth",
    },
    "QMIX": {
        "optuna": RESULTS / "results_iql_qmix" / "optuna" / "journal_tune_qmix",
        "evidence": sorted(glob.glob(str(RESULTS / "results_iql_qmix" / "journal_parallel" / "*")))[-1]
                    if glob.glob(str(RESULTS / "results_iql_qmix" / "journal_parallel" / "*")) else None,
        "checkpoint": RESULTS / "results_iql_qmix" / "optuna_then_pipeline" /
                      "20260418_140806" / "final_combined" / "checkpoints",
        "reward_file": "qmix_training_rewards.csv",
        "model_file": "unified_qmix_model.pth",
    },
}

ALGO_COLORS = {
    "MAGAT-D3QN": "#E63946", "MAPPO": "#457B9D", "VDN": "#2A9D8F",
    "IQL": "#E9C46A", "QMIX": "#F4A261",
}
ALGO_ORDER = ["MAGAT-D3QN", "MAPPO", "VDN", "QMIX", "IQL"]


def log(msg): print(f"[combine] {msg}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Optuna data
# ═══════════════════════════════════════════════════════════════════════════
def collect_optuna(out_csv):
    frames_trials, frames_pareto, frames_repr = [], [], []
    for algo, src in SOURCES.items():
        odir = Path(src["optuna"])
        for name, container in [("trials_summary.csv", frames_trials),
                                ("pareto_front.csv", frames_pareto),
                                ("representative_pareto_points.csv", frames_repr)]:
            p = odir / name
            if p.exists():
                df = pd.read_csv(p)
                df.insert(0, "Algorithm", algo)
                container.append(df)
                log(f"  {algo}/{name}: {len(df)} rows")
    for fname, frames in [("all_trials_summary.csv", frames_trials),
                           ("all_pareto_fronts.csv", frames_pareto),
                           ("all_representative_pareto.csv", frames_repr)]:
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_csv / fname, index=False)
            log(f"  Wrote {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Training rewards
# ═══════════════════════════════════════════════════════════════════════════
def collect_training_rewards(out_csv):
    frames = []
    for algo, src in SOURCES.items():
        cp = Path(src["checkpoint"])
        rf = cp / src["reward_file"]
        if rf.exists():
            df = pd.read_csv(rf)
            df.insert(0, "Algorithm", algo)
            frames.append(df)
            log(f"  {algo} rewards: {len(df)} episodes")
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_csv / "all_training_rewards.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Copy checkpoints
# ═══════════════════════════════════════════════════════════════════════════
def copy_checkpoints(out_ckpt):
    for algo, src in SOURCES.items():
        cp = Path(src["checkpoint"]) / src["model_file"]
        if cp.exists():
            shutil.copy2(cp, out_ckpt / src["model_file"])
            log(f"  Copied {algo}: {src['model_file']}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Journal evidence
# ═══════════════════════════════════════════════════════════════════════════
def _find_evidence_csvs(evidence_dir, pattern):
    if evidence_dir is None:
        return []
    return sorted(glob.glob(os.path.join(str(evidence_dir), pattern), recursive=True))


def collect_evidence(out_csv):
    suites = {
        "generalization_master.csv": "generalization/**/csv/generalization_suite_summary.csv",
        "robustness_master.csv": "robustness/**/csv/robustness_suite_summary.csv",
        "complexity_master.csv": "complexity/csv/complexity_benchmark.csv",
    }
    for out_name, pattern in suites.items():
        frames = []
        for algo, src in SOURCES.items():
            ev = src["evidence"]
            for p in _find_evidence_csvs(ev, pattern):
                df = pd.read_csv(p)
                if "Algorithm" not in df.columns:
                    df.insert(0, "Algorithm", algo)
                frames.append(df)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(out_csv / out_name, index=False)
            log(f"  {out_name}: {len(combined)} rows, algos={list(combined.get('Algorithm', combined.get('Model', pd.Series())).unique())}")

    # Failure recovery: try suite_summary first, fall back to individual *_eval.csv
    fail_frames = []
    for algo, src in SOURCES.items():
        ev = src["evidence"]
        # Try suite summary first
        summaries = _find_evidence_csvs(ev, "failure_recovery/csv/failure_recovery_suite_summary.csv")
        if summaries:
            for p in summaries:
                df = pd.read_csv(p)
                if "Algorithm" not in df.columns:
                    df.insert(0, "Algorithm", algo)
                fail_frames.append(df)
        else:
            # Fall back to individual eval CSVs
            evals = _find_evidence_csvs(ev, "failure_recovery/csv/failure_recovery_*_eval.csv")
            for p in evals:
                df = pd.read_csv(p)
                if "Algorithm" not in df.columns:
                    df.insert(0, "Algorithm", algo)
                # Extract scenario name from filename
                fname = os.path.basename(p)  # failure_recovery_<scenario>_eval.csv
                scenario = fname.replace("failure_recovery_", "").replace("_eval.csv", "")
                if "Scenario" not in df.columns:
                    df["Scenario"] = scenario
                fail_frames.append(df)
    if fail_frames:
        combined = pd.concat(fail_frames, ignore_index=True)
        combined.to_csv(out_csv / "failure_recovery_master.csv", index=False)
        model_col = "Model" if "Model" in combined.columns else "Algorithm"
        log(f"  failure_recovery_master.csv: {len(combined)} rows, algos={list(combined[model_col].unique())}")

    # Model complexity stats
    frames = []
    for algo, src in SOURCES.items():
        ev = src["evidence"]
        for p in _find_evidence_csvs(ev, "complexity/csv/model_complexity_stats.csv"):
            df = pd.read_csv(p)
            if "Algorithm" not in df.columns:
                df.insert(0, "Algorithm", algo)
            frames.append(df)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_csv / "model_complexity_stats.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Final comparison table
# ═══════════════════════════════════════════════════════════════════════════
def build_comparison_table(out_csv):
    pareto_path = out_csv / "all_representative_pareto.csv"
    if not pareto_path.exists():
        pareto_path = out_csv / "all_pareto_fronts.csv"
    if not pareto_path.exists():
        return
    df = pd.read_csv(pareto_path)
    rows = []
    for algo in ALGO_ORDER:
        sub = df[df["Algorithm"] == algo]
        if sub.empty:
            continue
        best = sub.sort_values("obj_throughput", ascending=False).iloc[0]
        rows.append({
            "Algorithm": algo,
            "Best_Throughput_Mbps": round(best.get("obj_throughput", 0), 4),
            "Best_Delay_ms": round(best.get("obj_delay", 0), 4),
            "Best_Drops": best.get("obj_drops", 0),
            "Best_Collisions": best.get("obj_collisions", 0),
            "Hidden_Dim": best.get("param_hidden_dim", ""),
            "LR": best.get("param_lr", ""),
            "Batch_Size": best.get("param_batch_size", ""),
        })
    pd.DataFrame(rows).to_csv(out_csv / "final_comparison_table.csv", index=False)
    log(f"  final_comparison_table: {len(rows)} algorithms")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Plots
# ═══════════════════════════════════════════════════════════════════════════
def generate_plots(out_csv, out_plots):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                             "figure.dpi": 150, "savefig.bbox": "tight"})
    except ImportError:
        log("  matplotlib not available, skipping plots")
        return

    # --- Plot 1: Combined Pareto Front (2x2) ---
    pareto_file = out_csv / "all_pareto_fronts.csv"
    if pareto_file.exists():
        df = pd.read_csv(pareto_file)
        pairs = [("obj_throughput", "obj_delay", "Throughput (Mbps)", "Delay (ms)"),
                 ("obj_throughput", "obj_drops", "Throughput (Mbps)", "Drops"),
                 ("obj_throughput", "obj_collisions", "Throughput (Mbps)", "Collisions"),
                 ("obj_delay", "obj_drops", "Delay (ms)", "Drops")]
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle("Combined Pareto Fronts — All Algorithms", fontsize=15, fontweight="bold", y=0.98)
        for ax, (x, y, xl, yl) in zip(axes.flat, pairs):
            for algo in ALGO_ORDER:
                sub = df[df["Algorithm"] == algo]
                if sub.empty or x not in sub or y not in sub:
                    continue
                ax.scatter(sub[x], sub[y], label=algo, color=ALGO_COLORS[algo],
                           alpha=0.7, s=50, edgecolors="white", linewidth=0.5)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_plots / "combined_pareto_front.png")
        plt.close(fig)
        log("  Wrote combined_pareto_front.png")

    # --- Plot 2: Algorithm Comparison Bar ---
    comp_file = out_csv / "final_comparison_table.csv"
    if comp_file.exists():
        df = pd.read_csv(comp_file)
        metrics = ["Best_Throughput_Mbps", "Best_Delay_ms", "Best_Drops", "Best_Collisions"]
        labels = ["Throughput\n(Mbps) ↑", "Delay\n(ms) ↓", "Drops ↓", "Collisions ↓"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle("Best Trial Comparison Across Algorithms", fontsize=14, fontweight="bold")
        for ax, m, lbl in zip(axes, metrics, labels):
            vals = df.set_index("Algorithm").reindex(ALGO_ORDER)[m].fillna(0)
            colors = [ALGO_COLORS.get(a, "#999") for a in vals.index]
            bars = ax.bar(range(len(vals)), vals.values, color=colors, edgecolor="white")
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(vals.index, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(lbl)
            ax.grid(axis="y", alpha=0.3)
            for bar, v in zip(bars, vals.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{v:.2f}" if isinstance(v, float) else str(int(v)),
                        ha="center", va="bottom", fontsize=7)
        plt.tight_layout()
        fig.savefig(out_plots / "algorithm_comparison_bar.png")
        plt.close(fig)
        log("  Wrote algorithm_comparison_bar.png")

    # --- Plot 3: Training Reward Curves ---
    reward_file = out_csv / "all_training_rewards.csv"
    if reward_file.exists():
        df = pd.read_csv(reward_file)
        fig, ax = plt.subplots(figsize=(12, 6))
        for algo in ALGO_ORDER:
            sub = df[df["Algorithm"] == algo].sort_values("episode")
            if sub.empty:
                continue
            window = max(1, len(sub) // 50)
            smoothed = sub["reward"].rolling(window, min_periods=1).mean()
            ax.plot(sub["episode"], smoothed, label=algo, color=ALGO_COLORS[algo], linewidth=1.5)
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward (smoothed)")
        ax.set_title("Training Reward Curves — All Algorithms", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(out_plots / "training_reward_curves.png")
        plt.close(fig)
        log("  Wrote training_reward_curves.png")

    # --- Plot 4: Generalization Heatmap ---
    gen_file = out_csv / "generalization_master.csv"
    if gen_file.exists():
        df = pd.read_csv(gen_file)
        model_col = "Model" if "Model" in df.columns else "Algorithm"
        if "Scenario" in df.columns and "Throughput_Mbps" in df.columns:
            pivot = df.pivot_table(values="Throughput_Mbps", index=model_col,
                                   columns="Scenario", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)*1.2), max(4, len(pivot)*0.8)))
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    v = pivot.values[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, label="Throughput (Mbps)")
            ax.set_title("Generalization: Mean Throughput by Scenario", fontweight="bold")
            plt.tight_layout()
            fig.savefig(out_plots / "generalization_heatmap.png")
            plt.close(fig)
            log("  Wrote generalization_heatmap.png")

    # --- Plot 5: Robustness Degradation ---
    rob_file = out_csv / "robustness_master.csv"
    if rob_file.exists():
        df = pd.read_csv(rob_file)
        model_col = "Model" if "Model" in df.columns else "Algorithm"
        if "Scenario" in df.columns and "Throughput_Mbps" in df.columns:
            pivot = df.pivot_table(values="Throughput_Mbps", index=model_col,
                                   columns="Scenario", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(pivot.columns))
            w = 0.8 / max(len(pivot.index), 1)
            for i, model in enumerate(pivot.index):
                colors = ALGO_COLORS.get(model, "#999")
                ax.bar(x + i*w, pivot.loc[model].values, w, label=model, color=colors, edgecolor="white")
            ax.set_xticks(x + w * len(pivot.index) / 2)
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Throughput (Mbps)")
            ax.set_title("Robustness: Performance Under Perturbation", fontweight="bold")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            fig.savefig(out_plots / "robustness_degradation.png")
            plt.close(fig)
            log("  Wrote robustness_degradation.png")

    # --- Plot 6: Failure Recovery ---
    fail_file = out_csv / "failure_recovery_master.csv"
    if fail_file.exists():
        df = pd.read_csv(fail_file)
        model_col = "Model" if "Model" in df.columns else "Algorithm"
        if "Scenario" in df.columns and "Throughput_Mbps" in df.columns:
            scenarios = df["Scenario"].unique() if "Scenario" in df.columns else df["Type"].unique()
            n = len(scenarios)
            fig, axes = plt.subplots(1, min(n, 5), figsize=(min(n, 5)*4, 4), squeeze=False)
            scen_col = "Scenario" if "Scenario" in df.columns else "Type"
            for ax, sc in zip(axes.flat, scenarios):
                sub = df[df[scen_col] == sc]
                for model in sub[model_col].unique():
                    ms = sub[sub[model_col] == model].sort_values("Offered_Load_pps")
                    ax.plot(ms["Offered_Load_pps"], ms["Throughput_Mbps"],
                            label=model, color=ALGO_COLORS.get(model, "#999"), marker="o", ms=3)
                ax.set_title(sc.replace("_", " ").title(), fontsize=9)
                ax.set_xlabel("Load (pps)"); ax.set_ylabel("Throughput")
                ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
            fig.suptitle("Failure Recovery Performance", fontweight="bold")
            plt.tight_layout()
            fig.savefig(out_plots / "failure_recovery_comparison.png")
            plt.close(fig)
            log("  Wrote failure_recovery_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Combine multi-server MARL results")
    parser.add_argument("--out-dir", default=str(RESULTS / "results_combined"))
    args = parser.parse_args()

    out = Path(args.out_dir)
    out_csv = out / "csv"
    out_plots = out / "plots"
    out_ckpt = out / "checkpoints"
    out_json = out / "json"
    for d in [out_csv, out_plots, out_ckpt, out_json]:
        d.mkdir(parents=True, exist_ok=True)

    log("=== Step 1: Optuna Studies ===")
    collect_optuna(out_csv)

    log("=== Step 2: Training Rewards ===")
    collect_training_rewards(out_csv)

    log("=== Step 3: Checkpoints ===")
    copy_checkpoints(out_ckpt)

    log("=== Step 4: Journal Evidence ===")
    collect_evidence(out_csv)

    log("=== Step 5: Comparison Table ===")
    build_comparison_table(out_csv)

    log("=== Step 6: Plots ===")
    generate_plots(out_csv, out_plots)

    # Manifest
    manifest = {
        "sources": {k: {sk: str(sv) for sk, sv in v.items()} for k, v in SOURCES.items()},
        "output_dir": str(out),
        "files": [str(p.relative_to(out)) for p in sorted(out.rglob("*")) if p.is_file()],
    }
    with open(out_json / "combination_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"\n{'='*60}")
    log(f"Combined package: {out}")
    csv_count = len(list(out_csv.glob("*.csv")))
    plot_count = len(list(out_plots.glob("*.png")))
    ckpt_count = len(list(out_ckpt.glob("*.pth")))
    log(f"  CSVs: {csv_count}, Plots: {plot_count}, Checkpoints: {ckpt_count}")


if __name__ == "__main__":
    main()

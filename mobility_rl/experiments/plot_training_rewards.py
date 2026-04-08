"""
Training reward plot generator for unified experiment outputs.
"""

from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SARL_FILES = {
    "dqn_training_rewards.csv": "DQN",
    "ppo_training_rewards.csv": "PPO",
    "a2c_training_rewards.csv": "A2C",
    "mca_d3qn_training_rewards.csv": "MCA-D3QN",
    "tabular_training_rewards.csv": "Tabular",
}

MARL_FILES = {
    "iql_training_rewards.csv": "IQL",
    "vdn_training_rewards.csv": "VDN",
    "qmix_training_rewards.csv": "QMIX",
    "magat_d3qn_training_rewards.csv": "MAGAT-D3QN",
}


def _plot_group(file_map: dict[str, str], csv_dir: str, images_dir: str, out_name: str, title: str) -> str | None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False

    for filename, label in file_map.items():
        path = os.path.join(csv_dir, filename)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        x_col = "step" if "step" in df.columns else "episode" if "episode" in df.columns else df.columns[0]
        y_col = "reward" if "reward" in df.columns else df.columns[-1]
        ax.plot(df[x_col], df[y_col], label=label, linewidth=2)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Step" if "steps" in out_name.lower() else "Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(images_dir, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_plots(csv_dir: str, images_dir: str) -> List[str]:
    os.makedirs(images_dir, exist_ok=True)
    written: List[str] = []

    sarl_plot = _plot_group(
        SARL_FILES,
        csv_dir,
        images_dir,
        "training_rewards_sarl.png",
        "SARL Training Reward Curves",
    )
    if sarl_plot:
        written.append(sarl_plot)

    marl_plot = _plot_group(
        MARL_FILES,
        csv_dir,
        images_dir,
        "training_rewards_marl.png",
        "MARL Training Reward Curves",
    )
    if marl_plot:
        written.append(marl_plot)

    return written

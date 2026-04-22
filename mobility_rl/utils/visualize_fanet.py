#!/usr/bin/env python3
"""
visualize_fanet.py — Real-time & recorded FANET multi-agent visualization
==========================================================================

Renders the MARLMacEnv simulation showing:
  • UAV positions colored by cluster membership
  • Cluster-head leaders (star markers)
  • Interference graph edges between cluster leaders
  • Communication range circles around leaders
  • Motion trails for each UAV
  • Per-step metrics overlay (throughput, collisions, drops)
  • Cluster labels near leaders

Supports:
  • Live matplotlib window (--live)
  • MP4 video export  (--save  path.mp4)
  • Adjustable FPS    (--fps 15)
  • 2D top-down view  (default) or 3D perspective (--view 3d)
  • Custom episode length (--steps 200)

Usage
-----
    # Live window with random actions
    python utils/visualize_fanet.py --live --steps 100

    # Save MP4 (no window)
    python utils/visualize_fanet.py --save results/fanet_demo.mp4 --steps 150 --fps 20

    # Both live + save, 3D view
    python utils/visualize_fanet.py --live --save demo.mp4 --view 3d --steps 80

    # With a trained checkpoint (loads MAGAT-D3QN policy)
    python utils/visualize_fanet.py --live --checkpoint results/unified/checkpoints
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

from configs import config as params
from configs.cluster_config import ClusterConfig as CC


def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available on this system."""
    import shutil
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Color palette — visually distinct, colorblind-friendly
# ---------------------------------------------------------------------------
CLUSTER_COLORS = [
    "#E63946",  # red
    "#457B9D",  # steel blue
    "#2A9D8F",  # teal
    "#E9C46A",  # golden
    "#F4A261",  # sandy orange
    "#264653",  # dark blue-green
    "#A8DADC",  # light blue
    "#6A0572",  # purple
    "#1D3557",  # navy
    "#B5179E",  # magenta
    "#FB5607",  # vivid orange
    "#3A86FF",  # bright blue
]

TRAIL_ALPHA = 0.15
EDGE_COLOR = "#FF006E"
EDGE_ALPHA = 0.45
BG_COLOR = "#0D1117"
GRID_COLOR = "#21262D"
TEXT_COLOR = "#C9D1D9"
LEADER_EDGE_COLOR = "#FFD700"


def _get_color(cluster_id: int) -> str:
    """Return a color for the given cluster ID."""
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


# ---------------------------------------------------------------------------
# Data collector — records state at every timestep
# ---------------------------------------------------------------------------
class EpisodeRecorder:
    """Collects per-step data from a MARLMacEnv episode for animation."""

    def __init__(self):
        self.positions: list[np.ndarray] = []        # (steps, N, 3)
        self.velocities: list[np.ndarray] = []       # (steps, N, 3)
        self.assignments: list[np.ndarray] = []      # (steps, N)
        self.leaders: list[dict[int, int]] = []       # cid -> uav_idx
        self.active_cids: list[list[int]] = []
        self.edge_indices: list[np.ndarray] = []     # (2, E)
        self.summaries: list[dict[str, Any]] = []
        self.infos: list[dict[str, Any]] = []

    def snapshot(self, env) -> None:
        """Capture current environment state."""
        self.positions.append(env.mobility_model.positions.copy())
        self.velocities.append(env.mobility_model.velocities.copy())
        self.assignments.append(env.cluster_manager.assignment.copy())

        active = env.cluster_manager.get_active_cluster_ids()
        self.active_cids.append(list(active))

        leaders = {}
        for cid in active:
            leaders[cid] = env.cluster_manager.get_leader(cid)
        self.leaders.append(leaders)

        edge_idx, _ = env.cluster_manager.get_interference_graph()
        self.edge_indices.append(edge_idx.copy())

    def record_step_info(self, summary: dict, infos: dict) -> None:
        self.summaries.append(dict(summary))
        self.infos.append(dict(infos))

    @property
    def n_steps(self) -> int:
        return len(self.positions)

    @property
    def n_uavs(self) -> int:
        return self.positions[0].shape[0] if self.positions else 0


# ---------------------------------------------------------------------------
# Run episode & collect data
# ---------------------------------------------------------------------------
def run_episode(
    env,
    max_steps: int = 200,
    policy_fn=None,
    seed: int = 42,
) -> EpisodeRecorder:
    """
    Run one full episode through the environment and record all states.

    Parameters
    ----------
    env : MARLMacEnv
        The PettingZoo ParallelEnv instance.
    max_steps : int
        Maximum number of environment steps.
    policy_fn : callable, optional
        Function(obs_dict, env) -> actions_dict. If None, uses random actions.
    seed : int
        Seed for environment reset.

    Returns
    -------
    EpisodeRecorder with all timestep data.
    """
    recorder = EpisodeRecorder()

    obs, infos = env.reset(seed=seed)
    recorder.snapshot(env)
    recorder.record_step_info({}, infos)

    env.max_steps = max_steps

    for step in range(max_steps):
        # Select actions
        if policy_fn is not None:
            actions = policy_fn(obs, env)
        else:
            # Random actions for all possible agents
            actions = {
                agent: env.action_space(agent).sample()
                for agent in env.possible_agents
            }

        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Record state after step
        recorder.snapshot(env)
        summary = env.get_last_step_summary()
        recorder.record_step_info(summary, infos)

        # Check if episode is done
        if all(terminations.values()):
            break

    return recorder


# ---------------------------------------------------------------------------
# 2D Animator
# ---------------------------------------------------------------------------
class FANETAnimator2D:
    """
    Matplotlib-based 2D top-down animation of the FANET simulation.

    Features:
      - UAVs as colored scatter points (color = cluster)
      - Leaders as star markers with golden edge
      - Interference graph edges between leaders
      - R_I range circles around leaders
      - Motion trails (faded history)
      - Metrics overlay (throughput, drops, collisions)
      - Cluster ID labels near leaders
    """

    def __init__(
        self,
        recorder: EpisodeRecorder,
        fps: int = 15,
        trail_length: int = 15,
        show_range_circles: bool = True,
        show_trails: bool = True,
        show_edges: bool = True,
        show_labels: bool = True,
        show_metrics: bool = True,
        figsize: tuple[float, float] = (14, 10),
    ):
        self.rec = recorder
        self.fps = fps
        self.trail_length = trail_length
        self.show_range_circles = show_range_circles
        self.show_trails = show_trails
        self.show_edges = show_edges
        self.show_labels = show_labels
        self.show_metrics = show_metrics
        self.figsize = figsize

        # Precompute area bounds
        self.area_x = params.AREA_X
        self.area_y = params.AREA_Y

        self.fig = None
        self.ax = None

    def _setup_figure(self):
        """Create the figure and axis with dark theme styling."""
        self.fig, self.ax = plt.subplots(
            1, 1, figsize=self.figsize,
            facecolor=BG_COLOR,
        )
        ax = self.ax
        ax.set_facecolor(BG_COLOR)
        ax.set_xlim(-5, self.area_x + 5)
        ax.set_ylim(-5, self.area_y + 5)
        ax.set_aspect("equal")

        # Grid
        ax.grid(True, color=GRID_COLOR, linewidth=0.3, alpha=0.5)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
            spine.set_linewidth(0.5)

        ax.set_xlabel("X (m)", color=TEXT_COLOR, fontsize=10, fontweight="bold")
        ax.set_ylabel("Y (m)", color=TEXT_COLOR, fontsize=10, fontweight="bold")

        # --- Persistent artists ---
        # Subordinate UAVs (non-leaders)
        self.scat_sub = ax.scatter(
            [], [], s=40, zorder=5, edgecolors="white", linewidths=0.3, alpha=0.85,
        )
        # Leader UAVs (stars)
        self.scat_leaders = ax.scatter(
            [], [], s=180, marker="*", zorder=6,
            edgecolors=LEADER_EDGE_COLOR, linewidths=1.0,
        )
        # Interference edges (LineCollection)
        self.edge_collection = LineCollection(
            [], colors=EDGE_COLOR, linewidths=1.2, alpha=EDGE_ALPHA, zorder=3,
        )
        ax.add_collection(self.edge_collection)

        # Range circles (we'll rebuild per-frame)
        self.range_circles: list[Circle] = []

        # Trail lines per UAV
        self.trail_lines = {}

        # Cluster labels
        self.cluster_labels: list[plt.Text] = []

        # Title
        self.title = ax.set_title(
            "", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=10,
        )

        # Metrics text box
        self.metrics_text = ax.text(
            0.02, 0.98, "", transform=ax.transAxes,
            fontsize=9, color=TEXT_COLOR,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#161B22", edgecolor=GRID_COLOR, alpha=0.9),
        )

        # Legend placeholder
        self.legend_text = ax.text(
            0.98, 0.02, "", transform=ax.transAxes,
            fontsize=7, color="#8B949E",
            verticalalignment="bottom", horizontalalignment="right", fontfamily="monospace",
        )

        plt.tight_layout()

    def _update_frame(self, frame_idx: int):
        """Update all artists for a single animation frame."""
        rec = self.rec
        pos = rec.positions[frame_idx]      # (N, 3) — use x, y
        assign = rec.assignments[frame_idx]  # (N,)
        leaders = rec.leaders[frame_idx]     # {cid: uav_idx}
        active_cids = rec.active_cids[frame_idx]
        edge_index = rec.edge_indices[frame_idx]

        # --- Separate leaders vs subordinates ---
        leader_set = set(leaders.values())
        sub_mask = np.array([i not in leader_set for i in range(rec.n_uavs)])
        leader_mask = ~sub_mask

        # Colors per UAV based on cluster assignment
        colors_all = []
        for i in range(rec.n_uavs):
            cid = assign[i]
            if cid >= 0:
                colors_all.append(_get_color(cid))
            else:
                colors_all.append("#555555")  # orphan

        colors_sub = [colors_all[i] for i in range(rec.n_uavs) if sub_mask[i]]
        colors_lead = [colors_all[i] for i in range(rec.n_uavs) if leader_mask[i]]

        # Update subordinate scatter
        if sub_mask.any():
            self.scat_sub.set_offsets(pos[sub_mask, :2])
            self.scat_sub.set_facecolors(colors_sub)
        else:
            self.scat_sub.set_offsets(np.empty((0, 2)))

        # Update leader scatter
        if leader_mask.any():
            self.scat_leaders.set_offsets(pos[leader_mask, :2])
            self.scat_leaders.set_facecolors(colors_lead)
        else:
            self.scat_leaders.set_offsets(np.empty((0, 2)))

        # --- Interference edges ---
        if self.show_edges and edge_index.shape[1] > 0 and len(active_cids) > 0:
            segments = []
            for col_idx in range(edge_index.shape[1]):
                u_local = edge_index[0, col_idx]
                v_local = edge_index[1, col_idx]
                if u_local < len(active_cids) and v_local < len(active_cids):
                    cid_u = active_cids[u_local]
                    cid_v = active_cids[v_local]
                    if cid_u in leaders and cid_v in leaders:
                        p1 = pos[leaders[cid_u], :2]
                        p2 = pos[leaders[cid_v], :2]
                        # Avoid duplicate undirected edges
                        if cid_u < cid_v:
                            segments.append([p1, p2])
            self.edge_collection.set_segments(segments)
        else:
            self.edge_collection.set_segments([])

        # --- Range circles ---
        for circ in self.range_circles:
            circ.remove()
        self.range_circles.clear()
        if self.show_range_circles:
            for cid, leader_idx in leaders.items():
                circ = Circle(
                    pos[leader_idx, :2], CC.R_I,
                    fill=False, edgecolor=_get_color(cid),
                    linewidth=0.6, alpha=0.25, linestyle="--", zorder=2,
                )
                self.ax.add_patch(circ)
                self.range_circles.append(circ)

        # --- Trails ---
        if self.show_trails:
            start = max(0, frame_idx - self.trail_length)
            for i in range(rec.n_uavs):
                trail_x = [rec.positions[t][i, 0] for t in range(start, frame_idx + 1)]
                trail_y = [rec.positions[t][i, 1] for t in range(start, frame_idx + 1)]
                color = colors_all[i]
                if i in self.trail_lines:
                    self.trail_lines[i].set_data(trail_x, trail_y)
                    self.trail_lines[i].set_color(color)
                else:
                    line, = self.ax.plot(
                        trail_x, trail_y,
                        color=color, linewidth=0.8, alpha=TRAIL_ALPHA, zorder=1,
                    )
                    self.trail_lines[i] = line

        # --- Cluster labels ---
        for lbl in self.cluster_labels:
            lbl.remove()
        self.cluster_labels.clear()
        if self.show_labels:
            for cid, leader_idx in leaders.items():
                lx, ly = pos[leader_idx, 0], pos[leader_idx, 1]
                n_members = np.sum(assign == cid)
                txt = self.ax.text(
                    lx + 3, ly + 3, f"C{cid}({n_members})",
                    fontsize=7, color=_get_color(cid), fontweight="bold",
                    zorder=7, alpha=0.9,
                )
                self.cluster_labels.append(txt)

        # --- Title ---
        self.title.set_text(
            f"FANET Decentralized Cluster Simulation  •  "
            f"Step {frame_idx}/{rec.n_steps - 1}  •  "
            f"{len(active_cids)} clusters  •  {rec.n_uavs} UAVs"
        )

        # --- Metrics overlay ---
        if self.show_metrics and frame_idx < len(rec.summaries):
            s = rec.summaries[frame_idx]
            if s:
                lines = [
                    f"Throughput: {s.get('throughput_mbps', 0):.2f} Mbps",
                    f"Drops:      {s.get('drops', 0):.0f}",
                    f"Collisions: {s.get('collisions', 0):.0f}",
                    f"Coord Succ: {s.get('coord_success', 0):.2f}",
                    f"Clusters:   {s.get('num_clusters', 0)}",
                    f"Avg Size:   {s.get('avg_cluster_size', 0):.1f}",
                    f"Reassoc:    {s.get('reassociations', 0)}",
                    f"Handovers:  {s.get('handovers', 0)}",
                ]
                self.metrics_text.set_text("\n".join(lines))
            else:
                self.metrics_text.set_text("Initializing...")

        # Legend
        self.legend_text.set_text("★ = Cluster Head  •  ── = Interference Edge  •  ⚬ = R_I range")

        return (
            self.scat_sub, self.scat_leaders, self.edge_collection,
            self.title, self.metrics_text,
        )

    def animate(
        self,
        save_path: Optional[str] = None,
        show_live: bool = True,
    ) -> animation.FuncAnimation:
        """
        Build and optionally display/save the animation.

        Parameters
        ----------
        save_path : str, optional
            Path to save as .mp4 (requires ffmpeg) or .gif.
        show_live : bool
            Whether to display the animation in a live window.

        Returns
        -------
        matplotlib.animation.FuncAnimation
        """
        if not show_live and save_path:
            matplotlib.use("Agg")

        self._setup_figure()

        anim = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self.rec.n_steps,
            interval=1000 // self.fps,
            blit=False,
            repeat=False,
        )

        if save_path:
            ext = os.path.splitext(save_path)[1].lower()
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            if ext == ".gif" or not _has_ffmpeg():
                if ext != ".gif" and not _has_ffmpeg():
                    save_path = os.path.splitext(save_path)[0] + ".gif"
                    print("ffmpeg not found — falling back to GIF output.")
                writer = animation.PillowWriter(fps=self.fps)
            else:
                writer = animation.FFMpegWriter(
                    fps=self.fps,
                    metadata={"title": "FANET Simulation", "artist": "visualize_fanet"},
                    codec="libx264",
                    bitrate=3000,
                )
            print(f"Saving animation to {save_path} ({self.rec.n_steps} frames @ {self.fps} fps)...")
            anim.save(save_path, writer=writer, dpi=120)
            print(f"Saved: {save_path}")

        if show_live:
            plt.show()

        return anim


# ---------------------------------------------------------------------------
# 3D Animator (bonus)
# ---------------------------------------------------------------------------
class FANETAnimator3D:
    """Matplotlib 3D perspective animation of the FANET simulation."""

    def __init__(
        self,
        recorder: EpisodeRecorder,
        fps: int = 15,
        trail_length: int = 10,
        figsize: tuple[float, float] = (14, 10),
    ):
        self.rec = recorder
        self.fps = fps
        self.trail_length = trail_length
        self.figsize = figsize
        self.fig = None
        self.ax = None

    def _setup_figure(self):
        self.fig = plt.figure(figsize=self.figsize, facecolor=BG_COLOR)
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor=BG_COLOR)

        self.ax.set_xlim(0, params.AREA_X)
        self.ax.set_ylim(0, params.AREA_Y)
        self.ax.set_zlim(0, params.AREA_Z)
        self.ax.set_xlabel("X (m)", color=TEXT_COLOR, fontsize=9)
        self.ax.set_ylabel("Y (m)", color=TEXT_COLOR, fontsize=9)
        self.ax.set_zlabel("Z (m)", color=TEXT_COLOR, fontsize=9)
        self.ax.tick_params(colors=TEXT_COLOR, labelsize=7)

        # pane colors
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor(GRID_COLOR)
        self.ax.yaxis.pane.set_edgecolor(GRID_COLOR)
        self.ax.zaxis.pane.set_edgecolor(GRID_COLOR)

        self.title = self.ax.set_title("", color=TEXT_COLOR, fontsize=11, fontweight="bold")
        plt.tight_layout()

    def _update_frame(self, frame_idx: int):
        ax = self.ax
        ax.cla()

        ax.set_xlim(0, params.AREA_X)
        ax.set_ylim(0, params.AREA_Y)
        ax.set_zlim(0, params.AREA_Z)
        ax.set_xlabel("X (m)", color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel("Y (m)", color=TEXT_COLOR, fontsize=9)
        ax.set_zlabel("Z (m)", color=TEXT_COLOR, fontsize=9)
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(GRID_COLOR)
        ax.yaxis.pane.set_edgecolor(GRID_COLOR)
        ax.zaxis.pane.set_edgecolor(GRID_COLOR)

        rec = self.rec
        pos = rec.positions[frame_idx]
        assign = rec.assignments[frame_idx]
        leaders = rec.leaders[frame_idx]
        active_cids = rec.active_cids[frame_idx]
        edge_index = rec.edge_indices[frame_idx]

        leader_set = set(leaders.values())

        # Colors
        colors = [_get_color(assign[i]) if assign[i] >= 0 else "#555555" for i in range(rec.n_uavs)]

        # Subordinates
        for i in range(rec.n_uavs):
            if i not in leader_set:
                ax.scatter(
                    pos[i, 0], pos[i, 1], pos[i, 2],
                    c=colors[i], s=25, alpha=0.7, edgecolors="white", linewidths=0.2,
                )

        # Leaders (stars)
        for cid, leader_idx in leaders.items():
            ax.scatter(
                pos[leader_idx, 0], pos[leader_idx, 1], pos[leader_idx, 2],
                c=_get_color(cid), s=120, marker="*", edgecolors=LEADER_EDGE_COLOR,
                linewidths=0.8, zorder=10,
            )

        # Interference edges
        if edge_index.shape[1] > 0 and len(active_cids) > 0:
            seen = set()
            for col_idx in range(edge_index.shape[1]):
                u_local = edge_index[0, col_idx]
                v_local = edge_index[1, col_idx]
                if u_local < len(active_cids) and v_local < len(active_cids):
                    cid_u = active_cids[u_local]
                    cid_v = active_cids[v_local]
                    pair = tuple(sorted((cid_u, cid_v)))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    if cid_u in leaders and cid_v in leaders:
                        p1 = pos[leaders[cid_u]]
                        p2 = pos[leaders[cid_v]]
                        ax.plot(
                            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                            color=EDGE_COLOR, linewidth=1.0, alpha=0.4,
                        )

        # Trails
        start = max(0, frame_idx - self.trail_length)
        for i in range(rec.n_uavs):
            trail = np.array([rec.positions[t][i] for t in range(start, frame_idx + 1)])
            ax.plot(
                trail[:, 0], trail[:, 1], trail[:, 2],
                color=colors[i], linewidth=0.5, alpha=0.15,
            )

        ax.set_title(
            f"FANET 3D  •  Step {frame_idx}/{rec.n_steps - 1}  •  "
            f"{len(active_cids)} clusters  •  {rec.n_uavs} UAVs",
            color=TEXT_COLOR, fontsize=11, fontweight="bold",
        )

    def animate(
        self,
        save_path: Optional[str] = None,
        show_live: bool = True,
    ) -> animation.FuncAnimation:
        if not show_live and save_path:
            matplotlib.use("Agg")

        self._setup_figure()

        anim = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self.rec.n_steps,
            interval=1000 // self.fps,
            blit=False,
            repeat=False,
        )

        if save_path:
            ext = os.path.splitext(save_path)[1].lower()
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            if ext == ".gif" or not _has_ffmpeg():
                if ext != ".gif" and not _has_ffmpeg():
                    save_path = os.path.splitext(save_path)[0] + ".gif"
                    print("ffmpeg not found — falling back to GIF output.")
                writer = animation.PillowWriter(fps=self.fps)
            else:
                writer = animation.FFMpegWriter(
                    fps=self.fps, codec="libx264", bitrate=3000,
                )
            print(f"Saving 3D animation to {save_path}...")
            anim.save(save_path, writer=writer, dpi=120)
            print(f"Saved: {save_path}")

        if show_live:
            plt.show()

        return anim


# ---------------------------------------------------------------------------
# Policy loader helpers
# ---------------------------------------------------------------------------
def make_random_policy():
    """Return a policy function that selects random actions."""
    def policy_fn(obs_dict, env):
        return {
            agent: env.action_space(agent).sample()
            for agent in env.possible_agents
        }
    return policy_fn


def make_fixed_policy(action_id: int = 0):
    """Return a policy that always picks the same action for all agents."""
    def policy_fn(obs_dict, env):
        return {agent: action_id for agent in env.possible_agents}
    return policy_fn


def make_magat_policy(checkpoint_dir: str):
    """
    Load a trained MAGAT-D3QN policy from a checkpoint directory.

    Returns a policy function compatible with run_episode().
    """
    import torch
    from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork
    from configs.marl_config import MARLConfig
    from configs.cluster_config import ClusterConfig as CC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find checkpoint file
    ckpt_path = os.path.join(checkpoint_dir, "unified_gnn_marl_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"MAGAT checkpoint not found: {ckpt_path}")

    policy_net = MAGAT_D3QN_QNetwork(
        node_in_dim=CC.OBS_DIM_CLUSTER,
        hidden_dim=MARLConfig.HIDDEN_DIM,
        num_actions=CC.NUM_ACTIONS,
        heads=MARLConfig.GNN_HEADS,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()
    policy_net.reset_memory()

    def policy_fn(obs_dict, env):
        x, edge_index, alive_mask = env.get_global_graph_state()
        x_t = torch.FloatTensor(x).to(device)
        ei_t = torch.LongTensor(edge_index).to(device) if edge_index.shape[1] > 0 else torch.zeros((2, 0), dtype=torch.long, device=device)
        mask_t = torch.BoolTensor(alive_mask).to(device)

        with torch.no_grad():
            q_values = policy_net(x_t, ei_t, alive_mask=mask_t)

        actions = {}
        for k, agent_name in enumerate(env.possible_agents):
            if alive_mask[k]:
                actions[agent_name] = int(q_values[k].argmax().item())
            else:
                actions[agent_name] = 0
        return actions

    return policy_fn


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FANET Multi-Agent Simulation Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--live", action="store_true", help="Show live animation window")
    parser.add_argument("--save", type=str, default=None, help="Save animation to file (.mp4 or .gif)")
    parser.add_argument("--fps", type=int, default=15, help="Animation frames per second")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed")
    parser.add_argument("--view", choices=["2d", "3d"], default="2d", help="View mode")
    parser.add_argument("--trail-length", type=int, default=15, help="Number of past positions to show as trail")
    parser.add_argument("--no-trails", action="store_true", help="Disable motion trails")
    parser.add_argument("--no-edges", action="store_true", help="Disable interference graph edges")
    parser.add_argument("--no-circles", action="store_true", help="Disable range circles")
    parser.add_argument("--no-labels", action="store_true", help="Disable cluster labels")
    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics overlay")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to MAGAT-D3QN checkpoint directory (uses trained policy)"
    )
    parser.add_argument(
        "--policy", choices=["random", "fixed_tdma", "fixed_csma", "magat"],
        default="random",
        help="Policy to use for action selection",
    )
    parser.add_argument("--nodes", type=int, default=None, help="Override number of UAV nodes")
    args = parser.parse_args()

    if not args.live and not args.save:
        print("Specify --live and/or --save <path>. Use --help for options.")
        sys.exit(1)

    # Override node count if specified
    if args.nodes is not None:
        params.N = args.nodes

    # Ensure MAX_STEPS_PER_EP exists
    if not hasattr(params, "MAX_STEPS_PER_EP"):
        params.MAX_STEPS_PER_EP = args.steps

    # Create environment
    from envs.marl_mac_env import MARLMacEnv
    env = MARLMacEnv(seed=args.seed)

    # Select policy
    if args.checkpoint or args.policy == "magat":
        ckpt_dir = args.checkpoint or "results/unified/checkpoints"
        print(f"Loading MAGAT-D3QN policy from: {ckpt_dir}")
        policy_fn = make_magat_policy(ckpt_dir)
    elif args.policy == "fixed_tdma":
        from envs.burst_scheduler import encode_action, VALID_RHO_LEVELS
        action_id = encode_action(0, len(VALID_RHO_LEVELS) // 2)  # TDMA, mid rho
        policy_fn = make_fixed_policy(action_id)
        print(f"Using fixed TDMA policy (action={action_id})")
    elif args.policy == "fixed_csma":
        from envs.burst_scheduler import encode_action, VALID_RHO_LEVELS
        action_id = encode_action(1, len(VALID_RHO_LEVELS) // 2)  # CSMA, mid rho
        policy_fn = make_fixed_policy(action_id)
        print(f"Using fixed CSMA policy (action={action_id})")
    else:
        policy_fn = make_random_policy()
        print("Using random policy")

    # Run episode
    print(f"Running episode: {args.steps} steps, {params.N} UAVs, seed={args.seed}...")
    recorder = run_episode(env, max_steps=args.steps, policy_fn=policy_fn, seed=args.seed)
    print(f"Recorded {recorder.n_steps} frames for {recorder.n_uavs} UAVs")

    # Animate
    if args.view == "3d":
        animator = FANETAnimator3D(
            recorder,
            fps=args.fps,
            trail_length=args.trail_length,
            figsize=(14, 10),
        )
    else:
        animator = FANETAnimator2D(
            recorder,
            fps=args.fps,
            trail_length=args.trail_length,
            show_trails=not args.no_trails,
            show_edges=not args.no_edges,
            show_range_circles=not args.no_circles,
            show_labels=not args.no_labels,
            show_metrics=not args.no_metrics,
            figsize=(14, 10),
        )

    animator.animate(save_path=args.save, show_live=args.live)


if __name__ == "__main__":
    main()

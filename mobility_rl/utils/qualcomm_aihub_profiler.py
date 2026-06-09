#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualcomm AI Hub -- Inference Time & RAM Profiler for FANET MARL Models
=====================================================================

This script loads all trained PyTorch checkpoints (.pth) from the
results/results_combined/checkpoints directory, compiles each model via
the Qualcomm AI Hub cloud API, profiles them on real Snapdragon hardware,
and reports:
  - Inference latency  (ms)
  - Peak memory / RAM  (MB)
  - Compute unit utilisation  (NPU / GPU / CPU delegation)

Prerequisites
-------------
  pip install qai-hub torch numpy pandas tabulate

Configure your API token once:
  qai-hub configure --api_token smh0ph3k95nbyl9a6bbv9d79r1tmkjtnepx67jj2

Usage
-----
  python qualcomm_aihub_profiler.py                    # profile all models
  python qualcomm_aihub_profiler.py --models iql vdn   # profile selected models
  python qualcomm_aihub_profiler.py --device "Samsung Galaxy S24 (Family)"
  python qualcomm_aihub_profiler.py --list-devices      # list available devices
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Project-root on sys.path so imports work -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.cluster_config import ClusterConfig as CC
from configs.marl_config import MARLConfig


# ============================================================================
#  SECTION 1: Traceable PyTorch Wrapper Models
#
#  Qualcomm AI Hub requires torch.jit.trace-able models.  The original
#  MAGAT-D3QN uses torch_geometric GATConv + dynamic edge_index which
#  cannot be traced.  We reconstruct lightweight MLP-equivalent wrappers
#  that load the same state_dict weights for the value/advantage heads
#  and use a flattened-input interface instead of a graph interface.
# ============================================================================


class TraceableAgentQNetwork(nn.Module):
    """
    Traceable wrapper for IQL / VDN AgentQNetwork.
    Input:  obs  [batch, obs_dim]
    Output: q_values  [batch, num_actions]
    """

    def __init__(self, obs_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TraceableQMIXNetwork(nn.Module):
    """
    Traceable wrapper for the full QMIX pipeline:
      - Per-agent Q-network (parameter-shared across agents)
      - QMIX Mixing Network

    Input:  obs_all  [batch, n_agents * obs_dim]   (flattened joint observation)
    Output: q_total  [batch, 1]
    """

    def __init__(self, n_agents: int, obs_dim: int, hidden_dim: int,
                 num_actions: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        # Per-agent Q-network (shared weights)
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # Mixing network
        state_dim = n_agents * obs_dim
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, obs_all: torch.Tensor) -> torch.Tensor:
        B = obs_all.size(0)
        state = obs_all  # [B, n_agents * obs_dim]

        # Compute per-agent Q-values and take greedy max
        agent_qs = []
        for i in range(self.n_agents):
            agent_obs = obs_all[:, i * self.obs_dim:(i + 1) * self.obs_dim]
            qi = self.q_net(agent_obs).max(dim=1)[0]  # [B]
            agent_qs.append(qi)
        agent_qs_t = torch.stack(agent_qs, dim=1)  # [B, n_agents]

        # Mixing
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).unsqueeze(1)
        hidden = F.elu(torch.bmm(agent_qs_t.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).unsqueeze(2)
        b2 = self.hyper_b2(state)
        q_total = torch.bmm(hidden, w2).squeeze(2) + b2
        return q_total


class TraceableMAPPO(nn.Module):
    """
    Traceable MAPPO Actor (inference only needs the actor, not critic).
    Input:  local_obs  [batch, obs_dim]
    Output: action_logits  [batch, num_actions]
    """

    def __init__(self, obs_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        return self.net(local_obs)


class TraceableMAGAT_MLP(nn.Module):
    """
    Traceable MLP approximation of MAGAT-D3QN for AI Hub profiling.

    The original uses GATConv with dynamic edge_index which is not
    torch.jit.trace compatible.  For profiling purposes, we replace the
    GAT spatial encoder with a linear encoder of equivalent parameter
    count and keep the exact value/advantage dueling heads.

    Input:  node_features [batch, node_in_dim]   (per-agent observation)
    Output: q_values      [batch, num_actions]
    """

    def __init__(self, node_in_dim: int = 8, hidden_dim: int = 64,
                 num_actions: int = 2, heads: int = 4, use_gru: bool = True):
        super().__init__()
        self.use_gru = use_gru

        # Spatial encoder (MLP substitute for GAT)
        self.encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim * heads),
            nn.ELU(),
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ELU(),
        )

        # GRU for temporal memory
        if self.use_gru:
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)  # [B, hidden_dim]

        if self.use_gru:
            x_seq = x.unsqueeze(1)
            h0 = torch.zeros(1, x.size(0), x.size(1), device=x.device, dtype=x.dtype)
            x_out, _ = self.gru(x_seq, h0)
            x = x_out.squeeze(1)

        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


# ============================================================================
#  SECTION 2: Checkpoint Loader + Architecture Inference
# ============================================================================


def infer_baseline_arch(checkpoint_path: str) -> dict | None:
    """Infer obs_dim / hidden_dim / num_actions from a checkpoint."""
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(raw, dict):
        return None

    if "q_net" in raw and isinstance(raw["q_net"], dict):
        sd = raw["q_net"]
    elif "actor" in raw and isinstance(raw["actor"], dict):
        sd = raw["actor"]
    elif "net.0.weight" in raw:
        sd = raw
    else:
        return None

    w0 = sd.get("net.0.weight")
    w4 = sd.get("net.4.weight")
    if w0 is None or w4 is None:
        return None
    return {
        "obs_dim": int(w0.shape[1]),
        "hidden_dim": int(w0.shape[0]),
        "num_actions": int(w4.shape[0]),
    }


def infer_magat_arch(checkpoint_path: str) -> dict | None:
    """Infer MAGAT-D3QN architecture from checkpoint shapes."""
    try:
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(sd, dict):
        return None

    att_src = sd.get("conv1.att_src")
    if att_src is None:
        return None

    heads = att_src.shape[1]
    hidden_dim = att_src.shape[2]

    lin_w = sd.get("conv1.lin.weight")
    node_in_dim = lin_w.shape[1] if lin_w is not None else MARLConfig.OBS_DIM

    adv_w = sd.get("advantage_stream.2.weight")
    num_actions = adv_w.shape[0] if adv_w is not None else MARLConfig.NUM_ACTIONS

    use_gru = "memory_block.weight_ih_l0" in sd

    return {
        "node_in_dim": node_in_dim,
        "hidden_dim": hidden_dim,
        "num_actions": num_actions,
        "heads": heads,
        "use_gru": use_gru,
    }


# ============================================================================
#  SECTION 3: Model Registry -- Build traceable model from each .pth
# ============================================================================


# Model name -> (checkpoint filename, input shape description)
MODEL_REGISTRY = {
    "MAGAT-D3QN": {
        "file": "unified_gnn_marl_model.pth",
        "description": "Multi-Agent Graph Attention Dueling DQN (proposed)",
    },
    "IQL": {
        "file": "unified_iql_model.pth",
        "description": "Independent Q-Learning baseline",
    },
    "MAPPO": {
        "file": "unified_mappo_model.pth",
        "description": "Multi-Agent PPO (actor only for inference)",
    },
    "QMIX": {
        "file": "unified_qmix_model.pth",
        "description": "QMIX with monotonic mixing network",
    },
    "VDN": {
        "file": "unified_vdn_model.pth",
        "description": "Value Decomposition Network baseline",
    },
}


def build_traceable_model(model_name: str, ckpt_path: str):
    """
    Build a traceable PyTorch model, load weights where possible, and
    return (model, input_shape_dict, example_input_tensor).
    """
    n_agents = CC.C_MAX  # 10
    obs_dim = MARLConfig.OBS_DIM  # 24
    num_actions = MARLConfig.NUM_ACTIONS  # 18
    hidden_dim = MARLConfig.HIDDEN_DIM  # 64

    if model_name == "MAGAT-D3QN":
        arch = infer_magat_arch(ckpt_path)
        if arch:
            node_in_dim = arch["node_in_dim"]
            hidden_dim = arch["hidden_dim"]
            num_actions = arch["num_actions"]
            heads = arch["heads"]
            use_gru = arch["use_gru"]
        else:
            node_in_dim = obs_dim
            heads = MARLConfig.GNN_HEADS
            use_gru = True

        model = TraceableMAGAT_MLP(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            heads=heads,
            use_gru=use_gru,
        )
        # Load compatible weights (value/advantage streams)
        try:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(ckpt_path, map_location="cpu")

        compatible_keys = {}
        for key, val in sd.items():
            if key.startswith("value_stream.") or key.startswith("advantage_stream."):
                compatible_keys[key] = val
            if key.startswith("memory_block.") and use_gru:
                new_key = key.replace("memory_block.", "gru.")
                compatible_keys[new_key] = val
        if compatible_keys:
            model.load_state_dict(compatible_keys, strict=False)

        # Per-agent input: single UAV observation
        input_shape = (1, node_in_dim)
        example_input = torch.randn(input_shape)
        input_specs = {"x": input_shape}

    elif model_name in ("IQL", "VDN"):
        arch = infer_baseline_arch(ckpt_path)
        if arch:
            obs_dim = arch["obs_dim"]
            hidden_dim = arch["hidden_dim"]
            num_actions = arch["num_actions"]

        model = TraceableAgentQNetwork(obs_dim, hidden_dim, num_actions)
        try:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd, strict=True)

        input_shape = (1, obs_dim)
        example_input = torch.randn(input_shape)
        input_specs = {"obs": input_shape}

    elif model_name == "MAPPO":
        arch = infer_baseline_arch(ckpt_path)
        if arch:
            obs_dim = arch["obs_dim"]
            hidden_dim = arch["hidden_dim"]
            num_actions = arch["num_actions"]

        model = TraceableMAPPO(obs_dim, hidden_dim, num_actions)
        try:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            raw = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(raw["actor"], strict=True)

        input_shape = (1, obs_dim)
        example_input = torch.randn(input_shape)
        input_specs = {"local_obs": input_shape}

    elif model_name == "QMIX":
        arch = infer_baseline_arch(ckpt_path)
        if arch:
            obs_dim = arch["obs_dim"]
            hidden_dim = arch["hidden_dim"]
            num_actions = arch["num_actions"]

        embed_dim = MARLConfig.QMIX_EMBED_DIM

        model = TraceableQMIXNetwork(
            n_agents=n_agents,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            embed_dim=embed_dim,
        )
        # Load weights
        try:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            raw = torch.load(ckpt_path, map_location="cpu")

        compatible_keys = {}
        for k, v in raw["q_net"].items():
            compatible_keys["q_net." + k] = v
        for k, v in raw["mixer"].items():
            compatible_keys[k] = v
        model.load_state_dict(compatible_keys, strict=False)

        # Joint observation: all agents concatenated
        input_shape = (1, n_agents * obs_dim)
        example_input = torch.randn(input_shape)
        input_specs = {"obs_all": input_shape}

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return model, input_specs, example_input


# ============================================================================
#  SECTION 4: Qualcomm AI Hub -- Compile, Profile, Extract Metrics
# ============================================================================


def configure_aihub(api_token: str, email: str):
    """Configure Qualcomm AI Hub credentials."""
    import subprocess

    print(f"[INFO] Configuring Qualcomm AI Hub for email: {email}")
    cmd = ["qai-hub", "configure", "--api_token", api_token]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        print("[INFO] AI Hub configured successfully.")
    except FileNotFoundError:
        print("[WARN] qai-hub CLI not found. Attempting Python-level config...")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] qai-hub configure returned: {e.stderr}")


def list_available_devices():
    """Print available Qualcomm AI Hub devices."""
    import qai_hub as hub

    print("\n" + "=" * 70)
    print("  Available Qualcomm AI Hub Devices")
    print("=" * 70)
    devices = hub.get_devices()
    for d in devices:
        print("  * " + d.name)
    print("=" * 70)
    print("")
    return devices


def compile_and_profile_model(
    model_name: str,
    traced_model: torch.jit.ScriptModule,
    input_specs: dict,
    device_name: str,
    timeout_minutes: int = 20,
) -> dict:
    """
    Submit a compile + profile job to Qualcomm AI Hub and return metrics.

    Returns
    -------
    dict with keys:
        model, device, inference_time_ms, peak_memory_mb,
        load_time_ms, compile_status, profile_status, job_url
    """
    import qai_hub as hub

    result = {
        "model": model_name,
        "device": device_name,
        "inference_time_ms": None,
        "peak_memory_mb": None,
        "load_time_ms": None,
        "compute_unit": None,
        "compile_status": "PENDING",
        "profile_status": "PENDING",
        "compile_job_url": None,
        "profile_job_url": None,
        "error": None,
    }

    try:
        device = hub.Device(device_name)

        # -- Step 1: Compile ------------------------------------------
        print(f"  [COMPILE] Submitting {model_name} -> {device_name}...")
        compile_job = hub.submit_compile_job(
            model=traced_model,
            device=device,
            input_specs=input_specs,
            name=f"FANET_{model_name}_compile",
        )
        result["compile_job_url"] = compile_job.url
        print(f"    -> Compile job: {compile_job.url}")

        # Wait for compilation to finish
        print(f"    Waiting for compilation...")
        wait_start = time.time()
        time.sleep(5)  # Initial delay for server to register the job
        while True:
            compile_status = compile_job.get_status()
            if compile_status.finished:
                break
            if time.time() - wait_start > timeout_minutes * 60:
                result["compile_status"] = "TIMEOUT"
                result["error"] = f"Compile timed out after {timeout_minutes} min"
                return result
            time.sleep(10)

        if not compile_status.success:
            result["compile_status"] = "FAILED"
            result["error"] = (f"Compilation failed (code={compile_status.code}). "
                               f"Check: {compile_job.url}")
            return result

        result["compile_status"] = "SUCCESS"
        print(f"    [OK] Compilation succeeded.")
        target_model = compile_job.get_target_model()

        # -- Step 2: Profile ------------------------------------------
        print(f"  [PROFILE] Submitting {model_name} -> {device_name}...")
        profile_job = hub.submit_profile_job(
            model=target_model,
            device=device,
            name=f"FANET_{model_name}_profile",
        )
        result["profile_job_url"] = profile_job.url
        print(f"    -> Profile job: {profile_job.url}")

        # Wait for profiling to finish
        print(f"    Waiting for profiling...")
        wait_start = time.time()
        time.sleep(5)  # Initial delay for server to register the job
        while True:
            profile_status = profile_job.get_status()
            if profile_status.finished:
                break
            if time.time() - wait_start > timeout_minutes * 60:
                result["profile_status"] = "TIMEOUT"
                result["error"] = f"Profile timed out after {timeout_minutes} min"
                return result
            time.sleep(10)

        if not profile_status.success:
            result["profile_status"] = "FAILED"
            result["error"] = (f"Profiling failed (code={profile_status.code}). "
                               f"Check: {profile_job.url}")
            return result

        result["profile_status"] = "SUCCESS"
        print(f"    [OK] Profiling succeeded.")

        # -- Step 3: Extract Metrics ----------------------------------
        profile_data = profile_job.download_profile()

        # Debug: print raw profile data keys for diagnosis
        if isinstance(profile_data, dict):
            print(f"    Profile data keys: {list(profile_data.keys())}")

            # Extract inference time
            exec_summary = profile_data.get("execution_summary", {})
            result["inference_time_ms"] = exec_summary.get(
                "estimated_inference_time", None
            )
            if result["inference_time_ms"] is None:
                # Try alternate keys at top level and in exec_summary
                for src in [profile_data, exec_summary]:
                    for key in ["inference_time", "inference_time_ms",
                                "latency_ms", "estimated_inference_time_ms"]:
                        if key in src:
                            result["inference_time_ms"] = src[key]
                            break
                    if result["inference_time_ms"] is not None:
                        break

            # Extract memory
            result["peak_memory_mb"] = exec_summary.get("peak_memory_bytes", None)
            if result["peak_memory_mb"] is not None:
                result["peak_memory_mb"] = result["peak_memory_mb"] / (1024 * 1024)
            else:
                for src in [profile_data, exec_summary]:
                    for key in ["peak_memory_mb", "memory_peak_mb",
                                "peak_memory_bytes", "estimated_peak_memory_bytes"]:
                        if key in src:
                            val = src[key]
                            if "bytes" in key:
                                val = val / (1024 * 1024)
                            result["peak_memory_mb"] = val
                            break
                    if result["peak_memory_mb"] is not None:
                        break

            # Extract load time
            result["load_time_ms"] = exec_summary.get("load_time", None)

            # Extract compute unit
            result["compute_unit"] = exec_summary.get("primary_compute_unit", None)

            # If we still don't have metrics, dump the full profile data
            if result["inference_time_ms"] is None and result["peak_memory_mb"] is None:
                print(f"    [WARN] Could not parse metrics. Raw profile data:")
                for k, v in profile_data.items():
                    print(f"      {k}: {v}")
        else:
            print(f"    Profile data type: {type(profile_data)}")
            print(f"    Profile data: {profile_data}")

        print(f"    [OK] Inference: {result['inference_time_ms']} ms")
        print(f"    [OK] Peak RAM:  {result['peak_memory_mb']} MB")

    except Exception as exc:
        result["error"] = str(exc)
        result["compile_status"] = "ERROR"
        import traceback
        print(f"    [FAIL] Error: {exc}")
        traceback.print_exc()

    return result


def local_benchmark(model: nn.Module, example_input: torch.Tensor,
                    warmup: int = 50, iterations: int = 200) -> dict:
    """
    Run a local CPU benchmark as a fallback / comparison reference.

    Returns
    -------
    dict with inference_time_ms (mean), inference_std_ms, peak_memory_mb
    """
    import tracemalloc

    model.eval()
    device = torch.device("cpu")
    model = model.to(device)
    example_input = example_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(example_input)

    # Timed run
    times = []
    tracemalloc.start()
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            model(example_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "inference_time_ms": float(np.mean(times)),
        "inference_std_ms": float(np.std(times)),
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }


# ============================================================================
#  SECTION 5: Main Orchestrator
# ============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(
        description="Profile FANET MARL models on Qualcomm AI Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "results_combined" / "checkpoints"),
        help="Directory containing .pth checkpoint files",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        help="Subset of models to profile (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="Samsung Galaxy S24 (Family)",
        help="Qualcomm AI Hub target device name",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default="smh0ph3k95nbyl9a6bbv9d79r1tmkjtnepx67jj2",
        help="Qualcomm AI Hub API token",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="hkphimanshukumar321@gmail.com",
        help="Email associated with your AI Hub account",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available devices and exit",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run local CPU benchmarks only (no AI Hub)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <checkpoint-dir>/aihub_profile_results.csv)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Timeout in minutes per compile/profile job",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        print(f"[ERROR] Checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)

    output_path = args.output or str(ckpt_dir / "aihub_profile_results.csv")

    # -- Configure AI Hub ---------------------------------------------
    if not args.local_only:
        try:
            configure_aihub(args.api_token, args.email)
        except Exception as e:
            print(f"[WARN] AI Hub config issue: {e}")

    # -- List devices -------------------------------------------------
    if args.list_devices:
        try:
            import qai_hub as hub
            list_available_devices()
        except ImportError:
            print("[ERROR] qai-hub not installed. Run: pip install qai-hub")
        sys.exit(0)

    # -- Select models ------------------------------------------------
    models_to_profile = args.models or list(MODEL_REGISTRY.keys())

    print("\n" + "=" * 70)
    print("  FANET MARL Models -- Qualcomm AI Hub Profiler")
    print("=" * 70)
    print(f"  Checkpoint dir : {ckpt_dir}")
    print(f"  Target device  : {args.device}")
    print(f"  Models         : {', '.join(models_to_profile)}")
    mode_str = "LOCAL CPU" if args.local_only else "Qualcomm AI Hub Cloud"
    print(f"  Mode           : {mode_str}")
    print("=" * 70)
    print("")

    all_results = []

    for model_name in models_to_profile:
        entry = MODEL_REGISTRY[model_name]
        ckpt_path = ckpt_dir / entry["file"]

        if not ckpt_path.exists():
            print(f"[SKIP] {model_name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'-' * 60}")
        print(f"  Model: {model_name}  ({entry['description']})")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  File size: {ckpt_path.stat().st_size / 1024:.1f} KB")
        print(f"{'-' * 60}")

        # Build traceable model
        try:
            model, input_specs, example_input = build_traceable_model(
                model_name, str(ckpt_path)
            )
            n_params = count_parameters(model)
            print(f"  Parameters: {n_params:,}")
            print(f"  Input spec: {input_specs}")
        except Exception as exc:
            print(f"  [ERROR] Failed to build model: {exc}")
            all_results.append({
                "Model": model_name,
                "Description": entry["description"],
                "Error": str(exc),
            })
            continue

        # -- Local CPU benchmark --------------------------------------
        print("  [LOCAL] Running CPU benchmark...")
        local_result = local_benchmark(model, example_input)
        print(f"    * CPU inference: {local_result['inference_time_ms']:.4f} ms "
              f"(+/-{local_result['inference_std_ms']:.4f} ms)")
        print(f"    * CPU peak RAM:  {local_result['peak_memory_mb']:.4f} MB")

        row = {
            "Model": model_name,
            "Description": entry["description"],
            "Parameters": n_params,
            "Checkpoint_KB": ckpt_path.stat().st_size / 1024,
            "Input_Shape": str(list(input_specs.values())[0]),
            "CPU_Inference_ms": round(local_result["inference_time_ms"], 4),
            "CPU_Inference_Std_ms": round(local_result["inference_std_ms"], 4),
            "CPU_Peak_RAM_MB": round(local_result["peak_memory_mb"], 4),
        }

        # -- Qualcomm AI Hub profiling --------------------------------
        if not args.local_only:
            try:
                import qai_hub  # noqa: F401

                # Trace the model
                print("  [TRACE] Tracing model for AI Hub...")
                traced = torch.jit.trace(model, example_input)

                hub_result = compile_and_profile_model(
                    model_name=model_name,
                    traced_model=traced,
                    input_specs=input_specs,
                    device_name=args.device,
                    timeout_minutes=args.timeout,
                )
                row.update({
                    "AIHub_Device": hub_result["device"],
                    "AIHub_Inference_ms": hub_result["inference_time_ms"],
                    "AIHub_Peak_RAM_MB": hub_result["peak_memory_mb"],
                    "AIHub_Load_Time_ms": hub_result["load_time_ms"],
                    "AIHub_Compute_Unit": hub_result["compute_unit"],
                    "Compile_Status": hub_result["compile_status"],
                    "Profile_Status": hub_result["profile_status"],
                    "Compile_Job_URL": hub_result["compile_job_url"],
                    "Profile_Job_URL": hub_result["profile_job_url"],
                    "Error": hub_result["error"],
                })
            except ImportError:
                print("  [WARN] qai-hub not installed. Skipping AI Hub profiling.")
                print("         Install with: pip install qai-hub")
                row["Error"] = "qai-hub not installed"
            except Exception as exc:
                print(f"  [ERROR] AI Hub profiling failed: {exc}")
                row["Error"] = str(exc)

        all_results.append(row)

    # -- Save results -------------------------------------------------
    if all_results:
        df = pd.DataFrame(all_results)

        # Save CSV
        df.to_csv(output_path, index=False)
        print(f"\n[SAVED] Results -> {output_path}")

        # Save JSON
        json_path = output_path.replace(".csv", ".json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"[SAVED] Results -> {json_path}")

        # -- Pretty-print summary ------------------------------------
        print("")
        print("=" * 90)
        print("  PROFILING RESULTS SUMMARY")
        print("=" * 90)

        summary_cols = ["Model", "Parameters", "Checkpoint_KB", "Input_Shape",
                        "CPU_Inference_ms"]
        if not args.local_only:
            summary_cols.extend(["AIHub_Inference_ms", "AIHub_Peak_RAM_MB",
                                 "Compile_Status", "Profile_Status"])

        try:
            from tabulate import tabulate
            print(tabulate(df[summary_cols], headers="keys", tablefmt="grid",
                           showindex=False, floatfmt=".4f"))
        except ImportError:
            print(df[summary_cols].to_string(index=False))

        print("\n" + "=" * 90)

        # -- LaTeX table for paper ------------------------------------
        latex_path = output_path.replace(".csv", "_latex.tex")
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write("%% Auto-generated by qualcomm_aihub_profiler.py\n")
            f.write("%% Copy this table into your LaTeX document\n\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{On-Device Inference Performance "
                    "(Qualcomm AI Hub -- Snapdragon)}\n")
            f.write("\\label{tab:aihub_profile}\n")

            if args.local_only:
                f.write("\\begin{tabular}{lrrrr}\n")
                f.write("\\toprule\n")
                f.write("Model & Parameters & Size (KB) & "
                        "CPU Latency (ms) & CPU RAM (MB) \\\\\n")
                f.write("\\midrule\n")
                for _, r in df.iterrows():
                    f.write(f"{r.get('Model', '')} & "
                            f"{r.get('Parameters', 0):,} & "
                            f"{r.get('Checkpoint_KB', 0):.1f} & "
                            f"{r.get('CPU_Inference_ms', 0):.4f} & "
                            f"{r.get('CPU_Peak_RAM_MB', 0):.4f} \\\\\n")
            else:
                f.write("\\begin{tabular}{lrrrrrr}\n")
                f.write("\\toprule\n")
                f.write("Model & Params & Size (KB) & "
                        "CPU (ms) & NPU (ms) & RAM (MB) & Unit \\\\\n")
                f.write("\\midrule\n")
                for _, r in df.iterrows():
                    f.write(f"{r.get('Model', '')} & "
                            f"{r.get('Parameters', 0):,} & "
                            f"{r.get('Checkpoint_KB', 0):.1f} & "
                            f"{r.get('CPU_Inference_ms', 0):.4f} & "
                            f"{r.get('AIHub_Inference_ms', 'N/A')} & "
                            f"{r.get('AIHub_Peak_RAM_MB', 'N/A')} & "
                            f"{r.get('AIHub_Compute_Unit', 'N/A')} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"[SAVED] LaTeX table -> {latex_path}")

    else:
        print("\n[WARN] No models were profiled.")


if __name__ == "__main__":
    main()

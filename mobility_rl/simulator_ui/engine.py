"""Live decentralized simulator backend for the browser UI.

The public entrypoint remains ``SimulationEngine``, but the implementation now
owns a live ``MARLMacEnv`` session and exposes cluster-aware snapshots instead
of the old sink-centric simulator contract.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import sys
from zipfile import ZipFile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs import config as params
from configs.cluster_config import ClusterConfig as CC
from configs.marl_config import MARLConfig
from envs.burst_scheduler import VALID_RHO_LEVELS, encode_action
from envs.marl_mac_env import MARLMacEnv
from envs.sarl_central_env import SARLCentralEnv


BASE_PARAM_KEYS = (
    "N",
    "SEED",
    "SIM_TIME_S",
    "MAX_STEPS_PER_EP",
    "AREA_X",
    "AREA_Y",
    "AREA_Z",
    "MOBILITY_MODEL",
    "SPEED_MODE",
    "V_MIN",
    "V_MAX",
    "V_MEAN",
    "V_STD",
    "SPEED_UPDATE_INTERVAL",
    "GM_ALPHA",
    "RWP_PAUSE_TIME",
    "CIRC_RADIUS",
    "CIRC_OMEGA_MEAN",
    "CIRC_OMEGA_STD",
    "CIRC_CLIMB_RATE",
    "MOBILITY_DT",
    "PHY_RATE_BPS",
    "PAYLOAD_BYTES",
    "QMAX",
    "CW_MIN",
    "CW_MAX",
    "DIFS_SLOTS",
    "SIFS_SLOTS",
    "ACK_SLOTS",
    "ACK_TIMEOUT_SLOTS",
    "MAX_RETRY",
    "RTS_CTS_ENABLED",
    "ACK_ENABLED",
    "RTS_SLOTS",
    "CTS_SLOTS",
    "TDMA_GUARD_TIME_S",
    "OFFERED_PPS",
    "ENABLE_FADING",
    "FADING_MODEL",
    "NAKAGAMI_M",
    "NAKAGAMI_OMEGA",
    "RICIAN_K",
    "MODULATION",
    "W_DIST",
    "W_SINR",
    "W_MOB",
    "W_LOAD",
    "THETA_JOIN",
    "THETA_LEAVE",
    "HEALTH_THRESHOLD",
    "A_ENERGY",
    "A_DEGREE",
    "A_MOBSTAB",
    "A_QUEUE",
    "A_RISK",
)

ENV_OPTION_KEYS = (
    "topology_preset",
    "traffic_profile",
    "mobility_model",
    "speed_scale",
    "interference_scale",
    "coordination_capacity_scale",
    "graph_mode",
    "graph_missing_edge_prob",
    "graph_false_edge_prob",
    "graph_staleness_steps",
    "obs_staleness_steps",
    "handover_info_staleness_steps",
    "obs_noise_std",
    "failure_schedule",
    "use_burst_history",
    "offered_pps",
)

BOOLEAN_BASE_PARAMS = {
    "RTS_CTS_ENABLED",
    "ACK_ENABLED",
    "ENABLE_FADING",
}

FLOAT_BASE_PARAMS = {
    "SIM_TIME_S",
    "AREA_X",
    "AREA_Y",
    "AREA_Z",
    "V_MIN",
    "V_MAX",
    "V_MEAN",
    "V_STD",
    "SPEED_UPDATE_INTERVAL",
    "GM_ALPHA",
    "RWP_PAUSE_TIME",
    "CIRC_RADIUS",
    "CIRC_OMEGA_MEAN",
    "CIRC_OMEGA_STD",
    "CIRC_CLIMB_RATE",
    "MOBILITY_DT",
    "PHY_RATE_BPS",
    "TDMA_GUARD_TIME_S",
    "NAKAGAMI_M",
    "NAKAGAMI_OMEGA",
    "RICIAN_K",
    "W_DIST",
    "W_SINR",
    "W_MOB",
    "W_LOAD",
    "THETA_JOIN",
    "THETA_LEAVE",
    "HEALTH_THRESHOLD",
    "A_ENERGY",
    "A_DEGREE",
    "A_MOBSTAB",
    "A_QUEUE",
    "A_RISK",
}

INT_BASE_PARAMS = set(BASE_PARAM_KEYS) - FLOAT_BASE_PARAMS - BOOLEAN_BASE_PARAMS - {
    "MOBILITY_MODEL",
    "SPEED_MODE",
    "FADING_MODEL",
    "MODULATION",
}

FLOAT_ENV_OPTIONS = {
    "speed_scale",
    "interference_scale",
    "coordination_capacity_scale",
    "graph_missing_edge_prob",
    "graph_false_edge_prob",
    "obs_noise_std",
}

INT_ENV_OPTIONS = {
    "offered_pps",
    "graph_staleness_steps",
    "obs_staleness_steps",
    "handover_info_staleness_steps",
}

BOOLEAN_ENV_OPTIONS = {"use_burst_history"}

MOBILITY_MODEL_OPTIONS = ["gauss_markov", "random_waypoint", "random_walk", "circular"]
TRAFFIC_PROFILE_OPTIONS = ["smooth", "bursty_on_off", "heavy_tail"]
GRAPH_MODE_OPTIONS = ["dynamic", "none", "static", "shuffled"]
TOPOLOGY_PRESET_OPTIONS = ["default", "compact_dense", "sparse_separated", "asymmetric_hotspot"]
FAILURE_TARGET_OPTIONS = ["random", "max_backlog", "max_degree"]

DISPLAY_LABELS = {
    "N": "Nodes",
    "SEED": "Seed",
    "AREA_X": "Area X",
    "AREA_Y": "Area Y",
    "AREA_Z": "Area Z",
    "OFFERED_PPS": "Load pps",
    "mobility_model": "Mobility",
    "traffic_profile": "Traffic",
    "topology_preset": "Topology",
    "graph_mode": "Observed graph",
    "graph_missing_edge_prob": "Missing-edge prob",
    "graph_false_edge_prob": "False-edge prob",
    "graph_staleness_steps": "Graph stale steps",
    "obs_staleness_steps": "Obs stale steps",
    "handover_info_staleness_steps": "Handover stale steps",
    "obs_noise_std": "Obs noise",
    "speed_scale": "Speed scale",
    "interference_scale": "Interference scale",
    "coordination_capacity_scale": "Coordination capacity",
    "use_burst_history": "Burst history",
    "failure_schedule": "Failure schedule",
    "W_DIST": "Weight: Dist",
    "W_SINR": "Weight: SINR",
    "W_MOB": "Weight: Mobility",
    "W_LOAD": "Weight: Load",
    "THETA_JOIN": "Join Thresh",
    "THETA_LEAVE": "Leave Thresh",
    "HEALTH_THRESHOLD": "Health Thresh",
    "A_ENERGY": "Health: Energy",
    "A_DEGREE": "Health: Degree",
    "A_MOBSTAB": "Health: Mobility",
    "A_QUEUE": "Health: Queue",
    "A_RISK": "Health: Risk",
}

GROUP_LABELS = {
    "default": "Default",
    "generalization": "Generalization",
    "robustness": "Robustness",
    "failure_recovery": "Failure/Recovery",
}

MODEL_PRIORITIES = {
    "MAGAT-D3QN": 100,
    "QMIX": 90,
    "VDN": 85,
    "IQL": 80,
    "MCA-D3QN": 60,
    "PPO": 40,
    "A2C": 35,
    "DQN": 30,
}

POLICY_FILES = {
    "MAGAT-D3QN": {"filename": "unified_gnn_marl_model.pth", "model_type": "marl_gnn"},
    "QMIX": {"filename": "unified_qmix_model.pth", "model_type": "marl"},
    "VDN": {"filename": "unified_vdn_model.pth", "model_type": "marl"},
    "IQL": {"filename": "unified_iql_model.pth", "model_type": "marl"},
    "MCA-D3QN": {"filename": "unified_mca_d3qn_model.zip", "model_type": "sarl_custom"},
    "PPO": {"filename": "unified_ppo_model.zip", "model_type": "sarl"},
    "A2C": {"filename": "unified_a2c_model.zip", "model_type": "sarl"},
    "DQN": {"filename": "unified_dqn_model.zip", "model_type": "sarl"},
}

FIXED_POLICY_DESCRIPTORS = (
    {
        "id": "fixed:all_tdma_mid",
        "label": "All TDMA (mid rho)",
        "model_type": "fixed",
        "source": "fixed",
        "priority": 1,
        "available": True,
        "compatible": True,
        "compatibility_note": None,
        "fixed_action": encode_action(0, len(VALID_RHO_LEVELS) // 2),
        "fixed_mac_label": "TDMA",
    },
    {
        "id": "fixed:all_csma_mid",
        "label": "All CSMA (mid rho)",
        "model_type": "fixed",
        "source": "fixed",
        "priority": 0,
        "available": True,
        "compatible": True,
        "compatibility_note": None,
        "fixed_action": encode_action(1, len(VALID_RHO_LEVELS) // 2),
        "fixed_mac_label": "CSMA_CA",
    },
)


def _ensure_max_steps_default() -> None:
    if not hasattr(params, "MAX_STEPS_PER_EP"):
        sim_time_s = float(getattr(params, "SIM_TIME_S", 10.0))
        params.MAX_STEPS_PER_EP = max(1, int(np.ceil(sim_time_s / CC.BURST_TOTAL_TIME)))


_ensure_max_steps_default()

DEFAULT_BASE_PARAMS: dict[str, Any] = {}
for key in BASE_PARAM_KEYS:
    if hasattr(params, key):
        DEFAULT_BASE_PARAMS[key] = copy.deepcopy(getattr(params, key))
    elif hasattr(CC, key):
        DEFAULT_BASE_PARAMS[key] = copy.deepcopy(getattr(CC, key))


def _scenario_id(group: str, name: str) -> str:
    return f"{group}:{name}"


def _silent_log(_: str) -> None:
    return None


def _coerce_value(value: Any, *, as_bool: bool = False, as_int: bool = False, as_float: bool = False) -> Any:
    if as_bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if as_int:
        return int(float(value))
    if as_float:
        return float(value)
    return value


def _slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _display_label(key: str) -> str:
    return DISPLAY_LABELS.get(key, key.replace("_", " ").title())


def _compact_value_label(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _safe_torch_load(path: str | os.PathLike[str]) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_tensor_shape(state: Any, key: str) -> tuple[int, ...] | None:
    if not isinstance(state, dict):
        return None
    tensor = state.get(key)
    if tensor is None or not hasattr(tensor, "shape"):
        return None
    return tuple(int(dim) for dim in tensor.shape)


def _validate_marl_checkpoint(path: str | os.PathLike[str]) -> tuple[bool, str | None]:
    try:
        state = _safe_torch_load(path)
    except Exception as exc:
        return False, f"checkpoint could not be inspected: {exc}"

    obs_shape = _state_tensor_shape(state, "net.0.weight")
    action_shape = _state_tensor_shape(state, "net.4.weight")
    if obs_shape is None or action_shape is None or len(obs_shape) < 2 or len(action_shape) < 2:
        return False, "checkpoint layout does not match current MARL baseline loader"

    trained_obs_dim = int(obs_shape[1])
    trained_actions = int(action_shape[0])
    expected_obs_dim = int(MARLConfig.OBS_DIM)
    expected_actions = int(MARLConfig.NUM_ACTIONS)
    if trained_obs_dim != expected_obs_dim or trained_actions != expected_actions:
        return (
            False,
            f"trained for obs_dim={trained_obs_dim}, actions={trained_actions}; current env uses obs_dim={expected_obs_dim}, actions={expected_actions}",
        )
    return True, None


def _validate_magat_checkpoint(path: str | os.PathLike[str]) -> tuple[bool, str | None]:
    try:
        state = _safe_torch_load(path)
    except Exception as exc:
        return False, f"checkpoint could not be inspected: {exc}"

    obs_shape = _state_tensor_shape(state, "conv1.lin.weight")
    action_shape = _state_tensor_shape(state, "advantage_stream.2.weight")
    if obs_shape is None or action_shape is None or len(obs_shape) < 2 or len(action_shape) < 2:
        return False, "checkpoint layout does not match current MAGAT loader"

    trained_obs_dim = int(obs_shape[1])
    trained_actions = int(action_shape[0])
    expected_obs_dim = int(MARLConfig.OBS_DIM)
    expected_actions = int(MARLConfig.NUM_ACTIONS)
    if trained_obs_dim != expected_obs_dim or trained_actions != expected_actions:
        return (
            False,
            f"trained for obs_dim={trained_obs_dim}, actions={trained_actions}; current env uses obs_dim={expected_obs_dim}, actions={expected_actions}",
        )
    return True, None


def _validate_sb3_zip_checkpoint(path: str | os.PathLike[str]) -> tuple[bool, str | None]:
    expected_obs_shape = (int(CC.C_MAX * CC.OBS_DIM_CLUSTER),)
    expected_action_count = int(CC.NUM_ACTIONS)
    expected_action_width = int(CC.C_MAX)

    try:
        with ZipFile(path) as archive:
            payload = json.loads(archive.read("data"))
    except Exception as exc:
        return False, f"checkpoint archive could not be inspected: {exc}"

    observation_space = payload.get("observation_space", {})
    action_space = payload.get("action_space", {})
    obs_spaces = str(observation_space.get("spaces", ""))
    action_type = str(action_space.get(":type:", ""))
    action_n = str(action_space.get("n", ""))

    if "Discrete" in action_type:
        action_name = action_type.split("'")[-2] if "'" in action_type else action_type.rsplit(".", 1)[-1]
        return (
            False,
            f"trained for {action_name} with n={action_n}; current UI SARL session uses MultiDiscrete({expected_action_width} x {expected_action_count}) and obs {expected_obs_shape}",
        )

    expected_obs_text = str(expected_obs_shape[0])
    if expected_obs_text not in obs_spaces and expected_obs_text not in json.dumps(observation_space):
        return (
            False,
            f"trained for observation layout {obs_spaces or observation_space}; current UI SARL session uses flattened obs {expected_obs_shape}",
        )

    return True, None


def _validate_checkpoint_compatibility(label: str, meta: dict[str, Any], path: Path) -> tuple[bool, bool, str | None]:
    if meta["model_type"] == "marl_gnn":
        compatible, note = _validate_magat_checkpoint(path)
        return compatible, compatible, note
    if meta["model_type"] == "marl":
        compatible, note = _validate_marl_checkpoint(path)
        return compatible, compatible, note
    if meta["model_type"] == "sarl":
        compatible, note = _validate_sb3_zip_checkpoint(path)
        return compatible, compatible, note
    if meta["model_type"] == "sarl_custom":
        return False, False, "custom MCA-D3QN checkpoints are not loadable by the current UI loader"
    return True, True, None


def _build_base_param_schema() -> list[dict[str, Any]]:
    return [
        {"key": "N", "label": "Nodes", "type": "number", "min": 1, "step": 1},
        {"key": "SEED", "label": "Seed", "type": "number", "min": 0, "step": 1},
        {"key": "AREA_X", "label": "Area X", "type": "number", "min": 10, "step": 10},
        {"key": "AREA_Y", "label": "Area Y", "type": "number", "min": 10, "step": 10},
        {"key": "AREA_Z", "label": "Area Z", "type": "number", "min": 10, "step": 5},
        {"key": "OFFERED_PPS", "label": "Load pps", "type": "number", "min": 1, "step": 10},
        {"key": "W_DIST", "label": "Weight: Dist", "type": "number", "step": 0.05},
        {"key": "W_SINR", "label": "Weight: SINR", "type": "number", "step": 0.05},
        {"key": "W_MOB", "label": "Weight: Mobility", "type": "number", "step": 0.05},
        {"key": "W_LOAD", "label": "Weight: Load", "type": "number", "step": 0.05},
        {"key": "THETA_JOIN", "label": "Join Thresh", "type": "number", "step": 0.05},
        {"key": "THETA_LEAVE", "label": "Leave Thresh", "type": "number", "step": 0.05},
        {"key": "HEALTH_THRESHOLD", "label": "Health Thresh", "type": "number", "step": 0.05},
        {"key": "A_ENERGY", "label": "Health: Energy", "type": "number", "step": 0.05},
        {"key": "A_DEGREE", "label": "Health: Degree", "type": "number", "step": 0.05},
        {"key": "A_MOBSTAB", "label": "Health: Mobility", "type": "number", "step": 0.05},
        {"key": "A_QUEUE", "label": "Health: Queue", "type": "number", "step": 0.05},
        {"key": "A_RISK", "label": "Health: Risk", "type": "number", "step": 0.05},
    ]


def _build_env_option_schema() -> list[dict[str, Any]]:
    return [
        {
            "key": "mobility_model",
            "label": "Mobility override",
            "type": "select",
            "section": "Mobility",
            "options": MOBILITY_MODEL_OPTIONS,
        },
        {
            "key": "speed_scale",
            "label": "Speed scale",
            "type": "number",
            "section": "Mobility",
            "min": 0.2,
            "max": 3.0,
            "step": 0.1,
        },
        {
            "key": "traffic_profile",
            "label": "Traffic profile",
            "type": "select",
            "section": "Traffic",
            "options": TRAFFIC_PROFILE_OPTIONS,
        },
        {
            "key": "topology_preset",
            "label": "Topology preset",
            "type": "select",
            "section": "Topology",
            "options": TOPOLOGY_PRESET_OPTIONS,
        },
        {
            "key": "graph_mode",
            "label": "Observed graph mode",
            "type": "select",
            "section": "Graph / Obs",
            "options": GRAPH_MODE_OPTIONS,
        },
        {
            "key": "graph_missing_edge_prob",
            "label": "Missing-edge prob",
            "type": "number",
            "section": "Graph / Obs",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
        {
            "key": "graph_false_edge_prob",
            "label": "False-edge prob",
            "type": "number",
            "section": "Graph / Obs",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
        {
            "key": "graph_staleness_steps",
            "label": "Graph stale steps",
            "type": "number",
            "section": "Graph / Obs",
            "min": 0,
            "step": 1,
        },
        {
            "key": "obs_staleness_steps",
            "label": "Obs stale steps",
            "type": "number",
            "section": "Graph / Obs",
            "min": 0,
            "step": 1,
        },
        {
            "key": "handover_info_staleness_steps",
            "label": "Handover stale steps",
            "type": "number",
            "section": "Graph / Obs",
            "min": 0,
            "step": 1,
        },
        {
            "key": "obs_noise_std",
            "label": "Obs noise std",
            "type": "number",
            "section": "Graph / Obs",
            "min": 0.0,
            "step": 0.01,
        },
        {
            "key": "interference_scale",
            "label": "Interference scale",
            "type": "number",
            "section": "Coordination",
            "min": 0.1,
            "max": 3.0,
            "step": 0.05,
        },
        {
            "key": "coordination_capacity_scale",
            "label": "Coordination capacity scale",
            "type": "number",
            "section": "Coordination",
            "min": 0.1,
            "max": 3.0,
            "step": 0.05,
        },
        {
            "key": "use_burst_history",
            "label": "Use burst history",
            "type": "boolean",
            "section": "Coordination",
        },
        {
            "key": "failure_schedule",
            "label": "Failure schedule",
            "type": "json",
            "section": "Failure",
            "placeholder": '[{"step":20,"target":"random"}]',
        },
    ]


def build_preset_registry(base_nodes: int) -> list[dict[str, Any]]:
    from experiments.run_failure_recovery_suite import build_scenarios as build_failure_scenarios
    from experiments.run_generalization_suite import build_scenarios as build_generalization_scenarios
    from experiments.run_robustness_suite import build_scenarios as build_robustness_scenarios

    presets: list[dict[str, Any]] = [
        {
            "id": _scenario_id("default", "base"),
            "group": "default",
            "group_label": GROUP_LABELS["default"],
            "study_block": "base",
            "name": "base",
            "label": "Base environment",
            "description": "Default decentralized cluster-head environment.",
            "param_overrides": {},
            "env_options": {},
        }
    ]

    for scenario in build_generalization_scenarios(base_nodes):
        presets.append(
            {
                "id": _scenario_id("generalization", scenario["scenario"]),
                "group": "generalization",
                "group_label": GROUP_LABELS["generalization"],
                "study_block": scenario["study_block"],
                "name": scenario["scenario"],
                "label": scenario["scenario"].replace("_", " "),
                "description": f"{scenario['study_block']} preset",
                "param_overrides": copy.deepcopy(scenario["param_overrides"]),
                "env_options": copy.deepcopy(scenario["env_options"]),
            }
        )

    for scenario in build_robustness_scenarios():
        presets.append(
            {
                "id": _scenario_id("robustness", scenario["scenario"]),
                "group": "robustness",
                "group_label": GROUP_LABELS["robustness"],
                "study_block": scenario["study_block"],
                "name": scenario["scenario"],
                "label": scenario["scenario"].replace("_", " "),
                "description": f"{scenario['study_block']} preset",
                "param_overrides": {},
                "env_options": copy.deepcopy(scenario["env_options"]),
            }
        )

    for scenario in build_failure_scenarios(max_steps=100):
        presets.append(
            {
                "id": _scenario_id("failure_recovery", scenario["scenario"]),
                "group": "failure_recovery",
                "group_label": GROUP_LABELS["failure_recovery"],
                "study_block": scenario["study_block"],
                "name": scenario["scenario"],
                "label": scenario["scenario"].replace("_", " "),
                "description": f"{scenario['study_block']} preset",
                "param_overrides": {},
                "env_options": copy.deepcopy(scenario["env_options"]),
            }
        )

    return presets


def discover_policy_descriptors(results_root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    root = Path(results_root)
    latest: dict[str, tuple[float, Path]] = {}
    descriptors = [copy.deepcopy(item) for item in FIXED_POLICY_DESCRIPTORS]

    if root.exists():
        for label, meta in POLICY_FILES.items():
            for path in root.rglob(meta["filename"]):
                mtime = path.stat().st_mtime
                current = latest.get(label)
                if current is None or mtime > current[0]:
                    latest[label] = (mtime, path)

    for label, (_, path) in latest.items():
        meta = POLICY_FILES[label]
        available, compatible, note = _validate_checkpoint_compatibility(label, meta, path)
        descriptors.append(
            {
                "id": f"checkpoint:{_slugify(label)}",
                "label": label,
                "model_type": meta["model_type"],
                "source": "checkpoint",
                "priority": MODEL_PRIORITIES.get(label, 10),
                "available": available,
                "compatible": compatible,
                "compatibility_note": note,
                "checkpoint_dir": str(path.parent),
                "checkpoint_file": str(path),
                "checkpoint_mtime": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }
        )

    descriptors.sort(key=lambda item: (-item["priority"], item["label"]))
    return descriptors


def choose_default_policy_id(
    descriptors: list[dict[str, Any]],
    *,
    exclude_ids: set[str] | None = None,
) -> str:
    excluded = set(exclude_ids or ())
    checkpoints = [
        item
        for item in descriptors
        if item["source"] == "checkpoint"
        and item["id"] not in excluded
        and item.get("available", True)
        and item.get("compatible", True)
    ]
    available = {item["label"]: item["id"] for item in checkpoints}
    if "MAGAT-D3QN" in available:
        return available["MAGAT-D3QN"]
    for label in ("QMIX", "VDN", "IQL"):
        if label in available:
            return available[label]
    if "fixed:all_tdma_mid" not in excluded:
        return "fixed:all_tdma_mid"
    for descriptor in descriptors:
        if descriptor["id"] not in excluded:
            return descriptor["id"]
    raise ValueError("No policies are available")


class SimulationEngine:
    """Public façade used by the websocket server and tests."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.base_params = copy.deepcopy(DEFAULT_BASE_PARAMS)
        self.env_reset_options: dict[str, Any] = {}
        self.runtime = {
            "policy_id": None,
            "deterministic": True,
            "speed_factor": 1.0,
        }
        self.runtime_status = {
            "policy_warning": None,
        }
        self.config_change_log: list[dict[str, Any]] = []
        self.results_root = os.path.join(project_root, getattr(params, "RESULTS_ROOT", "results"))

        self._preset_registry = build_preset_registry(int(self.base_params.get("N", getattr(params, "N", 50))))
        self._preset_map = {item["id"]: item for item in self._preset_registry}
        self.selected_preset_id = _scenario_id("default", "base")

        self._policy_registry = discover_policy_descriptors(self.results_root)
        self._policy_map = {item["id"]: item for item in self._policy_registry}
        self.runtime["policy_id"] = choose_default_policy_id(self._policy_registry)

        self._policy_cache: dict[str, tuple[str, Any]] = {}
        self._session_kind = "marl"
        self._session_obs: Any = None
        self._session_done = False
        self.sarl_env: SARLCentralEnv | None = None
        self.marl_env: MARLMacEnv | None = None
        self.last_infos: dict[str, Any] = {}
        self.last_snapshot: dict[str, Any] | None = None
        self.tick_count = 0
        self.sim_time = 0.0
        self.history: list[dict[str, Any]] = []
        self.cluster_step_history: list[dict[str, Any]] = []
        self.graph_edge_history: list[dict[str, Any]] = []
        self.membership_history: list[dict[str, Any]] = []
        self._prev_summary_counts = {"splits": 0, "merges": 0, "reassociations": 0, "handovers": 0, "failure_events": 0}
        self._prev_leader_map: dict[int, int] = {}

        self._apply_config_overrides(config or {})
        self._rebuild_preset_registry_if_needed()
        self.reset()

    def _apply_config_overrides(self, config: dict[str, Any]) -> None:
        if not config:
            return

        nested_base = config.get("base_params", {})
        nested_env = config.get("env_reset_options", {})
        nested_runtime = config.get("runtime", {})

        self.base_params.update(copy.deepcopy(nested_base))
        self.env_reset_options.update(copy.deepcopy(nested_env))
        self.runtime.update(copy.deepcopy(nested_runtime))

        for key, value in config.items():
            if key in {"base_params", "env_reset_options", "runtime"}:
                continue
            if key in BASE_PARAM_KEYS:
                self.base_params[key] = copy.deepcopy(value)
            elif key in ENV_OPTION_KEYS:
                self.env_reset_options[key] = copy.deepcopy(value)
            elif key in {"policy_id", "deterministic", "speed_factor"}:
                self.runtime[key] = copy.deepcopy(value)

        if self.runtime.get("policy_id") not in self._policy_map:
            self.runtime["policy_id"] = choose_default_policy_id(self._policy_registry)
        if self.selected_preset_id not in self._preset_map:
            self.selected_preset_id = _scenario_id("default", "base")

    def _rebuild_preset_registry_if_needed(self) -> None:
        base_nodes = int(self.base_params.get("N", getattr(params, "N", 50)))
        current = self.selected_preset_id
        self._preset_registry = build_preset_registry(base_nodes)
        self._preset_map = {item["id"]: item for item in self._preset_registry}
        if current in self._preset_map:
            self.selected_preset_id = current
        else:
            self.selected_preset_id = _scenario_id("default", "base")

    def _current_preset(self) -> dict[str, Any]:
        return self._preset_map.get(self.selected_preset_id, self._preset_map[_scenario_id("default", "base")])

    def _current_policy_descriptor(self) -> dict[str, Any]:
        policy_id = str(self.runtime.get("policy_id") or choose_default_policy_id(self._policy_registry))
        descriptor = self._policy_map.get(policy_id)
        if descriptor is None or not descriptor.get("available", True) or not descriptor.get("compatible", True):
            policy_id = choose_default_policy_id(self._policy_registry)
            descriptor = self._policy_map[policy_id]
            self.runtime["policy_id"] = policy_id
        return descriptor

    def _mark_policy_unavailable(self, policy_id: str, reason: str) -> None:
        descriptor = self._policy_map.get(policy_id)
        if descriptor is None:
            return
        descriptor["available"] = False
        descriptor["compatible"] = False
        descriptor["compatibility_note"] = reason

    def _set_policy_warning(self, message: str | None) -> None:
        self.runtime_status["policy_warning"] = message

    def _normalize_failure_schedule(self, value: Any) -> list[dict[str, Any]]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            value = json.loads(stripped)
        if not isinstance(value, list):
            raise ValueError("failure_schedule must be a JSON list of event objects")

        normalized = []
        for idx, event in enumerate(value):
            if not isinstance(event, dict):
                raise ValueError(f"failure_schedule[{idx}] must be an object")
            if "step" not in event:
                raise ValueError(f"failure_schedule[{idx}] is missing 'step'")
            normalized_event = dict(event)
            normalized_event["step"] = int(normalized_event["step"])
            if "cluster_id" in normalized_event:
                normalized_event["cluster_id"] = int(normalized_event["cluster_id"])
            normalized.append(normalized_event)
        return normalized

    def _context_item(self, *, key: str, value: Any, source: str) -> list[dict[str, Any]]:
        label = _display_label(key)
        if key == "failure_schedule":
            rows = []
            for idx, event in enumerate(value or []):
                if not isinstance(event, dict):
                    continue
                if "cluster_id" in event:
                    value_label = f"step {int(event.get('step', 0))} cluster={int(event['cluster_id'])}"
                else:
                    value_label = f"step {int(event.get('step', 0))} target={event.get('target', 'random')}"
                rows.append(
                    {
                        "key": f"{key}_{idx}",
                        "base_key": key,
                        "label": label,
                        "value": copy.deepcopy(event),
                        "value_label": value_label,
                        "source": source,
                    }
                )
            return rows or [{"key": key, "base_key": key, "label": label, "value": [], "value_label": "none", "source": source}]

        return [
            {
                "key": key,
                "base_key": key,
                "label": label,
                "value": copy.deepcopy(value),
                "value_label": _compact_value_label(value),
                "source": source,
            }
        ]

    def _build_scenario_context(self) -> dict[str, Any]:
        preset = self._current_preset()
        items: list[dict[str, Any]] = []

        for key, value in preset.get("param_overrides", {}).items():
            items.extend(self._context_item(key=key, value=value, source="preset"))
        for key, value in preset.get("env_options", {}).items():
            items.extend(self._context_item(key=key, value=value, source="preset"))
        for key, value in self.env_reset_options.items():
            items.extend(self._context_item(key=key, value=value, source="manual"))

        summary_lines = [f"{item['source']}: {item['label']} = {item['value_label']}" for item in items]
        impairment_keys = {
            "graph_mode",
            "graph_missing_edge_prob",
            "graph_false_edge_prob",
            "graph_staleness_steps",
            "obs_staleness_steps",
            "handover_info_staleness_steps",
            "obs_noise_std",
            "failure_schedule",
            "interference_scale",
            "coordination_capacity_scale",
        }
        impairment_lines = [
            f"{item['label']} = {item['value_label']}"
            for item in items
            if item.get("base_key") == "failure_schedule" or item.get("base_key") in impairment_keys
        ]
        if not summary_lines:
            summary_lines = ["Base environment"]
        if not impairment_lines:
            impairment_lines = ["No active impairments"]

        return {
            "context_items": items,
            "summary_lines": summary_lines,
            "impairment_lines": impairment_lines,
        }

    def _build_graph_stats(
        self,
        *,
        true_edges: list[dict[str, int]],
        observed_edges: list[dict[str, int]],
        cluster_count: int,
    ) -> dict[str, Any]:
        pair_capacity = max(cluster_count * (cluster_count - 1) / 2.0, 1.0)
        true_pairs = {tuple(sorted((edge["source"], edge["target"]))) for edge in true_edges}
        observed_pairs = {tuple(sorted((edge["source"], edge["target"]))) for edge in observed_edges}
        return {
            "true_edge_count": len(true_pairs),
            "observed_edge_count": len(observed_pairs),
            "missing_edge_count": len(true_pairs - observed_pairs),
            "false_edge_count": len(observed_pairs - true_pairs),
            "true_density": round(len(true_pairs) / pair_capacity, 4),
            "observed_density": round(len(observed_pairs) / pair_capacity, 4),
            "graph_mode": self._effective_env_options().get("graph_mode", "dynamic"),
            "graph_missing_edge_prob": float(self._effective_env_options().get("graph_missing_edge_prob", 0.0)),
            "graph_false_edge_prob": float(self._effective_env_options().get("graph_false_edge_prob", 0.0)),
            "graph_staleness_steps": int(self._effective_env_options().get("graph_staleness_steps", 0)),
            "obs_staleness_steps": int(self._effective_env_options().get("obs_staleness_steps", 0)),
            "handover_info_staleness_steps": int(self._effective_env_options().get("handover_info_staleness_steps", 0)),
        }

    def _effective_base_params(self) -> dict[str, Any]:
        effective = copy.deepcopy(self.base_params)
        effective.update(copy.deepcopy(self._current_preset().get("param_overrides", {})))
        if "SIM_TIME_S" not in effective:
            effective["SIM_TIME_S"] = float(getattr(params, "SIM_TIME_S", 10.0))
        return effective

    def _effective_env_options(self) -> dict[str, Any]:
        effective = copy.deepcopy(self._current_preset().get("env_options", {}))
        effective.update(copy.deepcopy(self.env_reset_options))
        return effective

    def _apply_effective_params_to_globals(self) -> None:
        effective = self._effective_base_params()
        for key, value in effective.items():
            if hasattr(params, key):
                setattr(params, key, value)
            if hasattr(CC, key):
                setattr(CC, key, value)

        if "MAX_STEPS_PER_EP" in effective:
            params.MAX_STEPS_PER_EP = int(effective["MAX_STEPS_PER_EP"])
        else:
            sim_time_s = float(effective.get("SIM_TIME_S", getattr(params, "SIM_TIME_S", 10.0)))
            params.MAX_STEPS_PER_EP = max(1, int(np.ceil(sim_time_s / CC.BURST_TOTAL_TIME)))

    def _session_kind_for_policy(self, descriptor: dict[str, Any]) -> str:
        if descriptor["model_type"] in {"sarl", "sarl_custom"}:
            return "sarl"
        return "marl"

    def _log_change(self, *, target: str, key: str, old_value: Any, new_value: Any, mode: str = "reset") -> dict[str, Any]:
        entry = {
            "timestamp": round(self.sim_time, 4),
            "tick": self.tick_count,
            "target": target,
            "param": key,
            "old_value": old_value,
            "new_value": new_value,
            "apply_mode": mode,
        }
        self.config_change_log.append(entry)
        return entry

    def get_config(self) -> dict[str, Any]:
        effective_base = self._effective_base_params()
        effective_env = self._effective_env_options()
        policies = []
        for descriptor in self._policy_registry:
            policies.append(
                {
                    "id": descriptor["id"],
                    "label": descriptor["label"],
                    "model_type": descriptor["model_type"],
                    "source": descriptor["source"],
                    "checkpoint_dir": descriptor.get("checkpoint_dir"),
                    "available": bool(descriptor.get("available", True)),
                    "compatible": bool(descriptor.get("compatible", True)),
                    "compatibility_note": descriptor.get("compatibility_note"),
                    "loaded": descriptor["id"] in self._policy_cache,
                }
            )

        return {
            "schema_version": "marl_ui_v2",
            "base_params": copy.deepcopy(self.base_params),
            "effective_base_params": effective_base,
            "env_reset_options": copy.deepcopy(self.env_reset_options),
            "effective_env_reset_options": effective_env,
            "runtime": copy.deepcopy(self.runtime),
            "runtime_status": copy.deepcopy(self.runtime_status),
            "selected_preset_id": self.selected_preset_id,
            "selected_policy_id": self.runtime["policy_id"],
            "presets": copy.deepcopy(self._preset_registry),
            "policies": policies,
            "ui_schema": {
                "mobility_models": MOBILITY_MODEL_OPTIONS,
                "traffic_profiles": TRAFFIC_PROFILE_OPTIONS,
                "graph_modes": GRAPH_MODE_OPTIONS,
                "topology_presets": TOPOLOGY_PRESET_OPTIONS,
                "failure_target_options": FAILURE_TARGET_OPTIONS,
                "boolean_base_params": sorted(BOOLEAN_BASE_PARAMS),
                "base_param_schema": _build_base_param_schema(),
                "env_option_schema": _build_env_option_schema(),
                "notes": {
                    "base_reset": "Changing base config resets and rebuilds the live session.",
                    "env_reset": "Changing advanced env controls resets and rebuilds the live session.",
                    "action_semantics": "Each cluster head chooses MAC mode and rho per burst; the network does not switch MAC globally.",
                },
            },
        }

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.base_params["SEED"] = int(seed)
        if options:
            self.env_reset_options.update(copy.deepcopy(options))

        self._apply_effective_params_to_globals()
        descriptor = self._current_policy_descriptor()
        session_kind = self._session_kind_for_policy(descriptor)
        env_options = self._effective_env_options()
        env_seed = int(self._effective_base_params().get("SEED", getattr(params, "SEED", 42)))

        self.sarl_env = None
        self.marl_env = None
        self._session_kind = session_kind
        self._session_done = False
        self.last_infos = {}
        self.tick_count = 0
        self.sim_time = 0.0
        self.history = []
        self.cluster_step_history = []
        self.graph_edge_history = []
        self.membership_history = []
        self._prev_summary_counts = {"splits": 0, "merges": 0, "reassociations": 0, "handovers": 0, "failure_events": 0}

        if session_kind == "sarl":
            self.sarl_env = SARLCentralEnv(seed=env_seed)
            obs, _ = self.sarl_env.reset(seed=env_seed, options=env_options)
            self.marl_env = self.sarl_env.marl_env
            self._session_obs = obs
        else:
            self.marl_env = MARLMacEnv(seed=env_seed)
            obs, infos = self.marl_env.reset(seed=env_seed, options=env_options)
            self._session_obs = obs
            self.last_infos = infos

        self._reset_policy_runtime()
        self._prev_leader_map = self._leader_map()

        snapshot = self._build_snapshot(initial=True)
        self.last_snapshot = snapshot
        self._record_snapshot(snapshot, include_step_records=False)
        return snapshot

    def tick(self) -> dict[str, Any]:
        if self.marl_env is None:
            return self.reset()
        if self._session_done:
            return self.reset()

        descriptor = self._current_policy_descriptor()

        try:
            if self._session_kind == "sarl":
                action = self._select_sarl_action(descriptor)
                next_obs, _reward, terminated, truncated, info = self.sarl_env.step(action)  # type: ignore[union-attr]
                self._session_obs = next_obs
                self.last_infos = info.get("raw_infos", {})
                self._session_done = bool(terminated or truncated)
            else:
                actions = self._select_marl_actions(descriptor)
                next_obs, _rewards, _terms, _truncs, infos = self.marl_env.step(actions)
                self._session_obs = next_obs
                self.last_infos = infos
                self._session_done = not bool(self.marl_env.agents)
        except Exception:
            exc = sys.exc_info()[1]
            if descriptor["id"] != "fixed:all_tdma_mid":
                reason = str(exc) if exc is not None else "runtime failure during policy evaluation"
                self._mark_policy_unavailable(descriptor["id"], reason)
                fallback_id = choose_default_policy_id(self._policy_registry, exclude_ids={descriptor["id"]})
                self.runtime["policy_id"] = fallback_id
                self._session_kind = "marl"
                self._session_done = False
                self._set_policy_warning(f"{descriptor['label']} failed in the current UI session and was replaced by {self._policy_map[fallback_id]['label']}: {reason}")
                return self.reset()
            raise

        self.tick_count = int(self.marl_env.current_step)
        self.sim_time = round(self.tick_count * CC.BURST_TOTAL_TIME, 4)
        snapshot = self._build_snapshot(initial=False)
        self.last_snapshot = snapshot
        self._record_snapshot(snapshot, include_step_records=True)
        return snapshot

    def set_base_param(self, key: str, value: Any) -> dict[str, Any]:
        if key not in BASE_PARAM_KEYS:
            raise KeyError(f"Unsupported base param: {key}")
        old_value = self.base_params.get(key)
        if key in BOOLEAN_BASE_PARAMS:
            value = _coerce_value(value, as_bool=True)
        elif key in FLOAT_BASE_PARAMS:
            value = _coerce_value(value, as_float=True)
        elif key in INT_BASE_PARAMS:
            value = _coerce_value(value, as_int=True)
        self.base_params[key] = value
        entry = self._log_change(target="base_params", key=key, old_value=old_value, new_value=value)
        if key == "N":
            self._rebuild_preset_registry_if_needed()
        self.reset()
        return entry

    def set_env_option(self, key: str, value: Any) -> dict[str, Any]:
        if key not in ENV_OPTION_KEYS:
            raise KeyError(f"Unsupported env option: {key}")
        old_value = self.env_reset_options.get(key)
        if value is None or value == "__inherit__" or (isinstance(value, str) and not value.strip()):
            self.env_reset_options.pop(key, None)
            entry = self._log_change(target="env_reset_options", key=key, old_value=old_value, new_value=None)
            self.reset()
            return entry
        if key == "failure_schedule":
            value = self._normalize_failure_schedule(value)
        if key in BOOLEAN_ENV_OPTIONS:
            value = _coerce_value(value, as_bool=True)
        elif key in FLOAT_ENV_OPTIONS:
            value = _coerce_value(value, as_float=True)
        elif key in INT_ENV_OPTIONS:
            value = _coerce_value(value, as_int=True)
        self.env_reset_options[key] = copy.deepcopy(value)
        entry = self._log_change(target="env_reset_options", key=key, old_value=old_value, new_value=value)
        self.reset()
        return entry

    def set_runtime_value(self, key: str, value: Any) -> dict[str, Any]:
        old_value = self.runtime.get(key)
        if key == "policy_id":
            return self.select_policy(str(value))
        if key == "deterministic":
            value = _coerce_value(value, as_bool=True)
        elif key == "speed_factor":
            value = max(0.1, float(value))
        self.runtime[key] = value
        if key == "deterministic":
            self._reset_policy_runtime()
        if self.marl_env is not None:
            self.last_snapshot = self._build_snapshot(initial=True)
        return self._log_change(target="runtime", key=key, old_value=old_value, new_value=value, mode="live")

    def select_preset(self, preset_id: str) -> dict[str, Any]:
        if preset_id not in self._preset_map:
            raise KeyError(f"Unknown preset: {preset_id}")
        old_value = self.selected_preset_id
        self.selected_preset_id = preset_id
        entry = self._log_change(target="preset", key="selected_preset_id", old_value=old_value, new_value=preset_id)
        self.reset()
        return entry

    def select_policy(self, policy_id: str) -> dict[str, Any]:
        if policy_id not in self._policy_map:
            raise KeyError(f"Unknown policy: {policy_id}")
        old_value = self.runtime.get("policy_id")
        old_kind = self._session_kind_for_policy(self._current_policy_descriptor())
        self.runtime["policy_id"] = policy_id
        self._set_policy_warning(None)
        new_kind = self._session_kind_for_policy(self._current_policy_descriptor())
        entry = self._log_change(target="runtime", key="policy_id", old_value=old_value, new_value=policy_id, mode="live")
        self._reset_policy_runtime()
        if self.marl_env is None or new_kind != old_kind:
            self.reset()
        else:
            self.last_snapshot = self._build_snapshot(initial=True)
        return entry

    def update_config(self, key: str, value: Any, mode: str = "restart") -> dict[str, Any]:
        if key in BASE_PARAM_KEYS:
            return self.set_base_param(key, value)
        if key in ENV_OPTION_KEYS:
            return self.set_env_option(key, value)
        raise KeyError(f"Unsupported config key: {key}")

    def apply_pending_changes(self):
        return self.reset()

    # ------------------------------------------------------------------
    # Policy loading / action selection
    # ------------------------------------------------------------------
    def _reset_policy_runtime(self) -> None:
        descriptor = self._current_policy_descriptor()
        cached = self._policy_cache.get(descriptor["id"])
        if cached is None:
            return
        model_type, model = cached
        if model_type == "marl_gnn" and hasattr(model, "reset_memory"):
            model.reset_memory()
        if model_type == "marl" and self.runtime.get("deterministic") and hasattr(model, "eps_decay"):
            model.steps_done = int(model.eps_decay) * 10

    def _ensure_policy_loaded(self, descriptor: dict[str, Any]) -> tuple[str, Any] | None:
        if descriptor["source"] == "fixed":
            return None
        cached = self._policy_cache.get(descriptor["id"])
        if cached is not None:
            return cached

        from experiments.run_unified_experiment import load_eval_models

        models = load_eval_models(descriptor["checkpoint_dir"], _silent_log)
        selected = models.get(descriptor["label"])
        if selected is None:
            raise RuntimeError(f"Could not load policy {descriptor['label']} from {descriptor['checkpoint_dir']}")
        self._policy_cache[descriptor["id"]] = selected
        self._reset_policy_runtime()
        return selected

    def _alive_mask(self) -> np.ndarray:
        if self.marl_env is None:
            return np.zeros(CC.C_MAX, dtype=np.bool_)
        return np.array(
            [1 if self.marl_env.cluster_manager.get_members(i) else 0 for i in range(CC.C_MAX)],
            dtype=np.bool_,
        )

    def _select_marl_actions(self, descriptor: dict[str, Any]) -> dict[str, int]:
        assert self.marl_env is not None
        if descriptor["source"] == "fixed":
            action_id = int(descriptor["fixed_action"])
            return {agent: action_id for agent in self.marl_env.possible_agents}

        loaded = self._ensure_policy_loaded(descriptor)
        assert loaded is not None
        model_type, model = loaded

        if model_type == "marl":
            obs_all = np.stack([self._session_obs[a] for a in self.marl_env.possible_agents])
            alive_mask = self._alive_mask()
            if self.runtime.get("deterministic") and hasattr(model, "eps_decay"):
                model.steps_done = int(model.eps_decay) * 10
            actions = model.select_actions(obs_all, alive_mask=alive_mask)
            return {agent: int(actions[idx]) for idx, agent in enumerate(self.marl_env.possible_agents)}

        if model_type == "marl_gnn":
            import torch

            x, edge_index, alive_mask = self.marl_env.get_global_graph_state()
            x_t = torch.tensor(x, dtype=torch.float32)
            e_t = torch.tensor(edge_index, dtype=torch.long)
            alive_t = torch.tensor(alive_mask, dtype=torch.bool)
            with torch.no_grad():
                q_vals = model(x_t, e_t, alive_t)
            return {
                agent: int(q_vals[idx].argmax().item()) if alive_mask[idx] else 0
                for idx, agent in enumerate(self.marl_env.possible_agents)
            }

        raise RuntimeError(f"Unsupported MARL model type: {model_type}")

    def _select_sarl_action(self, descriptor: dict[str, Any]) -> np.ndarray:
        if descriptor["source"] == "fixed":
            action_id = int(descriptor["fixed_action"])
            return np.full(CC.C_MAX, action_id, dtype=np.int64)

        loaded = self._ensure_policy_loaded(descriptor)
        assert loaded is not None
        _model_type, model = loaded
        action, _ = model.predict(self._session_obs, deterministic=bool(self.runtime.get("deterministic", True)))
        return np.asarray(action, dtype=np.int64)

    # ------------------------------------------------------------------
    # Snapshot builders
    # ------------------------------------------------------------------
    def _leader_map(self) -> dict[int, int]:
        assert self.marl_env is not None
        cm = self.marl_env.cluster_manager
        return {cid: int(cm.get_leader(cid)) for cid in cm.get_active_cluster_ids()}

    def _degree_map(self, active_cids: list[int], edge_index: np.ndarray) -> dict[int, int]:
        degree = {int(cid): 0 for cid in active_cids}
        if edge_index.size == 0:
            return degree
        for src_local, _dst_local in edge_index.T:
            if src_local >= len(active_cids):
                continue
            degree[int(active_cids[src_local])] += 1
        return degree

    def _build_metrics(self) -> dict[str, Any]:
        assert self.marl_env is not None
        summary = self.marl_env.get_last_step_summary()
        diagnostics = self.marl_env.cluster_manager.get_diagnostics()
        if not summary:
            summary = {
                "step": int(self.marl_env.current_step),
                "offered_pps": float(self.marl_env.current_offered_pps),
                "throughput_mbps": 0.0,
                "drops": 0.0,
                "collisions": 0.0,
                "coord_success": 0.0,
                "num_clusters": int(diagnostics["num_clusters"]),
                "avg_cluster_size": float(diagnostics["mean_cluster_size"]),
                "cluster_size_std": float(diagnostics["cluster_size_std"]),
                "avg_graph_degree": float(diagnostics["avg_graph_degree"]),
                "graph_density": float(diagnostics["graph_density"]),
                "reassociations": int(diagnostics["reassociations"]),
                "splits": int(diagnostics["splits"]),
                "merges": int(diagnostics["merges"]),
                "handovers": int(diagnostics["handovers"]),
                "failure_events": 0,
            }

        cluster_records = self.marl_env.get_last_step_records()
        summary = dict(summary)
        if cluster_records:
            summary["avg_local_backlog"] = float(np.mean([row["local_backlog"] for row in cluster_records]))
            summary["avg_inter_backlog"] = float(np.mean([row["inter_backlog"] for row in cluster_records]))
        else:
            summary["avg_local_backlog"] = 0.0
            summary["avg_inter_backlog"] = 0.0
        summary["decision_overhead_bytes"] = float(self.marl_env.estimate_neighbor_summary_overhead_bytes())
        return summary

    def _mapped_edges(self, edge_index: np.ndarray, active_cids: list[int]) -> list[dict[str, int]]:
        unique_pairs = set()
        rows: list[dict[str, int]] = []
        if edge_index.size == 0:
            return rows
        for src_local, dst_local in edge_index.T:
            if src_local >= len(active_cids) or dst_local >= len(active_cids):
                continue
            src = int(active_cids[src_local])
            dst = int(active_cids[dst_local])
            pair = tuple(sorted((src, dst)))
            if src == dst or pair in unique_pairs:
                continue
            unique_pairs.add(pair)
            rows.append({"source": src, "target": dst})
        return rows

    def _build_nodes(self) -> list[dict[str, Any]]:
        assert self.marl_env is not None
        cm = self.marl_env.cluster_manager
        positions = np.asarray(self.marl_env.mobility_model.positions, dtype=np.float64)
        velocities = np.asarray(self.marl_env.mobility_model.velocities, dtype=np.float64)
        active_ids = set(cm.get_active_cluster_ids())
        nodes = []
        for idx in range(int(getattr(params, "N", positions.shape[0]))):
            cluster_id = int(cm.assignment[idx]) if idx < len(cm.assignment) else -1
            leader_id = int(cm.get_leader(cluster_id)) if cluster_id >= 0 else -1
            speed = float(np.linalg.norm(velocities[idx]))
            role_label = "leader" if idx == leader_id else ("member" if cluster_id >= 0 else "inactive")
            nodes.append(
                {
                    "id": int(idx),
                    "position": positions[idx].astype(float).tolist(),
                    "velocity": velocities[idx].astype(float).tolist(),
                    "speed": speed,
                    "cluster_id": cluster_id,
                    "leader": bool(idx == leader_id),
                    "role_label": role_label,
                    "queue": float(self.marl_env.simulated_queues[idx]),
                    "energy": float(cm.energy[idx]),
                    "active_cluster": bool(cluster_id in active_ids),
                }
            )
        return nodes

    def _cluster_health_inputs(
        self,
        *,
        cluster_state: Any,
        graph_degree: int,
        active_cluster_count: int,
    ) -> dict[str, float]:
        assert self.marl_env is not None
        cm = self.marl_env.cluster_manager
        leader = int(cluster_state.leader_idx)
        velocities = np.asarray(self.marl_env.mobility_model.velocities, dtype=np.float64)

        energy_norm = float(cm.energy[leader] / CC.E_INIT)
        leader_speed = float(np.linalg.norm(velocities[leader]))
        mobility_stability = float(1.0 / (1.0 + leader_speed / 30.0))
        queue_norm = float(self.marl_env.simulated_queues[leader] / 100.0)
        degree_norm = float(graph_degree / max(active_cluster_count - 1, 1))
        risk = float(max(1.0 - energy_norm, 0.0))

        return {
            "energy_norm": energy_norm,
            "degree_norm": degree_norm,
            "mobility_stability": mobility_stability,
            "queue_norm": queue_norm,
            "risk": risk,
            "leader_speed": leader_speed,
            "leader_queue": float(self.marl_env.simulated_queues[leader]),
            "leader_energy": float(cm.energy[leader]),
        }

    def _build_clusters(self) -> list[dict[str, Any]]:
        assert self.marl_env is not None
        cm = self.marl_env.cluster_manager
        active_ids = cm.get_active_cluster_ids()
        degree_map = self._degree_map(list(self.marl_env.true_active_cids), self.marl_env.true_edge_index)
        failures = {int(event["cluster_id"]) for event in getattr(self.marl_env, "failure_events_triggered", [])}
        clusters = []
        for cid in active_ids:
            cs = cm.clusters[cid]
            info = self.last_infos.get(f"cluster_{cid}", {})
            chosen_mac = info.get("chosen_mac")
            graph_degree = int(info.get("graph_degree", degree_map.get(cid, 0)))
            health_inputs = self._cluster_health_inputs(
                cluster_state=cs,
                graph_degree=graph_degree,
                active_cluster_count=len(active_ids),
            )
            clusters.append(
                {
                    "cluster_id": int(cid),
                    "leader_id": int(cs.leader_idx),
                    "members": [int(member) for member in cs.member_indices],
                    "cluster_size": int(len(cs.member_indices)),
                    "chosen_mac": int(chosen_mac) if chosen_mac is not None else None,
                    "mac_label": "TDMA" if chosen_mac == 0 else ("CSMA_CA" if chosen_mac == 1 else "N/A"),
                    "rho": float(info.get("rho", cs.recent_rho)),
                    "t1_time": float(info.get("t1_time", 0.0)),
                    "t2_time": float(info.get("t2_time", 0.0)),
                    "throughput_mbps": float(info.get("throughput_mbps", 0.0)),
                    "inter_throughput_mbps": float(info.get("inter_throughput_mbps", 0.0)),
                    "delay_ms": float(info.get("delay_ms", cs.agg_delay)),
                    "inter_delay_ms": float(info.get("inter_delay_ms", 0.0)),
                    "local_backlog": float(info.get("local_backlog", cs.agg_queue)),
                    "inter_backlog": float(info.get("inter_backlog", cs.coord_backlog + cs.relay_demand)),
                    "graph_degree": graph_degree,
                    "leader_health": float(info.get("leader_health", 0.0)),
                    "health_inputs": health_inputs,
                    "handover_flag": bool(info.get("handover_flag", cs.handover_flag)),
                    "failure_flag": bool(cid in failures),
                    "recent_t1_util": float(cs.recent_t1_util),
                    "recent_t2_util": float(cs.recent_t2_util),
                    "recent_coord_success": float(cs.recent_coord_success),
                }
            )
        return clusters

    def _build_events(self, summary: dict[str, Any]) -> dict[str, Any]:
        current_leaders = self._leader_map()
        changed_clusters = [
            cid
            for cid, leader_id in current_leaders.items()
            if cid in self._prev_leader_map and self._prev_leader_map[cid] != leader_id
        ]
        failures = [copy.deepcopy(event) for event in getattr(self.marl_env, "failure_events_triggered", [])]

        counts = {
            "splits": max(int(summary.get("splits", 0)) - self._prev_summary_counts["splits"], 0),
            "merges": max(int(summary.get("merges", 0)) - self._prev_summary_counts["merges"], 0),
            "reassociations": max(int(summary.get("reassociations", 0)) - self._prev_summary_counts["reassociations"], 0),
            "handovers": max(int(summary.get("handovers", 0)) - self._prev_summary_counts["handovers"], 0),
            "failure_events": max(int(summary.get("failure_events", 0)) - self._prev_summary_counts["failure_events"], 0),
        }
        tick = int(self.marl_env.current_step)
        sim_time = round(float(tick) * CC.BURST_TOTAL_TIME, 4)
        feed: list[dict[str, Any]] = []
        for cid in changed_clusters:
            feed.append(
                {
                    "id": f"{tick}:handover:{cid}",
                    "tick": tick,
                    "sim_time": sim_time,
                    "kind": "handover",
                    "severity": "warning",
                    "cluster_id": int(cid),
                    "message": f"Cluster {cid} changed leader in this burst.",
                }
            )
        for failure in failures:
            feed.append(
                {
                    "id": f"{tick}:failure:{failure.get('cluster_id', 'na')}",
                    "tick": tick,
                    "sim_time": sim_time,
                    "kind": "failure",
                    "severity": "danger",
                    "cluster_id": int(failure.get("cluster_id", -1)),
                    "message": f"Failure triggered at cluster {int(failure.get('cluster_id', -1))} via {failure.get('mode', 'explicit')}.",
                }
            )
        for key, label in (("splits", "split"), ("merges", "merge"), ("reassociations", "reassociation")):
            count = counts[key]
            if count > 0:
                suffix = "" if count == 1 else "s"
                feed.append(
                    {
                        "id": f"{tick}:{key}:{count}",
                        "tick": tick,
                        "sim_time": sim_time,
                        "kind": key,
                        "severity": "info",
                        "message": f"{count} {label}{suffix} recorded in this burst.",
                    }
                )

        self._prev_summary_counts = {
            "splits": int(summary.get("splits", 0)),
            "merges": int(summary.get("merges", 0)),
            "reassociations": int(summary.get("reassociations", 0)),
            "handovers": int(summary.get("handovers", 0)),
            "failure_events": int(summary.get("failure_events", 0)),
        }
        self._prev_leader_map = current_leaders

        return {
            "counts": counts,
            "handover_clusters": [int(cid) for cid in changed_clusters],
            "failure_events": failures,
            "feed": feed,
        }

    def _build_snapshot(self, initial: bool = False) -> dict[str, Any]:
        assert self.marl_env is not None
        effective = self._effective_base_params()
        metrics = self._build_metrics()
        clusters = self._build_clusters()
        nodes = self._build_nodes()
        scenario_context = self._build_scenario_context()
        true_active = list(self.marl_env.true_active_cids)
        obs_active = list(self.marl_env.obs_active_cids)
        true_edges = self._mapped_edges(self.marl_env.true_edge_index, true_active)
        observed_edges = self._mapped_edges(self.marl_env.obs_edge_index, obs_active)
        graph_stats = self._build_graph_stats(
            true_edges=true_edges,
            observed_edges=observed_edges,
            cluster_count=len(clusters),
        )
        events = {
            "counts": {"splits": 0, "merges": 0, "reassociations": 0, "handovers": 0, "failure_events": 0},
            "handover_clusters": [],
            "failure_events": [],
            "feed": [],
        }
        if not initial:
            events = self._build_events(metrics)

        descriptor = self._current_policy_descriptor()
        return {
            "schema_version": "marl_ui_v2",
            "tick": int(self.marl_env.current_step),
            "sim_time": round(float(self.marl_env.current_step) * CC.BURST_TOTAL_TIME, 4),
            "N": int(effective.get("N", getattr(params, "N", len(nodes)))),
            "bounds": [float(effective["AREA_X"]), float(effective["AREA_Y"]), float(effective["AREA_Z"])],
            "nodes": nodes,
            "clusters": clusters,
            "graphs": {
                "true_edges": true_edges,
                "observed_edges": observed_edges,
                "stats": graph_stats,
            },
            "metrics": metrics,
            "runtime": {
                "policy_id": descriptor["id"],
                "policy_label": descriptor["label"],
                "policy_type": descriptor["model_type"],
                "policy_source": descriptor["source"],
                "policy_available": bool(descriptor.get("available", True)),
                "policy_compatible": bool(descriptor.get("compatible", True)),
                "policy_compatibility_note": descriptor.get("compatibility_note"),
                "policy_warning": self.runtime_status.get("policy_warning"),
                "deterministic": bool(self.runtime.get("deterministic", True)),
                "speed_factor": float(self.runtime.get("speed_factor", 1.0)),
            },
            "scenario": {
                "preset_id": self.selected_preset_id,
                "preset_label": self._current_preset()["label"],
                "preset_group": self._current_preset()["group_label"],
                "study_block": self._current_preset().get("study_block"),
                "preset_description": self._current_preset().get("description"),
                "param_overrides": copy.deepcopy(self._current_preset()["param_overrides"]),
                "preset_env_options": copy.deepcopy(self._current_preset()["env_options"]),
                "manual_env_options": copy.deepcopy(self.env_reset_options),
                "env_options": copy.deepcopy(self._effective_env_options()),
                "context_items": scenario_context["context_items"],
                "summary_lines": scenario_context["summary_lines"],
                "impairment_lines": scenario_context["impairment_lines"],
            },
            "events": events,
        }

    def _record_snapshot(self, snapshot: dict[str, Any], *, include_step_records: bool) -> None:
        tick = int(snapshot["tick"])
        sim_time = float(snapshot["sim_time"])
        self.history.append(copy.deepcopy(snapshot))

        for node in snapshot["nodes"]:
            self.membership_history.append(
                {
                    "tick": tick,
                    "timestamp": sim_time,
                    "uav_id": int(node["id"]),
                    "cluster_id": int(node["cluster_id"]),
                    "is_leader": int(bool(node["leader"])),
                }
            )

        for edge_kind, edges in (("true", snapshot["graphs"]["true_edges"]), ("observed", snapshot["graphs"]["observed_edges"])):
            for edge in edges:
                self.graph_edge_history.append(
                    {
                        "tick": tick,
                        "timestamp": sim_time,
                        "edge_kind": edge_kind,
                        "source_cluster_id": int(edge["source"]),
                        "target_cluster_id": int(edge["target"]),
                    }
                )

        if include_step_records and self.marl_env is not None:
            self.cluster_step_history.extend(self.marl_env.get_last_step_records())

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(self, base_results_dir: str | None = None) -> str:
        if base_results_dir is None:
            base_results_dir = self.results_root

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        effective = self._effective_base_params()
        descriptor = self._current_policy_descriptor()
        run_dir = os.path.join(
            base_results_dir,
            f"UI_MARL_N{int(effective['N'])}_{_slugify(descriptor['label'])}",
            f"trial_{ts}",
        )
        csv_dir = os.path.join(run_dir, "csv")
        img_dir = os.path.join(run_dir, "images")
        log_dir = os.path.join(run_dir, "logs")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        metadata = {
            "timestamp": dt.datetime.now().isoformat(),
            "source": "simulator_ui_live_marl",
            "ticks_run": int(self.tick_count),
            "sim_time_s": float(self.sim_time),
            "config": self.get_config(),
            "selected_policy": descriptor,
            "selected_preset": self._current_preset(),
        }
        with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)

        rows_pos: list[dict[str, Any]] = []
        for snapshot in self.history:
            for node in snapshot["nodes"]:
                velocity = np.asarray(node["velocity"], dtype=np.float64)
                position = node["position"]
                rows_pos.append(
                    {
                        "timestamp": round(float(snapshot["sim_time"]), 6),
                        "uav_id": int(node["id"]),
                        "x": round(float(position[0]), 4),
                        "y": round(float(position[1]), 4),
                        "z": round(float(position[2]), 4),
                        "vx": round(float(velocity[0]), 4),
                        "vy": round(float(velocity[1]), 4),
                        "vz": round(float(velocity[2]), 4),
                        "speed": round(float(np.linalg.norm(velocity)), 4),
                    }
                )
        pd.DataFrame(rows_pos).to_csv(os.path.join(csv_dir, "mobility_positions.csv"), index=False)

        pd.DataFrame(self.cluster_step_history if self.cluster_step_history else []).to_csv(
            os.path.join(csv_dir, "cluster_step_records.csv"),
            index=False,
        )
        pd.DataFrame(self.graph_edge_history).to_csv(os.path.join(csv_dir, "cluster_graph_edges.csv"), index=False)
        pd.DataFrame(self.membership_history).to_csv(os.path.join(csv_dir, "node_cluster_membership.csv"), index=False)

        if self.config_change_log:
            pd.DataFrame(self.config_change_log).to_csv(os.path.join(csv_dir, "config_changes.csv"), index=False)

        return run_dir


__all__ = [
    "BASE_PARAM_KEYS",
    "ENV_OPTION_KEYS",
    "SimulationEngine",
    "build_preset_registry",
    "choose_default_policy_id",
    "discover_policy_descriptors",
]

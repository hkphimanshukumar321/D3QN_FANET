"""
Run the unified experiment from environment variables and dump artifact paths.
"""

from __future__ import annotations

import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from experiments.run_unified_experiment import run_unified_experiment


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() == "1"


def main():
    kwargs = {
        "dry_run": env_flag("DRY_RUN"),
        "force_retrain": env_flag("FORCE_RETRAIN"),
        "use_checkpoints_only": env_flag("SKIP_TRAINING"),
        "stochastic_eval": False,
    }

    if os.environ.get("PHY_RATE_MBPS"):
        kwargs["phy_rate_mbps"] = float(os.environ["PHY_RATE_MBPS"])
    if os.environ.get("NODES"):
        kwargs["nodes"] = int(os.environ["NODES"])
    if os.environ.get("QMAX"):
        kwargs["qmax"] = int(os.environ["QMAX"])
    if os.environ.get("SWEEP_MIN_PPS"):
        kwargs["sweep_min_pps"] = int(os.environ["SWEEP_MIN_PPS"])
    if os.environ.get("SWEEP_MAX_PPS"):
        kwargs["sweep_max_pps"] = int(os.environ["SWEEP_MAX_PPS"])
    if os.environ.get("SWEEP_STEPS"):
        kwargs["sweep_steps"] = int(os.environ["SWEEP_STEPS"])

    artifacts = run_unified_experiment(**kwargs)
    artifact_json = os.path.abspath(os.environ["ARTIFACT_JSON"])
    with open(artifact_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                **artifacts,
                "execution_mode": {
                    "marl_env": "MARLMacEnv",
                    "marl_control": "decentralized_cluster_head",
                    "sarl_env": "SARLCentralEnv",
                    "burst_split_action": "(m_k, rho_k)",
                },
            },
            f,
            indent=2,
        )
    print(artifact_json)


if __name__ == "__main__":
    main()

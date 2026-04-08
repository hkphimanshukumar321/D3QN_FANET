"""
MAGAT isolation ablation runner.

This entrypoint retrains/evaluates the key MAGAT variants under the already
implemented decentralized cluster-head environment.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from contextlib import contextmanager

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from configs import config as params
from configs.marl_config import MARLConfig
from experiments.run_unified_experiment import run_unified_experiment


@contextmanager
def temporary_attrs(target, **overrides):
    snapshot = {key: getattr(target, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(target, key, value)
        yield
    finally:
        for key, value in snapshot.items():
            setattr(target, key, value)


def build_variants():
    return [
        {
            "name": "full_magat",
            "marl_overrides": {
                "MAGAT_USE_GRAPH": True,
                "MAGAT_USE_ATTENTION": True,
                "MAGAT_USE_GRU": True,
                "MAGAT_USE_BURST_HISTORY": True,
            },
            "param_overrides": {"GRAPH_MODE": "dynamic"},
        },
        {
            "name": "no_graph_edges",
            "marl_overrides": {
                "MAGAT_USE_GRAPH": False,
                "MAGAT_USE_ATTENTION": False,
                "MAGAT_USE_GRU": True,
                "MAGAT_USE_BURST_HISTORY": True,
            },
            "param_overrides": {"GRAPH_MODE": "none"},
        },
        {
            "name": "graph_no_attention",
            "marl_overrides": {
                "MAGAT_USE_GRAPH": True,
                "MAGAT_USE_ATTENTION": False,
                "MAGAT_USE_GRU": True,
                "MAGAT_USE_BURST_HISTORY": True,
            },
            "param_overrides": {"GRAPH_MODE": "dynamic"},
        },
        {
            "name": "no_gru",
            "marl_overrides": {
                "MAGAT_USE_GRAPH": True,
                "MAGAT_USE_ATTENTION": True,
                "MAGAT_USE_GRU": False,
                "MAGAT_USE_BURST_HISTORY": True,
            },
            "param_overrides": {"GRAPH_MODE": "dynamic"},
        },
        {
            "name": "no_burst_history",
            "marl_overrides": {
                "MAGAT_USE_GRAPH": True,
                "MAGAT_USE_ATTENTION": True,
                "MAGAT_USE_GRU": True,
                "MAGAT_USE_BURST_HISTORY": False,
            },
            "param_overrides": {"GRAPH_MODE": "dynamic"},
        },
        {
            "name": "static_graph_only",
            "marl_overrides": {
                "MAGAT_USE_GRAPH": True,
                "MAGAT_USE_ATTENTION": True,
                "MAGAT_USE_GRU": True,
                "MAGAT_USE_BURST_HISTORY": True,
            },
            "param_overrides": {"GRAPH_MODE": "static"},
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="MAGAT isolation ablation suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--phy-rate-mbps", type=float, default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--qmax", type=int, default=None)
    parser.add_argument("--sweep-min-pps", type=int, default=None)
    parser.add_argument("--sweep-max-pps", type=int, default=None)
    parser.add_argument("--sweep-steps", type=int, default=None)
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = os.path.join(project_root, "results", "journal_magat_isolation", ts)
    os.makedirs(suite_dir, exist_ok=True)

    manifest = []
    for variant in build_variants():
        with temporary_attrs(MARLConfig, **variant["marl_overrides"]):
            with temporary_attrs(params, **variant["param_overrides"]):
                artifact = run_unified_experiment(
                    dry_run=args.dry_run,
                    force_retrain=args.force_retrain,
                    use_checkpoints_only=args.skip_training,
                    phy_rate_mbps=args.phy_rate_mbps,
                    nodes=args.nodes,
                    qmax=args.qmax,
                    sweep_min_pps=args.sweep_min_pps,
                    sweep_max_pps=args.sweep_max_pps,
                    sweep_steps=args.sweep_steps,
                    stochastic_eval=False,
                )
        manifest.append(
            {
                "variant": variant["name"],
                "marl_overrides": variant["marl_overrides"],
                "param_overrides": variant["param_overrides"],
                "artifact": artifact,
            }
        )

    with open(os.path.join(suite_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()

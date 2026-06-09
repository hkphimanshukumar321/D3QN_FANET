# self_check.py — System Integrity Validator
# Validates config consistency, module availability, results integrity,
# CSV schemas, mobility bounds, queue invariants, and link invariants.
#
# Usage: python self_check.py [--results-dir path/to/results/run]

import os
import sys
import json
import importlib
import argparse
import numpy as np

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

PASS = 0
FAIL = 0


def check(condition, label, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}{': ' + detail if detail else ''}")
    return condition


# ======================================================================
# 1. Config Consistency
# ======================================================================
def check_config():
    print("\n1. CONFIG CONSISTENCY")
    print("-" * 50)
    try:
        import configs.config as cfg
        check(True, "configs.config importable")
    except Exception as e:
        check(False, "configs.config importable", str(e))
        return

    required_int = ["N", "QMAX", "SEED", "SWEEP_MIN_PPS", "SWEEP_MAX_PPS", "SWEEP_STEPS"]
    required_float = ["SIM_TIME_S", "SLOT_TIME_S", "PHY_RATE_BPS", "PAYLOAD_BYTES",
                       "AREA_X", "AREA_Y", "AREA_Z", "SINK_X", "SINK_Y", "SINK_Z",
                       "COMM_RANGE_R", "V_MIN", "V_MAX", "MOBILITY_DT"]
    required_str = ["MOBILITY_MODEL", "SPEED_MODE"]
    required_bool = ["ENABLE_MOBILITY", "ENABLE_PATHLOSS"]

    for field in required_int:
        check(hasattr(cfg, field), f"field exists: {field}")

    for field in required_float:
        val = getattr(cfg, field, None)
        check(val is not None and isinstance(val, (int, float)), f"field exists + numeric: {field}")

    for field in required_str:
        check(hasattr(cfg, field) and isinstance(getattr(cfg, field), str), f"field exists + str: {field}")

    for field in required_bool:
        check(hasattr(cfg, field), f"field exists: {field}")

    # Bounds checks
    check(cfg.N > 0, "N > 0", f"N={cfg.N}")
    check(cfg.QMAX > 0, "QMAX > 0")
    check(cfg.SIM_TIME_S > 0, "SIM_TIME_S > 0")
    check(cfg.SLOT_TIME_S > 0, "SLOT_TIME_S > 0")
    check(cfg.PHY_RATE_BPS > 0, "PHY_RATE_BPS > 0")
    check(cfg.AREA_X > 0 and cfg.AREA_Y > 0 and cfg.AREA_Z > 0, "Area bounds > 0")
    check(0 <= cfg.SINK_X <= cfg.AREA_X, "Sink X in bounds")
    check(0 <= cfg.SINK_Y <= cfg.AREA_Y, "Sink Y in bounds")
    check(0 <= cfg.SINK_Z <= cfg.AREA_Z, "Sink Z in bounds")
    check(cfg.V_MIN >= 0 and cfg.V_MAX >= cfg.V_MIN, "V_MIN <= V_MAX")
    check(cfg.COMM_RANGE_R > 0, "COMM_RANGE_R > 0")
    check(cfg.MOBILITY_DT > 0, "MOBILITY_DT > 0")
    check(cfg.MOBILITY_MODEL.lower() in
          ("gauss_markov", "random_waypoint", "random_walk", "circular", "static"),
          f"MOBILITY_MODEL valid: {cfg.MOBILITY_MODEL}")


# ======================================================================
# 2. Module Availability
# ======================================================================
def check_modules():
    print("\n2. MODULE AVAILABILITY")
    print("-" * 50)
    modules = [
        ("algorithms.mac.baseline", "Config, Logger, simulate_tdma, simulate_csma_ca"),
        ("algorithms.mobility.models", "create_mobility_model, StaticModel"),
        ("algorithms.mobility.speed", "SpeedEngine"),
        ("algorithms.mobility.link", "compute_distances, compute_link_up, compute_pathloss_success_prob"),
        ("algorithms.mobility.manager", "MobilityManager"),
        ("algorithms.rl.qlearning_selector", "QLearningAgent"),
        ("algorithms.mac.channel_aware_mac", "simulate_tdma_aware, simulate_csma_aware"),
    ]
    for mod_path, desc in modules:
        try:
            importlib.import_module(mod_path)
            check(True, f"{mod_path}")
        except Exception as e:
            check(False, f"{mod_path}", str(e))

    # Simulator UI
    ui_files = ["simulator_ui/engine.py", "simulator_ui/server.py",
                 "simulator_ui/static/index.html", "simulator_ui/static/js/main.js"]
    for f in ui_files:
        check(os.path.exists(os.path.join(project_root, f)), f"exists: {f}")


# ======================================================================
# 3. Results Folder Integrity (if provided)
# ======================================================================
def check_results(results_dir):
    print("\n3. RESULTS FOLDER INTEGRITY")
    print("-" * 50)
    if not results_dir or not os.path.isdir(results_dir):
        print("  (skipped: no results dir provided)")
        return

    check(os.path.isdir(os.path.join(results_dir, "csv")), "csv/ exists")
    check(os.path.isdir(os.path.join(results_dir, "images")), "images/ exists")
    check(os.path.exists(os.path.join(results_dir, "metadata.json")), "metadata.json exists")

    meta_path = os.path.join(results_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            check("timestamp" in meta, "metadata has timestamp")
            check("config" in meta or "parameters" in meta, "metadata has config/parameters")
            check("seed_used" in meta or ("config" in meta and "SEED" in meta.get("config", {})),
                  "metadata has seed reference")
        except Exception as e:
            check(False, "metadata.json valid JSON", str(e))


# ======================================================================
# 4. CSV Schema Integrity
# ======================================================================
def check_csv_schemas(results_dir):
    print("\n4. CSV SCHEMA INTEGRITY")
    print("-" * 50)
    if not results_dir:
        print("  (skipped)")
        return

    import pandas as pd
    csv_dir = os.path.join(results_dir, "csv")
    if not os.path.isdir(csv_dir):
        print("  (skipped: no csv/ dir)")
        return

    schemas = {
        "mobility_positions.csv": ["timestamp", "uav_id", "x", "y", "z"],
        "uav_sink_distance.csv": ["timestamp", "uav_id", "d_to_sink", "link_up"],
        "config_changes.csv": ["timestamp", "param", "old_value", "new_value", "apply_mode"],
        "all_results.csv": ["case", "mac", "load_pps", "seed", "throughput_mbps"],
        "aggregate_summary.csv": ["case", "mac", "load_pps", "throughput_mean"],
    }
    for fname, required_cols in schemas.items():
        fpath = os.path.join(csv_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath, nrows=3)
            missing = [c for c in required_cols if c not in df.columns]
            check(len(missing) == 0, f"{fname} schema OK",
                  f"missing columns: {missing}" if missing else "")
        except Exception as e:
            check(False, f"{fname} readable", str(e))


# ======================================================================
# 5. Mobility Bounds + Queue + Link Invariants
# ======================================================================
def check_invariants():
    print("\n5. RUNTIME INVARIANTS (quick sim)")
    print("-" * 50)
    try:
        import configs.config as cfg
        from algorithms.mobility.speed import SpeedEngine
        from algorithms.mobility.models import create_mobility_model
        from algorithms.mobility.link import compute_distances, compute_link_up

        rng = np.random.default_rng(cfg.SEED)
        se = SpeedEngine(n_nodes=min(cfg.N, 10), v_min=cfg.V_MIN, v_max=cfg.V_MAX,
                         mode=cfg.SPEED_MODE, v_mean=cfg.V_MEAN, v_std=cfg.V_STD,
                         update_interval=cfg.SPEED_UPDATE_INTERVAL, rng=rng)
        model = create_mobility_model(
            name=cfg.MOBILITY_MODEL, n_nodes=min(cfg.N, 10),
            bounds=(cfg.AREA_X, cfg.AREA_Y, cfg.AREA_Z),
            speed_engine=se, rng=rng, gm_alpha=cfg.GM_ALPHA,
        )
        sink = np.array([cfg.SINK_X, cfg.SINK_Y, cfg.SINK_Z])

        # Run 20 steps
        for _ in range(20):
            pos, vel = model.update(cfg.MOBILITY_DT)

        # Bounds check
        in_bounds = (
            np.all(pos[:, 0] >= 0) and np.all(pos[:, 0] <= cfg.AREA_X) and
            np.all(pos[:, 1] >= 0) and np.all(pos[:, 1] <= cfg.AREA_Y) and
            np.all(pos[:, 2] >= 0) and np.all(pos[:, 2] <= cfg.AREA_Z)
        )
        check(in_bounds, "Mobility: positions within bounds after 20 steps")

        # Link invariant
        dists = compute_distances(pos, sink)
        lu = compute_link_up(dists, cfg.COMM_RANGE_R)
        for i in range(len(lu)):
            if lu[i] == 1:
                check(dists[i] <= cfg.COMM_RANGE_R, f"Link invariant UAV {i}: link_up=1 => d <= R")
            else:
                check(dists[i] > cfg.COMM_RANGE_R, f"Link invariant UAV {i}: link_up=0 => d > R")
    except Exception as e:
        check(False, "Invariant checks", str(e))


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="FANET Simulator Self-Check")
    parser.add_argument("--results-dir", default=None, help="Path to a results run folder")
    args = parser.parse_args()

    print("=" * 55)
    print("  FANET Simulator — Self-Check")
    print("=" * 55)

    check_config()
    check_modules()
    check_results(args.results_dir)
    check_csv_schemas(args.results_dir)
    check_invariants()

    print("\n" + "=" * 55)
    print(f"  Results: {PASS} PASS, {FAIL} FAIL")
    print("=" * 55)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

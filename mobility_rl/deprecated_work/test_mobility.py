# test_mobility.py — Standalone Validation Script for 3D UAV Mobility Module
# Run: python work/test_mobility.py
# Exits with code 0 if all tests pass, 1 on any failure.

import os
import sys
import numpy as np
import tempfile
import shutil

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from algorithms.mobility.speed import SpeedEngine
from algorithms.mobility.models import (
    GaussMarkov3D, RandomWaypoint3D, RandomWalk3D, CircularSpiral3D,
    create_mobility_model,
)
from algorithms.mobility.link import compute_distances, compute_link_up
from algorithms.mobility.manager import MobilityManager

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")


def test_speed_engine():
    print("\n=== Test 1: SpeedEngine — all modes ===")
    v_min, v_max = 5.0, 30.0
    n = 50
    dt = 0.5
    steps = 200

    for mode in ["uniform", "gaussian", "per_node_uniform", "piecewise"]:
        engine = SpeedEngine(n_nodes=n, v_min=v_min, v_max=v_max,
                             mode=mode, v_mean=15.0, v_std=5.0,
                             update_interval=2.0, rng=np.random.default_rng(42))
        all_speeds = []
        for _ in range(steps):
            s = engine.sample(dt)
            all_speeds.append(s.copy())

        all_speeds = np.concatenate(all_speeds)
        check(f"speed_mode={mode}: all >= V_MIN",
              np.all(all_speeds >= v_min - 1e-9),
              f"min={all_speeds.min():.4f}")
        check(f"speed_mode={mode}: all <= V_MAX",
              np.all(all_speeds <= v_max + 1e-9),
              f"max={all_speeds.max():.4f}")
        check(f"speed_mode={mode}: speed varies over time",
              np.std(all_speeds) > 0.01,
              f"std={np.std(all_speeds):.6f}")

    # Gaussian mean/std check
    engine = SpeedEngine(n_nodes=5000, v_min=v_min, v_max=v_max,
                         mode="gaussian", v_mean=15.0, v_std=5.0,
                         update_interval=0.1, rng=np.random.default_rng(7))
    samples = []
    for _ in range(100):
        samples.append(engine.sample(0.1))
    samples = np.concatenate(samples)
    mean_err = abs(samples.mean() - 15.0)
    check("gaussian: mean within 2.0 of V_MEAN",
          mean_err < 2.0, f"mean={samples.mean():.2f}, err={mean_err:.2f}")


def test_bounds_enforcement():
    print("\n=== Test 2: Bounds Enforcement — all 4 models ===")
    n = 20
    bounds = (500, 500, 200)
    v_min, v_max = 10.0, 50.0  # relatively fast to stress boundaries
    dt = 0.5
    steps = 1000

    for model_name in ["gauss_markov", "random_waypoint", "random_walk", "circular"]:
        rng = np.random.default_rng(123)
        speed_eng = SpeedEngine(n_nodes=n, v_min=v_min, v_max=v_max,
                                mode="uniform", update_interval=2.0, rng=rng)
        model = create_mobility_model(
            model_name, n, bounds, speed_eng, rng=rng,
            circ_radius=50.0, circ_omega_mean=0.2, circ_omega_std=0.05,
        )

        all_ok = True
        for step in range(steps):
            pos, vel = model.update(dt)
            if np.any(pos[:, 0] < -0.01) or np.any(pos[:, 0] > bounds[0] + 0.01):
                all_ok = False
                break
            if np.any(pos[:, 1] < -0.01) or np.any(pos[:, 1] > bounds[1] + 0.01):
                all_ok = False
                break
            if np.any(pos[:, 2] < -0.01) or np.any(pos[:, 2] > bounds[2] + 0.01):
                all_ok = False
                break

        check(f"model={model_name}: all positions in bounds after {steps} steps", all_ok)


def test_link_model():
    print("\n=== Test 3: Link Model — range-gated ===")
    sink = np.array([250.0, 250.0, 0.0])
    positions = np.array([
        [250, 250, 0],    # d=0   -> link_up
        [250, 250, 100],  # d=100 -> link_up
        [750, 250, 0],    # d=500 -> link_up (exactly at range)
        [751, 250, 0],    # d=501 -> link_down
        [0, 0, 200],      # d=~390 -> link_up
    ], dtype=float)
    comm_range = 500.0

    dists = compute_distances(positions, sink)
    link = compute_link_up(dists, comm_range)

    check("link_up[0] (d=0): up", link[0] == 1, f"link={link[0]}")
    check("link_up[1] (d=100): up", link[1] == 1, f"link={link[1]}")
    check("link_up[2] (d=500): up (boundary)", link[2] == 1, f"link={link[2]}")
    check("link_up[3] (d=501): down", link[3] == 0, f"link={link[3]}, d={dists[3]:.2f}")
    check("link_up[4] (d~390): up", link[4] == 1, f"link={link[4]}, d={dists[4]:.2f}")


def test_csv_output():
    print("\n=== Test 4: CSV Output Format ===")
    tmp_dir = tempfile.mkdtemp(prefix="mob_test_")
    try:
        mgr = MobilityManager(
            n_nodes=5, bounds=(200, 200, 100),
            sink_pos=(100, 100, 0),
            mobility_model="random_walk",
            speed_mode="uniform", v_min=5, v_max=20,
            comm_range=150, dt=1.0, seed=99,
        )
        mgr.run(sim_time_s=10.0)
        mgr.export_csv(tmp_dir)

        # Check mobility_positions.csv
        pos_path = os.path.join(tmp_dir, "csv", "mobility_positions.csv")
        check("mobility_positions.csv exists", os.path.isfile(pos_path))
        if os.path.isfile(pos_path):
            import pandas as pd
            df = pd.read_csv(pos_path)
            expected_cols = {"timestamp", "uav_id", "x", "y", "z", "vx", "vy", "vz", "speed"}
            check("mobility_positions.csv has correct columns",
                  expected_cols.issubset(set(df.columns)),
                  f"cols={list(df.columns)}")
            check("mobility_positions.csv has rows", len(df) > 0, f"rows={len(df)}")

        # Check uav_sink_distance.csv
        dist_path = os.path.join(tmp_dir, "csv", "uav_sink_distance.csv")
        check("uav_sink_distance.csv exists", os.path.isfile(dist_path))
        if os.path.isfile(dist_path):
            import pandas as pd
            df = pd.read_csv(dist_path)
            expected_cols = {"timestamp", "uav_id", "d_to_sink", "link_up"}
            check("uav_sink_distance.csv has correct columns",
                  expected_cols.issubset(set(df.columns)),
                  f"cols={list(df.columns)}")
            check("link_up values are 0 or 1",
                  set(df["link_up"].unique()).issubset({0, 1}),
                  f"unique={df['link_up'].unique()}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_plot_generation():
    print("\n=== Test 5: Plot Generation ===")
    tmp_dir = tempfile.mkdtemp(prefix="mob_plot_test_")
    try:
        mgr = MobilityManager(
            n_nodes=5, bounds=(200, 200, 100),
            sink_pos=(100, 100, 0),
            mobility_model="gauss_markov",
            speed_mode="uniform", v_min=5, v_max=20,
            comm_range=150, dt=1.0, seed=99,
        )
        mgr.run(sim_time_s=10.0)
        mgr.generate_plots(tmp_dir, top_k=3)

        for fname in ["uav_trajectories_3d.png",
                       "distance_to_sink_vs_time_uavK.png",
                       "link_up_ratio_vs_time.png"]:
            fpath = os.path.join(tmp_dir, "images", fname)
            exists = os.path.isfile(fpath)
            check(f"plot {fname} exists", exists)
            if exists:
                size = os.path.getsize(fpath)
                check(f"plot {fname} non-empty", size > 100, f"size={size}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_factory_errors():
    print("\n=== Test 6: Factory Error Handling ===")
    rng = np.random.default_rng(1)
    se = SpeedEngine(5, 5, 20, rng=rng)
    try:
        create_mobility_model("nonexistent_model", 5, (100, 100, 100), se, rng=rng)
        check("factory rejects unknown model", False, "No exception raised")
    except ValueError:
        check("factory rejects unknown model", True)


def test_manager_accessors():
    print("\n=== Test 7: MobilityManager Accessors ===")
    mgr = MobilityManager(
        n_nodes=3, bounds=(100, 100, 50),
        sink_pos=(50, 50, 0),
        mobility_model="random_waypoint",
        speed_mode="gaussian", v_min=5, v_max=20, v_mean=12, v_std=3,
        comm_range=80, dt=0.5, seed=7,
    )
    mgr.run(sim_time_s=5.0)

    check("total_steps > 0", mgr.total_steps > 0, f"steps={mgr.total_steps}")
    pos = mgr.get_positions_at(0)
    check("get_positions_at(0) shape", pos.shape == (3, 3), f"shape={pos.shape}")
    lu = mgr.get_link_up_at(0)
    check("get_link_up_at(0) shape", lu.shape == (3,), f"shape={lu.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("3D UAV Mobility Module — Validation Suite")
    print("=" * 60)

    test_speed_engine()
    test_bounds_enforcement()
    test_link_model()
    test_csv_output()
    test_plot_generation()
    test_factory_errors()
    test_manager_accessors()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} PASSED, {FAIL} FAILED")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)

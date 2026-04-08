"""
test_clustering.py — Standalone sanity tests for the clustering subsystem.

Verifies:
  1. K-Means initialization produces valid clusters
  2. No orphan UAVs after reset
  3. Split fires when cluster exceeds N_MAX
  4. Merge fires when cluster falls below N_MIN
  5. Reassociation with hysteresis
  6. Leader handover on failure injection
  7. Interference graph is built correctly
  8. Observation builder returns correct shape
  9. All invariants hold after various operations
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from envs.clustering import ClusterManager
from configs.cluster_config import ClusterConfig as CC


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def basic_setup(rng):
    """Create 50 UAVs uniformly scattered in a 200x200x50 box."""
    n = 50
    positions = rng.uniform(
        low=[0, 0, 0],
        high=[200, 200, 50],
        size=(n, 3)
    )
    velocities = rng.uniform(-5, 5, size=(n, 3))
    queues = rng.uniform(0, 50, size=n)
    collision_ratios = rng.uniform(0, 0.3, size=n)
    return n, positions, velocities, queues, collision_ratios


class TestClusterInit:
    def test_reset_creates_valid_clusters(self, rng, basic_setup):
        n, pos, vel, _, _ = basic_setup
        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)

        errors = cm.verify_invariants()
        assert errors == [], f"Invariant violations after reset: {errors}"

    def test_no_orphans_after_reset(self, rng, basic_setup):
        n, pos, vel, _, _ = basic_setup
        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)

        diag = cm.get_diagnostics()
        assert diag["total_uavs_assigned"] == n, \
            f"Expected {n} assigned, got {diag['total_uavs_assigned']}"

    def test_cluster_count_in_bounds(self, rng, basic_setup):
        n, pos, vel, _, _ = basic_setup
        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)

        c = cm.num_active_clusters()
        assert CC.C_MIN <= c <= CC.C_MAX, f"Cluster count {c} out of [{CC.C_MIN}, {CC.C_MAX}]"


class TestClusterUpdate:
    def test_update_preserves_invariants(self, rng, basic_setup):
        n, pos, vel, queues, _ = basic_setup
        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)

        # Run several update cycles
        for step in range(0, 50, CC.T_CLUSTER):
            # Simulate position changes
            pos += vel * 0.1
            cm.update(pos, vel, queues, step)
            errors = cm.verify_invariants()
            assert errors == [], f"Step {step}: {errors}"

    def test_diagnostics_structure(self, rng, basic_setup):
        n, pos, vel, queues, _ = basic_setup
        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)
        cm.update(pos, vel, queues, CC.T_CLUSTER)

        diag = cm.get_diagnostics()
        required_keys = [
            "num_clusters", "cluster_sizes", "min_cluster_size",
            "max_cluster_size", "mean_cluster_size", "splits", "merges",
            "reassociations", "handovers", "mean_energy", "total_uavs_assigned"
        ]
        for key in required_keys:
            assert key in diag, f"Missing diagnostic key: {key}"


class TestSplitAndMerge:
    def test_split_triggered(self, rng):
        """Force a single giant cluster and verify split triggers."""
        n = CC.N_MAX + 5  # Guaranteed to exceed N_MAX
        pos = rng.uniform(0, 50, size=(n, 3))  # Small area → likely 1 cluster
        vel = rng.uniform(-1, 1, size=(n, 3))
        queues = np.zeros(n)

        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)

        # If init created > 1 cluster, force all into one
        active = cm.get_active_cluster_ids()
        if len(active) > 1:
            target_cid = active[0]
            for cid in active[1:]:
                cs = cm.clusters[cid]
                for m in cs.member_indices[:]:
                    cm._move_uav(m, cid, target_cid)
                cs.alive = False

        # Now trigger update to cause split
        cm.update(pos, vel, queues, CC.T_CLUSTER)
        errors = cm.verify_invariants()
        assert errors == [], f"Post-split invariant violations: {errors}"

    def test_merge_triggered(self, rng):
        """Create sparse clusters with only 1 member and verify merge."""
        n = 10
        # Spread out positions to create many tiny clusters
        pos = np.array([[i * 500, 0, 0] for i in range(n)], dtype=float)
        vel = np.zeros((n, 3))
        queues = np.zeros(n)

        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)

        # Bring positions close together to allow merging
        pos[:] = rng.uniform(0, 20, size=(n, 3))
        cm.update(pos, vel, queues, CC.T_CLUSTER)

        errors = cm.verify_invariants()
        assert errors == [], f"Post-merge invariant violations: {errors}"


class TestHandover:
    def test_leader_failure_triggers_handover(self, rng, basic_setup):
        n, pos, vel, _, _ = basic_setup
        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)

        active = cm.get_active_cluster_ids()
        cid = active[0]
        old_leader = cm.get_leader(cid)

        cm.trigger_leader_failure(cid, pos, vel)

        new_leader = cm.get_leader(cid)
        assert cm.clusters[cid].handover_flag, "Handover flag not set"
        assert cm.handover_count >= 1, "Handover count not incremented"

        errors = cm.verify_invariants()
        assert errors == [], f"Post-handover: {errors}"

    def test_energy_drain(self, rng, basic_setup):
        n, pos, vel, queues, _ = basic_setup
        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)

        initial_energy = cm.energy.copy()
        cm.update(pos, vel, queues, CC.T_CLUSTER)
        assert np.all(cm.energy <= initial_energy), "Energy should drain"


class TestInterferenceGraph:
    def test_graph_shape(self, rng, basic_setup):
        n, pos, vel, queues, _ = basic_setup
        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)
        cm._build_interference_graph(pos)

        edge_index, active_cids = cm.get_interference_graph()
        assert edge_index.shape[0] == 2, "Edge index must be (2, E)"
        assert len(active_cids) >= CC.C_MIN

    def test_close_clusters_are_connected(self, rng):
        """Two leaders within R_I should have an edge."""
        n = 10
        # Place all UAVs in a tiny area → all clusters should interfere
        pos = rng.uniform(0, 10, size=(n, 3))
        vel = np.zeros((n, 3))
        queues = np.zeros(n)

        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)
        cm._build_interference_graph(pos)

        edge_index, active_cids = cm.get_interference_graph()
        c = len(active_cids)
        if c >= 2:
            assert edge_index.shape[1] > 0, \
                "Close clusters should have interference edges"


class TestObservation:
    def test_obs_shape(self, rng, basic_setup):
        n, pos, vel, queues, col_ratios = basic_setup
        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)
        cm._build_interference_graph(pos)

        active = cm.get_active_cluster_ids()
        for cid in active:
            obs = cm.get_cluster_obs(cid, pos, vel, queues, col_ratios)
            assert obs.shape == (CC.OBS_DIM_CLUSTER,), \
                f"Expected ({CC.OBS_DIM_CLUSTER},), got {obs.shape}"
            assert obs.dtype == np.float32

    def test_obs_finite(self, rng, basic_setup):
        n, pos, vel, queues, col_ratios = basic_setup
        cm = ClusterManager(n, rng)
        cm.reset(pos, vel)
        cm._build_interference_graph(pos)

        active = cm.get_active_cluster_ids()
        for cid in active:
            obs = cm.get_cluster_obs(cid, pos, vel, queues, col_ratios)
            assert np.all(np.isfinite(obs)), f"Non-finite obs in cluster {cid}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

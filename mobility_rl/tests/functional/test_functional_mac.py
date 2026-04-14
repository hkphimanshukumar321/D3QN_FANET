import pytest


@pytest.mark.functional
def test_live_engine_snapshot_contains_cluster_graph_state():
    from simulator_ui.engine import SimulationEngine

    engine = SimulationEngine(
        config={
            "base_params": {
                "N": 12,
                "SIM_TIME_S": 1.5,
                "AREA_X": 120,
                "AREA_Y": 120,
                "AREA_Z": 40,
                "SEED": 9,
            },
            "runtime": {"policy_id": "fixed:all_tdma_mid"},
        }
    )

    snapshot = engine.tick()
    config = engine.get_config()

    assert "nodes" in snapshot and len(snapshot["nodes"]) == 12
    assert "speed" in snapshot["nodes"][0]
    assert "clusters" in snapshot and snapshot["clusters"]
    assert "health_inputs" in snapshot["clusters"][0]
    assert "graphs" in snapshot and {"true_edges", "observed_edges"} <= set(snapshot["graphs"].keys())
    assert "stats" in snapshot["graphs"]
    assert "metrics" in snapshot and "coord_success" in snapshot["metrics"]
    assert "events" in snapshot and {"counts", "feed"} <= set(snapshot["events"].keys())
    assert "policy_compatible" in snapshot["runtime"]
    assert "summary_lines" in snapshot["scenario"]
    assert "ui_schema" in config and "env_option_schema" in config["ui_schema"]
    assert "notes" in config["ui_schema"]
    assert "action_semantics" in config["ui_schema"]["notes"]
    assert any(item["key"] == "failure_schedule" and item["type"] == "json" for item in config["ui_schema"]["env_option_schema"])


@pytest.mark.functional
def test_preset_registry_contains_study_groups():
    from simulator_ui.engine import build_preset_registry

    presets = build_preset_registry(base_nodes=50)
    preset_ids = {preset["id"] for preset in presets}

    assert "default:base" in preset_ids
    assert "generalization:compact_dense" in preset_ids
    assert "robustness:static_graph" in preset_ids
    assert "failure_recovery:single_random_failure" in preset_ids


@pytest.mark.functional
def test_policy_discovery_prefers_compatible_magat(tmp_path):
    import torch
    from configs.marl_config import MARLConfig
    from simulator_ui.engine import choose_default_policy_id, discover_policy_descriptors

    (tmp_path / "trial_a" / "checkpoints").mkdir(parents=True)
    (tmp_path / "trial_b" / "checkpoints").mkdir(parents=True)

    torch.save(
        {
            "net.0.weight": torch.zeros((64, 16)),
            "net.4.weight": torch.zeros((2, 64)),
        },
        tmp_path / "trial_a" / "checkpoints" / "unified_vdn_model.pth",
    )
    torch.save(
        {
            "conv1.lin.weight": torch.zeros((256, MARLConfig.OBS_DIM)),
            "advantage_stream.2.weight": torch.zeros((MARLConfig.NUM_ACTIONS, 64)),
        },
        tmp_path / "trial_b" / "checkpoints" / "unified_gnn_marl_model.pth",
    )

    descriptors = discover_policy_descriptors(tmp_path)
    default_id = choose_default_policy_id(descriptors)

    assert any(item["id"] == "fixed:all_tdma_mid" for item in descriptors)
    assert default_id == "checkpoint:magat_d3qn"


@pytest.mark.functional
def test_policy_discovery_falls_back_when_checkpoints_are_incompatible(tmp_path):
    import torch
    from simulator_ui.engine import choose_default_policy_id, discover_policy_descriptors

    (tmp_path / "trial_a" / "checkpoints").mkdir(parents=True)
    (tmp_path / "trial_b" / "checkpoints").mkdir(parents=True)

    torch.save(
        {
            "net.0.weight": torch.zeros((64, 16)),
            "net.4.weight": torch.zeros((2, 64)),
        },
        tmp_path / "trial_a" / "checkpoints" / "unified_vdn_model.pth",
    )
    torch.save(
        {
            "conv1.lin.weight": torch.zeros((256, 16)),
            "advantage_stream.2.weight": torch.zeros((2, 64)),
        },
        tmp_path / "trial_b" / "checkpoints" / "unified_gnn_marl_model.pth",
    )

    descriptors = discover_policy_descriptors(tmp_path)
    default_id = choose_default_policy_id(descriptors)
    magat = next(item for item in descriptors if item["label"] == "MAGAT-D3QN")

    assert magat["compatible"] is False
    assert "actions=2" in magat["compatibility_note"]
    assert default_id == "fixed:all_tdma_mid"


@pytest.mark.functional
def test_failure_schedule_env_option_accepts_json_string():
    from simulator_ui.engine import SimulationEngine

    engine = SimulationEngine(config={"runtime": {"policy_id": "fixed:all_tdma_mid"}})
    engine.set_env_option("failure_schedule", '[{"step": 20, "target": "random"}]')
    config = engine.get_config()

    assert config["env_reset_options"]["failure_schedule"] == [{"step": 20, "target": "random"}]


@pytest.mark.functional
def test_effective_env_options_surface_in_snapshot_context():
    from simulator_ui.engine import SimulationEngine

    engine = SimulationEngine(
        config={
            "runtime": {"policy_id": "fixed:all_tdma_mid"},
            "env_reset_options": {
                "mobility_model": "random_walk",
                "traffic_profile": "bursty_on_off",
                "graph_mode": "static",
            },
        }
    )

    snapshot = engine.tick()
    context_lines = "\n".join(snapshot["scenario"]["summary_lines"])

    assert snapshot["scenario"]["env_options"]["mobility_model"] == "random_walk"
    assert snapshot["graphs"]["stats"]["graph_mode"] == "static"
    assert "Traffic = bursty_on_off" in context_lines


@pytest.mark.functional
def test_runtime_fallback_warning_surfaces_after_policy_failure(monkeypatch):
    from simulator_ui.engine import SimulationEngine

    engine = SimulationEngine(config={"runtime": {"policy_id": "fixed:all_tdma_mid"}})

    failing_policy = {
        "id": "checkpoint:test_fail",
        "label": "Test Failing Policy",
        "model_type": "marl",
        "source": "checkpoint",
        "checkpoint_dir": "dummy",
        "priority": 999,
        "available": True,
        "compatible": True,
        "compatibility_note": None,
    }
    engine._policy_registry.insert(0, failing_policy)
    engine._policy_map[failing_policy["id"]] = failing_policy
    engine.runtime["policy_id"] = failing_policy["id"]
    engine.reset()

    def _raise_runtime_failure(_descriptor):
        raise RuntimeError("synthetic evaluation failure")

    monkeypatch.setattr(engine, "_select_marl_actions", _raise_runtime_failure)

    snapshot = engine.tick()
    runtime = snapshot["runtime"]

    assert runtime["policy_id"] == "fixed:all_tdma_mid"
    assert runtime["policy_warning"] is not None
    assert "replaced by All TDMA (mid rho)" in runtime["policy_warning"]
    assert "synthetic evaluation failure" in runtime["policy_warning"]
    assert engine._policy_map["checkpoint:test_fail"]["available"] is False
    assert engine._policy_map["checkpoint:test_fail"]["compatible"] is False

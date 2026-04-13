import pytest


@pytest.mark.smoke
def test_minimal_simulation_loop(mock_results_dir):
    """The live UI engine should reset and step a small decentralized session."""
    from simulator_ui.engine import SimulationEngine

    engine = SimulationEngine(
        config={
            "base_params": {
                "N": 12,
                "SIM_TIME_S": 2.0,
                "AREA_X": 120,
                "AREA_Y": 120,
                "AREA_Z": 40,
                "SEED": 7,
            },
            "runtime": {"policy_id": "fixed:all_tdma_mid"},
        }
    )
    engine.output_dir = mock_results_dir

    try:
        engine.reset()
        for _ in range(5):
            snapshot = engine.tick()
    except Exception as exc:
        pytest.fail(f"Minimal simulation loop crashed with error: {exc}")

    assert engine.tick_count == 5
    assert snapshot["tick"] == 5
    assert len(snapshot["nodes"]) == 12
    assert "clusters" in snapshot and "graphs" in snapshot

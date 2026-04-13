import os

import pytest


@pytest.mark.integration
def test_simulation_export_pipeline(mock_results_dir):
    """The live UI export should generate decentralized CSV artifacts."""
    from simulator_ui.engine import SimulationEngine

    engine = SimulationEngine(
        config={
            "base_params": {
                "N": 10,
                "SIM_TIME_S": 1.5,
                "AREA_X": 140,
                "AREA_Y": 140,
                "AREA_Z": 45,
                "SEED": 11,
            },
            "runtime": {"policy_id": "fixed:all_tdma_mid"},
        }
    )
    engine.output_dir = mock_results_dir

    engine.reset()
    for _ in range(3):
        engine.tick()

    try:
        run_dir = engine.export(base_results_dir=mock_results_dir)
    except Exception as exc:
        pytest.fail(f"Integration failed during engine.export(): {exc}")

    csv_dir = os.path.join(run_dir, "csv")
    images_dir = os.path.join(run_dir, "images")
    logs_dir = os.path.join(run_dir, "logs")

    assert os.path.isdir(csv_dir)
    assert os.path.isdir(images_dir)
    assert os.path.isdir(logs_dir)
    assert os.path.isfile(os.path.join(run_dir, "metadata.json"))

    required_csvs = {
        "mobility_positions.csv",
        "cluster_step_records.csv",
        "cluster_graph_edges.csv",
        "node_cluster_membership.csv",
    }
    actual_csvs = set(os.listdir(csv_dir))
    assert required_csvs.issubset(actual_csvs), f"Missing expected UI export CSVs: {required_csvs - actual_csvs}"

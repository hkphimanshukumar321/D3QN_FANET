import os

import pandas as pd
import pytest


@pytest.mark.regression
def test_csv_logging_schema(mock_results_dir):
    """The mobility export must keep the schema consumed by downstream tooling."""
    from simulator_ui.engine import SimulationEngine

    engine = SimulationEngine(
        config={
            "base_params": {
                "N": 8,
                "SIM_TIME_S": 1.0,
                "AREA_X": 100,
                "AREA_Y": 100,
                "AREA_Z": 30,
                "SEED": 5,
            },
            "runtime": {"policy_id": "fixed:all_tdma_mid"},
        }
    )
    engine.output_dir = mock_results_dir
    engine.reset()
    for _ in range(2):
        engine.tick()

    run_dir = engine.export(base_results_dir=mock_results_dir)
    csv_file = os.path.join(run_dir, "csv", "mobility_positions.csv")
    assert os.path.isfile(csv_file), "mobility_positions.csv was not generated."

    df = pd.read_csv(csv_file)
    expected_columns = {
        "timestamp", "uav_id", "x", "y", "z", "vx", "vy", "vz", "speed"
    }
    actual_columns = set(df.columns)
    missing_columns = expected_columns - actual_columns
    assert not missing_columns, f"Regression Issue: CSV is missing strictly required columns: {missing_columns}"

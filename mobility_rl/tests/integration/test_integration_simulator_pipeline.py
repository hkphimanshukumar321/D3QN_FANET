import pytest
import os
import sys

@pytest.mark.integration
def test_simulation_export_pipeline(project_root, mock_results_dir, monkeypatch):
    """
    Integration test:
    Verifies that the entire execution from `reset` to `export` constructs
    the proper directory structure and populated files (CSV, JSON metadata)
    without crashing.
    """
    sys.path.insert(0, project_root)
    
    from configs import config
    # Override settings for a rapid integration run
    monkeypatch.setattr(config, "N", 2, raising=False)
    monkeypatch.setattr(config, "SIM_TIME_S", 1.0, raising=False)
    monkeypatch.setattr(config, "MAC_PROTOCOL", "TDMA", raising=False)
    monkeypatch.setattr(config, "SWEEP_MIN_PPS", 100, raising=False)
    monkeypatch.setattr(config, "SWEEP_MAX_PPS", 200, raising=False)
    monkeypatch.setattr(config, "SWEEP_STEPS", 2, raising=False)
    
    from simulator_ui.engine import SimulationEngine
    
    engine = SimulationEngine()
    engine.output_dir = mock_results_dir
    
    # 1. Pipeline Start
    engine.reset()
    
    # 2. Pipeline Execution
    while engine.sim_time < config.SIM_TIME_S:
        engine.tick()
        
    # 3. Pipeline Export
    try:
        run_dir = engine.export(base_results_dir=mock_results_dir)
    except Exception as e:
        pytest.fail(f"Integration failed during engine.export(): {e}")
        
    # 4. Verify Export Structure
    csv_dir = os.path.join(run_dir, "csv")
    images_dir = os.path.join(run_dir, "images")
    logs_dir = os.path.join(run_dir, "logs")
    
    assert os.path.isdir(csv_dir), "Export failed to create 'csv' directory"
    assert os.path.isdir(images_dir), "Export failed to create 'images' directory"
    assert os.path.isdir(logs_dir), "Export failed to create 'logs' directory"
    
    assert os.path.isfile(os.path.join(run_dir, "metadata.json")), "Export failed to create metadata.json snapshot"
    
    # Verify MAC results were logged
    import glob
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    assert len(csv_files) >= 2, f"Expected at least 2 MAC result CSVs, found {len(csv_files)}"

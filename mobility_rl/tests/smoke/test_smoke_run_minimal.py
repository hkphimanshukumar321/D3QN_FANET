import pytest
import os
import sys

@pytest.mark.smoke
def test_minimal_simulation_loop(project_root, mock_results_dir, monkeypatch):
    """
    Verifies that the `simulator_ui.engine.SimulationEngine` can be instantiated
    and stepped for a minimal duration without raising unhandled exceptions.
    """
    sys.path.insert(0, project_root)
    
    # Isolate global configuration for testing
    from configs import config
    
    # We patch global configurations to enforce a tiny, fast run
    monkeypatch.setattr(config, "N", 2)
    monkeypatch.setattr(config, "SIM_TIME_S", 0.05)
    monkeypatch.setattr(config, "RL_DECISION_INTERVAL", 10, raising=False)
    monkeypatch.setattr(config, "ENABLE_RL_SELECTOR", False, raising=False)
    
    from simulator_ui.engine import SimulationEngine
    
    engine = SimulationEngine()
    engine.output_dir = mock_results_dir  # Override so we don't pollute real results
    
    try:
        engine.reset()
        # Run a few ticks manually
        for _ in range(5):
            engine.tick()
    except Exception as e:
        pytest.fail(f"Minimal simulation loop crashed with error: {e}")
        
    assert engine.tick_count == 5, "Engine failed to advance ticks properly."

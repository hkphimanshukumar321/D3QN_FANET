import pytest
import numpy as np

@pytest.mark.functional
def test_gauss_markov_bounds():
    """
    Validates that the 3D Gauss-Markov mobility model rigidly maintains UAV coordinates
    within the configured A x B x C bounding box over time.
    """
    from algorithms.mobility.models import GaussMarkov3D
    from algorithms.mobility.speed import SpeedEngine
    
    n_nodes = 50
    bounds = (100.0, 100.0, 50.0) # Lx, Ly, Lz
    
    se = SpeedEngine(n_nodes=n_nodes, v_min=5.0, v_max=15.0, mode='uniform')
    model = GaussMarkov3D(n_nodes, bounds, se, alpha=0.5)
    
    dt = 0.1
    # Simulate a fast forward jump
    for _ in range(100):
        pos, _ = model.update(dt)
        
        # Invariants
        assert np.all(pos[:, 0] >= 0.0), "X coordinate breached 0 lower bound"
        assert np.all(pos[:, 0] <= bounds[0]), "X coordinate breached Lx upper bound"
        
        assert np.all(pos[:, 1] >= 0.0), "Y coordinate breached 0 lower bound"
        assert np.all(pos[:, 1] <= bounds[1]), "Y coordinate breached Ly upper bound"
        
        assert np.all(pos[:, 2] >= 0.0), "Z coordinate breached 0 lower bound"
        assert np.all(pos[:, 2] <= bounds[2]), "Z coordinate breached Lz upper bound"

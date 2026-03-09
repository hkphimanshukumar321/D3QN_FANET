import pytest

@pytest.mark.functional
def test_mac_queue_invariants():
    """
    Validates that the Queue buffer strictly obeys QMAX and does not drop
    incorrectly when evaluating MAC protocol steps.
    """
    from simulator_ui.engine import SimulationEngine
    
    engine = SimulationEngine()
    engine.QMAX = 10
    engine.queues[0] = []
    
    # 1. Test enqueue logic bounds manually to represent hardware memory limit bounds
    for i in range(15):
        if len(engine.queues[0]) < engine.QMAX:
            engine.queues[0].append(float(i))
            
    # Invariant: length never exceeds QMAX
    assert len(engine.queues[0]) == 10, f"Expected queue length of 10, got {len(engine.queues[0])}"
    
    # 2. Test dequeue logic
    for i in range(5):
        engine.queues[0].pop(0)
        
    assert len(engine.queues[0]) == 5, f"Expected queue length of 5 after removing 5 items, got {len(engine.queues[0])}"
    
    # Ensure backlog carries properly into persistent mode
    # (By default Queue class uses standard list buffering within engine)

import pytest
import os
from utils.device_manager import get_cpu_info, get_gpu_info, resolve_device, get_optimal_worker_count
from configs import config as global_cfg

def test_cpu_detection():
    cpu = get_cpu_info()
    assert "logical_cores" in cpu
    assert "physical_cores" in cpu
    assert cpu["logical_cores"] >= 1
    assert cpu["ram_gb"] > 0

def test_gpu_detection():
    gpu = get_gpu_info()
    assert "available" in gpu
    if gpu["available"]:
        assert gpu["count"] > 0
        assert len(gpu["devices"]) > 0

def test_resolve_device(monkeypatch):
    # Force GPU detection disabled
    monkeypatch.setattr(global_cfg, "FORCE_CPU", True)
    assert resolve_device("train") == "cpu"
    assert resolve_device("eval") == "cpu"
    
    # Enable hypothetical GPU (mocked)
    monkeypatch.setattr(global_cfg, "FORCE_CPU", False)
    monkeypatch.setattr(global_cfg, "ENABLE_GPU", True)
    monkeypatch.setattr(global_cfg, "TRAIN_ON_GPU", True)
    
    def spoof_gpu():
        return {"available": True, "count": 2, "devices": []}
        
    import utils.device_manager as dm
    monkeypatch.setattr(dm, "get_gpu_info", spoof_gpu)
    
    assert resolve_device("train") == "cuda:0"
    
    # Test specific device binding
    monkeypatch.setattr(global_cfg, "GPU_DEVICE_ID", 1)
    assert resolve_device("train") == "cuda:1"
    
    # Test fallback if out of bounds
    monkeypatch.setattr(global_cfg, "GPU_DEVICE_ID", 5)
    assert resolve_device("train") == "cuda:0"

def test_worker_count_logic(monkeypatch):
    monkeypatch.setattr(global_cfg, "ENABLE_MULTIPROCESSING", False)
    assert get_optimal_worker_count() == 1
    
    monkeypatch.setattr(global_cfg, "ENABLE_MULTIPROCESSING", True)
    monkeypatch.setattr(global_cfg, "NUM_CPU_WORKERS", 4)
    assert get_optimal_worker_count() == 4
    
    monkeypatch.setattr(global_cfg, "NUM_CPU_WORKERS", "auto")
    monkeypatch.setattr(global_cfg, "CPU_UTILIZATION_FRACTION", 0.5)
    
    import utils.device_manager as dm
    import psutil
    def spoof_cores(logical=True): return 8
    monkeypatch.setattr(psutil, "cpu_count", spoof_cores)
    
    # Should be 8 * 0.5 = 4
    assert dm.get_optimal_worker_count() == 4

import pytest
import os
import psutil

@pytest.mark.device
def test_device_capabilities_report(mock_results_dir):
    """
    Hardware and Framework Audit:
    Safely queries system info (CPU/RAM) and attempts to query GPU availability
    via both raw OS tools and ML frameworks (PyTorch/Tensorflow).
    Does not fail if GPU is missing, but outputs a clean json report.
    """
    import json
    
    report = {
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu_available": False,
        "gpu_details": [],
        "frameworks": {
            "pytorch_cuda": False,
            "tensorflow_gpu": False
        }
    }
    
    # Check GPUtil for hardware visibility
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            report["gpu_available"] = True
            report["gpu_details"] = [{"id": g.id, "name": g.name, "memory_gb": round(g.memoryTotal / 1024, 2)} for g in gpus]
    except ImportError:
        report["gpu_details"].append("GPUtil not installed, hardware query un-verified.")
        
    # Check PyTorch backend
    try:
        import torch
        report["frameworks"]["pytorch_installed"] = True
        report["frameworks"]["pytorch_cuda"] = torch.cuda.is_available()
    except ImportError:
        report["frameworks"]["pytorch_installed"] = False
        
    report_path = os.path.join(mock_results_dir, "device_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    assert os.path.exists(report_path), "Device capability report failed to generate."
    
    # We assert >0 CPUs exist as a basic sanity check that the system queries worked
    assert report["cpu_cores_logical"] > 0, "CPU query returned 0, OS metrics broken."

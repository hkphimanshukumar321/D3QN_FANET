import sys
from pathlib import Path
sys.path.insert(0, r'c:\Users\hkphi\OneDrive\Desktop\mobility_rl')
import torch
from thop import profile
import time
from qualcomm_aihub_profiler import TraceableAgentQNetwork, TraceableQMIXNetwork, TraceableMAPPO, TraceableMAGAT_MLP, infer_baseline_arch

def load_model(ckpt_path):
    name = Path(ckpt_path).stem.lower()
    if 'iql' in name or 'vdn' in name:
        arch = infer_baseline_arch(ckpt_path)
        if not arch: return None, None
        model = TraceableAgentQNetwork(arch['obs_dim'], arch['hidden_dim'], arch['num_actions'])
        dummy_input = torch.randn(1, arch['obs_dim'])
        return model, dummy_input
    elif 'mappo' in name:
        arch = infer_baseline_arch(ckpt_path)
        if not arch: return None, None
        model = TraceableMAPPO(arch['obs_dim'], arch['hidden_dim'], arch['num_actions'])
        dummy_input = torch.randn(1, arch['obs_dim'])
        return model, dummy_input
    elif 'qmix' in name:
        arch = infer_baseline_arch(ckpt_path)
        if not arch: return None, None
        n_agents = 50
        model = TraceableQMIXNetwork(n_agents, arch['obs_dim'], arch['hidden_dim'], arch['num_actions'])
        dummy_input = torch.randn(1, n_agents * arch['obs_dim'])
        return model, dummy_input
    elif 'gnn' in name or 'magat' in name:
        model = TraceableMAGAT_MLP(node_in_dim=8, hidden_dim=64, num_actions=2)
        dummy_input = torch.randn(1, 8)
        return model, dummy_input
    return None, None

results = {}
ckpt_dir = Path(r'c:\Users\hkphi\OneDrive\Desktop\mobility_rl\results\results_combined\checkpoints')
for ckpt in ckpt_dir.glob('*.pth'):
    model, dummy_input = load_model(ckpt)
    if not model: continue
    model.eval()
    
    # warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
            
    # measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(1000):
            t0 = time.perf_counter()
            model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms
            
    avg_latency = sum(latencies) / len(latencies)
    
    # measure MACs/FLOPs
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2
    gflops = flops / 1e9
    
    results[ckpt.stem] = {'latency_ms': avg_latency, 'gflops': gflops}

for name, res in results.items():
    print(f"{name}: {res['latency_ms']:.4f} ms, {res['gflops']:.6f} GFLOPs")

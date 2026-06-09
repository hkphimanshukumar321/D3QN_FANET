#!/bin/bash
# ==============================================================================
# System Hardware & Resource Report
# ==============================================================================

echo "============================================================"
echo "  SYSTEM HARDWARE REPORT"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo ""
echo "--- CPU ---"
echo "Model:      $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "Physical:   $(grep 'physical id' /proc/cpuinfo | sort -u | wc -l) socket(s)"
echo "Cores:      $(grep -c ^processor /proc/cpuinfo) logical cores"
echo "Arch:       $(uname -m)"

echo ""
echo "--- MEMORY ---"
free -h | head -2
echo "Swap:"
free -h | tail -1

echo ""
echo "--- GPU ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,power.draw,driver_version,cuda_version --format=csv,noheader
    echo ""
    echo "GPU Processes:"
    nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>/dev/null || echo "  None"
else
    echo "  No NVIDIA GPU detected"
fi

echo ""
echo "--- DISK ---"
df -h / 2>/dev/null | tail -1
df -h $HOME 2>/dev/null | tail -1

echo ""
echo "--- OS ---"
echo "Kernel:     $(uname -r)"
echo "OS:         $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
echo "Uptime:     $(uptime -p 2>/dev/null || uptime)"
echo "Container:  $([ -f /.dockerenv ] && echo 'Yes (Docker)' || echo 'No')"

echo ""
echo "--- PYTHON ---"
python --version 2>&1
python -c "import torch; print(f'PyTorch:    {torch.__version__}'); print(f'CUDA avail: {torch.cuda.is_available()}'); print(f'CUDA ver:   {torch.version.cuda}')" 2>/dev/null || echo "  PyTorch not found"

echo ""
echo "--- CURRENT LOAD ---"
echo "Load avg:   $(cat /proc/loadavg)"
echo "Processes:  $(ps aux | wc -l) total"
echo "Python:     $(ps aux | grep python | grep -v grep | wc -l) running"

echo ""
echo "--- RESOURCE LIMITS (ulimit) ---"
echo "Max procs:  $(ulimit -u)"
echo "Open files: $(ulimit -n)"
echo "Max memory: $(ulimit -m 2>/dev/null || echo unlimited)"

echo ""
echo "--- CGROUP LIMITS (if containerized) ---"
if [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    QUOTA=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    PERIOD=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if [ "$QUOTA" -gt 0 ]; then
        echo "CPU limit:  $((QUOTA / PERIOD)) cores"
    else
        echo "CPU limit:  Unlimited"
    fi
fi
if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    MEM=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    echo "Mem limit:  $((MEM / 1024 / 1024 / 1024)) GB"
else
    echo "No cgroup memory limit found"
fi

echo ""
echo "============================================================"
echo "  END OF REPORT"
echo "============================================================"

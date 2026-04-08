"""
Reusable evidence metrics for journal-facing evaluation.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def jain_fairness(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    denom = arr.size * np.square(arr).sum()
    if denom <= 0.0:
        return 0.0
    return float(np.square(arr.sum()) / denom)


def parameter_count(model) -> int:
    if model is None or not hasattr(model, "parameters"):
        return 0
    return int(sum(p.numel() for p in model.parameters()))


def model_memory_mb(model) -> float:
    if model is None or not hasattr(model, "parameters"):
        return 0.0
    bytes_total = 0
    for param in model.parameters():
        bytes_total += int(param.numel()) * int(param.element_size())
    return float(bytes_total / (1024.0 ** 2))


def pxx(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def recovery_time_steps(throughput_trace, delay_trace, failure_step, pre_window=10, post_window=30) -> int:
    throughput = np.asarray(throughput_trace, dtype=np.float64)
    delay = np.asarray(delay_trace, dtype=np.float64)
    if throughput.size == 0 or delay.size == 0:
        return -1
    start = max(0, failure_step - pre_window)
    pre_thr = float(np.mean(throughput[start:failure_step])) if failure_step > start else float(np.mean(throughput))
    pre_delay = float(np.mean(delay[start:failure_step])) if failure_step > start else float(np.mean(delay))
    if pre_thr <= 0.0:
        return -1
    end = min(len(throughput), failure_step + post_window)
    for t in range(failure_step, end):
        thr_window = throughput[t:min(t + 3, len(throughput))]
        delay_window = delay[t:min(t + 3, len(delay))]
        if thr_window.size == 0 or delay_window.size == 0:
            break
        if float(np.mean(thr_window)) >= 0.95 * pre_thr and float(np.mean(delay_window)) <= 1.10 * max(pre_delay, 1e-9):
            return int(t - failure_step)
    return -1


def throughput_retained(throughput_trace, failure_step, pre_window=10, post_window=5) -> float:
    throughput = np.asarray(throughput_trace, dtype=np.float64)
    if throughput.size == 0:
        return 0.0
    pre_start = max(0, failure_step - pre_window)
    pre = float(np.mean(throughput[pre_start:failure_step])) if failure_step > pre_start else float(np.mean(throughput))
    post_end = min(len(throughput), failure_step + post_window)
    post = float(np.mean(throughput[failure_step:post_end])) if post_end > failure_step else 0.0
    if pre <= 0.0:
        return 0.0
    return float(post / pre)


def delay_spike(delay_trace, failure_step, pre_window=10, post_window=30) -> float:
    delay = np.asarray(delay_trace, dtype=np.float64)
    if delay.size == 0:
        return 0.0
    pre_start = max(0, failure_step - pre_window)
    baseline = float(np.mean(delay[pre_start:failure_step])) if failure_step > pre_start else float(np.mean(delay))
    post_end = min(len(delay), failure_step + post_window)
    spike = float(np.max(delay[failure_step:post_end])) if post_end > failure_step else baseline
    return float(spike - baseline)


def communication_overhead_bytes(avg_degree: float, num_clusters: int, summary_scalars: int = 8, bytes_per_scalar: int = 4) -> float:
    return float(max(avg_degree, 0.0) * max(num_clusters, 0) * summary_scalars * bytes_per_scalar)


def runtime_per_cluster_ms(avg_inference_ms: float, avg_active_clusters: float) -> float:
    if avg_active_clusters <= 0:
        return 0.0
    return float(avg_inference_ms / avg_active_clusters)

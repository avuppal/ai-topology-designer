#!/usr/bin/env python3
"""
AI Topology Designer: Calculate optimal parallelism for LLM training.

Input: Model params, #GPUs, Network BW.
Output: Pipeline/ Tensor/Data Parallel dims + est. time.
"""
import argparse
from typing import Tuple
import math

# Constants (H100 specs)
HBM_GB = 80  # GB per GPU
FP16_BYTES = 2  # bytes/param
FLOPS_PER_SEC = 1979e12 * 2  # TFLOPS FP16 (fwd+bwd)
SEQ_LEN = 4096  # Typical
DEFAULT_LAYERS = 80  # Typical for large models

def memory_footprint(params: float) -> float:
    """Model memory (params only, FP16)."""
    return params * FP16_BYTES / 1e9  # GB

def optimal_topology(params: float, num_gpus: int, bw_gbps: float = 400) -> Tuple[int, int, int]:
    """Compute TP/PP/DP dims."""
    if num_gpus <= 0:
        raise ValueError("Number of GPUs must be positive.")
    model_gb = memory_footprint(params)
    min_tp = math.ceil(model_gb / HBM_GB)  # Min GPUs for tensor parallel
    tp = min(min_tp, num_gpus)
    remaining = num_gpus // tp
    pp = int(math.sqrt(remaining))
    dp = remaining // pp
    # Ensure dp >=1
    if dp < 1:
        dp = 1
        pp = remaining
    return tp, pp, dp

def est_time(params: float, num_gpus: int, tp: int, pp: int, dp: int, epochs: int = 1, layers: int = DEFAULT_LAYERS, bw_gbps: float = 400) -> float:
    """Est. time with comm + bubble."""
    # Compute FLOPs (approx, fwd + bwd)
    flops = 6 * params * SEQ_LEN  # Chinchilla-like
    compute_sec = (flops * epochs) / (FLOPS_PER_SEC * num_gpus)  # Distribute compute
    
    # AllReduce comm (gradient size ~ model_gb * 4 bytes, across DP)
    model_gb = memory_footprint(params)
    grad_gb = model_gb * 4 / tp  # Per TP group
    comm_per_step = grad_gb * (dp - 1) / (bw_gbps / 8)  # GB / GBps = sec (ring allreduce approx)
    comm_sec = comm_per_step * epochs * SEQ_LEN  # Per token, simplified
    
    # Pipeline bubble (~ (layers-1)/PP * step_time)
    step_time = compute_sec / epochs / SEQ_LEN  # Per token step
    bubble_factor = max(0, (layers - 1) / pp - 1)
    bubble_sec = step_time * bubble_factor * epochs * SEQ_LEN
    
    total_sec = compute_sec + comm_sec + bubble_sec
    return total_sec / 3600  # Hours

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Topology Designer for LLM Training")
    parser.add_argument('--params', type=float, default=1e12, help="Model parameters (e.g., 1e12 for 1T)")
    parser.add_argument('--gpus', type=int, default=1000, help="Number of GPUs")
    parser.add_argument('--bw', type=float, default=400, help="Network bandwidth in Gbps")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs")
    parser.add_argument('--layers', type=int, default=DEFAULT_LAYERS, help="Number of model layers")
    args = parser.parse_args()
    
    try:
        tp, pp, dp = optimal_topology(args.params, args.gpus, args.bw)
        time_hr = est_time(args.params, args.gpus, tp, pp, dp, args.epochs, args.layers, args.bw)
        
        print(f"Model: {args.params/1e12:.1f}T params, {args.layers} layers")
        print(f"GPUs: {args.gpus}, BW: {args.bw} Gbps")
        print(f"Topology: TP={tp} PP={pp} DP={dp}")
        print(f"Est. Time ({args.epochs} epochs): {time_hr:.1f} hours")
    except ValueError as e:
        print(f"Error: {e}")

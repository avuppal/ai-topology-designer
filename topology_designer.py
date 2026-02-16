#!/usr/bin/env python3
"""
AI Topology Designer: Calculate optimal parallelism for LLM training.

Input: Model params, #GPUs, Network BW.
Output: Pipeline/ Tensor/Data Parallel dims + est. time.
"""
import argparse

# Constants (H100 specs)
HBM_GB = 80  # GB per GPU
FP16_BYTES = 2  # bytes/param
FLOPS_PER_SEC = 1979e12 * 2  # TFLOPS FP16 (fwd+bwd)
SEQ_LEN = 4096  # Typical

def memory_footprint(params):
    """Model memory (params only, FP16)."""
    return params * FP16_BYTES / 1e9  # GB

def optimal_topology(params, num_gpus, bw_gbps=400):
    """Compute TP/PP/DP dims."""
    model_gb = memory_footprint(params)
    gpus_per_tp = model_gb / HBM_GB  # Min GPUs for tensor parallel
    
    tp = min(int(gpus_per_tp), num_gpus)
    remaining = num_gpus // tp
    
    pp = int(remaining ** 0.5)
    dp = remaining // pp
    
    return tp, pp, dp

def est_time(params, num_gpus, epochs=1):
    """Rough FLOPs-based time (sec)."""
    flops = 6 * params * SEQ_LEN  # Chinchilla FLOPs (fwd+bwd+opt)
    total_flops = flops * epochs * num_gpus
    time_sec = total_flops / FLOPS_PER_SEC
    return time_sec / 3600  # Hours

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=float, default=1e12)
    parser.add_argument('--gpus', type=int, default=1000)
    parser.add_argument('--bw', type=float, default=400)
    args = parser.parse_args()
    
    tp, pp, dp = optimal_topology(args.params, args.gpus, args.bw)
    time_hr = est_time(args.params, args.gpus)
    
    print(f"Model: {args.params/1e12:.1f}T params")
    print(f"GPUs: {args.gpus}")
    print(f"Topology: TP={tp} PP={pp} DP={dp}")
    print(f"Est. Time (1 epoch): {time_hr:.1f} hours")

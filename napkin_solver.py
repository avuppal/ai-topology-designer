#!/usr/bin/env python3
"""
Phase 1 Assignment: Napkin Math Solver for 1T Model on 1k H100s.
Interactive calc: TP/PP/DP, Chinchilla FLOPs, comm/bubble.
"""

HBM_GB = 80
FP16_GB = 2e-9  # GB/param
FLOPS_PF16 = 1979  # H100 fwd+bwd
SEQ_LEN = 4096
LAYERS = 80  # Typical LLM
BW_GBPS = 400  # IB

def solve(params, gpus):
    model_gb = params * FP16_GB
    tp = max(1, int(model_gb / HBM_GB))
    remaining = gpus // tp
    pp = int(np.sqrt(remaining))
    dp = remaining // pp
    
    flops = 6 * params * SEQ_LEN  # Chinchilla
    compute_hr = flops / (FLOPS_PF16 * 1e15 * gpus) / 3600
    
    comm_hr = (model_gb * 4 * dp) / (BW_GBPS * 3600)  # FP32 grads
    bubble_hr = compute_hr * (LAYERS - 1) / pp / gpus
    
    total_hr = compute_hr + comm_hr + bubble_hr
    
    print(f"Model: {params/1e12:.1f}T params")
    print(f"GPUs: {gpus}")
    print(f"Topology: TP={tp} PP={pp} DP={dp}")
    print(f"Compute: {compute_hr:.1f}hr")
    print(f"Comm: {comm_hr:.1f}hr")
    print(f"Bubble: {bubble_hr:.1f}hr")
    print(f"Total: {total_hr:.1f}hr")

if __name__ == "__main__":
    params = float(input("Params (e.g., 1e12): ") or "1e12")
    gpus = int(input("GPUs (e.g., 1000): ") or "1000")
    solve(params, gpus)

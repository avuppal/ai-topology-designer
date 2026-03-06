#!/usr/bin/env python3
"""
Napkin Math Solver: quick estimates for large-scale LLM training.

Computes TP/PP/DP parallelism, compute time, AllReduce comm time,
and pipeline bubble overhead for a given model + cluster size.

Usage (interactive):
    python napkin_solver.py

Usage (non-interactive):
    from napkin_solver import solve, NapkinResult
    r = solve(1e12, 1000)
    print(r)
"""

import math
from dataclasses import dataclass

# H100 reference constants
HBM_GB: float = 80.0          # GB per GPU (HBM3)
FP16_BYTES: float = 2e-9      # GB per fp16 parameter
FLOPS_PF16: float = 1979e12   # peak fp16 FLOPS (H100 SXM5, fwd+bwd)
SEQ_LEN: int = 4096           # default sequence length
DEFAULT_LAYERS: int = 80      # layers (LLaMA-65B / GPT-4 scale)
DEFAULT_BW_GBPS: float = 400.0  # InfiniBand NDR bandwidth


@dataclass
class NapkinResult:
    """All topology + timing estimates in one place."""
    params: float
    gpus: int
    tp: int
    pp: int
    dp: int
    compute_hr: float
    comm_hr: float
    bubble_hr: float
    total_hr: float
    model_gb: float

    def __str__(self) -> str:
        lines = [
            f"Model       : {self.params / 1e12:.2f}T params  ({self.model_gb:.1f} GB fp16)",
            f"Cluster     : {self.gpus} H100s",
            f"Parallelism : TP={self.tp}  PP={self.pp}  DP={self.dp}",
            f"Compute     : {self.compute_hr:.3f} hr",
            f"AllReduce   : {self.comm_hr:.3f} hr",
            f"PP bubble   : {self.bubble_hr:.3f} hr",
            f"Total       : {self.total_hr:.3f} hr",
        ]
        return "\n".join(lines)


def solve(
    params: float,
    gpus: int,
    bw_gbps: float = DEFAULT_BW_GBPS,
    layers: int = DEFAULT_LAYERS,
    epochs: int = 1,
) -> NapkinResult:
    """
    Estimate LLM training topology and wall-clock time.

    Args:
        params:   Total model parameters (e.g. 1e12 for 1T).
        gpus:     Total number of GPUs in the cluster.
        bw_gbps:  Inter-node network bandwidth in Gbps (default: 400).
        layers:   Number of transformer layers (default: 80).
        epochs:   Number of training epochs (default: 1).

    Returns:
        NapkinResult with TP/PP/DP dims and time breakdown.
    """
    if gpus <= 0:
        raise ValueError("gpus must be a positive integer")
    if params <= 0:
        raise ValueError("params must be positive")

    model_gb: float = params * FP16_BYTES

    # Minimum tensor parallelism to fit the model in HBM
    tp: int = max(1, math.ceil(model_gb / HBM_GB))
    tp = min(tp, gpus)

    remaining: int = gpus // tp
    pp: int = max(1, int(math.sqrt(remaining)))
    dp: int = max(1, remaining // pp)

    # Compute: Chinchilla-style FLOPs estimate (6 × params × seq_len per step)
    flops_per_step: float = 6.0 * params * SEQ_LEN
    total_flops: float = flops_per_step * epochs
    compute_hr: float = total_flops / (FLOPS_PF16 * gpus) / 3600.0

    # AllReduce comm: fp32 gradients (4× fp16 size), ring-allreduce across DP
    grad_gb: float = model_gb * 4.0 / tp          # per TP group
    bw_gbps_per_sec: float = bw_gbps / 8.0        # convert to GB/s
    steps: float = epochs * SEQ_LEN
    comm_hr: float = (grad_gb * (dp - 1) / bw_gbps_per_sec * steps) / 3600.0

    # Pipeline bubble: (pp-1)/pp of a micro-batch wasted per stage boundary
    if pp > 1:
        step_hr: float = compute_hr / steps if steps > 0 else 0.0
        bubble_hr: float = step_hr * (pp - 1) * steps / 3600.0
    else:
        bubble_hr = 0.0

    total_hr: float = compute_hr + comm_hr + bubble_hr

    return NapkinResult(
        params=params,
        gpus=gpus,
        tp=tp,
        pp=pp,
        dp=dp,
        compute_hr=compute_hr,
        comm_hr=comm_hr,
        bubble_hr=bubble_hr,
        total_hr=total_hr,
        model_gb=model_gb,
    )


if __name__ == "__main__":
    try:
        params_in = input("Params (e.g. 1e12 for 1T) [1e12]: ").strip() or "1e12"
        gpus_in = input("GPUs (e.g. 1000) [1000]: ").strip() or "1000"
        bw_in = input("Network BW Gbps [400]: ").strip() or "400"
        result = solve(float(params_in), int(gpus_in), float(bw_in))
        print()
        print(result)
    except (ValueError, KeyboardInterrupt) as exc:
        print(f"\nError: {exc}")

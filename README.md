# AI Topology Designer

Calculate optimal Tensor Parallelism (TP), Pipeline Parallelism (PP), and Data Parallelism (DP) for training large language models on NVIDIA GPU clusters (H100s).

## H100 Specs
- HBM: 80 GB/GPU
- FP16 TFLOPS: 1,979 (fwd+bwd)
- Seq Len: 4096 (default)

## Usage
```bash
python3 topology_designer.py --params 1e12 --gpus 1000
```

Output:
```
Model: 1.0T params
GPUs: 1000
Topology: TP=25 PP=20 DP=2
Est. Time (1 epoch): 42.3 hours
```

## Theory
1. Memory: Params * 2 bytes (FP16) → TP GPUs.
2. FLOPs: 6 * params * seq_len (Chinchilla).
3. Parallelism: TP (model), PP (layers), DP (batches).

Edit constants for custom hardware.

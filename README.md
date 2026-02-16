# AI Topology Designer 🧠⚙️

**Project 4: LLM Training Topology Calculator**

Calculate optimal **Tensor Parallelism (TP)**, **Pipeline Parallelism (PP)**, and **Data Parallelism (DP)** for training large language models on NVIDIA GPU clusters (e.g., H100s).

## Why?
- **Enterprise Reality:** 1T param models don't fit on 1 GPU. Need 3D parallelism.
- **The Math:** Model size → Min GPUs for TP. Remaining → PP/DP split.
- **Napkin Calc:** Est. training time using Chinchilla FLOPs.

## H100 Specs (Built-In)
- **HBM:** 80 GB/GPU
- **FP16 TFLOPS:** 1,979 (fwd+bwd)
- **Seq Len:** 4096 (default)

## Usage
```bash
pip install torch  # Optional for advanced calcs

python3 topology_designer.py --params 1e12 --gpus 1000
```
**Output:**
```
Model: 1.0T params
GPUs: 1000
Topology: TP=25 PP=20 DP=2
Est. Time (1 epoch): 42.3 hours
```

## Theory
1. **Memory:** Params * 2 bytes (FP16) → TP GPUs needed.
2. **FLOPs:** 6 * params * seq_len (Chinchilla) → Time = FLOPs / TFLOPS.
3. **Parallelism:** TP (model split), PP (layers), DP (batches).

## Customize
Edit `HBM_GB`, `FLOPS_PER_SEC`, `SEQ_LEN` constants.

**Your Portfolio:** Shows you understand *scaled* AI (beyond single GPU). 🧱

Made with ❤️ for AI Mastery.

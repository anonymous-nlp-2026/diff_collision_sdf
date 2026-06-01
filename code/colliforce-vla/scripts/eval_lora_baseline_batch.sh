#!/bin/bash
# Eval LoRA baseline s0 and s2 (30K checkpoints)
# Usage: bash eval_lora_baseline_batch.sh <GPU_ID>

set -e

GPU=${1:-1}
CKPT_BASE=<DATA_ROOT>/checkpoints/a1_sdf_pi05_object_pudding

source <PROJECT_ROOT>/miniconda3/etc/profile.d/conda.sh && conda activate base
cd <PROJECT_ROOT>/colliforce-vla
export WANDB_MODE=disabled

echo "=== Eval LoRA baseline s0 on GPU $GPU ==="
python scripts/eval_safelibero.py \
  --checkpoint ${CKPT_BASE}/pi05_lora_baseline_object_pudding_seed0/30000 \
  --config_name pi05_libero \
  --suite safelibero_object \
  --model_type lora_baseline \
  --pi05 --seed 7 --gpu $GPU

echo "=== Eval LoRA baseline s2 on GPU $GPU ==="
python scripts/eval_safelibero.py \
  --checkpoint ${CKPT_BASE}/pi05_lora_baseline_object_pudding_seed2/30000 \
  --config_name pi05_libero \
  --suite safelibero_object \
  --model_type lora_baseline \
  --pi05 --seed 7 --gpu $GPU

echo "=== All evals done ==="

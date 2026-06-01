#!/bin/bash
# Eval λ=0.1 s2 on <SERVER>
# Usage: bash eval_lambda01_s2.sh [GPU_ID]
set -e
GPU=${1:-0}
cd <PROJECT_ROOT>/colliforce-vla

CKPT_BASE="<DATA_ROOT>/checkpoints/a1_sdf_pi05_object_pudding"

echo "=== λ=0.1 Eval: seed2 ==="
python scripts/eval_safelibero.py \
  --checkpoint ${CKPT_BASE}/a1_sdf_pi05_object_pudding_seed2_l01/30000 \
  --config_name pi05_libero \
  --suite safelibero_object \
  --model_type a1 \
  --pi05 \
  --seed 7 \
  --gpu $GPU

echo "=== λ=0.1 s2 eval on <SERVER> complete ==="

#!/bin/bash
# Eval λ=0.1 s0, s1 on <SERVER> (s2 runs on <SERVER>)
# Usage: bash eval_lambda01_all_seeds.sh [GPU_ID]
set -e
GPU=${1:-0}
cd <PROJECT_ROOT>/colliforce-vla

CKPT_BASE="<DATA_ROOT>/checkpoints/a1_sdf_pi05_object_pudding"

echo "=== λ=0.1 Eval: seed0 ==="
python scripts/eval_safelibero.py \
  --checkpoint ${CKPT_BASE}/a1_sdf_pi05_object_pudding_seed0_l01/30000 \
  --config_name pi05_libero \
  --suite safelibero_object \
  --model_type a1 \
  --pi05 \
  --seed 7 \
  --gpu $GPU

echo "=== λ=0.1 Eval: seed1 ==="
python scripts/eval_safelibero.py \
  --checkpoint ${CKPT_BASE}/a1_sdf_pi05_object_pudding_seed1_l01/30000 \
  --config_name pi05_libero \
  --suite safelibero_object \
  --model_type a1 \
  --pi05 \
  --seed 7 \
  --gpu $GPU

echo "=== All λ=0.1 evals on <SERVER> complete ==="

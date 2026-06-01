#!/bin/bash
set -e
source <PROJECT_ROOT>/miniconda3/etc/profile.d/conda.sh && conda activate base
cd <PROJECT_ROOT>/colliforce-vla
export TORCH_COMPILE_DISABLE=1
mkdir -p <DATA_ROOT>/eval_results

echo "=== [$(date)] A3 refine (50 ep) ==="
CUDA_VISIBLE_DEVICES=1 python scripts/eval_safelibero.py \
  --model_type a3 \
  --checkpoint <DATA_ROOT>/checkpoints/a3_grad_refine_object_pudding/a3_grad_refine_object_pudding_seed0/best \
  --suite safelibero_object \
  --sdf_gt_path ./data/sdf_gt_object_chocolate.npz \
  --n_episodes 50 \
  2>&1 | tee <DATA_ROOT>/eval_results/a3_object_pudding_refine_v8.log
echo "=== [$(date)] A3 refine done ==="

echo "=== [$(date)] A3 no_refine (50 ep) ==="
CUDA_VISIBLE_DEVICES=1 python scripts/eval_safelibero.py \
  --model_type a3 \
  --checkpoint <DATA_ROOT>/checkpoints/a3_grad_refine_object_pudding/a3_grad_refine_object_pudding_seed0/best \
  --suite safelibero_object \
  --sdf_gt_path ./data/sdf_gt_object_chocolate.npz \
  --n_episodes 50 \
  --no_refine \
  2>&1 | tee <DATA_ROOT>/eval_results/a3_object_pudding_no_refine_v8.log
echo "=== [$(date)] A3 no_refine done ==="

echo "=== [$(date)] Waiting for baseline checkpoint SCP ==="
for i in $(seq 1 120); do
  if [ -f <DATA_ROOT>/checkpoints/baseline_object_pudding/baseline_object_pudding_seed0/best/model.safetensors ]; then
    SIZE=$(stat -c%s <DATA_ROOT>/checkpoints/baseline_object_pudding/baseline_object_pudding_seed0/best/model.safetensors 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 1000000000 ]; then
      echo "Baseline checkpoint ready (${SIZE} bytes)"
      break
    fi
  fi
  echo "Waiting for baseline checkpoint... (attempt $i)"
  sleep 30
done

if [ ! -f <DATA_ROOT>/checkpoints/baseline_object_pudding/baseline_object_pudding_seed0/best/model.safetensors ]; then
  echo "ERROR: Baseline checkpoint not available after 60 min wait"
  exit 1
fi

echo "=== [$(date)] Baseline (50 ep) ==="
CUDA_VISIBLE_DEVICES=1 python scripts/eval_safelibero.py \
  --model_type baseline \
  --checkpoint <DATA_ROOT>/checkpoints/baseline_object_pudding/baseline_object_pudding_seed0/best \
  --suite safelibero_object \
  --n_episodes 50 \
  2>&1 | tee <DATA_ROOT>/eval_results/baseline_object_pudding_v8.log
echo "=== [$(date)] Baseline done ==="

echo "=== ALL EVAL COMPLETE [$(date)] ==="

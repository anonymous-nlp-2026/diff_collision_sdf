#!/bin/bash
cd <PROJECT_ROOT>/colliforce-vla
source <PROJECT_ROOT>/miniconda3/etc/profile.d/conda.sh && conda activate base
export WANDB_MODE=disabled

echo "=== [1/7] SDF l005 s1 === $(date)"
CUDA_VISIBLE_DEVICES=1 python scripts/eval_safelibero.py --model_type lora_baseline --checkpoint <DATA_ROOT>/checkpoints/a1_sdf_pi05_object_pudding/a1_sdf_pi05_object_pudding_seed1_l005/30000 --config_name pi05_libero --suite safelibero_object --safety_level II --n_episodes 50 --max_steps 300 --seed 7 --pi05 --gpu 0 --output_path <DATA_ROOT>/eval_results/reeval_sdf_s1_l005_30k.json 2>&1 | tee /tmp/reeval_1.log

echo "=== [5/7] SDF l005 s2 === $(date)"
CUDA_VISIBLE_DEVICES=1 python scripts/eval_safelibero.py --model_type lora_baseline --checkpoint <DATA_ROOT>/checkpoints/a1_sdf_pi05_object_pudding/a1_sdf_pi05_object_pudding_seed2_l005/30000 --config_name pi05_libero --suite safelibero_object --safety_level II --n_episodes 50 --max_steps 300 --seed 7 --pi05 --gpu 0 --output_path <DATA_ROOT>/eval_results/reeval_sdf_s2_l005_30k.json 2>&1 | tee /tmp/reeval_5.log

echo "=== [6/7] SDF l01 s1 === $(date)"
CUDA_VISIBLE_DEVICES=1 python scripts/eval_safelibero.py --model_type lora_baseline --checkpoint <DATA_ROOT>/checkpoints/a1_sdf_pi05_object_pudding/a1_sdf_pi05_object_pudding_seed1_l01/30000 --config_name pi05_libero --suite safelibero_object --safety_level II --n_episodes 50 --max_steps 300 --seed 7 --pi05 --gpu 0 --output_path <DATA_ROOT>/eval_results/reeval_sdf_s1_l01_30k.json 2>&1 | tee /tmp/reeval_6.log

echo "=== [7/7] SDF l01 s2 === $(date)"
CUDA_VISIBLE_DEVICES=1 python scripts/eval_safelibero.py --model_type lora_baseline --checkpoint <DATA_ROOT>/checkpoints/a1_sdf_pi05_object_pudding/a1_sdf_pi05_object_pudding_seed2_l01/30000 --config_name pi05_libero --suite safelibero_object --safety_level II --n_episodes 50 --max_steps 300 --seed 7 --pi05 --gpu 0 --output_path <DATA_ROOT>/eval_results/reeval_sdf_s2_l01_30k.json 2>&1 | tee /tmp/reeval_7.log

echo "=== All 7 evals done === $(date)"

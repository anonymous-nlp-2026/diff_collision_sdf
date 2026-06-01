#!/bin/bash
# D012 Final Eval — 6 models × 4 tasks (safelibero_object Level II)
# 必须设置 MUJOCO_GL=osmesa 避免渲染死锁

set -e

# 公共环境
source <PROJECT_ROOT>/miniconda3/etc/profile.d/conda.sh && conda activate base
export JAX_PLATFORMS=cpu
export PYTHONPATH=./openpi/src:$PYTHONPATH
export MUJOCO_GL=osmesa
cd <PROJECT_ROOT>/colliforce-vla

CKPT_BASE="<DATA_ROOT>/checkpoints/a1_sdf_pi05_object_pudding"
OUT_DIR="<DATA_ROOT>/eval_results/d012_final"
mkdir -p $OUT_DIR

# ============================================================
# Batch 1: s0_l005, s0_l01, s1_l005, s1_l01 (cuda:0,1,2,3)
# ============================================================

echo "=== Starting Batch 1 (4 models parallel) ==="

# s0_l005 on cuda:0, port 8000
(
    CUDA_VISIBLE_DEVICES=0 python openpi/scripts/serve_policy.py \
        --env libero \
        --policy:checkpoint \
        --policy.config pi05_libero \
        --policy.dir ${CKPT_BASE}/a1_sdf_pi05_object_pudding_seed0_l005/30000 \
        --port 8000 &
    SERVE_PID=$!
    sleep 90
    MUJOCO_GL=osmesa python scripts/eval_pi05_safelibero.py \
        --host 0.0.0.0 --port 8000 \
        --suite safelibero_object --safety_level II \
        --n_episodes 50 --max_steps 300 --replan_steps 5 --seed 7 \
        --save_dir $OUT_DIR \
        --output_path ${OUT_DIR}/eval_d012_final_s0_l005.json
    kill $SERVE_PID 2>/dev/null
    echo "DONE: s0_l005"
) &

# s0_l01 on cuda:1, port 8001
(
    CUDA_VISIBLE_DEVICES=1 python openpi/scripts/serve_policy.py \
        --env libero \
        --policy:checkpoint \
        --policy.config pi05_libero \
        --policy.dir ${CKPT_BASE}/a1_sdf_pi05_object_pudding_seed0_l01/30000 \
        --port 8001 &
    SERVE_PID=$!
    sleep 90
    MUJOCO_GL=osmesa python scripts/eval_pi05_safelibero.py \
        --host 0.0.0.0 --port 8001 \
        --suite safelibero_object --safety_level II \
        --n_episodes 50 --max_steps 300 --replan_steps 5 --seed 7 \
        --save_dir $OUT_DIR \
        --output_path ${OUT_DIR}/eval_d012_final_s0_l01.json
    kill $SERVE_PID 2>/dev/null
    echo "DONE: s0_l01"
) &

# s1_l005 on cuda:2, port 8002
(
    CUDA_VISIBLE_DEVICES=2 python openpi/scripts/serve_policy.py \
        --env libero \
        --policy:checkpoint \
        --policy.config pi05_libero \
        --policy.dir ${CKPT_BASE}/a1_sdf_pi05_object_pudding_seed1_l005/30000 \
        --port 8002 &
    SERVE_PID=$!
    sleep 90
    MUJOCO_GL=osmesa python scripts/eval_pi05_safelibero.py \
        --host 0.0.0.0 --port 8002 \
        --suite safelibero_object --safety_level II \
        --n_episodes 50 --max_steps 300 --replan_steps 5 --seed 7 \
        --save_dir $OUT_DIR \
        --output_path ${OUT_DIR}/eval_d012_final_s1_l005.json
    kill $SERVE_PID 2>/dev/null
    echo "DONE: s1_l005"
) &

# s1_l01 on cuda:3, port 8003
(
    CUDA_VISIBLE_DEVICES=3 python openpi/scripts/serve_policy.py \
        --env libero \
        --policy:checkpoint \
        --policy.config pi05_libero \
        --policy.dir ${CKPT_BASE}/a1_sdf_pi05_object_pudding_seed1_l01/30000 \
        --port 8003 &
    SERVE_PID=$!
    sleep 90
    MUJOCO_GL=osmesa python scripts/eval_pi05_safelibero.py \
        --host 0.0.0.0 --port 8003 \
        --suite safelibero_object --safety_level II \
        --n_episodes 50 --max_steps 300 --replan_steps 5 --seed 7 \
        --save_dir $OUT_DIR \
        --output_path ${OUT_DIR}/eval_d012_final_s1_l01.json
    kill $SERVE_PID 2>/dev/null
    echo "DONE: s1_l01"
) &

# 等待 Batch 1 全部完成
wait
echo "=== Batch 1 complete ==="

# ============================================================
# Batch 2: s2_l005, s2_l01 (cuda:0,1)
# ============================================================

echo "=== Starting Batch 2 (2 models parallel) ==="

# s2_l005 on cuda:0, port 8000
(
    CUDA_VISIBLE_DEVICES=0 python openpi/scripts/serve_policy.py \
        --env libero \
        --policy:checkpoint \
        --policy.config pi05_libero \
        --policy.dir ${CKPT_BASE}/a1_sdf_pi05_object_pudding_seed2_l005/30000 \
        --port 8000 &
    SERVE_PID=$!
    sleep 90
    MUJOCO_GL=osmesa python scripts/eval_pi05_safelibero.py \
        --host 0.0.0.0 --port 8000 \
        --suite safelibero_object --safety_level II \
        --n_episodes 50 --max_steps 300 --replan_steps 5 --seed 7 \
        --save_dir $OUT_DIR \
        --output_path ${OUT_DIR}/eval_d012_final_s2_l005.json
    kill $SERVE_PID 2>/dev/null
    echo "DONE: s2_l005"
) &

# s2_l01 on cuda:1, port 8001
(
    CUDA_VISIBLE_DEVICES=1 python openpi/scripts/serve_policy.py \
        --env libero \
        --policy:checkpoint \
        --policy.config pi05_libero \
        --policy.dir ${CKPT_BASE}/a1_sdf_pi05_object_pudding_seed2_l01/30000 \
        --port 8001 &
    SERVE_PID=$!
    sleep 90
    MUJOCO_GL=osmesa python scripts/eval_pi05_safelibero.py \
        --host 0.0.0.0 --port 8001 \
        --suite safelibero_object --safety_level II \
        --n_episodes 50 --max_steps 300 --replan_steps 5 --seed 7 \
        --save_dir $OUT_DIR \
        --output_path ${OUT_DIR}/eval_d012_final_s2_l01.json
    kill $SERVE_PID 2>/dev/null
    echo "DONE: s2_l01"
) &

wait
echo "=== Batch 2 complete ==="
echo "=== ALL EVAL DONE ==="

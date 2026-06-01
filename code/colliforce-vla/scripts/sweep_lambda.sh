#!/bin/bash
# ============================================================================
# Lambda Refine Sweep Script
# 串行遍历 lambda_refine 值列表，在指定场景上逐个训练
#
# 用法:
#   bash scripts/sweep_lambda.sh --lambdas "0.01 0.05 0.1 0.5" --scene spatial_bowl --seed 0 --gpu 0
#   bash scripts/sweep_lambda.sh --lambdas "0.01 0.1" --scene object_pudding --seed 0 --gpu 1
#
# 参数:
#   --lambdas       lambda_refine 值列表（空格分隔，引号包裹）  默认 "0.01 0.05 0.1 0.5"
#   --scene         场景名                                      必填
#   --seed          随机种子                                    默认 0
#   --gpu           GPU 编号                                    默认 0
#   --batch_size    批大小                                      默认 32
#   --save_interval 保存间隔                                    默认 2000
#   --max_steps     最大训练步数                                默认 30000
#
# 时间估算:
#   每个 lambda 值约 29h（30K steps × ~3.5s/it），4 个 lambda 串行 ≈ 116h 单卡
# ============================================================================

set -euo pipefail

# ---------- conda 环境激活 ----------
source <PROJECT_ROOT>/miniconda3/etc/profile.d/conda.sh && conda activate base

# ---------- 默认值 ----------
LAMBDAS="0.01 0.05 0.1 0.5"
SCENE=""
SEED=0
GPU=0
BATCH_SIZE=32
SAVE_INTERVAL=2000
MAX_STEPS=30000

# ---------- 参数解析 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lambdas)      LAMBDAS="$2";       shift 2 ;;
        --scene)        SCENE="$2";         shift 2 ;;
        --seed)         SEED="$2";          shift 2 ;;
        --gpu)          GPU="$2";           shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";    shift 2 ;;
        --save_interval) SAVE_INTERVAL="$2"; shift 2 ;;
        --max_steps)    MAX_STEPS="$2";     shift 2 ;;
        --help|-h)
            sed -n '2,/^# ====/p' "$0" | head -n -1
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$SCENE" ]]; then
    echo "Error: --scene is required" >&2
    exit 1
fi

# ---------- 项目根目录 ----------
PROJECT_DIR="<PROJECT_ROOT>/colliforce-vla"
cd "$PROJECT_DIR"

echo "========================================"
echo "Lambda Sweep: scene=${SCENE} seed=${SEED} gpu=${GPU}"
echo "Lambdas: ${LAMBDAS}"
echo "========================================"

# ---------- 环境变量 ----------
export JAX_PLATFORMS=cpu
export WANDB_MODE=offline

# ---------- 串行遍历 ----------
for LAMBDA in $LAMBDAS; do
    EXP_NAME="a3_lambda${LAMBDA}_${SCENE}_s${SEED}"
    LOG_DIR="outputs/${EXP_NAME}"
    mkdir -p "$LOG_DIR"

    echo ""
    echo "----------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${EXP_NAME} (lambda_refine=${LAMBDA})"
    echo "----------------------------------------"

    CUDA_VISIBLE_DEVICES=${GPU} python scripts/train_a3_hook.py \
        --scene "${SCENE}" \
        --seed "${SEED}" \
        --gpu 0 \
        --batch_size "${BATCH_SIZE}" \
        --num_train_steps "${MAX_STEPS}" \
        --save_interval "${SAVE_INTERVAL}" \
        --lambda_refine "${LAMBDA}" \
        --exp_name "${EXP_NAME}" \
        2>&1 | tee "${LOG_DIR}/train.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${EXP_NAME} (exit=$?)"
done

echo ""
echo "========================================"
echo "All lambda sweep runs completed."
echo "========================================"

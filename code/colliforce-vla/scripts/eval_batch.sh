#!/usr/bin/env bash
# ==============================================================================
# eval_batch.sh — 批量运行 SafeLIBERO 评估
#
# 三层循环：method × scene × seed
# 对每个组合自动推导 checkpoint 路径、选择 eval 参数、调用 eval_safelibero.py，
# 并将 TSR/CAR/avg_smoothness 等指标追加到 CSV 汇总文件。
#
# 参数：
#   --checkpoint_root   checkpoint 根目录（默认 <DATA_ROOT>/checkpoints/）
#   --methods           method 列表，空格分隔（默认 "baseline a1 a3 a3_no_refine"）
#   --scenes            场景列表，空格分隔（默认 "spatial_bowl object_pudding"）
#   --seeds             seed 列表，空格分隔（默认 "0 1"）
#   --n_episodes        每个 task 的 episode 数（默认 50）
#   --gpu               GPU ID（默认 0）
#   --output_csv        CSV 输出路径（默认 <DATA_ROOT>/eval_results/summary.csv）
#   --step              指定 checkpoint step（默认取最大数值子目录）
#   --dry_run           只打印命令不执行
#
# 示例：
#   # 评估所有 baseline
#   bash scripts/eval_batch.sh --methods "baseline" --scenes "spatial_bowl object_pudding" --seeds "0 1" --gpu 3
#
#   # 评估所有方法
#   bash scripts/eval_batch.sh --methods "baseline a1 a3 a3_no_refine" --scenes "spatial_bowl" --seeds "0" --gpu 3
#
#   # dry run 查看会执行哪些命令
#   bash scripts/eval_batch.sh --methods "baseline a1" --scenes "spatial_bowl" --seeds "0" --dry_run
# ==============================================================================

set -euo pipefail

# ======================== 默认参数 ========================
CHECKPOINT_ROOT="<DATA_ROOT>/checkpoints"
METHODS="baseline a1 a3 a3_no_refine"
SCENES="spatial_bowl object_pudding"
SEEDS="0 1"
N_EPISODES=50
GPU=0
OUTPUT_CSV="<DATA_ROOT>/eval_results/summary.csv"
STEP=""
DRY_RUN=false

# ======================== 参数解析 ========================
print_help() {
    sed -n '2,/^# ====.*===$/p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_root) CHECKPOINT_ROOT="$2"; shift 2 ;;
        --methods)         METHODS="$2"; shift 2 ;;
        --scenes)          SCENES="$2"; shift 2 ;;
        --seeds)           SEEDS="$2"; shift 2 ;;
        --n_episodes)      N_EPISODES="$2"; shift 2 ;;
        --gpu)             GPU="$2"; shift 2 ;;
        --output_csv)      OUTPUT_CSV="$2"; shift 2 ;;
        --step)            STEP="$2"; shift 2 ;;
        --dry_run)         DRY_RUN=true; shift ;;
        --help|-h)         print_help ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ======================== 场景映射 ========================
get_suite() {
    local scene="$1"
    case "$scene" in
        spatial_bowl)    echo "safelibero_spatial" ;;
        object_pudding)  echo "safelibero_object" ;;
        object_*)        echo "safelibero_object" ;;
        goal_*)          echo "safelibero_goal" ;;
        long_*)          echo "safelibero_long" ;;
        *) echo "safelibero_spatial" ;;
    esac
}

get_safety_level() {
    local scene="$1"
    case "$scene" in
        spatial_*)  echo "II" ;;
        object_*)   echo "I" ;;
        goal_*)     echo "II" ;;
        long_*)     echo "II" ;;
        *)          echo "II" ;;
    esac
}

get_config_name() {
    local method="$1" scene="$2"
    local base_method="$method"
    [[ "$method" == "a3_no_refine" ]] && base_method="a3"
    case "$base_method" in
        baseline) echo "baseline_${scene}" ;;
        a1)       echo "baseline_${scene}" ;;
        a3)       echo "baseline_${scene}" ;;
        *)        echo "baseline_${scene}" ;;
    esac
}

# 获取 checkpoint 目录名前缀
get_ckpt_dir_name() {
    local method="$1" scene="$2"
    case "$method" in
        baseline)      echo "baseline_${scene}" ;;
        a1)            echo "a1_sdf_${scene}" ;;
        a3|a3_no_refine) echo "a3_grad_refine_${scene}" ;;
        *)             echo "${method}_${scene}" ;;
    esac
}

# 找最新 step 子目录（数值最大的）
find_latest_step() {
    local ckpt_dir="$1"
    if [[ -n "$STEP" ]]; then
        echo "$STEP"
        return
    fi
    # ls 数值目录，取最大值；忽略 best 等非数值目录
    local latest
    latest=$(ls -1 "$ckpt_dir" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
    if [[ -z "$latest" ]]; then
        # fallback: 如果有 best 目录
        if [[ -d "${ckpt_dir}/best" ]]; then
            echo "best"
        else
            echo ""
        fi
    else
        echo "$latest"
    fi
}

# ======================== SDF GT 路径 ========================
get_sdf_gt_path() {
    local scene="$1"
    local gt_path="./data/sdf_gt_${scene}.npz"
    if [[ -f "$gt_path" ]]; then
        echo "$gt_path"
    else
        echo ""
    fi
}

# ======================== CSV 初始化 ========================
CSV_DIR=$(dirname "$OUTPUT_CSV")
mkdir -p "$CSV_DIR"

if [[ ! -f "$OUTPUT_CSV" ]]; then
    echo "method,scene,seed,checkpoint_step,TSR,CAR,ETS,avg_smoothness,eval_time" > "$OUTPUT_CSV"
fi

# ======================== 辅助函数 ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_safelibero.py"

extract_metric() {
    local json_file="$1" metric="$2"
    python3 -c "
import json, sys, numpy as np
with open('$json_file') as f:
    data = json.load(f)
results = data['results']
if isinstance(results[0], dict) and '$metric' in results[0]:
    # 多 task 汇总
    vals = [r['$metric'] for r in results]
    print(f'{np.mean(vals):.6f}')
else:
    print('N/A')
"
}

extract_ets() {
    local json_file="$1"
    python3 -c "
import json, numpy as np
with open('$json_file') as f:
    data = json.load(f)
results = data['results']
# ETS = TSR * CAR (Effective Task Safety)
tsrs = [r.get('TSR', 0) for r in results]
cars = [r.get('CAR', 0) for r in results]
ets_vals = [t * c for t, c in zip(tsrs, cars)]
print(f'{np.mean(ets_vals):.6f}')
"
}

# ======================== 主循环 ========================
TOTAL=0
SUCCESS=0
FAIL=0

for method in $METHODS; do
    for scene in $SCENES; do
        for seed in $SEEDS; do
            TOTAL=$((TOTAL + 1))

            echo "========================================"
            echo "[${TOTAL}] method=${method} scene=${scene} seed=${seed}"
            echo "========================================"

            # --- 推导路径 ---
            ckpt_dir_name=$(get_ckpt_dir_name "$method" "$scene")
            seed_dir="${CHECKPOINT_ROOT}/${ckpt_dir_name}/${ckpt_dir_name}_seed${seed}"

            if [[ ! -d "$seed_dir" ]]; then
                echo "SKIP: checkpoint dir not found: ${seed_dir}"
                FAIL=$((FAIL + 1))
                continue
            fi

            step=$(find_latest_step "$seed_dir")
            if [[ -z "$step" ]]; then
                echo "SKIP: no step subdirectory in ${seed_dir}"
                FAIL=$((FAIL + 1))
                continue
            fi

            ckpt_path="${seed_dir}/${step}"
            suite=$(get_suite "$scene")
            safety_level=$(get_safety_level "$scene")
            config_name=$(get_config_name "$method" "$scene")

            # --- 构建 eval 命令 ---
            CMD="python3 ${EVAL_SCRIPT}"
            CMD+=" --checkpoint ${ckpt_path}"
            CMD+=" --config_name ${config_name}"
            CMD+=" --suite ${suite}"
            CMD+=" --safety_level ${safety_level}"
            CMD+=" --n_episodes ${N_EPISODES}"
            CMD+=" --seed ${seed}"
            CMD+=" --gpu ${GPU}"

            # method 特定参数
            case "$method" in
                baseline)
                    CMD+=" --model_type baseline"
                    ;;
                a1)
                    CMD+=" --model_type a1"
                    sdf_ckpt="${ckpt_path}/sdf_decoder.pt"
                    if [[ -f "$sdf_ckpt" ]]; then
                        CMD+=" --sdf_checkpoint ${sdf_ckpt}"
                    fi
                    ;;
                a3)
                    CMD+=" --model_type a3"
                    sdf_gt=$(get_sdf_gt_path "$scene")
                    if [[ -n "$sdf_gt" ]]; then
                        CMD+=" --sdf_gt_path ${sdf_gt}"
                    fi
                    ;;
                a3_no_refine)
                    CMD+=" --model_type a3 --no_refine"
                    sdf_gt=$(get_sdf_gt_path "$scene")
                    if [[ -n "$sdf_gt" ]]; then
                        CMD+=" --sdf_gt_path ${sdf_gt}"
                    fi
                    ;;
            esac

            echo "CMD: ${CMD}"

            if $DRY_RUN; then
                echo "[DRY RUN] Skipping execution."
                continue
            fi

            # --- 执行 eval ---
            START_TIME=$(date +%s)

            if ! eval "$CMD"; then
                echo "FAIL: eval crashed for method=${method} scene=${scene} seed=${seed}"
                FAIL=$((FAIL + 1))
                continue
            fi

            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))

            # --- 提取结果 ---
            ckpt_basename=$(basename "$ckpt_path")
            model_tag=""
            [[ "$method" != "baseline" ]] && model_tag="_${method}"
            [[ "$method" == "a3_no_refine" ]] && model_tag="_a3"
            result_json="${CSV_DIR}/eval_${suite}_all_level${safety_level}_${ckpt_basename}${model_tag}_seed${seed}.json"

            if [[ ! -f "$result_json" ]]; then
                # 尝试 save_dir 默认路径
                result_json="<DATA_ROOT>/eval_results/eval_${suite}_all_level${safety_level}_${ckpt_basename}${model_tag}_seed${seed}.json"
            fi

            if [[ -f "$result_json" ]]; then
                TSR=$(extract_metric "$result_json" "TSR")
                CAR=$(extract_metric "$result_json" "CAR")
                ETS=$(extract_ets "$result_json")
                AVG_SMOOTH=$(extract_metric "$result_json" "avg_smoothness")
            else
                echo "WARNING: result json not found, using N/A"
                TSR="N/A"
                CAR="N/A"
                ETS="N/A"
                AVG_SMOOTH="N/A"
            fi

            # --- 追加 CSV ---
            echo "${method},${scene},${seed},${step},${TSR},${CAR},${ETS},${AVG_SMOOTH},${ELAPSED}s" >> "$OUTPUT_CSV"
            echo "=> TSR=${TSR} CAR=${CAR} ETS=${ETS} smoothness=${AVG_SMOOTH} time=${ELAPSED}s"

            SUCCESS=$((SUCCESS + 1))
        done
    done
done

echo ""
echo "========================================"
echo "Batch eval complete: ${SUCCESS} success, ${FAIL} fail, ${TOTAL} total"
echo "Results saved to: ${OUTPUT_CSV}"
echo "========================================"

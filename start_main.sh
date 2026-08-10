#!/bin/bash

set -u

# ==============================================================================
# 1. Base configuration
# ==============================================================================
DATA_ROOT="${DATA_ROOT:-./datasets}"
PY_SCRIPT="${PY_SCRIPT:-main.py}"
PROMPT_PATH="${PROMPT_PATH:-./prompt_template/user_template.txt}"
SKIP_DATASETS="${SKIP_DATASETS:-}"

# ==============================================================================
# 2. GPU / service configuration
# ==============================================================================
LLM_GPUS="${LLM_GPUS:-0}"
XGB_GPUS="${XGB_GPUS:-0}"
LLM_BASE_PORT="${LLM_BASE_PORT:-8000}"

MODEL_NAME_TEMPLATE="${MODEL_NAME_TEMPLATE:-qwen3_gpu0}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "$MODEL_NAME_TEMPLATE" | sed 's/%d/gpu/g' | tr '/' '_' | tr '-' '_')}"

read -ra LLM_GPU_ARR <<< "$LLM_GPUS"
read -ra XGB_GPU_ARR <<< "$XGB_GPUS"
NUM_XGB=${#XGB_GPU_ARR[@]}

# ==============================================================================
# 3. Experiment arguments for main.py
# ==============================================================================
STEPS="${STEPS:-25}"
MASKED="${MASKED:-True}"
REJECT_WEAK="${REJECT_WEAK:-False}"
TRACK_TEST_CURVE="${TRACK_TEST_CURVE:-True}"
LLM_PRESET="${LLM_PRESET:-coding}"
TEMPERATURE="${TEMPERATURE:-}"
RANDOM_SEED="${RANDOM_SEED:-42}"

TASK_MODELS=(${TASK_MODELS:-xgboost})

# Sweep controls
ABLATION_MODES="${ABLATION_MODES:-ours}"
NOISE_LEVELS="${NOISE_LEVELS:-0.1}"

# ==============================================================================
# ★ 4. Quick Run Configuration (For Reviewers) ★
# ==============================================================================
TARGET_DATASET="${TARGET_DATASET:-}"
TARGET_SEED="${TARGET_SEED:-}"

# ==============================================================================
# 5. Results base directory (exp_name 会在模型循环中动态设置)
# ==============================================================================
BASE_RESULTS="${BASE_RESULTS:-./results/$(basename "$PY_SCRIPT" .py)}"
HYPER_TAG="steps${STEPS}_preset_${LLM_PRESET}"
[ "$REJECT_WEAK" == "True" ] && HYPER_TAG="${HYPER_TAG}_reject"

# ==============================================================================
# 6. Validation
# ==============================================================================
if [ ! -f "$PY_SCRIPT" ]; then
    echo "Error: Python script '$PY_SCRIPT' not found."
    exit 1
fi
if [ ! -f "$PROMPT_PATH" ]; then
    echo "Error: Prompt template '$PROMPT_PATH' not found."
    exit 1
fi
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root '$DATA_ROOT' not found."
    exit 1
fi
if [ "$NUM_XGB" -eq 0 ]; then
    echo "Error: XGB_GPUS must provide at least one GPU id."
    exit 1
fi

# ==============================================================================
# 7. Build mode / noise combinations
# ==============================================================================
declare -a MODE_NOISE_TASKS=()
read -ra ABLATION_MODE_ARR <<< "$ABLATION_MODES"
read -ra NOISE_LEVEL_ARR <<< "$NOISE_LEVELS"

for ablation_mode in "${ABLATION_MODE_ARR[@]}"; do
    case "$ablation_mode" in
        ours|random_grouping|rank_only|noise_sensitivity)
            for noise_level in "${NOISE_LEVEL_ARR[@]}"; do
                MODE_NOISE_TASKS+=("${ablation_mode}|${noise_level}")
            done
            ;;
        no_grouping|no_noise)
            MODE_NOISE_TASKS+=("${ablation_mode}|0.1")
            ;;
        *)
            echo "Error: Unsupported ablation mode '$ablation_mode'."
            exit 1
            ;;
    esac
done

# ==============================================================================
# 8. Collect all tasks (dataset × seed × ablation × noise)
# ==============================================================================
declare -a TASKS=()
for dataset_dir in "$DATA_ROOT"/*; do
    [ ! -d "$dataset_dir" ] && continue
    dataset_name=$(basename "$dataset_dir")

    if [[ " ${SKIP_DATASETS} " =~ " ${dataset_name} " ]]; then
        continue
    fi
    if [ -n "$TARGET_DATASET" ] && [ "$dataset_name" != "$TARGET_DATASET" ]; then
        continue
    fi

    for seed_dir in "$dataset_dir"/*; do
        [ ! -d "$seed_dir" ] && continue
        seed=$(basename "$seed_dir")

        if [ -n "$TARGET_SEED" ]; then
            if [[ ! "$seed" =~ ${TARGET_SEED}$ ]]; then
                continue
            fi
        elif [ -n "$TARGET_DATASET" ]; then
            if [[ ! "$seed" =~ 0$ ]]; then
                continue
            fi
        fi

        for mode_noise in "${MODE_NOISE_TASKS[@]}"; do
            IFS='|' read -r ablation_mode noise_level <<< "$mode_noise"
            TASKS+=("${seed_dir}|${dataset_name}|${seed}|${ablation_mode}|${noise_level}")
        done
    done
done

TOTAL=${#TASKS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "No tasks found. Check DATA_ROOT, SKIP_DATASETS, TARGET_DATASET, or TARGET_SEED."
    exit 0
fi

echo "========================================================"
echo "Starting SEQUENTIAL experiments"
echo "Data Root:         $DATA_ROOT"
echo "Target Dataset:    ${TARGET_DATASET:-[ALL DATASETS]}"
if [ -n "$TARGET_DATASET" ] && [ -z "$TARGET_SEED" ]; then
    echo "Target Seed:       0 (Default for single dataset run)"
else
    echo "Target Seed:       ${TARGET_SEED:-[ALL SEEDS]}"
fi
echo "XGB GPUs:          ${XGB_GPU_ARR[*]} (Round-robin allocation)"
echo "LLM Base Port:     $LLM_BASE_PORT"
echo "Ablation Modes:    $ABLATION_MODES"
echo "Models to test:    ${TASK_MODELS[*]}"
echo "Total Tasks:       $TOTAL (will run for each model)"
echo "========================================================"

llm_gpu_default=${LLM_GPU_ARR[0]}
served_model=$(printf "$MODEL_NAME_TEMPLATE" "$llm_gpu_default")
port=$LLM_BASE_PORT

# ==============================================================================
# 9. Main loops: 遍历每个模型，再遍历所有任务
# ==============================================================================
for task_model in "${TASK_MODELS[@]}"; do
    EXP_NAME="$task_model"
    TASK_MODEL="$task_model"
    FINAL_SAVE_DIR="${BASE_RESULTS}/${EXP_NAME}/${MODEL_TAG}/${HYPER_TAG}"

    mkdir -p "$FINAL_SAVE_DIR"
    cp "$PY_SCRIPT" "$FINAL_SAVE_DIR/script_backup.py"
    cp "$PROMPT_PATH" "$FINAL_SAVE_DIR/prompt_backup.txt"

    echo "========================================================"
    echo "Running model: $task_model  |  Results -> $FINAL_SAVE_DIR"
    echo "========================================================"

    for (( i=0; i<TOTAL; i++ )); do
        IFS='|' read -r seed_path dataset_name seed ablation_mode noise_level <<< "${TASKS[$i]}"

        xgb_idx=$(( i % NUM_XGB ))
        xgb_gpu=${XGB_GPU_ARR[$xgb_idx]}

        mkdir -p "${FINAL_SAVE_DIR}/stdout_logs/${dataset_name}"
        noise_tag=${noise_level//./p}
        log_file="${FINAL_SAVE_DIR}/stdout_logs/${dataset_name}/seq_${ablation_mode}_noise_${noise_tag}_seed_${seed}.log"

        echo ">> [Task $((i+1))/$TOTAL] Model=$task_model " \
             "Dataset=$dataset_name Seed=$seed Mode=$ablation_mode Noise=$noise_level GPU=$xgb_gpu"

        cmd=(
            python -u "$PY_SCRIPT"
            --dataset_path "$seed_path"
            --masked "$MASKED"
            --steps "$STEPS"
            --model "$served_model"
            --results_dir "$FINAL_SAVE_DIR"
            --reject_weak "$REJECT_WEAK"
            --port "$port"
            --prompt_path "$PROMPT_PATH"
            --ablation_mode "$ablation_mode"
            --noise_level "$noise_level"
            --random_seed "$RANDOM_SEED"
            --track_test_curve "$TRACK_TEST_CURVE"
            --llm_preset "$LLM_PRESET"
            --task_model "$TASK_MODEL"
        )
        [ -n "$TEMPERATURE" ] && cmd+=(--temperature "$TEMPERATURE")

        CUDA_VISIBLE_DEVICES=$xgb_gpu "${cmd[@]}" 2>&1 | tee "$log_file"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo ">> [SUCCESS] Task $((i+1))/$TOTAL completed."
        else
            echo ">> [FAILED]  Task $((i+1))/$TOTAL failed. Stopping execution."
            exit 1
        fi
    done
    echo ">> [Model: $task_model] All tasks done. Generating plots..."
    python plot_results.py \
        --results_dir "$FINAL_SAVE_DIR" \
        --model_name "$task_model"   # 可选，如果绘图脚本需要区分

    if [ $? -eq 0 ]; then
        echo ">> Plotting done for $task_model."
    else
        echo ">> Plotting failed for $task_model, but continuing to next model."
    fi
done

echo ">> All tasks for all models done."

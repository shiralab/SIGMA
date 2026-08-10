#!/bin/bash
# specify GPUs: ./start_llm_service.sh 0 1 2 3
# Default: GPU 0, Prot: 8000

MODEL="Qwen/Qwen3-4B-Instruct-2507"

# 1. Parse Arguments: Default to (0) if no arguments are passed
if [ "$#" -eq 0 ]; then
    GPUS=(0)
    echo ">> No GPUs specified. Defaulting to GPU 0."
else
    GPUS=("$@")
    echo ">> Using specified GPUs: ${GPUS[*]}"
fi

# 2. Start Servers
for GPU_ID in "${GPUS[@]}"; do
    PORT=$((8000 + GPU_ID))
    SERVED_NAME="qwen3_gpu${GPU_ID}"
    echo ">> Starting server on GPU $GPU_ID, Port $PORT, Name $SERVED_NAME"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --served-model-name "$SERVED_NAME" \
        --seed 42 \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.85 \
        --enable-prefix-caching \
        --port "$PORT" \
        --max-model-len 65536 &
done

# 3. Wait for Readiness
echo ">> Waiting for all servers to be ready..."
for i in $(seq 1 60); do
    ALL_OK=true
    for GPU_ID in "${GPUS[@]}"; do
        PORT=$((8000 + GPU_ID))
        curl -s "http://localhost:${PORT}/health" > /dev/null || ALL_OK=false
    done
    if $ALL_OK; then
        echo ">> All servers ready after $((i * 10)) seconds"
        break
    fi
    echo ">> Waiting... ($i/60)"
    sleep 10
done

# 4. Final Health Check
echo ">> ========================================="
for GPU_ID in "${GPUS[@]}"; do
    PORT=$((8000 + GPU_ID))
    SERVED_NAME="qwen3_gpu${GPU_ID}"
    if curl -s "http://localhost:${PORT}/health" > /dev/null; then
        echo ">> [OK]   GPU $GPU_ID | Port $PORT | Model $SERVED_NAME"
    else
        echo ">> [WARN] GPU $GPU_ID | Port $PORT not responding"
    fi
done
echo ">> ========================================="
echo ">> Startup script finished!"
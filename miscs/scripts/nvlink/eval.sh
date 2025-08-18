#!/bin/bash
# Evaluation execution script for lmms-eval

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source configuration
source "$SCRIPT_DIR/config.sh"

# Arguments
model_name=$1
task_name=$2
num_gpus=${3:-1}
limit=${4:-""}

echo "=========================================="
echo "Starting lmms-eval evaluation"
echo "=========================================="
echo "Model: $model_name"
echo "Task: $task_name"
echo "GPUs: $num_gpus"
echo "Limit: ${limit:-no limit}"
echo "=========================================="

# Load environment variables
load_env_file

# Get model type and args
model_type=${MODEL_TYPE_MAP[$model_name]}
safe_model_name=$(get_safe_model_name "$model_name")
model_args=$(get_model_args "$model_name")

# Check if flash-attn is actually available and adjust args if needed
if [[ "$model_type" == "qwen2_5_vl" || "$model_type" == "qwen2_vl" ]]; then
    if ! python -c "import flash_attn" 2>/dev/null; then
        echo "flash-attn not available, using sdpa instead"
        # Replace flash_attention_2 with sdpa in model_args
        model_args=$(echo "$model_args" | sed 's/attn_implementation=flash_attention_2/attn_implementation=sdpa/g')
    else
        echo "Using flash_attention_2 for better performance"
    fi
fi

# Set output paths
output_dir="$OUTPUT_BASE_DIR/${safe_model_name}/${task_name}"
mkdir -p "$output_dir"

# Change to repo directory
cd "$REPO_PATH"

# Activate the uv environment
# TODO: Change by model_type
source .uv/qwen/bin/activate
# uv pip install -e .
# uv sync --active --extra qwen
# uv sync --active --extra qwen --extra flash_attn

# This happens inside SLURM job where GPU is available
model_type=${MODEL_TYPE_MAP[$model_name]}

# Build the command
CMD="python -m lmms_eval"
CMD="$CMD --model $model_type"
CMD="$CMD --model_args $model_args"
CMD="$CMD --tasks $task_name"
CMD="$CMD --batch_size $DEFAULT_BATCH_SIZE"
CMD="$CMD --output_path $output_dir"
CMD="$CMD --log_samples"
CMD="$CMD --log_samples_suffix ${task_name}_${safe_model_name}"

# Add limit if specified
if [ -n "$limit" ]; then
    CMD="$CMD --limit $limit"
fi

# Add device configuration for multi-GPU
if [ "$num_gpus" -gt 1 ]; then
    # For multi-GPU, we'll use accelerate launch
    CMD="accelerate launch --num_processes=$num_gpus --main_process_port 29500 -m lmms_eval"
    CMD="$CMD --model $model_type"
    CMD="$CMD --model_args $model_args"
    CMD="$CMD --tasks $task_name"
    CMD="$CMD --batch_size $DEFAULT_BATCH_SIZE"
    CMD="$CMD --output_path $output_dir"
    CMD="$CMD --log_samples"
    CMD="$CMD --log_samples_suffix ${task_name}_${safe_model_name}"
    
    if [ -n "$limit" ]; then
        CMD="$CMD --limit $limit"
    fi
fi

echo "=========================================="
echo "Executing command:"
echo "$CMD"
echo "=========================================="

# Run the evaluation and capture exit code
$CMD
EXIT_CODE=$?

echo "=========================================="
echo "Evaluation completed with exit code: $EXIT_CODE"
echo "Results saved to: $output_dir"
echo "=========================================="

# Exit with the same code
exit $EXIT_CODE
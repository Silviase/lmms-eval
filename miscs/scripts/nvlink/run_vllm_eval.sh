#!/bin/bash
#SBATCH --job-name=vllm_eval
#SBATCH --output=vllm_eval_%j.out
#SBATCH --error=vllm_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8

# vLLM Evaluation Script for NVLink Multi-GPU Setup
# This script runs lmms-eval with vLLM backend on multiple GPUs connected via NVLink
export HF_HOME=/data/silviase/.hf_cache

# Navigate to project directory
cd /home/silviase/LIMIT-Lab/typographic_atk/lmms-eval

# Activate virtual environment and sync dependencies
echo "========================================="
echo "Setting up environment..."
echo "========================================="
source .uv/qwen/bin/activate
# uv sync --active --extra qwen
# uv sync --active --extra qwen --extra flash_attn --extra vllm
echo ""

# Environment setup for NVLink/multi-GPU communication
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=INFO  # Set to DEBUG for more verbose output
# CUDA_VISIBLE_DEVICES is managed by SLURM, not setting explicitly

# Load environment variables from .env file if it exists
if [ -f "/home/silviase/LIMIT-Lab/typographic_atk/lmms-eval/.env" ]; then
    export $(grep -v '^#' /home/silviase/LIMIT-Lab/typographic_atk/lmms-eval/.env | xargs)
fi

# Hugging Face cache directory (can be overridden by .env)
export HF_HOME="${HF_HOME:-/data/silviase/.cache/huggingface}"

# Login to Hugging Face if token is available
if [ -n "$HF_TOKEN" ]; then
    hf auth login --token $HF_TOKEN
fi

# Print environment info
echo "========================================="
echo "Environment Information"
echo "========================================="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo ""

# Check GPU availability
echo "========================================="
echo "GPU Information"
echo "========================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Check CUDA in PyTorch
echo "========================================="
echo "PyTorch CUDA Check"
echo "========================================="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch CUDA check failed"
echo ""

# Script parameters
MODEL_NAME="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
TASKS="${2:-chartqa,docvqa_val,infovqa_val,textvqa}"
BATCH_SIZE="${3:-1024}"
TENSOR_PARALLEL="${4:-8}"  # Number of GPUs for tensor parallelism
OUTPUT_DIR="${5:-./results_vllm}"
LIMIT="${6:-}"  # Optional limit for debugging

# Build model arguments
MODEL_ARGS="model=${MODEL_NAME}"
MODEL_ARGS="${MODEL_ARGS},tensor_parallel_size=${TENSOR_PARALLEL}"
MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=0.9"
MODEL_ARGS="${MODEL_ARGS},trust_remote_code=True"
MODEL_ARGS="${MODEL_ARGS},max_pixels=12845056"  # For high-res images

# Optional: Add data parallel for even more GPUs
# MODEL_ARGS="${MODEL_ARGS},data_parallel_size=2"

# Build command
CMD="python -m lmms_eval"
CMD="${CMD} --model vllm"
CMD="${CMD} --model_args ${MODEL_ARGS}"
CMD="${CMD} --tasks ${TASKS}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --log_samples"
CMD="${CMD} --log_samples_suffix vllm_nvlink"
CMD="${CMD} --output_path ${OUTPUT_DIR}"

# Add limit if specified
if [ -n "${LIMIT}" ]; then
    CMD="${CMD} --limit ${LIMIT}"
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Log configuration
echo "========================================="
echo "vLLM Evaluation Configuration"
echo "========================================="
echo "Model: ${MODEL_NAME}"
echo "Tasks: ${TASKS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Tensor Parallel: ${TENSOR_PARALLEL}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Output: ${OUTPUT_DIR}"
if [ -n "${LIMIT}" ]; then
    echo "Limit: ${LIMIT}"
fi
echo "========================================="
echo ""

# Run evaluation
echo "Starting evaluation..."
echo "Command: ${CMD}"
echo ""

${CMD}

echo ""
echo "Evaluation complete! Results saved to: ${OUTPUT_DIR}"
#!/bin/bash
#SBATCH --job-name=197_test_vllm_internvl3
#SBATCH --output=vllm_test_%j.out
#SBATCH --error=vllm_test_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Quick Test Script for vLLM Evaluation
# Tests basic functionality with minimal resources

# Navigate to project directory
cd /home/silviase/LIMIT-Lab/typographic_atk/lmms-eval

# Activate virtual environment and sync dependencies
echo "========================================="
echo "Setting up environment..."
echo "========================================="
source .uv/qwen/bin/activate
uv sync --active --extra qwen
uv sync --active --extra qwen --extra flash_attn --extra vllm
echo ""

# Environment setup
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=INFO
export HF_HOME="/data/silviase/.cache/huggingface"
# CUDA_VISIBLE_DEVICES is managed by SLURM, not setting explicitly

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

# Test parameters (small model, single task, limited samples)
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"  # Smallest model
TASKS="mme"  # Single, simple task
BATCH_SIZE=4
TENSOR_PARALLEL=1  # Single GPU
OUTPUT_DIR="./results_vllm_test_$(date +%Y%m%d_%H%M%S)"
LIMIT=5  # Only process 5 samples for quick test

# Build model arguments
MODEL_ARGS="model=${MODEL_NAME}"
MODEL_ARGS="${MODEL_ARGS},tensor_parallel_size=${TENSOR_PARALLEL}"
MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=0.9"
MODEL_ARGS="${MODEL_ARGS},trust_remote_code=True"
MODEL_ARGS="${MODEL_ARGS},max_pixels=6422528"  # Lower resolution for quick test

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Log configuration
echo "========================================="
echo "Quick Test Configuration"
echo "========================================="
echo "Model: ${MODEL_NAME}"
echo "Tasks: ${TASKS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Tensor Parallel: ${TENSOR_PARALLEL}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Output: ${OUTPUT_DIR}"
echo "Limit: ${LIMIT} samples"
echo "========================================="
echo ""

# Run evaluation
echo "Starting quick test evaluation..."
CMD="python -m lmms_eval"
CMD="${CMD} --model vllm"
CMD="${CMD} --model_args ${MODEL_ARGS}"
CMD="${CMD} --tasks ${TASKS}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --limit ${LIMIT}"
CMD="${CMD} --log_samples"
CMD="${CMD} --log_samples_suffix vllm_quick_test"
CMD="${CMD} --output_path ${OUTPUT_DIR}"

echo "Command: ${CMD}"
echo ""

# Execute with timing
start_time=$(date +%s)
${CMD}
exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "========================================="
if [ ${exit_code} -eq 0 ]; then
    echo "✅ Quick test completed successfully!"
    echo "Duration: ${duration} seconds"
    echo "Results saved to: ${OUTPUT_DIR}"
else
    echo "❌ Quick test failed with exit code: ${exit_code}"
    echo "Duration: ${duration} seconds"
    echo "Check the error messages above for details."
fi
echo "========================================="

exit ${exit_code}
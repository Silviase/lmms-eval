#!/bin/bash
#SBATCH --job-name=197_test_vllm_internvl3
#SBATCH --output=vllm_test_%j.out
#SBATCH --error=vllm_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

export HF_HOME=/data/silviase/.hf_cache
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Navigate to project directory
cd /home/silviase/LIMIT-Lab/typographic_atk/lmms-eval

# Activate virtual environment
source .uv/qwen/bin/activate
uv sync --active --extra qwen
uv sync --active --extra qwen --extra flash_attn --extra vllm

# Set debug logging
export VLLM_LOGGING_LEVEL=DEBUG

# Print environment info
echo "========================================="
echo "Environment Information"
echo "========================================="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Pytorch version: $(python -c 'import torch; print(torch.__version__)')"
echo -e "uv pip list:\n $(uv pip list)"
echo ""

# Check GPU availability
echo "========================================="
echo "GPU Information"
echo "========================================="
nvidia-smi
echo ""

# Check CUDA in PyTorch
echo "========================================="
echo "PyTorch CUDA Check"
echo "========================================="
uv run --active python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Test vLLM model compatibility
echo "========================================="
echo "Testing vLLM Model Compatibility"
echo "========================================="
echo "Model: OpenGVLab/InternVL3-8B"
echo ""

uv run --active python miscs/scripts/nvlink/check_vllm_model.py --trust-remote-code OpenGVLab/InternVL3-8B

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model test completed successfully!"
else
    echo ""
    echo "❌ Model test failed. Check the error messages above."
fi

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
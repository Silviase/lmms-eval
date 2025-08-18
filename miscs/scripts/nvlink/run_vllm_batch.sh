#!/bin/bash
#SBATCH --job-name=vllm_batch
#SBATCH --output=vllm_batch_%j.out
#SBATCH --error=vllm_batch_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# Batch vLLM Evaluation Script for Multiple Models and Tasks
# Optimized for NVLink multi-GPU systems
export HF_HOME=/data/silviase/.hf_cache

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

# Base configuration
OUTPUT_BASE="./results_vllm_batch"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${OUTPUT_BASE}/logs_${TIMESTAMP}"

# Create directories
mkdir -p ${OUTPUT_BASE}
mkdir -p ${LOG_DIR}

# Define models to evaluate
MODELS=(
    # "Qwen/Qwen2.5-VL-3B-Instruct:2:16"     # model:tensor_parallel:batch_size
    # "Qwen/Qwen2.5-VL-7B-Instruct:4:8"
    "Qwen/Qwen2.5-VL-32B-Instruct:4:4"
    "OpenGVLab/InternVL3-14B:2:8"
    # Add more models here
    # "meta-llama/Llama-3.2-11B-Vision-Instruct:4:8"
    # "microsoft/Phi-3.5-vision-instruct:2:16"
)

# Define task groups
declare -A TASK_GROUPS
TASK_GROUPS["text_vqa"]="chartqa,docvqa_val,infovqa_val,textvqa"
TASK_GROUPS["general_vqa"]="mme,gqa,vqav2"
TASK_GROUPS["academic"]="mmmu_val,mathvista_testmini,ai2d"
TASK_GROUPS["reasoning"]="mme_cot_reason,mathverse_testmini"

# Function to run evaluation
run_evaluation() {
    local model_spec=$1
    local task_group_name=$2
    local tasks=$3
    
    # Parse model specification
    IFS=':' read -r model_name tensor_parallel batch_size <<< "${model_spec}"
    
    # Clean model name for directory
    model_dir=$(echo ${model_name} | sed 's/\//_/g')
    
    # Output paths
    output_dir="${OUTPUT_BASE}/${model_dir}/${task_group_name}_${TIMESTAMP}"
    log_file="${LOG_DIR}/${model_dir}_${task_group_name}.log"
    
    echo "=========================================" | tee -a ${log_file}
    echo "Starting: ${model_name} on ${task_group_name}" | tee -a ${log_file}
    echo "Time: $(date)" | tee -a ${log_file}
    echo "=========================================" | tee -a ${log_file}
    
    # Build model arguments
    MODEL_ARGS="model=${model_name}"
    MODEL_ARGS="${MODEL_ARGS},tensor_parallel_size=${tensor_parallel}"
    MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=0.9"
    MODEL_ARGS="${MODEL_ARGS},trust_remote_code=True"
    
    # Adjust max_pixels based on model
    if [[ ${model_name} == *"32B"* ]]; then
        MODEL_ARGS="${MODEL_ARGS},max_pixels=6422528"  # Lower for 32B model
    else
        MODEL_ARGS="${MODEL_ARGS},max_pixels=12845056"
    fi
    
    # Run evaluation
    python -m lmms_eval \
        --model vllm \
        --model_args ${MODEL_ARGS} \
        --tasks ${tasks} \
        --batch_size ${batch_size} \
        --log_samples \
        --log_samples_suffix vllm_${task_group_name} \
        --output_path ${output_dir} \
        2>&1 | tee -a ${log_file}
    
    echo "Completed: ${model_name} on ${task_group_name}" | tee -a ${log_file}
    echo "" | tee -a ${log_file}
}

# Main execution
echo "Starting batch vLLM evaluation"
echo "Timestamp: ${TIMESTAMP}"
echo "Output directory: ${OUTPUT_BASE}"
echo ""

# Iterate through models and task groups
for model_spec in "${MODELS[@]}"; do
    for task_group_name in "${!TASK_GROUPS[@]}"; do
        tasks="${TASK_GROUPS[${task_group_name}]}"
        run_evaluation "${model_spec}" "${task_group_name}" "${tasks}"
    done
done

echo "All evaluations complete!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "Logs saved to: ${LOG_DIR}"
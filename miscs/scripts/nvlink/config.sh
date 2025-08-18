#!/bin/bash
# Configuration file for lmms-eval GPU evaluation scripts
# Focused on typographic attack research

# === Path Configuration ===
export ROOT_DIR="/home/silviase/LIMIT-Lab/typographic_atk"
export DATA_DIR="/data/silviase"
export REPO_PATH="$ROOT_DIR/lmms-eval"
export SCRIPT_PATH="$REPO_PATH/miscs/scripts/nvlink"

# HuggingFace cache directories
export HF_HOME="$DATA_DIR/.hf_cache"
export HF_DATASETS_CACHE="$DATA_DIR/datasets"
export HF_HUB_CACHE="$DATA_DIR/models"
export APPTAINER_CACHEDIR="$DATA_DIR/apptainer_cache"

# CUDA configuration
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

# === Model List (for typographic attack research) ===
declare -a model_list=(
    # Qwen2.5-VL series (excellent OCR capabilities)
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen/Qwen2.5-VL-32B-Instruct"
    
    # InternVL2 series (strong vision understanding)
    "OpenGVLab/InternVL3-8B"
    "OpenGVLab/InternVL3-38B"
    "OpenGVLab/InternVL3-78B"
    
)

# === Model to lmms-eval model name mapping ===
# These map to the keys in AVAILABLE_SIMPLE_MODELS in lmms_eval/models/__init__.py
declare -A MODEL_TYPE_MAP=(
    ["Qwen/Qwen2.5-VL-3B-Instruct"]="qwen2_5_vl"
    ["Qwen/Qwen2.5-VL-7B-Instruct"]="qwen2_5_vl"
    ["Qwen/Qwen2.5-VL-32B-Instruct"]="qwen2_5_vl"
    ["Qwen/Qwen2-VL-7B-Instruct"]="qwen2_vl"
    ["OpenGVLab/InternVL3-8B"]="internvl3"
    ["OpenGVLab/InternVL3-38B"]="internvl3"
    ["OpenGVLab/InternVL3-78B"]="internvl3"
)

# === Model GPU Requirements ===
declare -A model_gpu_map=(
    # Qwen models (1 GPU)
    ["Qwen/Qwen2.5-VL-3B-Instruct"]=1
    ["Qwen/Qwen2.5-VL-7B-Instruct"]=1
    ["Qwen/Qwen2.5-VL-32B-Instruct"]=2
    
    # InternVL3
    ["OpenGVLab/InternVL3-8B"]=1
    ["OpenGVLab/InternVL3-38B"]=4
    ["OpenGVLab/InternVL3-78B"]=8
)

# === Task List (typographic attack focus) ===
declare -a task_list=(
    # Primary OCR and text understanding tasks
    "chartqa"           # Chart understanding with text
    "docvqa_val"       # Document Visual QA
    "infovqa_val"      # Infographic VQA
    "textvqa"          # Text-based VQA
    
    # Additional relevant tasks
    "mme"              # MME benchmark (includes text recognition)
    "ai2d"             # Science diagrams with text
    "mathvista"        # Mathematical reasoning with diagrams
    
    # General VQA for baseline
    "vqa"              # VQAv2
    "gqa"              # GQA
)

# === Short task list for quick testing ===
declare -a quick_task_list=(
    "chartqa"
    "docvqa_val"
    "infovqa_val"
    "textvqa"
)

# === Memory requirements per task (in GB) ===
declare -A task_memory_map=(
    ["chartqa"]=32
    ["docvqa_val"]=32
    ["infovqa_val"]=48
    ["textvqa"]=32
    ["mme"]=32
    ["ai2d"]=32
    ["mathvista"]=48
    ["vqa"]=32
    ["gqa"]=32
)

# === Function to get model args for lmms-eval ===
get_model_args() {
    local model_name=$1
    local model_type=${MODEL_TYPE_MAP[$model_name]}
    
    # Base args
    local args="pretrained=$model_name"
    
    # Add model-specific args
    case $model_type in
        qwen2_5_vl|qwen2_vl)
            # Qwen models need specific settings
            # Try flash_attention_2 first (faster), fallback to sdpa if not available
            # max_pixels based on Qwen examples
            # Note: flash-attn will be installed in eval.sh if not present
            args="$args,max_pixels=12845056,attn_implementation=flash_attention_2"
            ;;
        internvl2)
            # InternVL2 models
            args="$args,device_map=auto"
            ;;
        llama_vision)
            # Llama vision models
            args="$args,device_map=auto"
            ;;
        llava_onevision)
            # LLaVA OneVision
            args="$args,device_map=auto"
            ;;
        llava_hf)
            # LLaVA HF
            args="$args,device_map=auto"
            ;;
        *)
            # Default args
            ;;
    esac
    
    echo "$args"
}

# === Function to get safe model name for file paths ===
get_safe_model_name() {
    local model_name=$1
    echo "$model_name" | sed 's/\//-/g'
}

# === Function to load .env file if exists ===
load_env_file() {
    if [ -f "$REPO_PATH/.env" ]; then
        export $(grep -v '^#' "$REPO_PATH/.env" | xargs)
    fi
}

# === Default batch size configuration ===
export DEFAULT_BATCH_SIZE=1
export DEFAULT_LIMIT=""  # Empty means no limit (full evaluation)
export TEST_LIMIT=5      # For testing purposes

# === Output configuration ===
export OUTPUT_BASE_DIR="$REPO_PATH/outputs"
export LOG_BASE_DIR="$REPO_PATH/logs"

# Create necessary directories
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$LOG_BASE_DIR"
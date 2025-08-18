#!/usr/bin/env bash
# SLURM job submission script for lmms-eval
# Usage: bash sbatch.sh MODEL_NAME [NUM_GPUS] [--tasks TASK1,TASK2,...] [--test]

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source configuration
source "$SCRIPT_DIR/config.sh"

# Initialize variables
SPECIFIED_TASKS=()
TASKS_MODE="quick"  # Default to quick tasks for typographic attack research
TEST_MODE=false
ALL_MODE=false

# === Argument Parsing ===
if [ $# -lt 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 MODEL_NAME [NUM_GPUS] [--tasks TASK1,TASK2,...] [--test] [--all-tasks]"
    echo "       $0 --all [--tasks TASK1,TASK2,...] [--test]"
    echo ""
    echo "Options:"
    echo "  MODEL_NAME     The model to evaluate"
    echo "  NUM_GPUS       Number of GPUs to use (optional, auto-detected)"
    echo "  --all          Submit all models for evaluation"
    echo "  --tasks        Comma-separated list of tasks (default: quick typographic tasks)"
    echo "  --all-tasks    Run all available tasks (not just quick list)"
    echo "  --test         Run with limit=$TEST_LIMIT for testing"
    echo "  --help, -h     Show this help message"
    echo ""
    echo "Quick tasks (default for typographic attack):"
    for task in "${quick_task_list[@]}"; do
        echo "  - $task"
    done
    echo ""
    echo "All available tasks:"
    for task in "${task_list[@]}"; do
        echo "  - $task"
    done
    echo ""
    echo "Available models:"
    for model in "${model_list[@]}"; do
        gpu_count=${model_gpu_map[$model]:-1}
        echo "  - $model (GPUs: $gpu_count)"
    done
    echo ""
    echo "Examples:"
    echo "  # Single model with auto GPU, quick tasks"
    echo "  $0 'Qwen/Qwen2.5-VL-3B-Instruct'"
    echo ""
    echo "  # Single model with specific tasks and test mode"
    echo "  $0 'OpenGVLab/InternVL2-2B' 1 --tasks chartqa,docvqa_val --test"
    echo ""
    echo "  # All models with quick tasks"
    echo "  $0 --all"
    echo ""
    echo "  # All models with all tasks"
    echo "  $0 --all --all-tasks"
    exit 0
fi

# Check if running in --all mode
if [ "$1" = "--all" ]; then
    ALL_MODE=true
    MODEL_NAME=""
    NUM_GPUS=""
    shift  # Remove --all from arguments
else
    ALL_MODE=false
    MODEL_NAME=$1
    NUM_GPUS=${2:-"auto"}
    shift  # Remove model name
    if [ "$NUM_GPUS" != "auto" ] && [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
        shift  # Remove NUM_GPUS if it's a number
    fi
fi

# Parse remaining options
while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)
            TASKS_MODE="specified"
            IFS=',' read -ra SPECIFIED_TASKS <<< "$2"
            shift 2
            ;;
        --all-tasks)
            TASKS_MODE="all"
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine which tasks to run
if [ "$TASKS_MODE" = "specified" ]; then
    # Validate specified tasks
    for task in "${SPECIFIED_TASKS[@]}"; do
        task_found=false
        for valid_task in "${task_list[@]}"; do
            if [ "$task" = "$valid_task" ]; then
                task_found=true
                break
            fi
        done
        if [ "$task_found" = false ]; then
            echo "❌ Invalid task: $task"
            echo "📋 Available tasks:"
            for valid_task in "${task_list[@]}"; do
                echo "   - $valid_task"
            done
            exit 1
        fi
    done
    tasks_to_run=("${SPECIFIED_TASKS[@]}")
elif [ "$TASKS_MODE" = "all" ]; then
    tasks_to_run=("${task_list[@]}")
else
    # Default to quick tasks
    tasks_to_run=("${quick_task_list[@]}")
fi

# Set limit based on test mode
if [ "$TEST_MODE" = true ]; then
    LIMIT=$TEST_LIMIT
    echo "🧪 TEST MODE: Running with limit=$LIMIT"
else
    LIMIT=""
fi

# Function to submit jobs for a single model
submit_single_model() {
    local model_name=$1
    local num_gpus=$2
    local safe_model_name=$(get_safe_model_name "$model_name")
    
    echo "🚀 Submitting jobs for model: $model_name"
    echo "🖥️  Number of GPUs: $num_gpus"
    echo "📋 Number of tasks: ${#tasks_to_run[@]}"
    echo "📌 Tasks: ${tasks_to_run[*]}"
    if [ "$TEST_MODE" = true ]; then
        echo "🧪 Test mode: limit=$LIMIT"
    fi
    echo ""
    
    for task in "${tasks_to_run[@]}"; do
        # Create output directories
        output_dir="$OUTPUT_BASE_DIR/${safe_model_name}/${task}"
        log_dir="$LOG_BASE_DIR/${safe_model_name}"
        mkdir -p "$output_dir"
        mkdir -p "$log_dir"
        
        echo "  📝 Submitting task: $task"
        
        # Get memory requirement for task
        mem_gb=${task_memory_map[$task]:-32}
        
        # Prepare job name with 197 prefix for consistency with llm-jp-eval-mm
        # Format: 197_lmms-eval_<model>_<task>
        job_name="197_lmms-eval_${safe_model_name}_${task}"
        
        # Submit job
        sbatch_cmd="sbatch"
        sbatch_cmd="$sbatch_cmd --job-name=\"$job_name\""
        sbatch_cmd="$sbatch_cmd --output=$log_dir/${task}.out"
        sbatch_cmd="$sbatch_cmd --error=$log_dir/${task}.err"
        sbatch_cmd="$sbatch_cmd --time=24:00:00"
        sbatch_cmd="$sbatch_cmd --gres=gpu:$num_gpus"
        sbatch_cmd="$sbatch_cmd --ntasks=1"
        sbatch_cmd="$sbatch_cmd --cpus-per-task=8"
        sbatch_cmd="$sbatch_cmd --mem=${mem_gb}G"
        sbatch_cmd="$sbatch_cmd --wrap=\"bash $SCRIPT_DIR/eval.sh '$model_name' '$task' '$num_gpus' '$LIMIT'\""
        
        echo "    Command: $sbatch_cmd"
        eval $sbatch_cmd
    done
    
    echo ""
    echo "✅ All tasks submitted for $model_name!"
}

# Main execution
if [ "$ALL_MODE" = true ]; then
    # Submit all models
    echo "🚀 Starting batch submission for all models"
    echo "📊 Total models to evaluate: ${#model_list[@]}"
    echo "📋 Tasks per model: ${#tasks_to_run[@]}"
    echo ""
    
    counter=0
    for model in "${model_list[@]}"; do
        num_gpus=${model_gpu_map[$model]:-1}
        counter=$((counter + 1))
        echo "[$counter/${#model_list[@]}] Processing $model with $num_gpus GPU(s)..."
        submit_single_model "$model" "$num_gpus"
        echo ""
        sleep 1  # Small delay between submissions
    done
    
    echo "✅ All models have been submitted!"
    echo "📊 Total models submitted: $counter"
    echo "📊 Total jobs submitted: $((counter * ${#tasks_to_run[@]}))"
else
    # Single model mode
    # Validate model name
    model_found=false
    for model in "${model_list[@]}"; do
        if [ "$MODEL_NAME" = "$model" ]; then
            model_found=true
            break
        fi
    done
    
    if [ "$model_found" = false ]; then
        echo "❌ Model '$MODEL_NAME' is not in the available model list."
        echo "📋 Available models:"
        for model in "${model_list[@]}"; do
            gpu_count=${model_gpu_map[$model]:-1}
            echo "   - $model (GPUs: $gpu_count)"
        done
        exit 1
    fi
    
    # Auto-detect GPU count if needed
    if [ "$NUM_GPUS" = "auto" ]; then
        NUM_GPUS=${model_gpu_map[$MODEL_NAME]:-1}
        echo "🔍 Auto-detected GPU count: $NUM_GPUS"
    fi
    
    # Validate GPU count
    if ! [[ "$NUM_GPUS" =~ ^[1-8]$ ]]; then
        echo "❌ NUM_GPUS must be between 1 and 8"
        exit 1
    fi
    
    # Submit single model
    submit_single_model "$MODEL_NAME" "$NUM_GPUS"
    
    echo "📊 Total jobs submitted: ${#tasks_to_run[@]}"
fi

echo ""
echo "💡 Monitor jobs with: squeue -u $USER"
echo "💡 Check logs in: $LOG_BASE_DIR"
echo "💡 Results will be in: $OUTPUT_BASE_DIR"
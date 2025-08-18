# lmms-eval GPU Evaluation Scripts for Typographic Attack Research

This directory contains SLURM-based scripts for running lmms-eval on GPU clusters, specifically configured for typographic attack research.

## Directory Structure

```
nvlink/
├── config.sh         # Configuration file with models, tasks, and settings
├── sbatch.sh         # Main job submission script
├── eval.sh           # Evaluation execution script
├── test_single.sh    # Direct test script (no SLURM)
├── monitor_jobs.sh   # Job monitoring utility
└── README.md         # This file
```

## Quick Start

### 1. Test the Setup (Without SLURM)

First, test with a small model and limited samples:

```bash
cd /home/silviase/LIMIT-Lab/typographic_atk/lmms-eval
bash miscs/scripts/nvlink/test_single.sh
```

This will run Qwen2.5-VL-3B on chartqa task with limit=3.

### 2. Submit a Single Model Job

Submit evaluation for a specific model with default tasks (chartqa, docvqa_val, infovqa_val, textvqa):

```bash
bash miscs/scripts/nvlink/sbatch.sh "Qwen/Qwen2.5-VL-3B-Instruct"
```

### 3. Submit with Specific Tasks

```bash
bash miscs/scripts/nvlink/sbatch.sh "OpenGVLab/InternVL2-2B" --tasks chartqa,docvqa_val
```

### 4. Test Mode with SLURM

Run with limited samples (limit=5) for testing:

```bash
bash miscs/scripts/nvlink/sbatch.sh "Qwen/Qwen2.5-VL-7B-Instruct" --test
```

### 5. Submit All Models

Evaluate all configured models:

```bash
# With default quick tasks
bash miscs/scripts/nvlink/sbatch.sh --all

# With all available tasks
bash miscs/scripts/nvlink/sbatch.sh --all --all-tasks

# In test mode
bash miscs/scripts/nvlink/sbatch.sh --all --test
```

## Monitoring Jobs

### Check Job Status

```bash
# Show comprehensive status
bash miscs/scripts/nvlink/monitor_jobs.sh

# Watch live updates
watch -n 10 bash miscs/scripts/nvlink/monitor_jobs.sh

# Show only job details
bash miscs/scripts/nvlink/monitor_jobs.sh details

# Check for errors
bash miscs/scripts/nvlink/monitor_jobs.sh errors

# Check output files
bash miscs/scripts/nvlink/monitor_jobs.sh outputs
```

### Cancel Jobs

```bash
# Cancel all lmms jobs
bash miscs/scripts/nvlink/monitor_jobs.sh cancel

# Cancel specific job
scancel <job_id>
```

## Available Models

Models optimized for typographic attack research:

| Model | GPUs | Description |
|-------|------|-------------|
| Qwen/Qwen2.5-VL-3B-Instruct | 1 | Smallest, fast, good OCR |
| Qwen/Qwen2.5-VL-7B-Instruct | 1 | Balanced size and performance |
| Qwen/Qwen2.5-VL-32B-Instruct | 4 | Larger, stronger performance |
| OpenGVLab/InternVL2-1B | 1 | Very small, for quick tests |
| OpenGVLab/InternVL2-2B | 1 | Small but capable |
| OpenGVLab/InternVL2-8B | 1 | Good balance |
| meta-llama/Llama-3.2-11B-Vision-Instruct | 2 | Strong reasoning |
| lmms-lab/llava-onevision-qwen2-7b-ov | 1 | Latest LLaVA variant |

## Available Tasks

### Quick Tasks (Default for Typographic Attack)
- `chartqa` - Chart understanding with text
- `docvqa_val` - Document Visual QA
- `infovqa_val` - Infographic VQA  
- `textvqa` - Text-based VQA

### Additional Tasks
- `mme` - MME benchmark (includes text recognition)
- `ai2d` - Science diagrams with text
- `mathvista` - Mathematical reasoning with diagrams
- `vqa` - VQAv2 (general baseline)
- `gqa` - GQA (general baseline)

## Output Locations

- **Results**: `/home/silviase/LIMIT-Lab/typographic_atk/lmms-eval/outputs/<model>/<task>/`
- **Logs**: `/home/silviase/LIMIT-Lab/typographic_atk/lmms-eval/logs/<model>/<task>.{out,err}`

## Configuration

Edit `config.sh` to:
- Add/remove models
- Modify task lists
- Adjust GPU allocations
- Change memory requirements
- Update paths

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size in config.sh or allocate more GPUs
2. **Module not found**: Ensure you're in the correct directory and environment is activated
3. **SLURM errors**: Check `.err` files in the logs directory

### Debug Commands

```bash
# Check if environment works
cd /home/silviase/LIMIT-Lab/typographic_atk/lmms-eval
source dev/bin/activate
python -m lmms_eval --help

# Test minimal evaluation
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks chartqa \
    --limit 1 \
    --batch_size 1
```

## Tips

1. Start with test mode (`--test`) to verify everything works
2. Use smaller models (1B-3B) for initial experiments
3. Monitor GPU memory usage with `nvidia-smi`
4. Check logs immediately if jobs fail quickly
5. Use `--tasks` to run specific tasks instead of all

## Research Notes

These scripts are configured specifically for typographic attack research:
- Focus on OCR and text understanding tasks
- Conservative GPU/memory allocation for stability
- Batch size=1 to handle various image sizes
- Comprehensive logging for debugging

For questions or issues, check the logs first, then consult the lmms-eval documentation.
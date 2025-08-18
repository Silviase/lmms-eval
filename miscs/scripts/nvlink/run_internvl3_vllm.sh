#!/bin/bash
# Run InternVL3-8B evaluation with vLLM for typographic attack research

# Submit job for InternVL3-8B with all typographic attack relevant tasks
sbatch run_vllm_eval.sh \
    "OpenGVLab/InternVL3-8B" \
    "textvqa,infovqa_val,docvqa_val,chartqa" \
    1024 \
    1 \
    ./results_internvl3_vllm

echo "Submitted InternVL3-8B evaluation job"
echo "Tasks: textvqa, infovqa_val, docvqa_val, chartqa"
echo "Results will be saved to: ./results_internvl3_vllm"

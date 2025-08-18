#!/usr/bin/env python3
"""
Check available models in lmms-eval
"""

import sys
import os

# Add parent directory to path to import lmms_eval
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    from lmms_eval.models import AVAILABLE_SIMPLE_MODELS, AVAILABLE_CHAT_TEMPLATE_MODELS
    
    print("=" * 60)
    print("Available Models in lmms-eval")
    print("=" * 60)
    
    print("\n[Simple Models]")
    print("-" * 40)
    for key, value in sorted(AVAILABLE_SIMPLE_MODELS.items()):
        print(f"  {key:<30} -> {value}")
    
    print(f"\nTotal: {len(AVAILABLE_SIMPLE_MODELS)} simple models")
    
    print("\n[Chat Template Models]")
    print("-" * 40)
    for key, value in sorted(AVAILABLE_CHAT_TEMPLATE_MODELS.items()):
        mark = " ✓" if key == "vllm" else ""
        print(f"  {key:<30} -> {value}{mark}")
    
    print(f"\nTotal: {len(AVAILABLE_CHAT_TEMPLATE_MODELS)} chat template models")
    
    print("\n" + "=" * 60)
    print("Usage:")
    print("  python -m lmms_eval --model <model_name> ...")
    print("\nFor vLLM:")
    print("  python -m lmms_eval --model vllm --model_args ... --tasks ...")
    print("=" * 60)
    
except ImportError as e:
    print(f"Error importing lmms_eval modules: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)
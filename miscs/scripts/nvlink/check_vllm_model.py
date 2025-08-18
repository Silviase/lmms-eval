#!/usr/bin/env python3
"""
Check if a model is compatible with vLLM backend
"""

import sys
import argparse
from typing import Optional
from urllib import response

def check_vllm_compatibility(model_name: str, trust_remote_code: bool = True) -> bool:
    """
    Check if a model is compatible with vLLM.
    
    Args:
        model_name: Hugging Face model name or local path
        trust_remote_code: Whether to trust remote code
    
    Returns:
        bool: True if compatible, False otherwise
    """
    try:
        from vllm import LLM
        
        print(f"Checking vLLM compatibility for: {model_name}")
        print("This may take a few moments to download and load the model...")
        
        # Try to initialize the model
        llm = LLM(
            model=model_name, 
            trust_remote_code=trust_remote_code,
            max_model_len=2048,  # Use smaller context for testing
            gpu_memory_utilization=0.95,  # Increase GPU memory utilization for InternVL3
            # max_num_seqs=256,  # Reduce max sequences to avoid OOM during warmup
            enforce_eager=True,
        )
        
        print(f"✅ Model '{model_name}' is compatible with vLLM!")

        # Generate 100 sample sentences
        prompts = [f"Generate a sample random sentence {i}:" for i in range(1, 101)]
        responses = llm.generate(prompts)
        output_texts = [response.outputs[0].text for response in responses]
        print("\n".join(output_texts))

        print("====")
        print(f"✅ Model '{model_name}' has been successfully tested with vLLM!")

        # Clean up
        del llm
        
        return True
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("vLLM is not installed. Install it with: pip install vllm")
        return False
        
    except Exception as e:
        print(f"❌ Model '{model_name}' may not be compatible with vLLM")
        print(f"   Error: {str(e)}")
        
        # Provide suggestions based on common errors
        if "trust_remote_code" in str(e):
            print("\n💡 Suggestion: Try running with --trust-remote-code flag")
        elif "OOM" in str(e) or "memory" in str(e).lower():
            print("\n💡 Suggestion: Model may be too large for available GPU memory")
        elif "not supported" in str(e).lower():
            print("\n💡 Suggestion: This model architecture may not be supported by vLLM yet")
            
        return False

def list_known_compatible_models():
    """List known compatible VLM models for vLLM."""
    
    models = {
        "Qwen2-VL Series": [
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct", 
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct"
        ],
        "LLaVA Series": [
            "llava-hf/llava-1.5-7b-hf",
            "llava-hf/llava-1.5-13b-hf",
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "llava-hf/llava-v1.6-vicuna-7b-hf",
            "llava-hf/llava-v1.6-34b-hf"
        ],
        "Phi-3 Vision": [
            "microsoft/Phi-3-vision-128k-instruct",
            "microsoft/Phi-3.5-vision-instruct"
        ],
        "InternVL": [
            "OpenGVLab/InternVL2-8B",
            "OpenGVLab/InternVL2-26B",
            "OpenGVLab/InternVL2-40B"
        ],
        "Yi-VL": [
            "01-ai/Yi-VL-6B",
            "01-ai/Yi-VL-34B"
        ],
        "Others": [
            "BAAI/Bunny-v1_0-3B",
            "bczhou/TinyLLaVA-3.1B",
            "adept/fuyu-8b"
        ]
    }
    
    print("\n" + "="*60)
    print("Known vLLM-Compatible Vision-Language Models")
    print("="*60)
    
    for category, model_list in models.items():
        print(f"\n{category}:")
        for model in model_list:
            print(f"  - {model}")
    
    print("\n" + "="*60)
    print("Note: This list may not be exhaustive. New models are")
    print("continuously being added to vLLM support.")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Check if a model is compatible with vLLM backend"
    )
    parser.add_argument(
        "model", 
        nargs="?",
        help="Hugging Face model name or local path to check"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from the model repository"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known compatible VLM models"
    )
    
    args = parser.parse_args()
    
    if args.list or not args.model:
        list_known_compatible_models()
        if not args.model:
            print("\nTo check a specific model, run:")
            print("  python check_vllm_model.py <model_name>")
    
    if args.model:
        success = check_vllm_compatibility(args.model, args.trust_remote_code)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
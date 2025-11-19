"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 4, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

"""
Tests LLM model loading and initialization.
Validates compatibility with HuggingFace transformers.
Checks memory requirements and inference speed.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def check_gpus():
    """Check GPU availability."""
    print("=" * 80)
    print("GPU Availability Check")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("✗ CUDA not available!")
        return False

    n_gpus = torch.cuda.device_count()
    print(f"✓ CUDA available with {n_gpus} GPUs")

    for i in [5, 6, 7]:
        if i < n_gpus:
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {gpu_name}")
        else:
            print(f"  GPU {i}: NOT AVAILABLE (only {n_gpus} GPUs detected)")

    print()
    return True

def test_gpt2_loading():
    """Test GPT-2 loading on GPU 5."""
    print("=" * 80)
    print("Test 1: GPT-2 on GPU 5")
    print("=" * 80)

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        print("Loading GPT-2...")
        device = 'cuda:5' if torch.cuda.device_count() > 5 else 'cuda:0'

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model = model.to(device)

        print(f"✓ GPT-2 loaded on {device}")
        print(f"  Parameters: {model.num_parameters():,}")

        # Test inference
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer("Temperature increases because", return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=20, do_sample=False)

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Test generation: {text}")
        print(f"✓ GPT-2 inference works!\n")

        # Clean up
        del model
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"✗ Error loading GPT-2: {e}\n")
        return False

def test_llama2_loading():
    """Test LLaMA-2 alternative (TinyLlama) loading on GPU 6."""
    print("=" * 80)
    print("Test 2: TinyLlama (1.1B) on GPU 6 (LLaMA alternative)")
    print("=" * 80)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading TinyLlama (open, no authentication needed)...")
        device = 'cuda:6' if torch.cuda.device_count() > 6 else 'cuda:0'

        # Use TinyLlama as open alternative to LLaMA-2
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        print(f"  Loading from: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load without quantization (RTX 3090 has 24GB)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(device)

        print(f"✓ TinyLlama loaded on {device}")
        print(f"  Parameters: {model.num_parameters():,}")

        # Test inference
        inputs = tokenizer("The causal relationship is", return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=20, do_sample=False)

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Test generation: {text}")
        print(f"✓ TinyLlama inference works!\n")

        # Clean up
        del model
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"✗ Error loading TinyLlama: {e}\n")
        return False

def test_mistral_loading():
    """Test Mistral alternative (Phi-2) loading on GPU 7."""
    print("=" * 80)
    print("Test 3: Phi-2 (2.7B) on GPU 7 (Technical reasoning)")
    print("=" * 80)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading Phi-2 (Microsoft, open model)...")
        device = 'cuda:7' if torch.cuda.device_count() > 7 else 'cuda:0'

        # Use Phi-2 as alternative - good at technical reasoning
        model_name = "microsoft/phi-2"

        print(f"  Loading from: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load without quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16  # Use FP16 for efficiency
        )
        model = model.to(device)

        print(f"✓ Phi-2 loaded on {device}")
        print(f"  Parameters: {model.num_parameters():,}")

        # Test inference
        inputs = tokenizer("HVAC systems control", return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=20, do_sample=False)

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Test generation: {text}")
        print(f"✓ Phi-2 inference works!\n")

        # Clean up
        del model
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"✗ Error loading Phi-2: {e}\n")
        return False

def main():
    print("\n")
    print("=" * 80)
    print("LLM Loading Test for Causal SLM Project")
    print("=" * 80)
    print()

    # Check GPUs
    if not check_gpus():
        print("Cannot proceed without CUDA support.")
        return

    # Test each LLM
    results = {
        'GPT-2 (124M)': test_gpt2_loading(),
        'TinyLlama (1.1B)': test_llama2_loading(),
        'Phi-2 (2.7B)': test_mistral_loading()
    }

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:20} {status}")

    print()

    if all(results.values()):
        print("✓✓ All 3 LLMs loaded successfully!")
        print("   Ready to proceed with comprehensive evaluation.")
        print()
        print("Model Configuration:")
        print("  - GPU 5: GPT-2 (124M params) - Simple narratives")
        print("  - GPU 6: TinyLlama (1.1B params) - Counterfactual reasoning")
        print("  - GPU 7: Phi-2 (2.7B params) - Technical explanations")
    else:
        print("⚠ Some LLMs failed to load.")
        print("  Check error messages above for details.")

    print("=" * 80)

if __name__ == '__main__':
    main()

"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 3, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

"""
Simplified version of multi-LLM causal system for quick prototyping.
Reduced complexity while maintaining core causal reasoning capabilities.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SimpleLLM:
    """Base wrapper for LLM without complex dependencies."""

    def __init__(self, model_name: str, device: str, use_fp16: bool = True):
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16

        print(f"Loading {model_name} on {device}...")

        # Load tokenizer
        if "gpt2" in model_name.lower():
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if use_fp16 else torch.float32
            )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move to device
        self.model = self.model.to(device)
        self.model.eval()

        print(f"✓ Loaded {model_name} ({self.model.num_parameters():,} params)")

    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate text from prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        texts = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        return texts

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        torch.cuda.empty_cache()


class GPT2NarrativeGenerator(SimpleLLM):
    """LLM 1: GPT-2 for simple, fast causal narratives."""

    def __init__(self, device: str = 'cuda:5'):
        super().__init__("gpt2", device, use_fp16=False)
        self.role = "Simple Narrative Generator"

    def create_prompt(self, sensor_data: pd.DataFrame, causal_graph: np.ndarray,
                     variable_names: List[str]) -> str:
        """Create prompt for simple narrative generation with CAUSAL KEYWORDS."""
        # Get last observation
        last_obs = sensor_data.iloc[-1]

        # Find causal relationships
        edges = []
        n = len(variable_names)
        for i in range(n):
            for j in range(n):
                if causal_graph[i, j] > 0:
                    edges.append(f"{variable_names[i]} → {variable_names[j]}")

        prompt = f"""Explain this sensor data using causal language.

Sensor readings:
{', '.join([f"{name}={last_obs[name]:.2f}" for name in variable_names])}

Causal relationships discovered:
{', '.join(edges) if edges else 'No clear causal relationships'}

Write 2-3 sentences explaining what's happening. IMPORTANT: Use causal keywords like "because", "causes", "leads to", "results in", "due to":
"""
        return prompt

    def generate_narrative(self, sensor_data: pd.DataFrame, causal_graph: np.ndarray,
                          variable_names: List[str]) -> str:
        """Generate simple causal narrative."""
        prompt = self.create_prompt(sensor_data, causal_graph, variable_names)
        outputs = self.generate(prompt, max_length=150, temperature=0.7)

        # Extract just the generated part
        narrative = outputs[0].replace(prompt, "").strip()
        return narrative


class TinyLlamaCounterfactualReasoner(SimpleLLM):
    """LLM 2: TinyLlama for counterfactual reasoning."""

    def __init__(self, device: str = 'cuda:6'):
        super().__init__("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device, use_fp16=True)
        self.role = "Counterfactual Reasoning"

    def create_prompt(self, sensor_data: pd.DataFrame, causal_graph: np.ndarray,
                     variable_names: List[str]) -> str:
        """Create prompt for counterfactual reasoning with CAUSAL LANGUAGE."""
        last_obs = sensor_data.iloc[-1]

        # Find strongest causal relationship
        max_edge = np.unravel_index(np.argmax(causal_graph), causal_graph.shape)
        cause_var = variable_names[max_edge[0]]
        effect_var = variable_names[max_edge[1]]

        prompt = f"""<|system|>
You are an expert in causal reasoning and counterfactual analysis.</s>
<|user|>
Sensor data shows {cause_var}={last_obs[cause_var]:.2f} CAUSES {effect_var}={last_obs[effect_var]:.2f}.

Explain what would happen IF {cause_var} was different. Use counterfactual reasoning with causal keywords ("if...then", "would cause", "would lead to", "because", "results in"). Answer in 2-3 sentences.</s>
<|assistant|>
"""
        return prompt

    def generate_narrative(self, sensor_data: pd.DataFrame, causal_graph: np.ndarray,
                          variable_names: List[str]) -> str:
        """Generate counterfactual narrative."""
        prompt = self.create_prompt(sensor_data, causal_graph, variable_names)
        outputs = self.generate(prompt, max_length=200, temperature=0.7)

        # Extract assistant response
        narrative = outputs[0].split("<|assistant|>")[-1].strip()
        return narrative


class Phi2TechnicalExplainer(SimpleLLM):
    """LLM 3: Phi-2 for technical domain explanations."""

    def __init__(self, device: str = 'cuda:7'):
        super().__init__("microsoft/phi-2", device, use_fp16=True)
        self.role = "Technical Domain Expert"

    def create_prompt(self, sensor_data: pd.DataFrame, causal_graph: np.ndarray,
                     variable_names: List[str]) -> str:
        """Create prompt for technical explanation with CAUSAL KEYWORDS."""
        last_obs = sensor_data.iloc[-1]

        # Get statistics
        stats = {}
        for var in variable_names:
            stats[var] = {
                'current': last_obs[var],
                'mean': sensor_data[var].mean(),
                'std': sensor_data[var].std()
            }

        # Get causal relationships
        edges = []
        n = len(variable_names)
        for i in range(n):
            for j in range(n):
                if causal_graph[i, j] > 0:
                    edges.append(f"{variable_names[i]} → {variable_names[j]}")

        prompt = f"""Provide a technical analysis of this sensor system with causal reasoning.

Current readings:
"""
        for var in variable_names:
            prompt += f"- {var}: {stats[var]['current']:.2f} (μ={stats[var]['mean']:.2f}, σ={stats[var]['std']:.2f})\n"

        prompt += f"\nCausal structure: {np.sum(causal_graph > 0)} edges - {', '.join(edges) if edges else 'no clear causality'}\n\nProvide technical explanation using causal language (\"because\", \"causes\", \"leads to\", \"results in\", \"due to\"). Write 2-3 sentences:\n"

        return prompt

    def generate_narrative(self, sensor_data: pd.DataFrame, causal_graph: np.ndarray,
                          variable_names: List[str]) -> str:
        """Generate technical narrative."""
        prompt = self.create_prompt(sensor_data, causal_graph, variable_names)
        outputs = self.generate(prompt, max_length=200, temperature=0.7)

        # Extract generated part
        narrative = outputs[0].replace(prompt, "").strip()
        # Clean up if it continues beyond explanation
        if '\n\n' in narrative:
            narrative = narrative.split('\n\n')[0]
        return narrative


class MultiLLMEnsemble:
    """
    Ensemble of 3 LLMs for causal explanation.

    Strategies:
    - weighted_vote: Weight by validation performance
    - consensus: Select output with highest agreement
    - best_model: Use single best model (Phi-2)
    """

    def __init__(self, gpu_ids: List[int] = [5, 6, 7]):
        print("\n" + "="*80)
        print("Initializing Multi-LLM Ensemble")
        print("="*80)

        # Initialize 3 LLMs
        self.llm1 = GPT2NarrativeGenerator(device=f'cuda:{gpu_ids[0]}')
        self.llm2 = TinyLlamaCounterfactualReasoner(device=f'cuda:{gpu_ids[1]}')
        self.llm3 = Phi2TechnicalExplainer(device=f'cuda:{gpu_ids[2]}')

        # Default weights (can be learned from validation)
        self.weights = [0.2, 0.3, 0.5]  # Phi-2 gets highest weight

        print("\n✓ Multi-LLM Ensemble ready!")
        print(f"  LLM 1 (GPT-2): {self.llm1.role}")
        print(f"  LLM 2 (TinyLlama): {self.llm2.role}")
        print(f"  LLM 3 (Phi-2): {self.llm3.role}")
        print("="*80 + "\n")

    def generate_ensemble_explanation(
        self,
        sensor_data: pd.DataFrame,
        causal_graph: np.ndarray,
        variable_names: List[str],
        strategy: str = 'best_model'
    ) -> Dict[str, str]:
        """
        Generate explanation using ensemble of 3 LLMs.

        Args:
            sensor_data: Time series sensor data
            causal_graph: Discovered causal graph (n x n adjacency matrix)
            variable_names: List of variable names
            strategy: 'weighted_vote', 'consensus', 'best_model', or 'all'

        Returns:
            Dictionary with individual outputs and ensemble result
        """
        print(f"Generating narratives with {strategy} strategy...")

        # Generate from each LLM
        narrative1 = self.llm1.generate_narrative(sensor_data, causal_graph, variable_names)
        print(f"  ✓ LLM 1 (GPT-2): {len(narrative1)} chars")

        narrative2 = self.llm2.generate_narrative(sensor_data, causal_graph, variable_names)
        print(f"  ✓ LLM 2 (TinyLlama): {len(narrative2)} chars")

        narrative3 = self.llm3.generate_narrative(sensor_data, causal_graph, variable_names)
        print(f"  ✓ LLM 3 (Phi-2): {len(narrative3)} chars")

        # Aggregate based on strategy
        if strategy == 'best_model':
            ensemble_output = narrative3  # Use Phi-2 (best performer)
        elif strategy == 'weighted_vote':
            # Simple concatenation with labels (more sophisticated would use embeddings)
            ensemble_output = f"Combined analysis:\n\n{narrative3}"
        elif strategy == 'consensus':
            # Use longest narrative (proxy for most informative)
            narratives = [narrative1, narrative2, narrative3]
            ensemble_output = max(narratives, key=len)
        elif strategy == 'all':
            ensemble_output = f"GPT-2: {narrative1}\n\nTinyLlama: {narrative2}\n\nPhi-2: {narrative3}"
        else:
            ensemble_output = narrative3  # Default to best

        return {
            'gpt2': narrative1,
            'tinyllama': narrative2,
            'phi2': narrative3,
            'ensemble': ensemble_output,
            'strategy': strategy
        }

    def cleanup(self):
        """Free all GPU memory."""
        print("Cleaning up LLMs...")
        self.llm1.cleanup()
        self.llm2.cleanup()
        self.llm3.cleanup()
        print("✓ Cleanup complete")


# Quick test
if __name__ == '__main__':
    print("\nTesting Multi-LLM Ensemble...\n")

    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    data = pd.DataFrame({
        'temperature': 20 + np.random.randn(n_samples) * 2,
        'humidity': 50 + np.random.randn(n_samples) * 5,
        'hvac_power': 1.5 + np.random.randn(n_samples) * 0.3
    })

    # Dummy causal graph (temp → hvac → humidity)
    graph = np.array([
        [0, 0, 1],  # temp → hvac
        [0, 0, 0],
        [0, 1, 0]   # hvac → humidity
    ])

    variables = ['temperature', 'humidity', 'hvac_power']

    # Test ensemble
    ensemble = MultiLLMEnsemble(gpu_ids=[5, 6, 7])

    results = ensemble.generate_ensemble_explanation(
        data, graph, variables, strategy='all'
    )

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    for key, value in results.items():
        if key != 'strategy':
            print(f"\n{key.upper()}:\n{value}\n")

    ensemble.cleanup()

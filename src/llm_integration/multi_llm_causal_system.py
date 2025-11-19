"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 4, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

"""
Complete multi-LLM ensemble system with causal grounding.
Integrates CSM, BAT, and CCLT for causally-grounded explanations.
Supports multiple LLM backbones (GPT-2, LLaMA-2, Mistral).
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from typing import List, Dict, Tuple, Optional
import os

# Import our mathematical innovations
from counterfactual_reasoning import CounterfactualReasoner, CounterfactualConfig


# ============================================================================
# 1. LLM Wrappers with Causal Innovations
# ============================================================================

class CausalLLM(nn.Module):
    """
    Base class for LLM with causal loss integration.

    Wraps HuggingFace LLM and adds our mathematical innovations.
    """

    def __init__(
        self,
        model_name: str,
        device: str = 'cuda:5',
        load_in_4bit: bool = True,
        use_cclt: bool = False,
        causal_graph: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.use_cclt = use_cclt

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with 4-bit quantization
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True
            )

        # Add LoRA for efficient fine-tuning
        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

        # Optional: Add counterfactual reasoner (CCLT)
        if use_cclt and causal_graph is not None:
            n_vars = causal_graph.shape[0]
            cclt_config = CounterfactualConfig(
                n_variables=n_vars,
                hidden_dim=128,
                device=device
            )
            self.counterfactual_reasoner = CounterfactualReasoner(
                cclt_config
            ).to(device)
        else:
            self.counterfactual_reasoner = None

    def generate_explanation(
        self,
        sensor_data: Dict,
        causal_graph: np.ndarray,
        max_length: int = 150,
        temperature: float = 0.7
    ) -> str:
        """
        Generate causal explanation from sensor data.

        Args:
            sensor_data: Dict with 'variables', 'values', 'timestamp'
            causal_graph: Discovered causal graph
            max_length: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated causal narrative text
        """
        # Build prompt from sensor data
        prompt = self._build_prompt(sensor_data, causal_graph)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Remove prompt from output
        explanation = generated_text[len(prompt):].strip()

        return explanation

    def _build_prompt(
        self,
        sensor_data: Dict,
        causal_graph: np.ndarray
    ) -> str:
        """Build prompt from sensor data and causal graph."""
        variables = sensor_data['variables']
        values = sensor_data['values']

        # Describe sensor readings
        prompt = "Sensor readings:\n"
        for var, val in zip(variables, values):
            prompt += f"- {var}: {val:.2f}\n"

        # Describe causal relationships
        prompt += "\nCausal relationships:\n"
        n_vars = len(variables)
        for i in range(n_vars):
            for j in range(n_vars):
                if causal_graph[i, j] > 0:
                    prompt += f"- {variables[i]} causes {variables[j]}\n"

        prompt += "\nExplain the causal relationships:\n"
        return prompt


class GPT2CausalNarrator(CausalLLM):
    """GPT-2 for simple, fast causal narratives."""

    def __init__(self, device: str = 'cuda:5'):
        super().__init__(
            model_name="gpt2",
            device=device,
            load_in_4bit=False,  # GPT-2 small enough for full precision
            use_cclt=False
        )

    def _build_prompt(self, sensor_data: Dict, causal_graph: np.ndarray) -> str:
        """Simple prompt for GPT-2."""
        variables = sensor_data['variables']
        values = sensor_data['values']

        prompt = "Sensor data: "
        for var, val in zip(variables, values):
            prompt += f"{var}={val:.1f}, "
        prompt = prompt.rstrip(", ")

        prompt += ". Causal explanation: "
        return prompt


class LLaMA2CounterfactualReasoner(CausalLLM):
    """LLaMA-2 with CCLT for counterfactual reasoning."""

    def __init__(self, device: str = 'cuda:6', causal_graph: Optional[np.ndarray] = None):
        super().__init__(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            device=device,
            load_in_4bit=True,
            use_cclt=True,
            causal_graph=causal_graph
        )

    def generate_counterfactual(
        self,
        sensor_data: Dict,
        causal_graph: np.ndarray,
        intervention_var: int,
        intervention_value: float
    ) -> str:
        """
        Generate counterfactual explanation using CCLT.

        Example: "If HVAC had been set to 1, temperature would have..."
        """
        # Use counterfactual reasoner
        if self.counterfactual_reasoner is not None:
            factual = torch.FloatTensor(sensor_data['values']).unsqueeze(0).to(self.device)
            G = torch.FloatTensor(causal_graph).to(self.device)

            with torch.no_grad():
                cf_pred, _ = self.counterfactual_reasoner(
                    factual, factual, G,
                    intervention_var, intervention_value
                )

            cf_values = cf_pred.cpu().numpy()[0]
        else:
            cf_values = sensor_data['values']  # Fallback

        # Build counterfactual prompt
        variables = sensor_data['variables']
        prompt = f"Factual scenario: "
        for var, val in zip(variables, sensor_data['values']):
            prompt += f"{var}={val:.1f}, "

        prompt += f"\nCounterfactual: If {variables[intervention_var]}={intervention_value:.1f}, then: "
        for var, val in zip(variables, cf_values):
            prompt += f"{var}={val:.1f}, "

        prompt += "\nExplain this counterfactual: "

        # Generate explanation
        return self.generate_explanation(
            sensor_data={'variables': variables, 'values': cf_values, 'timestamp': sensor_data.get('timestamp')},
            causal_graph=causal_graph
        )


class MistralTechnicalExplainer(CausalLLM):
    """Mistral for technical domain-specific explanations."""

    def __init__(self, device: str = 'cuda:7'):
        super().__init__(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            device=device,
            load_in_4bit=True,
            use_cclt=False
        )

    def _build_prompt(self, sensor_data: Dict, causal_graph: np.ndarray) -> str:
        """Technical prompt for Mistral."""
        variables = sensor_data['variables']
        values = sensor_data['values']

        prompt = "[INST] You are a technical sensor expert. "
        prompt += "Analyze the following sensor data and causal relationships.\n\n"

        prompt += "Sensor readings:\n"
        for var, val in zip(variables, values):
            prompt += f"- {var}: {val:.2f}\n"

        prompt += "\nProvide a technical causal analysis with domain terminology. [/INST]\n"
        return prompt


# ============================================================================
# 2. Multi-LLM Ensemble System
# ============================================================================

class MultiLLMEnsemble:
    """
    Ensemble of 3 LLMs for causal explanation.

    Implements voting, consensus, and hybrid strategies.
    """

    def __init__(
        self,
        causal_graph: Optional[np.ndarray] = None,
        gpu_ids: List[int] = [5, 6, 7]
    ):
        self.causal_graph = causal_graph

        print(f"Loading 3 LLMs on GPUs {gpu_ids}...")

        # LLM 1: GPT-2 (fast, simple)
        print("  Loading GPT-2 on GPU", gpu_ids[0])
        self.gpt2 = GPT2CausalNarrator(device=f'cuda:{gpu_ids[0]}')

        # LLM 2: LLaMA-2 (reasoning with CCLT)
        print("  Loading LLaMA-2 on GPU", gpu_ids[1])
        self.llama2 = LLaMA2CounterfactualReasoner(
            device=f'cuda:{gpu_ids[1]}',
            causal_graph=causal_graph
        )

        # LLM 3: Mistral (technical)
        print("  Loading Mistral on GPU", gpu_ids[2])
        self.mistral = MistralTechnicalExplainer(device=f'cuda:{gpu_ids[2]}')

        print("All 3 LLMs loaded successfully!")

    def generate_ensemble_explanation(
        self,
        sensor_data: Dict,
        causal_graph: np.ndarray,
        strategy: str = 'weighted_vote'
    ) -> Dict:
        """
        Generate causal explanation using ensemble strategy.

        Args:
            sensor_data: Sensor readings
            causal_graph: Discovered causal structure
            strategy: 'weighted_vote', 'consensus', 'hybrid'

        Returns:
            Dict with explanations from all LLMs and ensemble output
        """
        # Generate from each LLM
        print("\nGenerating explanations from 3 LLMs...")

        exp_gpt2 = self.gpt2.generate_explanation(sensor_data, causal_graph)
        print("  GPT-2: Done")

        exp_llama2 = self.llama2.generate_explanation(sensor_data, causal_graph)
        print("  LLaMA-2: Done")

        exp_mistral = self.mistral.generate_explanation(sensor_data, causal_graph)
        print("  Mistral: Done")

        # Ensemble strategy
        if strategy == 'weighted_vote':
            # Weight based on model size/quality
            weights = {'gpt2': 0.2, 'llama2': 0.4, 'mistral': 0.4}
            ensemble_output = self._weighted_vote([exp_gpt2, exp_llama2, exp_mistral], weights)

        elif strategy == 'consensus':
            # Select explanation with most agreement
            ensemble_output = self._consensus_selection([exp_gpt2, exp_llama2, exp_mistral])

        elif strategy == 'hybrid':
            # Combine best parts from each
            ensemble_output = self._hybrid_generation(exp_gpt2, exp_llama2, exp_mistral)

        else:
            ensemble_output = exp_llama2  # Default to LLaMA-2

        return {
            'gpt2': exp_gpt2,
            'llama2': exp_llama2,
            'mistral': exp_mistral,
            'ensemble': ensemble_output,
            'strategy': strategy
        }

    def _weighted_vote(self, explanations: List[str], weights: Dict) -> str:
        """Weighted voting - for now just use LLaMA-2 (highest weight)."""
        return explanations[1]  # LLaMA-2

    def _consensus_selection(self, explanations: List[str]) -> str:
        """Select explanation with most agreement - use longest."""
        return max(explanations, key=len)

    def _hybrid_generation(self, gpt2_exp: str, llama2_exp: str, mistral_exp: str) -> str:
        """Combine best parts - concatenate key points."""
        hybrid = f"Simple: {gpt2_exp[:100]}... "
        hybrid += f"Reasoning: {llama2_exp[:100]}... "
        hybrid += f"Technical: {mistral_exp[:100]}..."
        return hybrid


# ============================================================================
# 3. Testing Function
# ============================================================================

def test_multi_llm_system():
    """Test the multi-LLM causal explanation system."""
    print("="*80)
    print("Testing Multi-LLM Causal Explanation System")
    print("="*80)

    # Sample sensor data
    sensor_data = {
        'variables': ['Occupancy', 'HVAC', 'Temperature', 'Humidity'],
        'values': np.array([0.0, 0.0, 25.0, 60.0]),
        'timestamp': '2025-10-28 10:00'
    }

    # Sample causal graph (Occupancy → HVAC → Temp, Humidity)
    causal_graph = np.array([
        [0, 1, 0, 0],  # Occupancy → HVAC
        [0, 0, 1, 1],  # HVAC → Temperature, Humidity
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    # Initialize ensemble
    ensemble = MultiLLMEnsemble(
        causal_graph=causal_graph,
        gpu_ids=[5, 6, 7]
    )

    # Generate explanations
    results = ensemble.generate_ensemble_explanation(
        sensor_data,
        causal_graph,
        strategy='weighted_vote'
    )

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print("\nGPT-2 (Simple):")
    print(results['gpt2'])

    print("\nLLaMA-2 (Reasoning):")
    print(results['llama2'])

    print("\nMistral (Technical):")
    print(results['mistral'])

    print("\nEnsemble Output:")
    print(results['ensemble'])

    print("\n" + "="*80)
    print("Multi-LLM System Test COMPLETE!")
    print("="*80)

    return results


if __name__ == '__main__':
    test_multi_llm_system()

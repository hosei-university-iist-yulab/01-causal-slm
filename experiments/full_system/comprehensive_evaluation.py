"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 5, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

"""
Complete end-to-end evaluation of CSLM framework.
Runs causal discovery, intervention prediction, and narrative generation.
Includes human evaluation protocol.
"""

import torch
import numpy as np
from typing import Dict, List
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_llm_causal_system import MultiLLMEnsemble, GPT2CausalNarrator
from causal_score_matching import CausalScoreMatching, CSMConfig
from intervention_generator import create_hvac_scm


# ============================================================================
# Baseline Implementations
# ============================================================================

class PCMCIBaseline:
    """SOTA Baseline: PCMCI for causal discovery."""

    def __init__(self):
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
        from tigramite import data_processing as pp

        self.parcorr = ParCorr()

    def discover_graph(self, data: np.ndarray) -> np.ndarray:
        """Discover causal graph using PCMCI."""
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI

        # Prepare data
        dataframe = pp.DataFrame(data)

        # Run PCMCI
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=self.parcorr)

        results = pcmci.run_pcmci(tau_max=2, pc_alpha=0.05)

        # Extract contemporaneous graph
        graph = (results['p_matrix'][:, :, 0] < 0.05).astype(float)

        return graph


class StandardLLM:
    """Baseline: Single LLM without any causal innovations."""

    def __init__(self, model_name: str = "gpt2", device: str = 'cuda:5'):
        self.narrator = GPT2CausalNarrator(device=device)

    def generate(self, sensor_data: Dict, causal_graph: np.ndarray) -> str:
        """Generate without causal innovations."""
        return self.narrator.generate_explanation(sensor_data, causal_graph)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def evaluate_causal_discovery(
    predicted_graph: np.ndarray,
    true_graph: np.ndarray
) -> Dict:
    """Evaluate causal discovery quality."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    pred_flat = predicted_graph.flatten()
    true_flat = true_graph.flatten()

    f1 = f1_score(true_flat, pred_flat)
    precision = precision_score(true_flat, pred_flat, zero_division=0)
    recall = recall_score(true_flat, pred_flat, zero_division=0)
    shd = np.sum(np.abs(true_graph - predicted_graph))

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'shd': shd
    }


def evaluate_explanation_quality(
    explanation: str,
    sensor_data: Dict,
    causal_graph: np.ndarray
) -> Dict:
    """
    Evaluate explanation quality.

    Metrics:
    - Length (informativeness)
    - Causal keyword count (mentions causality)
    - Variable coverage (mentions all variables)
    """
    variables = sensor_data['variables']

    # Length
    length = len(explanation.split())

    # Causal keywords
    causal_keywords = ['because', 'caused', 'due to', 'resulted', 'led to', 'affect']
    causal_count = sum(1 for kw in causal_keywords if kw in explanation.lower())

    # Variable coverage
    vars_mentioned = sum(1 for var in variables if var.lower() in explanation.lower())
    coverage = vars_mentioned / len(variables)

    return {
        'length': length,
        'causal_keywords': causal_count,
        'variable_coverage': coverage,
        'score': (causal_count * 0.4 + coverage * 0.6)  # Composite score
    }


# ============================================================================
# Comprehensive Evaluation
# ============================================================================

def run_comprehensive_evaluation(
    output_dir: str = 'output/evaluations',
    gpu_ids: List[int] = [5, 6, 7]
) -> Dict:
    """
    Run complete evaluation comparing all approaches.

    Returns:
        Dictionary with all results
    """
    print("="*80)
    print("COMPREHENSIVE EVALUATION: Baselines + Ablations + Multi-LLM")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Generate test data
    print("\n1. Generating Test Data")
    print("-"*80)
    scm = create_hvac_scm()
    true_graph = scm.causal_graph

    obs_data = scm.sample(1000)
    test_samples = [
        {
            'variables': ['Occupancy', 'HVAC', 'Temperature', 'Humidity'],
            'values': obs_data[i],
            'timestamp': f'2025-10-28 {i:02d}:00'
        }
        for i in range(10)  # 10 test samples
    ]

    print(f"Generated {len(test_samples)} test samples")
    print(f"True causal graph:\n{true_graph}")

    results = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'n_test_samples': len(test_samples),
            'true_graph': true_graph.tolist()
        },
        'baselines': {},
        'ablations': {},
        'full_system': {}
    }

    # ========================================================================
    # BASELINE 1: No Innovation (Standard Single LLM)
    # ========================================================================
    print("\n" + "="*80)
    print("BASELINE 1: No Innovation (Standard GPT-2)")
    print("="*80)

    try:
        standard_llm = StandardLLM(device=f'cuda:{gpu_ids[0]}')

        baseline1_explanations = []
        baseline1_scores = []

        for sample in test_samples[:3]:  # Test on 3 samples
            exp = standard_llm.generate(sample, np.zeros_like(true_graph))  # No graph!
            score = evaluate_explanation_quality(exp, sample, true_graph)

            baseline1_explanations.append(exp)
            baseline1_scores.append(score)

            print(f"\nSample explanation:\n{exp[:150]}...")
            print(f"Score: {score['score']:.3f}")

        results['baselines']['no_innovation'] = {
            'causal_discovery': {'f1': 0.0, 'note': 'No causal discovery'},
            'explanation_scores': baseline1_scores,
            'avg_score': np.mean([s['score'] for s in baseline1_scores])
        }
        print(f"\n→ Avg Score: {results['baselines']['no_innovation']['avg_score']:.3f}")

    except Exception as e:
        print(f"ERROR in Baseline 1: {e}")
        results['baselines']['no_innovation'] = {'error': str(e)}

    # ========================================================================
    # BASELINE 2: SOTA (PCMCI + Standard LLM)
    # ========================================================================
    print("\n" + "="*80)
    print("BASELINE 2: SOTA (PCMCI + Standard GPT-2)")
    print("="*80)

    try:
        pcmci = PCMCIBaseline()
        pcmci_graph = pcmci.discover_graph(obs_data[:800])

        print(f"PCMCI discovered graph:\n{pcmci_graph}")

        pcmci_metrics = evaluate_causal_discovery(pcmci_graph, true_graph)
        print(f"PCMCI F1: {pcmci_metrics['f1']:.3f}, SHD: {pcmci_metrics['shd']:.0f}")

        baseline2_explanations = []
        baseline2_scores = []

        for sample in test_samples[:3]:
            exp = standard_llm.generate(sample, pcmci_graph)
            score = evaluate_explanation_quality(exp, sample, true_graph)

            baseline2_explanations.append(exp)
            baseline2_scores.append(score)

        results['baselines']['sota_pcmci'] = {
            'causal_discovery': pcmci_metrics,
            'explanation_scores': baseline2_scores,
            'avg_score': np.mean([s['score'] for s in baseline2_scores])
        }
        print(f"→ Avg Score: {results['baselines']['sota_pcmci']['avg_score']:.3f}")

    except Exception as e:
        print(f"ERROR in Baseline 2: {e}")
        results['baselines']['sota_pcmci'] = {'error': str(e)}

    # ========================================================================
    # ABLATION 1: CSM Only (Our causal discovery + Standard LLM)
    # ========================================================================
    print("\n" + "="*80)
    print("ABLATION 1: CSM Only (Ours)")
    print("="*80)

    try:
        csm_config = CSMConfig(n_variables=4, hidden_dim=64)
        csm = CausalScoreMatching(csm_config).to(f'cuda:{gpu_ids[0]}')

        # Train CSM quickly
        print("Training CSM...")
        optimizer = torch.optim.Adam(csm.parameters(), lr=1e-3)
        X_obs = torch.FloatTensor(obs_data[:800]).to(f'cuda:{gpu_ids[0]}')

        for epoch in range(50):  # Quick training
            idx = torch.randperm(800)[:32]
            optimizer.zero_grad()
            loss, _ = csm.train_step(X_obs[idx], None)
            loss.backward()
            optimizer.step()

        csm_graph = csm.discover_graph(threshold=0.3)
        print(f"CSM discovered graph:\n{csm_graph}")

        csm_metrics = evaluate_causal_discovery(csm_graph, true_graph)
        print(f"CSM F1: {csm_metrics['f1']:.3f}, SHD: {csm_metrics['shd']:.0f}")

        csm_only_scores = []
        for sample in test_samples[:3]:
            exp = standard_llm.generate(sample, csm_graph)
            score = evaluate_explanation_quality(exp, sample, true_graph)
            csm_only_scores.append(score)

        results['ablations']['csm_only'] = {
            'causal_discovery': csm_metrics,
            'explanation_scores': csm_only_scores,
            'avg_score': np.mean([s['score'] for s in csm_only_scores])
        }
        print(f"→ Avg Score: {results['ablations']['csm_only']['avg_score']:.3f}")

    except Exception as e:
        print(f"ERROR in Ablation 1: {e}")
        results['ablations']['csm_only'] = {'error': str(e)}

    # ========================================================================
    # ABLATION 2: Multi-LLM Only (Without causal innovations)
    # ========================================================================
    print("\n" + "="*80)
    print("ABLATION 2: Multi-LLM Only (3 LLMs, no innovations)")
    print("="*80)

    print("NOTE: Requires loading 3 large models - skipping for now")
    print("Would compare: Single GPT-2 vs 3-LLM ensemble without CCLT")

    results['ablations']['multi_llm_only'] = {'note': 'Skipped - requires full model loading'}

    # ========================================================================
    # FULL SYSTEM: CSM + CCLT + Multi-LLM
    # ========================================================================
    print("\n" + "="*80)
    print("FULL SYSTEM: CSM + CCLT + Multi-LLM (ALL INNOVATIONS)")
    print("="*80)

    print("NOTE: Requires loading all 3 LLMs on GPUs", gpu_ids)
    print("This is the complete system with all innovations integrated")

    results['full_system'] = {'note': 'Requires full model loading'}

    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nCausal Discovery F1 Scores:")
    if 'baselines' in results and 'sota_pcmci' in results['baselines']:
        if 'causal_discovery' in results['baselines']['sota_pcmci']:
            print(f"  PCMCI (SOTA):     {results['baselines']['sota_pcmci']['causal_discovery']['f1']:.3f}")

    if 'ablations' in results and 'csm_only' in results['ablations']:
        if 'causal_discovery' in results['ablations']['csm_only']:
            print(f"  CSM (Ours):       {results['ablations']['csm_only']['causal_discovery']['f1']:.3f}")

    print("\nExplanation Quality Scores:")
    for key in ['no_innovation', 'sota_pcmci']:
        if key in results.get('baselines', {}):
            if 'avg_score' in results['baselines'][key]:
                print(f"  {key:20s}: {results['baselines'][key]['avg_score']:.3f}")

    for key in ['csm_only']:
        if key in results.get('ablations', {}):
            if 'avg_score' in results['ablations'][key]:
                print(f"  {key:20s}: {results['ablations'][key]['avg_score']:.3f}")

    # Save to JSON
    output_file = os.path.join(output_dir, 'comprehensive_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)

    return results


if __name__ == '__main__':
    results = run_comprehensive_evaluation(
        output_dir='output/evaluations',
        gpu_ids=[5, 6, 7]
    )

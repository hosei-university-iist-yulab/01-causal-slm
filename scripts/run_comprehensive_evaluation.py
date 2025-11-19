"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 14, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from llm_integration.multi_llm_system_simple import MultiLLMEnsemble, Phi2TechnicalExplainer


def load_all_synthetic_datasets() -> Dict:
    """Load all 4 synthetic datasets with ground truth."""
    datasets = {}
    data_dir = Path('data')

    dataset_names = [
        'synthetic_hvac_simple',
        'synthetic_industrial_simple',
        'synthetic_hvac_complex',
        'synthetic_environmental_complex'
    ]

    print("Loading datasets:")
    for name in dataset_names:
        df = pd.read_csv(data_dir / f'{name}.csv')
        graph = np.load(data_dir / f'{name}_graph.npy')
        variables = [col for col in df.columns if col != 'timestamp']

        datasets[name] = {
            'data': df,
            'ground_truth': graph,
            'variables': variables
        }
        print(f"  ✓ {name}: {len(df)} samples, {len(variables)} variables, {np.sum(graph > 0)} edges")

    return datasets


def discover_causal_graph_simple(data: pd.DataFrame, variables: List[str]) -> np.ndarray:
    """Simple correlation-based causal discovery."""
    X = data[variables].values
    corr_matrix = np.corrcoef(X.T)

    # Create directed graph
    threshold = 0.3
    graph = np.abs(corr_matrix) * (np.abs(corr_matrix) > threshold)
    graph = np.triu(graph, k=1)  # Make acyclic

    return graph


def evaluate_causal_discovery(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Compute causal discovery metrics."""
    threshold = 0.3
    pred_binary = (pred > threshold).astype(int)
    true_binary = (true > 0).astype(int)

    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    shd = fp + fn

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'shd': int(shd),
        'true_edges': int(np.sum(true_binary)),
        'pred_edges': int(np.sum(pred_binary))
    }


def evaluate_narrative_quality(narrative: str, variables: List[str]) -> Dict[str, float]:
    """Simple narrative quality metrics."""
    # Check if narrative mentions variables
    var_coverage = sum(1 for v in variables if v.lower() in narrative.lower()) / len(variables)

    # Check for causal keywords
    causal_keywords = ['because', 'causes', 'due to', 'if', 'then', 'when', 'results in', 'leads to']
    causal_keyword_count = sum(1 for kw in causal_keywords if kw in narrative.lower())

    # Length (normalized)
    length = len(narrative)

    return {
        'variable_coverage': float(var_coverage),
        'causal_keywords': int(causal_keyword_count),
        'length': int(length),
        'completeness_score': float(var_coverage * min(causal_keyword_count / 3, 1.0))
    }


def baseline1_no_innovation(data: pd.DataFrame, variables: List[str]) -> Tuple[str, Dict]:
    """BASELINE 1: Standard single LLM without causal structure."""
    print("\n  Running BASELINE 1: No Innovation (Single LLM, no causal graph)...")

    # Use Phi-2 without causal graph
    llm = Phi2TechnicalExplainer(device='cuda:7')

    # Simple prompt without causal information
    last_obs = data[variables].iloc[-1]
    prompt = f"""Analyze this sensor data:

Current readings: {', '.join([f'{v}={last_obs[v]:.2f}' for v in variables])}

Provide a brief explanation (2-3 sentences):
"""

    narrative = llm.generate(prompt, max_length=150, temperature=0.7)[0]
    narrative = narrative.replace(prompt, "").strip()

    llm.cleanup()

    return narrative, {'method': 'baseline1_no_innovation', 'uses_causal_graph': False}


def baseline2_single_best_with_causal(data: pd.DataFrame, variables: List[str],
                                      causal_graph: np.ndarray) -> Tuple[str, Dict]:
    """BASELINE 2: Single best LLM (Phi-2) WITH causal graph."""
    print("\n  Running BASELINE 2: Single Best LLM (Phi-2) with causal graph...")

    llm = Phi2TechnicalExplainer(device='cuda:7')
    narrative = llm.generate_narrative(data, causal_graph, variables)
    llm.cleanup()

    return narrative, {'method': 'baseline2_single_best', 'uses_causal_graph': True}


def baseline3_multi_llm_no_causal(data: pd.DataFrame, variables: List[str]) -> Tuple[str, Dict]:
    """BASELINE 3: Multi-LLM ensemble WITHOUT causal graph."""
    print("\n  Running BASELINE 3: Multi-LLM without causal graph...")

    # Create dummy graph (all zeros)
    dummy_graph = np.zeros((len(variables), len(variables)))

    ensemble = MultiLLMEnsemble(gpu_ids=[5, 6, 7])
    results = ensemble.generate_ensemble_explanation(
        data, dummy_graph, variables, strategy='best_model'
    )
    ensemble.cleanup()

    return results['ensemble'], {'method': 'baseline3_multi_no_causal', 'uses_causal_graph': False}


def full_system_multi_llm_with_causal(data: pd.DataFrame, variables: List[str],
                                      causal_graph: np.ndarray) -> Tuple[str, Dict]:
    """FULL SYSTEM: Multi-LLM ensemble WITH causal graph."""
    print("\n  Running FULL SYSTEM: Multi-LLM with causal graph...")

    ensemble = MultiLLMEnsemble(gpu_ids=[5, 6, 7])
    results = ensemble.generate_ensemble_explanation(
        data, causal_graph, variables, strategy='best_model'
    )
    ensemble.cleanup()

    return results['ensemble'], {'method': 'full_system', 'uses_causal_graph': True}


def evaluate_single_dataset(dataset_name: str, dataset: Dict) -> Dict:
    """Evaluate all methods on a single dataset."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*80}")

    data = dataset['data']
    true_graph = dataset['ground_truth']
    variables = dataset['variables']

    print(f"  Samples: {len(data)}, Variables: {len(variables)}, True edges: {np.sum(true_graph > 0)}")

    # Discover causal graph
    print("\n  Discovering causal graph...")
    discovered_graph = discover_causal_graph_simple(data, variables)
    causal_metrics = evaluate_causal_discovery(discovered_graph, true_graph)
    print(f"    Discovered edges: {causal_metrics['pred_edges']}, F1: {causal_metrics['f1_score']:.3f}")

    # Run all methods
    results = {
        'dataset': dataset_name,
        'n_samples': len(data),
        'n_variables': len(variables),
        'causal_discovery': causal_metrics,
        'methods': {}
    }

    # Baseline 1: No innovation
    narrative1, meta1 = baseline1_no_innovation(data, variables)
    quality1 = evaluate_narrative_quality(narrative1, variables)
    results['methods']['baseline1_no_innovation'] = {
        'narrative': narrative1,
        'quality': quality1,
        'metadata': meta1
    }
    print(f"    ✓ Baseline 1: {quality1['length']} chars, {quality1['causal_keywords']} causal keywords")

    # Baseline 2: Single best with causal
    narrative2, meta2 = baseline2_single_best_with_causal(data, variables, discovered_graph)
    quality2 = evaluate_narrative_quality(narrative2, variables)
    results['methods']['baseline2_single_best'] = {
        'narrative': narrative2,
        'quality': quality2,
        'metadata': meta2
    }
    print(f"    ✓ Baseline 2: {quality2['length']} chars, {quality2['causal_keywords']} causal keywords")

    # Baseline 3: Multi-LLM no causal
    narrative3, meta3 = baseline3_multi_llm_no_causal(data, variables)
    quality3 = evaluate_narrative_quality(narrative3, variables)
    results['methods']['baseline3_multi_no_causal'] = {
        'narrative': narrative3,
        'quality': quality3,
        'metadata': meta3
    }
    print(f"    ✓ Baseline 3: {quality3['length']} chars, {quality3['causal_keywords']} causal keywords")

    # Full System: Multi-LLM with causal
    narrative4, meta4 = full_system_multi_llm_with_causal(data, variables, discovered_graph)
    quality4 = evaluate_narrative_quality(narrative4, variables)
    results['methods']['full_system'] = {
        'narrative': narrative4,
        'quality': quality4,
        'metadata': meta4
    }
    print(f"    ✓ Full System: {quality4['length']} chars, {quality4['causal_keywords']} causal keywords")

    return results


def compare_methods(all_results: List[Dict]) -> Dict:
    """Compare all methods across datasets."""
    print(f"\n{'='*80}")
    print("COMPARISON ACROSS ALL DATASETS")
    print(f"{'='*80}\n")

    methods = ['baseline1_no_innovation', 'baseline2_single_best',
               'baseline3_multi_no_causal', 'full_system']

    comparison = {method: {
        'avg_completeness': 0.0,
        'avg_causal_keywords': 0.0,
        'avg_variable_coverage': 0.0,
        'avg_length': 0.0
    } for method in methods}

    n_datasets = len(all_results)

    for method in methods:
        for result in all_results:
            quality = result['methods'][method]['quality']
            comparison[method]['avg_completeness'] += quality['completeness_score']
            comparison[method]['avg_causal_keywords'] += quality['causal_keywords']
            comparison[method]['avg_variable_coverage'] += quality['variable_coverage']
            comparison[method]['avg_length'] += quality['length']

        # Average
        for key in comparison[method]:
            comparison[method][key] /= n_datasets

    # Print comparison table
    print(f"{'Method':<30} {'Completeness':<15} {'Causal KW':<12} {'Var Coverage':<15} {'Length':<10}")
    print("-" * 85)
    for method in methods:
        print(f"{method:<30} "
              f"{comparison[method]['avg_completeness']:<15.3f} "
              f"{comparison[method]['avg_causal_keywords']:<12.1f} "
              f"{comparison[method]['avg_variable_coverage']:<15.3f} "
              f"{comparison[method]['avg_length']:<10.1f}")

    return comparison


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION: Multi-LLM System with Baselines")
    print("="*80)
    print("\nComparing:")
    print("  BASELINE 1: No innovation (single LLM, no causal graph)")
    print("  BASELINE 2: Single best LLM (Phi-2) with causal graph")
    print("  BASELINE 3: Multi-LLM without causal graph")
    print("  FULL SYSTEM: Multi-LLM with causal graph")
    print("\nDatasets: 4 synthetic datasets with ground truth")
    print("="*80)

    # Load datasets
    datasets = load_all_synthetic_datasets()

    # Evaluate each dataset
    all_results = []
    for name, dataset in datasets.items():
        result = evaluate_single_dataset(name, dataset)
        all_results.append(result)

    # Compare methods
    comparison = compare_methods(all_results)

    # Save results
    output_dir = Path('output/evaluations')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'comprehensive_evaluation_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'datasets': all_results,
            'comparison': comparison
        }, f, indent=2)

    print(f"\n{'='*80}")
    print("RESULTS SAVED")
    print(f"{'='*80}")
    print(f"  File: {results_file}")
    print("\n✓✓ COMPREHENSIVE EVALUATION COMPLETE! ✓✓\n")


if __name__ == '__main__':
    main()

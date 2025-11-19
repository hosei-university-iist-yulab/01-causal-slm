#!/usr/bin/env python3
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
import json
from pathlib import Path
from datetime import datetime
from sklearn.covariance import GraphicalLassoCV
import gc

# Import our modules
from innovations.causal_score_matching import CausalScoreMatching, CSMConfig
from llm_integration.multi_llm_system_simple import (
    MultiLLMEnsemble,
    GPT2NarrativeGenerator,
    TinyLlamaCounterfactualReasoner,
    Phi2TechnicalExplainer
)


def evaluate_causal_graph(pred: np.ndarray, true: np.ndarray) -> dict:
    """Compute causal discovery metrics."""
    pred_binary = (pred > 0.3).astype(int)
    true_binary = (true > 0).astype(int)

    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    shd = fp + fn

    return {
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'shd': int(shd),
        'n_edges': int(np.sum(pred_binary))
    }


def correlation_discovery(data, variables, threshold=0.5):
    """T1: Correlation baseline."""
    df = pd.DataFrame(data, columns=variables)
    corr = df.corr().abs().values
    np.fill_diagonal(corr, 0)
    graph = (corr > threshold).astype(float) * corr
    graph = np.triu(graph, k=1)
    return graph


def pcmci_discovery(data, variables, threshold=0.1):
    """T2: PCMCI (simplified using GraphicalLassoCV)."""
    X = data
    try:
        glasso = GraphicalLassoCV(cv=5, max_iter=100, tol=1e-3)
        glasso.fit(X)
        precision = glasso.precision_
        graph = np.abs(precision) > threshold
        graph = graph.astype(float) * np.abs(precision)
        graph = np.triu(graph, k=1)
        return graph
    except Exception as e:
        print(f"  Warning: PCMCI failed ({e}), using correlation fallback")
        return correlation_discovery(data, variables)


def csm_discovery(data, variables, device='cuda:5', n_epochs=500):
    """T3: CSM (lightweight training for ablations)."""
    try:
        config = CSMConfig(
            n_variables=len(variables),
            hidden_dim=64,
            n_layers=2,
            learning_rate=0.001,
            device=device
        )
        csm = CausalScoreMatching(config)

        # Lightweight training
        X_tensor = torch.FloatTensor(data).to(device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(n_epochs):
            for batch in dataloader:
                batch_data = batch[0]
                losses = csm.train_step(batch_data, x_0_counterfactual=None)

        discovered_graph = csm.get_causal_graph(threshold=0.3)

        # Cleanup
        del csm, X_tensor, dataset, dataloader
        torch.cuda.empty_cache()
        gc.collect()

        return discovered_graph
    except Exception as e:
        print(f"  Warning: CSM failed ({e}), using correlation fallback")
        return correlation_discovery(data, variables)


def load_dataset(dataset_name):
    """Load dataset (synthetic with ground truth, or real without)."""
    data_dir = Path(__file__).parent.parent / 'data'

    # Load data
    csv_path = data_dir / f'{dataset_name}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    variables = [col for col in df.columns if col not in ['sample', 'time', 'timestamp']]
    data = df[variables].values

    # Load ground truth graph (if exists)
    gt_path = data_dir / f'{dataset_name}_graph.npy'
    if gt_path.exists():
        ground_truth = np.load(gt_path)
    else:
        ground_truth = None

    return data, variables, ground_truth


def evaluate_narrative_quality(narrative, variables):
    """Evaluate narrative quality metrics."""
    if not narrative or len(narrative.strip()) == 0:
        return {
            'length': 0,
            'causal_keywords': 0,
            'variables_mentioned': 0,
            'completeness': 0.0
        }

    # Count causal keywords
    causal_keywords = ['because', 'causes', 'cause', 'leads to', 'results in',
                      'due to', 'if', 'then', 'would', 'counterfactual']
    keyword_count = sum(narrative.lower().count(kw) for kw in causal_keywords)

    # Count variables mentioned
    var_count = sum(1 for var in variables if var.lower() in narrative.lower())

    # Completeness score
    completeness = var_count / len(variables) if variables else 0.0

    return {
        'length': len(narrative),
        'causal_keywords': keyword_count,
        'variables_mentioned': var_count,
        'completeness': completeness
    }


def run_ablation_on_dataset(dataset_name, data, variables, ground_truth):
    """Run comprehensive ablation study on one dataset."""

    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*80}")
    print(f"Variables: {len(variables)}, Samples: {len(data)}")
    if ground_truth is not None:
        print(f"True edges: {np.sum(ground_truth > 0)}")
    print()

    df = pd.DataFrame(data, columns=variables)

    results = {
        'dataset': dataset_name,
        'n_variables': len(variables),
        'n_samples': len(data),
        'true_edges': int(np.sum(ground_truth > 0)) if ground_truth is not None else None,
        'ablations': {}
    }

    # ========================================================================
    # PART 1: INDIVIDUAL COMPONENTS
    # ========================================================================
    print("="*80)
    print("PART 1: INDIVIDUAL COMPONENTS")
    print("="*80)

    print("\n1.1 Theories Only (No LLM)")
    print("-"*40)

    theories = {
        'T1_correlation': lambda: correlation_discovery(data, variables),
        'T2_pcmci': lambda: pcmci_discovery(data, variables),
        'T3_csm': lambda: csm_discovery(data, variables, device='cuda:5', n_epochs=500)
    }

    theory_graphs = {}
    for theory_name, theory_func in theories.items():
        print(f"{theory_name}...", end=" ")
        graph = theory_func()
        theory_graphs[theory_name] = graph
        n_edges = np.sum(graph > 0)

        metrics = {}
        if ground_truth is not None:
            metrics = evaluate_causal_graph(graph, ground_truth)

        results['ablations'][theory_name] = {
            'type': 'theory_only',
            'n_edges': int(n_edges),
            'causal_metrics': metrics
        }

        print(f"Edges={n_edges}", end="")
        if metrics:
            print(f", F1={metrics.get('f1', 0):.3f}")
        else:
            print()

    print("\n1.2 LLMs Only (No Causal Graph)")
    print("-"*40)

    empty_graph = np.zeros((len(variables), len(variables)))

    llms = {
        'L1_gpt2': lambda: GPT2NarrativeGenerator(device='cuda:5'),
        'L2_tinyllama': lambda: TinyLlamaCounterfactualReasoner(device='cuda:6'),
        'L3_phi2': lambda: Phi2TechnicalExplainer(device='cuda:7')
    }

    for llm_name, llm_func in llms.items():
        print(f"{llm_name}...", end=" ")
        llm = llm_func()
        narrative = llm.generate_narrative(df, empty_graph, variables)

        narrative_metrics = evaluate_narrative_quality(narrative, variables)

        results['ablations'][llm_name] = {
            'type': 'llm_only',
            'narrative_length': narrative_metrics['length'],
            'causal_keywords': narrative_metrics['causal_keywords'],
            'completeness': narrative_metrics['completeness']
        }

        print(f"Length={narrative_metrics['length']}, Keywords={narrative_metrics['causal_keywords']}")

        llm.cleanup()
        del llm
        torch.cuda.empty_cache()
        gc.collect()

    # ========================================================================
    # PART 2: THEORY × LLM CROSS-TESTING (9 combinations)
    # ========================================================================
    print("\n"+"="*80)
    print("PART 2: THEORY × LLM CROSS-TESTING (3×3 = 9 combinations)")
    print("="*80)
    print()

    for theory_name, graph in theory_graphs.items():
        print(f"\n{theory_name}:")
        for llm_name, llm_func in llms.items():
            combo_name = f"{theory_name}×{llm_name}"
            print(f"  {combo_name}...", end=" ")

            llm = llm_func()
            narrative = llm.generate_narrative(df, graph, variables)

            causal_metrics = {}
            if ground_truth is not None:
                causal_metrics = evaluate_causal_graph(graph, ground_truth)

            narrative_metrics = evaluate_narrative_quality(narrative, variables)

            results['ablations'][combo_name] = {
                'type': 'theory_llm_cross',
                'theory': theory_name,
                'llm': llm_name,
                'n_edges': int(np.sum(graph > 0)),
                'causal_metrics': causal_metrics,
                'narrative_length': narrative_metrics['length'],
                'causal_keywords': narrative_metrics['causal_keywords'],
                'completeness': narrative_metrics['completeness']
            }

            f1_str = f"F1={causal_metrics.get('f1', 0):.3f}" if causal_metrics else "NoGT"
            print(f"{f1_str}, Len={narrative_metrics['length']}, Kw={narrative_metrics['causal_keywords']}")

            llm.cleanup()
            del llm
            torch.cuda.empty_cache()
            gc.collect()

    # ========================================================================
    # PART 3: COMBINATIONS
    # ========================================================================
    print("\n"+"="*80)
    print("PART 3: THEORY PAIRS × LLM PAIRS")
    print("="*80)

    theory_pairs = [
        ('T1_correlation', 'T2_pcmci'),
        ('T1_correlation', 'T3_csm'),
        ('T2_pcmci', 'T3_csm')
    ]

    llm_pairs = [
        ('L1_gpt2', 'L2_tinyllama'),
        ('L1_gpt2', 'L3_phi2'),
        ('L2_tinyllama', 'L3_phi2')
    ]

    print("\n3.1 Theory Pairs")
    print("-"*40)
    for t1, t2 in theory_pairs:
        pair_name = f"[{t1}+{t2}]"
        print(f"{pair_name}...", end=" ")

        graph1 = theory_graphs[t1]
        graph2 = theory_graphs[t2]
        ensemble_graph = (graph1 + graph2) / 2.0
        ensemble_graph = (ensemble_graph > 0.3).astype(float) * ensemble_graph

        n_edges = np.sum(ensemble_graph > 0)
        causal_metrics = {}
        if ground_truth is not None:
            causal_metrics = evaluate_causal_graph(ensemble_graph, ground_truth)

        results['ablations'][pair_name] = {
            'type': 'theory_pair',
            'theories': [t1, t2],
            'n_edges': int(n_edges),
            'causal_metrics': causal_metrics
        }

        f1_str = f"F1={causal_metrics.get('f1', 0):.3f}" if causal_metrics else "NoGT"
        print(f"Edges={n_edges}, {f1_str}")

    print("\n3.2 LLM Pairs")
    print("-"*40)
    test_graph = theory_graphs['T1_correlation']

    for l1, l2 in llm_pairs:
        pair_name = f"[{l1}+{l2}]"
        print(f"{pair_name}...", end=" ")

        llm1 = llms[l1]()
        llm2 = llms[l2]()

        narrative1 = llm1.generate_narrative(df, test_graph, variables)
        narrative2 = llm2.generate_narrative(df, test_graph, variables)
        combined_narrative = f"{narrative1}\n\n{narrative2}"

        narrative_metrics = evaluate_narrative_quality(combined_narrative, variables)

        results['ablations'][pair_name] = {
            'type': 'llm_pair',
            'llms': [l1, l2],
            'narrative_length': narrative_metrics['length'],
            'causal_keywords': narrative_metrics['causal_keywords'],
            'completeness': narrative_metrics['completeness']
        }

        print(f"Len={narrative_metrics['length']}, Kw={narrative_metrics['causal_keywords']}")

        llm1.cleanup()
        llm2.cleanup()
        del llm1, llm2
        torch.cuda.empty_cache()
        gc.collect()

    # ========================================================================
    # PART 4: FULL SYSTEM
    # ========================================================================
    print("\n"+"="*80)
    print("PART 4: FULL SYSTEM (All Theories + All LLMs)")
    print("="*80)
    print()

    print("Full System: [T1+T2+T3] × [L1+L2+L3]")

    # Ensemble all theory graphs
    full_graph = (theory_graphs['T1_correlation'] +
                  theory_graphs['T2_pcmci'] +
                  theory_graphs['T3_csm']) / 3.0
    full_graph = (full_graph > 0.3).astype(float) * full_graph

    # Use MultiLLMEnsemble
    ensemble = MultiLLMEnsemble(gpu_ids=[5, 6, 7])
    ensemble_results = ensemble.generate_ensemble_explanation(
        df, full_graph, variables, strategy='all'
    )

    # Evaluate
    causal_metrics = {}
    if ground_truth is not None:
        causal_metrics = evaluate_causal_graph(full_graph, ground_truth)

    all_narratives = "\n\n".join([
        ensemble_results.get('gpt2', ''),
        ensemble_results.get('llama2', ''),  # tinyllama maps to llama2
        ensemble_results.get('mistral', '')  # phi2 maps to mistral
    ])

    narrative_metrics = evaluate_narrative_quality(all_narratives, variables)

    results['ablations']['FULL_SYSTEM'] = {
        'type': 'full_system',
        'theories': ['T1_correlation', 'T2_pcmci', 'T3_csm'],
        'llms': ['L1_gpt2', 'L2_tinyllama', 'L3_phi2'],
        'n_edges': int(np.sum(full_graph > 0)),
        'causal_metrics': causal_metrics,
        'narrative_length': narrative_metrics['length'],
        'causal_keywords': narrative_metrics['causal_keywords'],
        'completeness': narrative_metrics['completeness']
    }

    f1_str = f"F1={causal_metrics.get('f1', 0):.3f}" if causal_metrics else "NoGT"
    print(f"Edges={np.sum(full_graph > 0)}, {f1_str}")
    print(f"Len={narrative_metrics['length']}, Kw={narrative_metrics['causal_keywords']}")

    ensemble.cleanup()
    del ensemble
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*80)
print("COMPREHENSIVE ABLATION STUDY - PROFESSIONAL ANALYSIS")
print("="*80)
print()
print("Test Structure:")
print("  1. Individual Components (6 tests)")
print("  2. Theory × LLM Cross-Testing (9 tests)")
print("  3. Theory Pairs × LLM Pairs")
print("  4. Full System (1 test)")
print()
print("Theories: T1=Correlation, T2=PCMCI, T3=CSM")
print("LLMs: L1=GPT-2, L2=TinyLlama, L3=Phi-2")
print("="*80)
print()

# Test datasets: 2 synthetic + 1 real
test_datasets = [
    'synthetic_hvac_simple',  # Synthetic 1
    'synthetic_industrial_simple',  # Synthetic 2
    'real_hvac_energy'  # Real
]

print(f"Datasets: {len(test_datasets)}")
for ds in test_datasets:
    print(f"  - {ds}")
print()

# Run ablation on all datasets
all_results = {}

for dataset_name in test_datasets:
    print(f"\n{'#'*80}")
    print(f"# PROCESSING: {dataset_name}")
    print(f"{'#'*80}\n")

    try:
        data, variables, ground_truth = load_dataset(dataset_name)
        dataset_results = run_ablation_on_dataset(dataset_name, data, variables, ground_truth)
        all_results[dataset_name] = dataset_results

        print(f"\n✓ {dataset_name} complete!")

    except Exception as e:
        print(f"\n✗ {dataset_name} failed: {e}")
        import traceback
        traceback.print_exc()
        continue


# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n"+"="*80)
print("SAVING RESULTS")
print("="*80)

output_dir = Path(__file__).parent.parent / 'output' / 'evaluations'
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = output_dir / f'comprehensive_ablations_{timestamp}.json'

with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")


# ============================================================================
# SUMMARY TABLE (ALL DATASETS)
# ============================================================================
print("\n"+"="*80)
print("SUMMARY TABLE - ALL DATASETS")
print("="*80)
print()

for dataset_name, dataset_results in all_results.items():
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"{'Component':<40} {'Type':<15} {'F1':>6} {'Edges':>6} {'Length':>8} {'Keywords':>8}")
    print("-"*90)

    for name, data in dataset_results['ablations'].items():
        comp_type = data.get('type', 'unknown')[:14]
        f1 = data.get('causal_metrics', {}).get('f1', 0.0)
        edges = data.get('n_edges', 0)
        length = data.get('narrative_length', 0)
        keywords = data.get('causal_keywords', 0)

        f1_str = f"{f1:.3f}" if f1 > 0 else "N/A"
        edges_str = f"{edges}" if edges > 0 else "N/A"
        length_str = f"{length}" if length > 0 else "N/A"
        keywords_str = f"{keywords}" if keywords > 0 else "N/A"

        print(f"{name:<40} {comp_type:<15} {f1_str:>6} {edges_str:>6} {length_str:>8} {keywords_str:>8}")

print("\n"+"="*80)
print("✓ COMPREHENSIVE ABLATION STUDY COMPLETE!")
print("="*80)

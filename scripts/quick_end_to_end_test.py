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

# Import our innovations
from innovations.causal_score_matching import CausalScoreMatching, CSMConfig
from llm_integration.multi_llm_system_simple import MultiLLMEnsemble


def load_dataset(dataset_name: str = 'synthetic_hvac_simple'):
    """Load dataset and ground truth graph."""
    data_dir = Path('data')

    # Load data
    df = pd.read_csv(data_dir / f'{dataset_name}.csv')
    print(f"✓ Loaded {dataset_name}.csv: {df.shape}")

    # Load ground truth graph
    graph_path = data_dir / f'{dataset_name}_graph.npy'
    if graph_path.exists():
        true_graph = np.load(graph_path)
        print(f"✓ Loaded ground truth graph: {true_graph.shape}")
    else:
        true_graph = None
        print(f"⚠ No ground truth graph found")

    # Get variable names (exclude timestamp)
    variable_names = [col for col in df.columns if col != 'timestamp']

    return df, true_graph, variable_names


def discover_causal_graph(data: pd.DataFrame, variable_names: list):
    """Discover causal graph using simple correlation-based method."""
    print("\n" + "="*80)
    print("STEP 1: Causal Discovery (Correlation-based for quick test)")
    print("="*80)

    # Prepare data (exclude timestamp)
    X = data[variable_names].values

    print(f"Discovering causal structure from {X.shape[0]} samples, {X.shape[1]} variables...")

    # Simple correlation-based causal discovery for quick test
    # In full evaluation, this will use CSM
    corr_matrix = np.corrcoef(X.T)

    # Create directed graph: correlation > threshold → causal edge
    threshold = 0.3
    discovered_graph = np.abs(corr_matrix) * (np.abs(corr_matrix) > threshold)

    # Make acyclic by removing lower triangle
    discovered_graph = np.triu(discovered_graph, k=1)

    print(f"\n✓ Discovered {np.sum(discovered_graph > 0)} causal edges")
    print(f"  Graph sparsity: {np.sum(discovered_graph > 0) / (X.shape[1]**2):.2%}")
    print(f"  Note: Using simple correlation-based method for quick test")
    print(f"        Full evaluation will use CSM (Innovation 1)")

    return discovered_graph


def generate_narratives(data: pd.DataFrame, causal_graph: np.ndarray,
                       variable_names: list):
    """Generate causal narratives with 3 LLMs."""
    print("\n" + "="*80)
    print("STEP 2: Generate Narratives with Multi-LLM System")
    print("="*80)

    # Initialize ensemble
    ensemble = MultiLLMEnsemble(gpu_ids=[5, 6, 7])

    # Generate with all strategies
    strategies = ['best_model', 'all']
    all_results = {}

    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        results = ensemble.generate_ensemble_explanation(
            data, causal_graph, variable_names, strategy=strategy
        )
        all_results[strategy] = results

    # Cleanup
    ensemble.cleanup()

    return all_results


def evaluate_causal_discovery(discovered_graph: np.ndarray,
                              true_graph: np.ndarray):
    """Evaluate causal discovery accuracy."""
    print("\n" + "="*80)
    print("STEP 3: Evaluate Causal Discovery")
    print("="*80)

    # Threshold discovered graph
    threshold = 0.3
    pred_binary = (discovered_graph > threshold).astype(int)
    true_binary = (true_graph > 0).astype(int)

    # Compute metrics
    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Structural Hamming Distance
    shd = fp + fn

    metrics = {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'shd': int(shd)
    }

    print(f"\nCausal Discovery Metrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  SHD: {shd}")
    print(f"  True edges: {np.sum(true_binary > 0)}")
    print(f"  Discovered edges: {np.sum(pred_binary > 0)}")

    return metrics


def save_results(dataset_name: str, discovered_graph: np.ndarray,
                narratives: dict, metrics: dict):
    """Save all outputs."""
    print("\n" + "="*80)
    print("STEP 4: Save Results")
    print("="*80)

    # Create output directories
    output_dir = Path('output')
    (output_dir / 'causal_graphs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'narratives').mkdir(parents=True, exist_ok=True)
    (output_dir / 'evaluations').mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{dataset_name}_{timestamp}"

    # Save discovered graph
    graph_path = output_dir / 'causal_graphs' / f'{prefix}_discovered_graph.npy'
    np.save(graph_path, discovered_graph)
    print(f"✓ Saved discovered graph: {graph_path}")

    # Save narratives
    for strategy, results in narratives.items():
        narrative_path = output_dir / 'narratives' / f'{prefix}_{strategy}.txt'
        with open(narrative_path, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Strategy: {strategy}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("="*80 + "\n\n")

            for llm_name, narrative in results.items():
                if llm_name != 'strategy':
                    f.write(f"{llm_name.upper()}:\n{narrative}\n\n")

        print(f"✓ Saved narrative ({strategy}): {narrative_path}")

    # Save metrics
    metrics_path = output_dir / 'evaluations' / f'{prefix}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics: {metrics_path}")

    return prefix


def main():
    print("\n" + "="*80)
    print("QUICK END-TO-END TEST: CSM + Multi-LLM System")
    print("="*80)
    print("Testing complete pipeline:")
    print("  1. Load synthetic dataset with ground truth")
    print("  2. Discover causal graph with CSM")
    print("  3. Generate narratives with 3 LLMs")
    print("  4. Evaluate and save results")
    print("="*80 + "\n")

    # Configuration
    dataset_name = 'synthetic_hvac_simple'
    print(f"Dataset: {dataset_name}\n")

    # Load data
    data, true_graph, variable_names = load_dataset(dataset_name)
    print(f"Variables: {variable_names}\n")

    # Step 1: Causal Discovery
    discovered_graph = discover_causal_graph(data, variable_names)

    # Step 2: Generate Narratives
    narratives = generate_narratives(data, discovered_graph, variable_names)

    # Step 3: Evaluate (if ground truth available)
    if true_graph is not None:
        metrics = evaluate_causal_discovery(discovered_graph, true_graph)
    else:
        metrics = {'note': 'No ground truth available for this dataset'}

    # Step 4: Save Results
    prefix = save_results(dataset_name, discovered_graph, narratives, metrics)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Complete pipeline executed successfully!")
    print(f"  Dataset: {dataset_name}")
    print(f"  Variables: {len(variable_names)}")
    print(f"  Samples: {len(data)}")
    if true_graph is not None:
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"  Output prefix: {prefix}")
    print("\nOutput files:")
    print(f"  - output/causal_graphs/{prefix}_discovered_graph.npy")
    print(f"  - output/narratives/{prefix}_*.txt")
    print(f"  - output/evaluations/{prefix}_metrics.json")
    print("="*80 + "\n")

    print("✓✓ END-TO-END TEST COMPLETE! ✓✓\n")


if __name__ == '__main__':
    main()

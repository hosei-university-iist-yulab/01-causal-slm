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
from pathlib import Path
import json
from datetime import datetime

from llm_integration.multi_llm_system_simple import MultiLLMEnsemble

def discover_causal_graph_simple(data: pd.DataFrame, variables: list) -> np.ndarray:
    """Simple correlation-based causal discovery."""
    X = data[variables].values
    corr_matrix = np.corrcoef(X.T)
    threshold = 0.3
    graph = np.abs(corr_matrix) * (np.abs(corr_matrix) > threshold)
    graph = np.triu(graph, k=1)
    return graph

def evaluate_real_dataset(dataset_name: str, data: pd.DataFrame, variables: list):
    """Evaluate on real dataset."""
    print(f"\n{'='*80}")
    print(f"REAL DATASET: {dataset_name}")
    print(f"{'='*80}")
    print(f"Samples: {len(data)}, Variables: {len(variables)}")

    # Discover causal graph
    print("\nDiscovering causal structure...")
    graph = discover_causal_graph_simple(data, variables)
    n_edges = np.sum(graph > 0)
    print(f"✓ Discovered {n_edges} causal edges")

    # Show discovered relationships
    print("\nDiscovered causal relationships:")
    for i in range(len(variables)):
        for j in range(len(variables)):
            if graph[i,j] > 0:
                print(f"  {variables[i]} → {variables[j]} (strength: {graph[i,j]:.3f})")

    # Generate narratives
    print("\nGenerating narratives from 3 LLMs...")
    ensemble = MultiLLMEnsemble(gpu_ids=[5, 6, 7])

    results = ensemble.generate_ensemble_explanation(
        data, graph, variables, strategy='all'
    )

    ensemble.cleanup()

    # Save results
    output_dir = Path('output/real_datasets')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save narratives
    narrative_file = output_dir / f'{dataset_name}_{timestamp}_narratives.txt'
    with open(narrative_file, 'w') as f:
        f.write(f"Real Dataset Evaluation: {dataset_name}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Variables: {', '.join(variables)}\n")
        f.write(f"Discovered edges: {n_edges}\n\n")
        f.write(f"Causal relationships:\n")
        for i in range(len(variables)):
            for j in range(len(variables)):
                if graph[i,j] > 0:
                    f.write(f"  {variables[i]} → {variables[j]}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"GENERATED NARRATIVES\n")
        f.write(f"{'='*80}\n\n")

        for llm_name, narrative in results.items():
            if llm_name != 'strategy':
                f.write(f"{llm_name.upper()}:\n{narrative}\n\n{'='*80}\n\n")

    print(f"\n✓ Results saved to: {narrative_file}")

    return results

def main():
    print("\n" + "="*80)
    print("REAL DATASET EVALUATION")
    print("="*80)
    print("\nEvaluating multi-LLM system on real-world data")
    print("Note: No ground truth available - qualitative analysis only\n")

    data_dir = Path('data')

    # Dataset 1: Real HVAC Energy
    print("\n" + "="*80)
    print("1. REAL HVAC ENERGY DATASET")
    print("="*80)

    df_hvac = pd.read_csv(data_dir / 'real_hvac_energy.csv')
    hvac_vars = [col for col in df_hvac.columns if col != 'timestamp']

    hvac_results = evaluate_real_dataset('real_hvac_energy', df_hvac, hvac_vars)

    # Dataset 2: Real Environmental
    print("\n" + "="*80)
    print("2. REAL ENVIRONMENTAL AIR QUALITY DATASET")
    print("="*80)

    df_env = pd.read_csv(data_dir / 'real_environmental_airquality.csv')
    env_vars = [col for col in df_env.columns if col != 'timestamp']

    env_results = evaluate_real_dataset('real_environmental_airquality', df_env, env_vars)

    print("\n" + "="*80)
    print("REAL DATASET EVALUATION COMPLETE")
    print("="*80)
    print("\nResults saved to: output/real_datasets/")
    print("\nQualitative Analysis:")
    print("  - Check narrative coherence")
    print("  - Verify domain knowledge")
    print("  - Compare with domain expert expectations")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()

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
from typing import Dict, Tuple

from innovations.causal_score_matching import CausalScoreMatching, CSMConfig
from innovations.intervention_generator import InterventionGenerator, InterventionConfig, create_hvac_scm


def train_csm_on_data(
    data: pd.DataFrame,
    variables: list,
    device: str = 'cuda:5',
    n_epochs: int = 500,
    batch_size: int = 32,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Train CSM on dataset and return discovered causal graph.

    Args:
        data: Dataset with variable columns
        variables: List of variable names
        device: GPU device
        n_epochs: Training epochs
        batch_size: Batch size
        verbose: Print progress

    Returns:
        discovered_graph: Causal adjacency matrix
        metrics: Training metrics
    """
    X = data[variables].values
    n_vars = len(variables)

    if verbose:
        print(f"  Training CSM on {X.shape[0]} samples, {n_vars} variables...")

    # Configure CSM
    config = CSMConfig(
        n_variables=n_vars,
        hidden_dim=128,
        n_layers=3,
        n_timesteps=1000,
        lambda_interventional=1.0,
        lambda_graph=0.1,
        lambda_acyclicity=1.0,
        device=device
    )

    csm = CausalScoreMatching(config)

    # Training loop
    history = {
        'loss_total': [],
        'loss_obs': [],
        'loss_int': [],
        'loss_graph': [],
        'loss_acyc': []
    }

    # Convert to tensor
    X_tensor = torch.FloatTensor(X).to(device)
    n_samples = X.shape[0]

    for epoch in range(n_epochs):
        # Sample batch
        indices = torch.randint(0, n_samples, (batch_size,))
        batch = X_tensor[indices]

        # Training step (observational only for simplicity)
        losses = csm.train_step(batch, x_0_counterfactual=None)

        # Log
        for key, value in losses.items():
            if key in history:
                history[key].append(value)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: Loss = {losses['total']:.4f}")

    # Get discovered graph
    discovered_graph = csm.get_causal_graph(threshold=0.3)

    if verbose:
        n_edges = np.sum(discovered_graph > 0)
        print(f"  ✓ CSM training complete: {n_edges} edges discovered")

    metrics = {
        'final_loss': history['loss_total'][-1] if history['loss_total'] else 0,
        'n_edges_discovered': int(np.sum(discovered_graph > 0))
    }

    return discovered_graph, metrics


def compare_correlation_vs_csm(data: pd.DataFrame, variables: list,
                               true_graph: np.ndarray = None) -> Dict:
    """Compare correlation-based vs CSM discovery."""
    print("\n" + "="*80)
    print("COMPARISON: Correlation vs CSM")
    print("="*80)

    X = data[variables].values

    # Method 1: Correlation (baseline)
    print("\n1. Correlation-based discovery...")
    corr_matrix = np.corrcoef(X.T)
    corr_graph = np.abs(corr_matrix) * (np.abs(corr_matrix) > 0.3)
    corr_graph = np.triu(corr_graph, k=1)
    n_corr_edges = np.sum(corr_graph > 0)
    print(f"   Discovered {n_corr_edges} edges")

    # Method 2: CSM
    print("\n2. CSM-based discovery...")
    csm_graph, csm_metrics = train_csm_on_data(
        data, variables, device='cuda:5', n_epochs=500, verbose=True
    )

    # Compare
    results = {
        'correlation': {
            'graph': corr_graph,
            'n_edges': n_corr_edges
        },
        'csm': {
            'graph': csm_graph,
            'n_edges': csm_metrics['n_edges_discovered']
        }
    }

    # If ground truth available, compute metrics
    if true_graph is not None:
        from sklearn.metrics import precision_score, recall_score, f1_score

        true_binary = (true_graph > 0).astype(int).flatten()

        # Correlation metrics
        corr_binary = (corr_graph > 0.3).astype(int).flatten()
        corr_f1 = f1_score(true_binary, corr_binary, zero_division=0)
        corr_prec = precision_score(true_binary, corr_binary, zero_division=0)
        corr_rec = recall_score(true_binary, corr_binary, zero_division=0)

        # CSM metrics
        csm_binary = (csm_graph > 0.3).astype(int).flatten()
        csm_f1 = f1_score(true_binary, csm_binary, zero_division=0)
        csm_prec = precision_score(true_binary, csm_binary, zero_division=0)
        csm_rec = recall_score(true_binary, csm_binary, zero_division=0)

        results['correlation']['f1'] = corr_f1
        results['correlation']['precision'] = corr_prec
        results['correlation']['recall'] = corr_rec

        results['csm']['f1'] = csm_f1
        results['csm']['precision'] = csm_prec
        results['csm']['recall'] = csm_rec

        # Print comparison
        print(f"\n{'Method':<20} {'F1':<10} {'Precision':<12} {'Recall':<10}")
        print("-" * 55)
        print(f"{'Correlation':<20} {corr_f1:<10.3f} {corr_prec:<12.3f} {corr_rec:<10.3f}")
        print(f"{'CSM (Innovation 1)':<20} {csm_f1:<10.3f} {csm_prec:<12.3f} {csm_rec:<10.3f}")

        improvement = ((csm_f1 - corr_f1) / corr_f1 * 100) if corr_f1 > 0 else 0
        print(f"\nImprovement: {improvement:+.1f}%")

    return results


if __name__ == '__main__':
    # Test on synthetic_hvac_simple
    print("\n" + "="*80)
    print("CSM INTEGRATION TEST")
    print("="*80)

    from pathlib import Path

    data_dir = Path('data')
    dataset_name = 'synthetic_hvac_simple'

    # Load data
    df = pd.read_csv(data_dir / f'{dataset_name}.csv')
    true_graph = np.load(data_dir / f'{dataset_name}_graph.npy')
    variables = [col for col in df.columns if col != 'timestamp']

    print(f"\nDataset: {dataset_name}")
    print(f"  Samples: {len(df)}")
    print(f"  Variables: {len(variables)}")
    print(f"  True edges: {np.sum(true_graph > 0)}")

    # Compare methods
    results = compare_correlation_vs_csm(df, variables, true_graph)

    print("\n✓ CSM integration test complete!")

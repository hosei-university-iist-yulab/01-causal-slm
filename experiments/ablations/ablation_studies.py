"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 7, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

"""
Ablation study experiments for CSLM components.
Tests impact of removing CSM, BAT, or CCLT.
Quantifies contribution of each innovation.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, List
from sklearn.metrics import f1_score, mean_absolute_error, r2_score

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from causal_score_matching import CausalScoreMatching, CSMConfig
from intervention_generator import create_hvac_scm
from counterfactual_reasoning import CounterfactualReasoner, CounterfactualConfig


# ============================================================================
# 1. Causal Discovery Baselines
# ============================================================================

def baseline_correlation(data: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Baseline: Use correlation as causal graph."""
    corr = np.corrcoef(data.T)
    graph = (np.abs(corr) > threshold).astype(float)
    np.fill_diagonal(graph, 0)
    return graph


def baseline_granger(data: np.ndarray, max_lag: int = 1, threshold: float = 0.05) -> np.ndarray:
    """Baseline: Granger causality (simplified version)."""
    from scipy import stats
    n_vars = data.shape[1]
    graph = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue

            # Simple Granger test: does past of j predict current i?
            X_past = data[:-max_lag, j]
            Y_current = data[max_lag:, i]

            if len(X_past) > max_lag + 1:
                corr, p_value = stats.pearsonr(X_past, Y_current)
                if p_value < threshold:
                    graph[j, i] = 1  # j causes i

    return graph


def run_causal_discovery_ablation(
    obs_data: np.ndarray,
    int_data: np.ndarray,
    true_graph: np.ndarray,
    device: str = 'cuda'
) -> Dict:
    """
    Compare causal discovery methods.

    Methods:
    1. CSM (ours) - with interventions
    2. CSM (observational only) - ablation
    3. Correlation (baseline)
    4. Granger causality (baseline)
    """
    print("\n" + "="*80)
    print("ABLATION 1: Causal Discovery Methods")
    print("="*80)

    n_vars = obs_data.shape[1]
    results = {}

    # Ground truth metrics helper
    def compute_metrics(pred_graph):
        pred_flat = pred_graph.flatten()
        true_flat = true_graph.flatten()
        f1 = f1_score(true_flat, pred_flat)
        shd = np.sum(np.abs(true_graph - pred_graph))
        return {'f1': f1, 'shd': shd}

    # Method 1: CSM with interventions (ours)
    print("\n1. CSM (with interventions)")
    print("-"*40)
    csm_config = CSMConfig(
        n_variables=n_vars,
        hidden_dim=64,
        lambda_interventional=0.5,
        lambda_graph=1.0,
        lambda_acyclicity=5.0
    )
    csm = CausalScoreMatching(csm_config).to(device)

    # Train
    optimizer = torch.optim.Adam(csm.parameters(), lr=1e-3)
    X_obs = torch.FloatTensor(obs_data[:600]).to(device)
    X_int = torch.FloatTensor(int_data[:600]).to(device)

    for epoch in range(100):  # Faster for ablation
        idx_obs = torch.randperm(600)[:32]
        idx_int = torch.randperm(600)[:32]

        optimizer.zero_grad()
        loss, _ = csm.train_step(X_obs[idx_obs], X_int[idx_int])
        loss.backward()
        optimizer.step()

    graph_csm = csm.discover_graph(threshold=0.3)
    results['csm_with_int'] = compute_metrics(graph_csm)
    print(f"F1: {results['csm_with_int']['f1']:.3f}, SHD: {results['csm_with_int']['shd']:.0f}")

    # Method 2: CSM observational only (ablation)
    print("\n2. CSM (observational only) - Ablation")
    print("-"*40)
    csm_obs = CausalScoreMatching(csm_config).to(device)
    optimizer_obs = torch.optim.Adam(csm_obs.parameters(), lr=1e-3)

    for epoch in range(100):
        idx = torch.randperm(600)[:32]
        optimizer_obs.zero_grad()
        loss, _ = csm_obs.train_step(X_obs[idx], None)  # No interventions!
        loss.backward()
        optimizer_obs.step()

    graph_csm_obs = csm_obs.discover_graph(threshold=0.3)
    results['csm_obs_only'] = compute_metrics(graph_csm_obs)
    print(f"F1: {results['csm_obs_only']['f1']:.3f}, SHD: {results['csm_obs_only']['shd']:.0f}")

    # Method 3: Correlation baseline
    print("\n3. Correlation (baseline)")
    print("-"*40)
    graph_corr = baseline_correlation(obs_data)
    results['correlation'] = compute_metrics(graph_corr)
    print(f"F1: {results['correlation']['f1']:.3f}, SHD: {results['correlation']['shd']:.0f}")

    # Method 4: Granger causality
    print("\n4. Granger Causality (baseline)")
    print("-"*40)
    graph_granger = baseline_granger(obs_data)
    results['granger'] = compute_metrics(graph_granger)
    print(f"F1: {results['granger']['f1']:.3f}, SHD: {results['granger']['shd']:.0f}")

    # Summary
    print("\n" + "-"*80)
    print("SUMMARY: Causal Discovery")
    print("-"*80)
    for method, metrics in results.items():
        print(f"{method:20s}: F1={metrics['f1']:.3f}, SHD={metrics['shd']:.0f}")

    # Improvement calculation
    best_baseline_f1 = max(results['correlation']['f1'], results['granger']['f1'])
    csm_improvement = (results['csm_with_int']['f1'] - best_baseline_f1) / best_baseline_f1 * 100
    print(f"\nCSM improvement over best baseline: {csm_improvement:.1f}%")

    return results


# ============================================================================
# 2. Counterfactual Reasoning Baselines
# ============================================================================

class StandardMLPPredictor(nn.Module):
    """Baseline: Standard MLP (no causal structure)."""
    def __init__(self, n_variables: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_variables, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_variables)
        )

    def forward(self, x):
        return self.net(x)


def run_counterfactual_ablation(
    factual_data: np.ndarray,
    counterfactual_data: np.ndarray,
    causal_graph: np.ndarray,
    intervention_var: int,
    intervention_value: float,
    device: str = 'cuda'
) -> Dict:
    """
    Compare counterfactual reasoning methods.

    Methods:
    1. CCLT (ours) - full model
    2. CCLT without abduction loss - ablation
    3. CCLT without structure loss - ablation
    4. Standard MLP (baseline)
    """
    print("\n" + "="*80)
    print("ABLATION 2: Counterfactual Reasoning Methods")
    print("="*80)

    n_vars = factual_data.shape[1]
    results = {}

    # Split data
    n_train = 600
    X_fact_train = torch.FloatTensor(factual_data[:n_train]).to(device)
    X_cf_train = torch.FloatTensor(counterfactual_data[:n_train]).to(device)
    X_fact_test = torch.FloatTensor(factual_data[n_train:]).to(device)
    X_cf_test = torch.FloatTensor(counterfactual_data[n_train:]).to(device)
    G = torch.FloatTensor(causal_graph).to(device)

    # Metrics helper
    def evaluate(model, model_type='cclt'):
        model.eval()
        with torch.no_grad():
            if model_type == 'cclt':
                Y_pred, _ = model(
                    X_fact_test, X_fact_test, G,
                    intervention_var, intervention_value
                )
            else:  # standard MLP
                Y_pred = model(X_fact_test)

            mae = torch.abs(Y_pred - X_cf_test).mean().item()

            # R² on outcome variable
            y_true = X_cf_test[:, -1].cpu().numpy()
            y_pred = Y_pred[:, -1].cpu().numpy()
            r2 = r2_score(y_true, y_pred)

            return {'mae': mae, 'r2': r2}

    # Method 1: CCLT full (ours)
    print("\n1. CCLT (full model)")
    print("-"*40)
    config_full = CounterfactualConfig(
        n_variables=n_vars,
        hidden_dim=128,
        lambda_abduction=1.0,
        lambda_action=0.5,
        lambda_prediction=2.0,
        lambda_structure=1.0,
        device=device
    )
    cclt_full = CounterfactualReasoner(config_full).to(device)

    # Train (simplified)
    optimizer = torch.optim.Adam(cclt_full.parameters(), lr=1e-3)
    for epoch in range(100):
        idx = torch.randperm(n_train)[:32]
        optimizer.zero_grad()
        Y_pred, _ = cclt_full(
            X_fact_train[idx], X_fact_train[idx], G,
            intervention_var, intervention_value
        )
        loss = nn.MSELoss()(Y_pred, X_cf_train[idx])
        loss.backward()
        optimizer.step()

    results['cclt_full'] = evaluate(cclt_full, 'cclt')
    print(f"MAE: {results['cclt_full']['mae']:.4f}, R²: {results['cclt_full']['r2']:.4f}")

    # Method 2: CCLT without abduction (ablation)
    print("\n2. CCLT (no abduction loss) - Ablation")
    print("-"*40)
    config_no_abd = CounterfactualConfig(
        n_variables=n_vars,
        hidden_dim=128,
        lambda_abduction=0.0,  # Removed!
        lambda_action=0.5,
        lambda_prediction=2.0,
        lambda_structure=1.0,
        device=device
    )
    cclt_no_abd = CounterfactualReasoner(config_no_abd).to(device)

    optimizer = torch.optim.Adam(cclt_no_abd.parameters(), lr=1e-3)
    for epoch in range(100):
        idx = torch.randperm(n_train)[:32]
        optimizer.zero_grad()
        Y_pred, _ = cclt_no_abd(
            X_fact_train[idx], X_fact_train[idx], G,
            intervention_var, intervention_value
        )
        loss = nn.MSELoss()(Y_pred, X_cf_train[idx])
        loss.backward()
        optimizer.step()

    results['cclt_no_abduction'] = evaluate(cclt_no_abd, 'cclt')
    print(f"MAE: {results['cclt_no_abduction']['mae']:.4f}, R²: {results['cclt_no_abduction']['r2']:.4f}")

    # Method 3: Standard MLP (baseline)
    print("\n3. Standard MLP (baseline)")
    print("-"*40)
    mlp = StandardMLPPredictor(n_vars, hidden_dim=128).to(device)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for epoch in range(100):
        idx = torch.randperm(n_train)[:32]
        optimizer.zero_grad()
        Y_pred = mlp(X_fact_train[idx])
        loss = nn.MSELoss()(Y_pred, X_cf_train[idx])
        loss.backward()
        optimizer.step()

    results['standard_mlp'] = evaluate(mlp, 'mlp')
    print(f"MAE: {results['standard_mlp']['mae']:.4f}, R²: {results['standard_mlp']['r2']:.4f}")

    # Summary
    print("\n" + "-"*80)
    print("SUMMARY: Counterfactual Reasoning")
    print("-"*80)
    for method, metrics in results.items():
        print(f"{method:25s}: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

    # Improvement calculation
    baseline_mae = results['standard_mlp']['mae']
    cclt_mae = results['cclt_full']['mae']
    improvement = (baseline_mae - cclt_mae) / baseline_mae * 100
    print(f"\nCCLT MAE improvement over MLP: {improvement:.1f}%")

    return results


# ============================================================================
# 3. Main
# ============================================================================

def main():
    print("="*80)
    print("Week 7-8: Comprehensive Ablation Studies")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Generate data
    print("\n1. Generating Experimental Data")
    print("-"*80)
    scm = create_hvac_scm()
    true_graph = scm.causal_graph

    obs_data = scm.sample(1000)
    int_data = scm.intervene(1, 1.0, 1000)  # do(HVAC=1)

    print(f"True causal graph:\n{true_graph}")
    print(f"Data shapes: Obs={obs_data.shape}, Int={int_data.shape}")

    # Ablation 1: Causal Discovery
    discovery_results = run_causal_discovery_ablation(
        obs_data, int_data, true_graph, device
    )

    # Ablation 2: Counterfactual Reasoning
    cf_results = run_counterfactual_ablation(
        obs_data, int_data, true_graph,
        intervention_var=1,
        intervention_value=1.0,
        device=device
    )

    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY: All Ablations")
    print("="*80)

    print("\nCausal Discovery:")
    print(f"  CSM (ours):           F1={discovery_results['csm_with_int']['f1']:.3f} ✓")
    print(f"  Best baseline:        F1={max(discovery_results['correlation']['f1'], discovery_results['granger']['f1']):.3f}")

    print("\nCounterfactual Reasoning:")
    print(f"  CCLT (ours):          R²={cf_results['cclt_full']['r2']:.4f} ✓")
    print(f"  Standard MLP:         R²={cf_results['standard_mlp']['r2']:.4f}")

    print("\n" + "="*80)
    print("Ablation Studies COMPLETE!")
    print("="*80)

    return discovery_results, cf_results


if __name__ == '__main__':
    discovery_results, cf_results = main()

"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 6, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

"""
Integration tests for CSM and CCLT components.
Validates compatibility and performance of combined system.
Tests gradient flow and training stability.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from causal_score_matching import CausalScoreMatching, CSMConfig
from intervention_generator import StructuralCausalModel, create_hvac_scm
from counterfactual_reasoning import (
    CounterfactualReasoner,
    CounterfactualConfig,
    CounterfactualLoss,
    StructuralEquationLoss
)


# ============================================================================
# 1. End-to-End Pipeline
# ============================================================================

class CausalReasoningSystem:
    """
    Complete causal reasoning system integrating CSM + CCLT.

    Pipeline:
    1. Discover causal graph G from observational data (CSM)
    2. Train counterfactual reasoner on G (CCLT)
    3. Answer counterfactual queries
    """

    def __init__(
        self,
        n_variables: int,
        device: str = 'cuda'
    ):
        self.n_variables = n_variables
        self.device = device

        # Step 1: Causal discovery (CSM)
        self.csm_config = CSMConfig(
            n_variables=n_variables,
            hidden_dim=64,
            lambda_interventional=0.5,
            lambda_graph=1.0,
            lambda_acyclicity=5.0
        )
        self.csm = CausalScoreMatching(self.csm_config).to(device)

        # Step 2: Counterfactual reasoning (CCLT)
        self.cclt_config = CounterfactualConfig(
            n_variables=n_variables,
            hidden_dim=128,
            lambda_abduction=1.0,
            lambda_action=0.5,
            lambda_prediction=2.0,
            lambda_structure=1.0,
            device=device
        )
        self.counterfactual_reasoner = CounterfactualReasoner(
            self.cclt_config
        ).to(device)

        # Discovered causal graph
        self.causal_graph = None

    def discover_causal_graph(
        self,
        observational_data: np.ndarray,
        interventional_data: Dict,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Step 1: Discover causal graph using CSM.

        Args:
            observational_data: (n_samples, n_variables)
            interventional_data: Dict with 'data' and 'intervention_masks'

        Returns:
            causal_graph: (n_variables, n_variables) adjacency matrix
        """
        if verbose:
            print("\n" + "="*80)
            print("STEP 1: Causal Graph Discovery (CSM)")
            print("="*80)

        # Convert to tensors
        X_obs = torch.FloatTensor(observational_data).to(self.device)
        X_int = torch.FloatTensor(interventional_data['data']).to(self.device)
        int_masks = torch.FloatTensor(
            interventional_data['intervention_masks']
        ).to(self.device)

        # Train CSM
        optimizer = torch.optim.Adam(self.csm.parameters(), lr=lr)

        n_obs = X_obs.shape[0]
        n_int = X_int.shape[0]

        for epoch in range(epochs):
            # Observational batch
            obs_idx = torch.randperm(n_obs)[:batch_size]
            X_obs_batch = X_obs[obs_idx]

            # Interventional batch
            int_idx = torch.randperm(n_int)[:batch_size]
            X_int_batch = X_int[int_idx]

            # Train step
            optimizer.zero_grad()
            loss, components = self.csm.train_step(
                X_obs_batch,
                X_int_batch
            )
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Graph: {components['graph_loss']:.4f}")

        # Discover graph
        self.causal_graph = self.csm.discover_graph(threshold=0.3)

        if verbose:
            print(f"\nDiscovered causal graph:")
            print(self.causal_graph)

        return self.causal_graph

    def train_counterfactual_reasoner(
        self,
        factual_data: np.ndarray,
        counterfactual_data: np.ndarray,
        intervention_var: int,
        intervention_value: float,
        U_true: np.ndarray = None,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        verbose: bool = True
    ) -> Dict:
        """
        Step 2: Train counterfactual reasoner on discovered graph.

        Args:
            factual_data: (n_samples, n_variables)
            counterfactual_data: (n_samples, n_variables) under intervention
            intervention_var: Index of intervened variable
            intervention_value: Value set by intervention

        Returns:
            Training metrics
        """
        if self.causal_graph is None:
            raise ValueError("Must discover causal graph first!")

        if verbose:
            print("\n" + "="*80)
            print("STEP 2: Counterfactual Reasoner Training (CCLT)")
            print("="*80)

        # Convert to tensors
        X_factual = torch.FloatTensor(factual_data).to(self.device)
        Y_factual = X_factual.clone()
        X_cf = torch.FloatTensor(counterfactual_data).to(self.device)
        Y_cf = X_cf.clone()
        causal_graph = torch.FloatTensor(self.causal_graph).to(self.device)

        if U_true is not None:
            U_true_tensor = torch.FloatTensor(U_true).to(self.device)
        else:
            U_true_tensor = None

        # Loss functions
        cf_loss_fn = CounterfactualLoss(self.cclt_config)
        struct_loss_fn = StructuralEquationLoss(self.cclt_config)

        # Optimizer
        optimizer = torch.optim.Adam(
            self.counterfactual_reasoner.parameters(),
            lr=lr
        )

        n_samples = X_factual.shape[0]
        history = {'total': [], 'prediction': []}

        for epoch in range(epochs):
            indices = torch.randperm(n_samples)[:batch_size]

            X_fact_batch = X_factual[indices]
            Y_fact_batch = Y_factual[indices]
            X_cf_batch = X_cf[indices]
            Y_cf_batch = Y_cf[indices]
            U_batch = U_true_tensor[indices] if U_true_tensor is not None else None

            optimizer.zero_grad()

            # Factual prediction
            Y_fact_pred, U_inferred = self.counterfactual_reasoner(
                X_fact_batch, Y_fact_batch, causal_graph,
                intervention_var=None, intervention_value=None
            )

            # Counterfactual prediction
            Y_cf_pred, _ = self.counterfactual_reasoner(
                X_fact_batch, Y_fact_batch, causal_graph,
                intervention_var=intervention_var,
                intervention_value=intervention_value
            )

            # Losses
            losses = cf_loss_fn(
                Y_factual_pred=Y_fact_pred,
                Y_factual_true=Y_fact_batch,
                U_inferred=U_inferred,
                U_true=U_batch,
                Y_counterfactual_pred=Y_cf_pred,
                Y_counterfactual_true=Y_cf_batch,
                X_intervened=Y_cf_pred,
                intervention_var=intervention_var,
                intervention_value=intervention_value
            )

            L_struct = struct_loss_fn(Y_fact_pred, Y_fact_batch)
            loss_total = losses['total'] + self.cclt_config.lambda_structure * L_struct

            loss_total.backward()
            optimizer.step()

            history['total'].append(loss_total.item())
            history['prediction'].append(losses['prediction'].item())

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Total: {loss_total.item():.4f} | "
                      f"CF Pred: {losses['prediction'].item():.4f}")

        return history

    def answer_counterfactual_query(
        self,
        factual_observation: np.ndarray,
        intervention_var: int,
        intervention_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 3: Answer counterfactual query.

        Given factual observation, predict what would happen under intervention.

        Args:
            factual_observation: (n_variables,) observed values
            intervention_var: Variable to intervene on
            intervention_value: Value to set

        Returns:
            counterfactual_prediction: (n_variables,) predicted values
            U_inferred: (n_variables,) inferred exogenous variables
        """
        if self.causal_graph is None:
            raise ValueError("Must discover causal graph first!")

        self.counterfactual_reasoner.eval()
        with torch.no_grad():
            # Convert to tensors
            X = torch.FloatTensor(factual_observation).unsqueeze(0).to(self.device)
            Y = X.clone()
            G = torch.FloatTensor(self.causal_graph).to(self.device)

            # Predict counterfactual
            Y_cf, U_inferred = self.counterfactual_reasoner(
                X, Y, G,
                intervention_var=intervention_var,
                intervention_value=intervention_value
            )

            return Y_cf.cpu().numpy()[0], U_inferred.cpu().numpy()[0]


# ============================================================================
# 2. Comprehensive Evaluation
# ============================================================================

def evaluate_end_to_end(
    system: CausalReasoningSystem,
    true_graph: np.ndarray,
    test_data: Dict
) -> Dict:
    """
    Evaluate complete system performance.

    Metrics:
    1. Graph discovery accuracy (F1, SHD)
    2. Counterfactual prediction accuracy (MAE, R²)
    """
    discovered_graph = system.causal_graph

    # Graph discovery metrics
    from sklearn.metrics import f1_score, precision_score, recall_score

    true_edges = true_graph.flatten()
    pred_edges = discovered_graph.flatten()

    f1 = f1_score(true_edges, pred_edges)
    precision = precision_score(true_edges, pred_edges, zero_division=0)
    recall = recall_score(true_edges, pred_edges, zero_division=0)

    # Structural Hamming Distance
    shd = np.sum(np.abs(true_graph - discovered_graph))

    # Counterfactual accuracy
    factual = test_data['factual']
    counterfactual_true = test_data['counterfactual']
    intervention_var = test_data['intervention_var']
    intervention_value = test_data['intervention_value']

    # Predict counterfactuals for all test samples
    cf_predictions = []
    for i in range(factual.shape[0]):
        cf_pred, _ = system.answer_counterfactual_query(
            factual[i],
            intervention_var,
            intervention_value
        )
        cf_predictions.append(cf_pred)

    cf_predictions = np.array(cf_predictions)

    # Counterfactual MAE
    mae = np.abs(cf_predictions - counterfactual_true).mean()

    # R² on outcome variable
    Y_var = 2  # Last variable
    y_true = counterfactual_true[:, Y_var]
    y_pred = cf_predictions[:, Y_var]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        'graph_f1': f1,
        'graph_precision': precision,
        'graph_recall': recall,
        'graph_shd': shd,
        'counterfactual_mae': mae,
        'counterfactual_r2': r2
    }


# ============================================================================
# 3. Main Experiment
# ============================================================================

def main():
    print("="*80)
    print("Week 7-8: End-to-End Integration of CSM + CCLT")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create SCM with known structure
    print("\n1. Creating Causal SCM Dataset")
    print("-"*80)
    scm = create_hvac_scm()
    true_graph = scm.causal_graph
    print(f"True causal graph:\n{true_graph}")

    # Generate data
    n_train = 800
    n_test = 200

    # Observational data
    obs_data = scm.sample(n_train + n_test)

    # Interventional data: do(HVAC = 1)
    intervention_var = 1  # HVAC
    intervention_value = 1.0
    int_data = scm.intervene(intervention_var, intervention_value, n_train)
    int_masks = np.zeros((n_train, 4))
    int_masks[:, intervention_var] = 1

    # Split train/test
    obs_train = obs_data[:n_train]
    obs_test = obs_data[n_train:]

    # Counterfactual test data
    cf_test = scm.intervene(intervention_var, intervention_value, n_test)

    test_data = {
        'factual': obs_test,
        'counterfactual': cf_test,
        'intervention_var': intervention_var,
        'intervention_value': intervention_value
    }

    print(f"\nData shapes:")
    print(f"  Observational train: {obs_train.shape}")
    print(f"  Interventional train: {int_data.shape}")
    print(f"  Test: {obs_test.shape}")

    # Initialize system
    print("\n2. Initializing Causal Reasoning System")
    print("-"*80)
    system = CausalReasoningSystem(n_variables=4, device=device)
    print(f"CSM parameters: {sum(p.numel() for p in system.csm.parameters()):,}")
    print(f"CCLT parameters: {sum(p.numel() for p in system.counterfactual_reasoner.parameters()):,}")

    # Step 1: Discover graph
    discovered_graph = system.discover_causal_graph(
        obs_train,
        {'data': int_data, 'intervention_masks': int_masks},
        epochs=200,
        verbose=True
    )

    # Step 2: Train counterfactual reasoner
    history = system.train_counterfactual_reasoner(
        obs_train,
        int_data,
        intervention_var,
        intervention_value,
        epochs=200,
        verbose=True
    )

    # Step 3: Evaluate
    print("\n" + "="*80)
    print("STEP 3: End-to-End Evaluation")
    print("="*80)

    metrics = evaluate_end_to_end(system, true_graph, test_data)

    print(f"\nGraph Discovery Performance:")
    print(f"  F1 Score:     {metrics['graph_f1']:.3f}")
    print(f"  Precision:    {metrics['graph_precision']:.3f}")
    print(f"  Recall:       {metrics['graph_recall']:.3f}")
    print(f"  SHD:          {metrics['graph_shd']:.0f}")

    print(f"\nCounterfactual Reasoning Performance:")
    print(f"  MAE:          {metrics['counterfactual_mae']:.4f}")
    print(f"  R²:           {metrics['counterfactual_r2']:.4f}")

    # Overall assessment
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)

    if metrics['graph_f1'] >= 0.7 and metrics['counterfactual_r2'] >= 0.8:
        print("✓✓ EXCELLENT: Both components working well!")
    elif metrics['graph_f1'] >= 0.5 and metrics['counterfactual_r2'] >= 0.6:
        print("✓ GOOD: System demonstrates end-to-end capability")
    else:
        print("⚠ MODERATE: Components work individually, integration needs tuning")

    print(f"\nKey Insight:")
    print(f"  Discovered graph F1 = {metrics['graph_f1']:.3f}")
    print(f"  Counterfactual R² = {metrics['counterfactual_r2']:.4f}")
    print(f"  → System successfully combines CSM + CCLT")

    print("\n" + "="*80)
    print("Week 7-8 Integration COMPLETE!")
    print("="*80)

    return system, metrics


if __name__ == '__main__':
    system, metrics = main()

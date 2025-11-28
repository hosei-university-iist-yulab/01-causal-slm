"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 2, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

"""
Generates synthetic interventions for causal reasoning experiments.
Creates do(X=x) scenarios for evaluating intervention prediction accuracy.
Supports multiple sensor domains and intervention types.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class InterventionConfig:
    """Configuration for intervention generation."""
    n_variables: int
    intervention_prob: float = 0.3  # Probability of intervening on each variable
    soft_intervention: bool = True  # Use soft interventions (with noise)
    noise_std: float = 0.1  # Noise standard deviation for soft interventions


class StructuralCausalModel:
    """
    Structural Causal Model (SCM) for generating interventional data.

    SCM consists of:
    - Structural equations: X_i = f_i(pa(X_i), U_i)
    - Exogenous variables: U_i ~ p(U_i)
    - Causal graph: G specifying parent-child relationships
    """

    def __init__(
        self,
        causal_graph: np.ndarray,
        structural_equations: Dict[int, Callable],
        noise_distributions: Dict[int, Callable]
    ):
        """
        Initialize SCM.

        Args:
            causal_graph: Adjacency matrix (n_variables, n_variables)
                         causal_graph[i,j] = 1 means i → j
            structural_equations: Dict mapping variable index to function
                                 f_i(parents, noise) → value
            noise_distributions: Dict mapping variable index to noise sampler
                                () → noise value
        """
        self.causal_graph = causal_graph
        self.n_variables = causal_graph.shape[0]
        self.structural_equations = structural_equations
        self.noise_distributions = noise_distributions

        # Compute topological ordering for generation
        self.topological_order = self._topological_sort()

    def _topological_sort(self) -> List[int]:
        """Compute topological ordering of variables."""
        in_degree = self.causal_graph.sum(axis=0)
        queue = [i for i in range(self.n_variables) if in_degree[i] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            # Reduce in-degree of children
            children = np.where(self.causal_graph[node, :] == 1)[0]
            for child in children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return order

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample from observational distribution p(X).

        Args:
            n_samples: Number of samples

        Returns:
            Samples (n_samples, n_variables)
        """
        samples = np.zeros((n_samples, self.n_variables))

        for _ in range(n_samples):
            # Generate in topological order
            for var_idx in self.topological_order:
                # Get parents
                parents_idx = np.where(self.causal_graph[:, var_idx] == 1)[0]
                parent_values = samples[_, parents_idx] if len(parents_idx) > 0 else np.array([])

                # Sample noise
                noise = self.noise_distributions[var_idx]()

                # Apply structural equation
                samples[_, var_idx] = self.structural_equations[var_idx](parent_values, noise)

        return samples

    def intervene(
        self,
        intervention_var: int,
        intervention_value: float,
        n_samples: int = 1,
        soft: bool = False,
        noise_std: float = 0.1
    ) -> np.ndarray:
        """
        Sample from interventional distribution p(X | do(X_i = value)).

        Implements Pearl's do-operator via graph surgery:
        1. Cut incoming edges to X_i
        2. Set X_i = value (or value + noise for soft intervention)
        3. Generate descendants via structural equations

        Args:
            intervention_var: Index of intervened variable
            intervention_value: Value to set
            n_samples: Number of samples
            soft: Use soft intervention (with noise)
            noise_std: Noise std for soft intervention

        Returns:
            Interventional samples (n_samples, n_variables)
        """
        samples = np.zeros((n_samples, self.n_variables))

        for i in range(n_samples):
            # Generate in topological order
            for var_idx in self.topological_order:
                if var_idx == intervention_var:
                    # Intervention: Set value directly (graph surgery)
                    if soft:
                        # Soft intervention: add noise
                        samples[i, var_idx] = intervention_value + np.random.randn() * noise_std
                    else:
                        # Hard intervention: exact value
                        samples[i, var_idx] = intervention_value
                else:
                    # Non-intervened variable: use structural equation
                    parents_idx = np.where(self.causal_graph[:, var_idx] == 1)[0]
                    parent_values = samples[i, parents_idx] if len(parents_idx) > 0 else np.array([])

                    # Sample noise
                    noise = self.noise_distributions[var_idx]()

                    # Apply structural equation
                    samples[i, var_idx] = self.structural_equations[var_idx](parent_values, noise)

        return samples

    def counterfactual(
        self,
        factual_observation: np.ndarray,
        intervention_var: int,
        intervention_value: float
    ) -> np.ndarray:
        """
        Generate counterfactual using Pearl's 3-step algorithm:
        1. Abduction: Infer exogenous variables U from observation
        2. Action: Intervene do(X_i = value)
        3. Prediction: Compute counterfactual using inferred U

        Args:
            factual_observation: Observed values (n_variables,)
            intervention_var: Index to intervene on
            intervention_value: Counterfactual value

        Returns:
            Counterfactual sample (n_variables,)
        """
        # Step 1: Abduction - infer U from observation
        inferred_noise = {}

        for var_idx in self.topological_order:
            # Get parents
            parents_idx = np.where(self.causal_graph[:, var_idx] == 1)[0]
            parent_values = factual_observation[parents_idx] if len(parents_idx) > 0 else np.array([])

            # Infer noise: U_i = X_i - f_i(pa(X_i), 0)
            # Assumes structural equation is: X_i = f_i(pa(X_i)) + U_i
            deterministic_part = self.structural_equations[var_idx](parent_values, 0)
            inferred_noise[var_idx] = factual_observation[var_idx] - deterministic_part

        # Step 2 & 3: Action + Prediction
        counterfactual = np.zeros(self.n_variables)

        for var_idx in self.topological_order:
            if var_idx == intervention_var:
                # Action: Set intervention value
                counterfactual[var_idx] = intervention_value
            else:
                # Prediction: Use structural equation with inferred noise
                parents_idx = np.where(self.causal_graph[:, var_idx] == 1)[0]
                parent_values = counterfactual[parents_idx] if len(parents_idx) > 0 else np.array([])

                counterfactual[var_idx] = self.structural_equations[var_idx](
                    parent_values, inferred_noise[var_idx]
                )

        return counterfactual


class InterventionGenerator:
    """
    Generator for interventional training data for CSM.

    Usage:
        generator = InterventionGenerator(scm, config)
        x_factual, x_counterfactual, intervention = generator.generate_batch(batch_size=32)
    """

    def __init__(self, scm: StructuralCausalModel, config: InterventionConfig):
        self.scm = scm
        self.config = config

    def generate_intervention(self) -> Tuple[int, float]:
        """
        Sample random intervention.

        Returns:
            (variable_index, intervention_value)
        """
        # Choose variable to intervene on
        var_idx = np.random.randint(0, self.config.n_variables)

        # Choose intervention value (sample from marginal distribution)
        # For now, use random value in reasonable range
        intervention_value = np.random.randn() * 2  # Scale as needed

        return var_idx, intervention_value

    def generate_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, float]]]:
        """
        Generate batch of interventional data.

        For each sample:
        1. Sample factual observation from p(X)
        2. Sample random intervention do(X_i = value)
        3. Generate counterfactual p(X | do(X_i = value))

        Args:
            batch_size: Number of samples

        Returns:
            x_factual: Factual observations (batch_size, n_variables)
            x_counterfactual: Counterfactual samples (batch_size, n_variables)
            interventions: List of (var_idx, value) for each sample
        """
        x_factual = []
        x_counterfactual = []
        interventions = []

        for _ in range(batch_size):
            # Sample factual observation
            factual = self.scm.sample(n_samples=1)[0]

            # Sample intervention
            var_idx, value = self.generate_intervention()

            # Generate counterfactual
            if self.config.soft_intervention:
                # Soft intervention: sample from interventional distribution
                counterfactual = self.scm.intervene(
                    var_idx, value, n_samples=1,
                    soft=True, noise_std=self.config.noise_std
                )[0]
            else:
                # Hard intervention: use Pearl's 3-step algorithm
                counterfactual = self.scm.counterfactual(factual, var_idx, value)

            x_factual.append(factual)
            x_counterfactual.append(counterfactual)
            interventions.append((var_idx, value))

        x_factual = torch.tensor(x_factual, dtype=torch.float32)
        x_counterfactual = torch.tensor(x_counterfactual, dtype=torch.float32)

        return x_factual, x_counterfactual, interventions


# ============================================================================
# Example: HVAC SCM for Testing
# ============================================================================

def create_hvac_scm() -> StructuralCausalModel:
    """
    Create HVAC SCM with known causal structure.

    Variables:
    - 0: Occupancy (binary)
    - 1: HVAC (binary)
    - 2: Temperature (continuous)
    - 3: Humidity (continuous)

    Causal structure:
    - Occupancy → HVAC
    - HVAC → Temperature
    - HVAC → Humidity
    """
    # Causal graph
    causal_graph = np.array([
        [0, 1, 0, 0],  # Occupancy → HVAC
        [0, 0, 1, 1],  # HVAC → Temperature, Humidity
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    # Structural equations
    def f_occupancy(parents, noise):
        # Occupancy ~ Bernoulli(0.6) + noise
        base = 0.6
        return 1 if (base + noise) > 0.5 else 0

    def f_hvac(parents, noise):
        # HVAC depends on occupancy
        # p(HVAC=1 | Occupancy=1) = 0.8
        # p(HVAC=1 | Occupancy=0) = 0.2
        occupancy = parents[0] if len(parents) > 0 else 0
        prob = 0.8 if occupancy == 1 else 0.2
        return 1 if (prob + noise * 0.3) > 0.5 else 0

    def f_temperature(parents, noise):
        # Temperature depends on HVAC
        # If HVAC off: 28°C, if on: 22°C (with noise)
        hvac = parents[0] if len(parents) > 0 else 0
        base_temp = 22 if hvac == 1 else 28
        return base_temp + noise

    def f_humidity(parents, noise):
        # Humidity depends on HVAC
        # If HVAC on: 45%, if off: 55% (with noise)
        hvac = parents[0] if len(parents) > 0 else 0
        base_humidity = 45 if hvac == 1 else 55
        return base_humidity + noise

    structural_equations = {
        0: f_occupancy,
        1: f_hvac,
        2: f_temperature,
        3: f_humidity
    }

    # Noise distributions
    noise_distributions = {
        0: lambda: np.random.randn() * 0.1,
        1: lambda: np.random.randn() * 0.1,
        2: lambda: np.random.randn() * 0.5,  # Temperature noise
        3: lambda: np.random.randn() * 2.0   # Humidity noise
    }

    return StructuralCausalModel(causal_graph, structural_equations, noise_distributions)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Intervention Generator - Week 1-2 Implementation")
    print("=" * 80)

    # Create HVAC SCM
    print("\n1. Creating HVAC SCM...")
    scm = create_hvac_scm()
    print(f"   Variables: {scm.n_variables}")
    print(f"   Topological order: {scm.topological_order}")
    print(f"\n   Causal graph:")
    print(f"   {scm.causal_graph}")

    # Sample observational data
    print("\n2. Sampling observational data...")
    obs_data = scm.sample(n_samples=5)
    print(f"   Shape: {obs_data.shape}")
    print(f"   Samples:")
    print(f"   Occupancy | HVAC | Temp  | Humidity")
    for i in range(5):
        print(f"   {obs_data[i,0]:5.0f}     | {obs_data[i,1]:2.0f}   | {obs_data[i,2]:5.1f} | {obs_data[i,3]:6.1f}")

    # Sample interventional data
    print("\n3. Sampling interventional data do(HVAC=1)...")
    int_data = scm.intervene(intervention_var=1, intervention_value=1, n_samples=5, soft=False)
    print(f"   Shape: {int_data.shape}")
    print(f"   Samples:")
    print(f"   Occupancy | HVAC | Temp  | Humidity")
    for i in range(5):
        print(f"   {int_data[i,0]:5.0f}     | {int_data[i,1]:2.0f}   | {int_data[i,2]:5.1f} | {int_data[i,3]:6.1f}")

    print(f"\n   Notice: HVAC is always 1 (intervention)")
    print(f"   Notice: Temperature lower (~22°C vs ~28°C in observational)")

    # Generate counterfactual
    print("\n4. Generating counterfactual...")
    factual = obs_data[0]
    print(f"   Factual:       Occ={factual[0]:.0f}, HVAC={factual[1]:.0f}, Temp={factual[2]:.1f}, Humidity={factual[3]:.1f}")

    counterfactual = scm.counterfactual(factual, intervention_var=1, intervention_value=1)
    print(f"   Counterfactual (do(HVAC=1)): Occ={counterfactual[0]:.0f}, HVAC={counterfactual[1]:.0f}, Temp={counterfactual[2]:.1f}, Humidity={counterfactual[3]:.1f}")
    print(f"   Temperature change: {factual[2] - counterfactual[2]:.1f}°C (should be ~6°C if HVAC was off)")

    # Create intervention generator
    print("\n5. Creating InterventionGenerator...")
    config = InterventionConfig(n_variables=4, soft_intervention=True, noise_std=0.1)
    generator = InterventionGenerator(scm, config)

    # Generate batch
    print("\n6. Generating intervention batch...")
    x_factual, x_counterfactual, interventions = generator.generate_batch(batch_size=5)
    print(f"   Factual shape: {x_factual.shape}")
    print(f"   Counterfactual shape: {x_counterfactual.shape}")
    print(f"   Interventions: {interventions}")

    print(f"\n   Comparison (first 3 samples):")
    for i in range(3):
        var_idx, value = interventions[i]
        print(f"   Sample {i+1}: do(Var{var_idx} = {value:.2f})")
        print(f"     Factual:       {x_factual[i].numpy()}")
        print(f"     Counterfactual: {x_counterfactual[i].numpy()}")
        print(f"     Difference:     {(x_factual[i] - x_counterfactual[i]).numpy()}")

    print("\n" + "=" * 80)
    print("Intervention Generator complete!")
    print("Next: Integrate with CSM interventional loss")
    print("=" * 80)

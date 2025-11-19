"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: October 30, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

"""
Implementation of Pearl's backdoor criterion for causal inference.
Identifies valid adjustment sets for estimating causal effects.
Used by BAT to construct attention masks.
"""

import numpy as np
from typing import Set, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BackdoorResult:
    """Result of backdoor criterion search."""
    exists: bool  # Does valid backdoor set exist?
    backdoor_set: Set[int]  # Minimal backdoor set (empty if none exists)
    all_backdoor_paths: List[List[int]]  # All backdoor paths X ← Z → Y


class CausalGraph:
    """
    Causal graph representation with path finding algorithms.

    Supports:
    - Directed paths
    - Backdoor paths (contains →)
    - Descendants computation
    - Backdoor criterion checking
    """

    def __init__(self, adjacency_matrix: np.ndarray):
        """
        Initialize causal graph.

        Args:
            adjacency_matrix: Binary adjacency matrix (n, n)
                             adj[i, j] = 1 means i → j
        """
        self.adj = adjacency_matrix.copy()
        self.n = adjacency_matrix.shape[0]

    def parents(self, node: int) -> Set[int]:
        """Get parents of node (incoming edges)."""
        return set(np.where(self.adj[:, node] == 1)[0])

    def children(self, node: int) -> Set[int]:
        """Get children of node (outgoing edges)."""
        return set(np.where(self.adj[node, :] == 1)[0])

    def descendants(self, node: int) -> Set[int]:
        """
        Get all descendants of node (reachable via directed paths).

        Uses BFS to find all reachable nodes.
        """
        desc = set()
        queue = [node]
        visited = {node}

        while queue:
            current = queue.pop(0)
            for child in self.children(current):
                if child not in visited:
                    visited.add(child)
                    desc.add(child)
                    queue.append(child)

        return desc

    def has_directed_path(self, source: int, target: int) -> bool:
        """Check if directed path exists from source to target."""
        if source == target:
            return True

        visited = {source}
        queue = [source]

        while queue:
            current = queue.pop(0)
            if current == target:
                return True

            for child in self.children(current):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        return False

    def find_backdoor_paths(self, X: int, Y: int) -> List[List[int]]:
        """
        Find all backdoor paths from X to Y.

        Backdoor path: Contains at least one arrow INTO X
        Example: X ← Z → Y or X ← Z → W → Y

        Args:
            X: Source node
            Y: Target node

        Returns:
            List of paths (each path is list of nodes)
        """
        backdoor_paths = []

        # Start from parents of X (arrows into X)
        for parent in self.parents(X):
            # DFS from parent to Y, avoiding X
            self._dfs_backdoor_paths(
                current=parent,
                target=Y,
                avoid={X},
                path=[X, parent],
                backdoor_paths=backdoor_paths
            )

        return backdoor_paths

    def _dfs_backdoor_paths(
        self,
        current: int,
        target: int,
        avoid: Set[int],
        path: List[int],
        backdoor_paths: List[List[int]]
    ):
        """DFS helper for finding backdoor paths."""
        if current == target:
            backdoor_paths.append(path.copy())
            return

        # Try going to children
        for child in self.children(current):
            if child not in avoid and child not in path:
                path.append(child)
                self._dfs_backdoor_paths(current=child, target=target, avoid=avoid, path=path, backdoor_paths=backdoor_paths)
                path.pop()

        # Try going to parents (backdoor continues)
        for parent in self.parents(current):
            if parent not in avoid and parent not in path:
                path.append(parent)
                self._dfs_backdoor_paths(current=parent, target=target, avoid=avoid, path=path, backdoor_paths=backdoor_paths)
                path.pop()


def check_backdoor_criterion(
    graph: CausalGraph,
    X: int,
    Y: int,
    Z: Set[int]
) -> bool:
    """
    Check if Z satisfies backdoor criterion for (X, Y).

    Backdoor Criterion (Pearl, 2009):
    Z is valid if:
    1. Z blocks all backdoor paths from X to Y
    2. No node in Z is a descendant of X

    Args:
        graph: Causal graph
        X: Treatment variable
        Y: Outcome variable
        Z: Proposed adjustment set

    Returns:
        True if Z satisfies backdoor criterion
    """
    # Check condition 2: No node in Z is descendant of X
    descendants_X = graph.descendants(X)
    if Z & descendants_X:  # Intersection not empty
        return False

    # Check condition 1: Z blocks all backdoor paths
    backdoor_paths = graph.find_backdoor_paths(X, Y)

    for path in backdoor_paths:
        # Check if this path is blocked by Z
        path_blocked = False

        # Path is blocked if any non-endpoint node is in Z
        for node in path[1:-1]:  # Exclude endpoints X and Y
            if node in Z:
                path_blocked = True
                break

        if not path_blocked:
            return False  # Found unblocked backdoor path

    return True


def find_minimal_backdoor_set(
    graph: CausalGraph,
    X: int,
    Y: int
) -> BackdoorResult:
    """
    Find minimal backdoor adjustment set for (X, Y).

    Algorithm (greedy search):
    1. Find all backdoor paths
    2. If none, return empty set (no confounding)
    3. Otherwise, greedily add nodes that block most paths
    4. Stop when all paths blocked

    Complexity: O(n * |paths|) = O(n²) for sparse graphs

    Args:
        graph: Causal graph
        X: Treatment variable
        Y: Outcome variable

    Returns:
        BackdoorResult with minimal backdoor set (or empty if none exists)
    """
    # Find all backdoor paths
    backdoor_paths = graph.find_backdoor_paths(X, Y)

    # No backdoor paths = no confounding
    if len(backdoor_paths) == 0:
        return BackdoorResult(
            exists=True,
            backdoor_set=set(),
            all_backdoor_paths=[]
        )

    # Get descendants of X (cannot be in backdoor set)
    descendants_X = graph.descendants(X)

    # Candidate nodes: all nodes except X, Y, descendants(X)
    candidates = set(range(graph.n)) - {X, Y} - descendants_X

    # Greedy search: add node that blocks most paths
    Z = set()
    remaining_paths = backdoor_paths.copy()

    while remaining_paths and candidates:
        # Count how many paths each candidate blocks
        best_node = None
        best_count = 0

        for node in candidates:
            count = sum(1 for path in remaining_paths if node in path[1:-1])
            if count > best_count:
                best_count = count
                best_node = node

        if best_count == 0:
            # No candidate blocks any remaining path
            break

        # Add best node
        Z.add(best_node)
        candidates.remove(best_node)

        # Remove blocked paths
        remaining_paths = [
            path for path in remaining_paths
            if not any(node in Z for node in path[1:-1])
        ]

    # Check if all paths blocked
    if remaining_paths:
        # Failed to find valid backdoor set
        return BackdoorResult(
            exists=False,
            backdoor_set=set(),
            all_backdoor_paths=backdoor_paths
        )

    # Verify with formal check
    assert check_backdoor_criterion(graph, X, Y, Z), "Backdoor set verification failed"

    return BackdoorResult(
        exists=True,
        backdoor_set=Z,
        all_backdoor_paths=backdoor_paths
    )


def compute_backdoor_matrix(graph: CausalGraph) -> np.ndarray:
    """
    Compute backdoor adjustment matrix for all pairs.

    Matrix[i, j] = |Z_ij| where Z_ij is minimal backdoor set for (i, j)
                 = -1 if no valid backdoor set exists

    Used in Formula 4:
        M_backdoor[i,j] = exp(-λ * |Z_ij|) if Z_ij exists, else 0

    Args:
        graph: Causal graph

    Returns:
        Matrix (n, n) with backdoor set sizes
    """
    n = graph.n
    backdoor_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                backdoor_matrix[i, j] = 0  # Self-loops: no adjustment needed
                continue

            result = find_minimal_backdoor_set(graph, i, j)

            if result.exists:
                backdoor_matrix[i, j] = len(result.backdoor_set)
            else:
                backdoor_matrix[i, j] = -1  # No valid set exists

    return backdoor_matrix


# ============================================================================
# Example Usage & Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Backdoor Criterion Search - Week 3-4 Implementation")
    print("=" * 80)

    # Example 1: Simple confounding
    print("\n1. Example: Simple Confounding")
    print("   Graph: Z → X, Z → Y (Z confounds X-Y relationship)")

    adj = np.array([
        [0, 1, 1],  # Z → X, Y
        [0, 0, 0],  # X → (none)
        [0, 0, 0]   # Y → (none)
    ])

    graph = CausalGraph(adj)
    print(f"   Adjacency matrix:")
    print(f"   {adj}")

    # Find backdoor set for X → Y
    result = find_minimal_backdoor_set(graph, X=1, Y=2)
    print(f"\n   Backdoor criterion for X=1, Y=2:")
    print(f"   - Valid backdoor set exists: {result.exists}")
    print(f"   - Minimal backdoor set: {result.backdoor_set}")
    print(f"   - Backdoor paths: {result.all_backdoor_paths}")
    print(f"   - Adjustment: Control for Z={result.backdoor_set}")

    # Example 2: HVAC system
    print("\n" + "-" * 80)
    print("2. Example: HVAC System")
    print("   Graph: Occupancy → HVAC → Temperature, Humidity")

    adj_hvac = np.array([
        [0, 1, 0, 0],  # Occupancy → HVAC
        [0, 0, 1, 1],  # HVAC → Temperature, Humidity
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    graph_hvac = CausalGraph(adj_hvac)
    print(f"   Adjacency matrix:")
    print(f"   {adj_hvac}")

    # Check several pairs
    pairs = [(1, 2), (1, 3), (0, 2), (0, 3)]
    pair_names = [("HVAC", "Temp"), ("HVAC", "Humidity"), ("Occupancy", "Temp"), ("Occupancy", "Humidity")]

    for (X, Y), (X_name, Y_name) in zip(pairs, pair_names):
        result = find_minimal_backdoor_set(graph_hvac, X, Y)
        print(f"\n   {X_name} → {Y_name}:")
        print(f"   - Backdoor set: {result.backdoor_set}")
        print(f"   - Backdoor paths: {len(result.all_backdoor_paths)}")

    # Example 3: Backdoor matrix computation
    print("\n" + "-" * 80)
    print("3. Computing Backdoor Matrix for HVAC")

    backdoor_matrix = compute_backdoor_matrix(graph_hvac)
    print(f"   Backdoor matrix (|Z_ij| for each pair):")
    print(f"   {backdoor_matrix}")
    print(f"\n   Interpretation:")
    print(f"   - 0: No adjustment needed (direct or no path)")
    print(f"   - k>0: Need to adjust for k variables")
    print(f"   - -1: No valid backdoor set exists")

    # Example 4: Complex confounding
    print("\n" + "-" * 80)
    print("4. Example: Multiple Confounders")
    print("   Graph: Z1 → X, Z1 → Y, Z2 → X, Z2 → Y")

    adj_complex = np.array([
        [0, 0, 1, 1],  # Z1 → X, Y
        [0, 0, 1, 1],  # Z2 → X, Y
        [0, 0, 0, 0],  # X → (none)
        [0, 0, 0, 0]   # Y → (none)
    ])

    graph_complex = CausalGraph(adj_complex)
    result = find_minimal_backdoor_set(graph_complex, X=2, Y=3)
    print(f"   Minimal backdoor set: {result.backdoor_set}")
    print(f"   Need to adjust for: {len(result.backdoor_set)} confounders")

    print("\n" + "=" * 80)
    print("Backdoor Criterion Search COMPLETE!")
    print("Next: Implement Formula 4 (backdoor attention mask)")
    print("=" * 80)

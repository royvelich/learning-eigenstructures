"""
Standalone script to test manifold-aware Hadamard-like probe functions on 3D meshes.

This script demonstrates generating probe functions that are "smooth" in the Hadamard sense:
- Piecewise constant on hierarchical partitions of the manifold
- Partitions respect the manifold structure via kNN graph

Usage:
    python hadamard_probe_test.py --mesh_path /path/to/mesh.obj
    python hadamard_probe_test.py --mesh_path /path/to/mesh.ply --n_probes 10 --max_depth 5

If no mesh_path is provided, a default mesh will be used if available.
"""

import argparse
import numpy as np
import polyscope as ps
from scipy.spatial import cKDTree
from typing import Tuple, Optional, List
import os


# ============================================================================
# Probe Function Generation
# ============================================================================

def build_knn_graph(points: np.ndarray, k: int = 15) -> np.ndarray:
    """
    Build k-nearest neighbor graph.

    Args:
        points: [n_points, 3] array of 3D coordinates
        k: number of neighbors

    Returns:
        edge_index: [2, n_edges] array of edges
    """
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k + 1)  # +1 because query includes self

    n_points = len(points)
    sources = np.repeat(np.arange(n_points), k)
    targets = indices[:, 1:].flatten()  # Exclude self-loops

    edge_index = np.stack([sources, targets], axis=0)
    return edge_index


def get_subgraph_edges(edge_index: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Extract edges that are within a subset of vertices.

    Args:
        edge_index: [2, n_edges] full graph edges
        indices: vertex indices of the subgraph

    Returns:
        Filtered edge_index containing only edges within the subset
    """
    index_set = set(indices.tolist())
    mask = np.array([
        (edge_index[0, i] in index_set) and (edge_index[1, i] in index_set)
        for i in range(edge_index.shape[1])
    ])
    return edge_index[:, mask]


def graph_aware_bisection(
    points: np.ndarray,
    edge_index: np.ndarray,
    indices: np.ndarray,
    method: str = "spatial"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bisect a subset of points using manifold-aware method.

    Args:
        points: All points [n_points, dim]
        edge_index: Graph edges [2, n_edges]
        indices: Indices of points to bisect
        method: "spatial" (random hyperplane) or "spectral" (Fiedler vector)

    Returns:
        left_indices, right_indices: Two partitions
    """
    if len(indices) < 2:
        return indices, np.array([], dtype=indices.dtype)

    subset_points = points[indices]

    if method == "spatial":
        # Random hyperplane through centroid
        centroid = subset_points.mean(axis=0)

        # Random direction
        direction = np.random.rand(points.shape[1])
        direction /= np.linalg.norm(direction) + 1e-10

        # Project and split at median
        projections = (subset_points - centroid) @ direction
        median_proj = np.median(projections)

        left_mask = projections <= median_proj

    elif method == "spectral":
        # Use Fiedler vector (second eigenvector of graph Laplacian)
        # This gives the optimal graph cut
        sub_edge_index = get_subgraph_edges(edge_index, indices)

        if sub_edge_index.shape[1] == 0:
            # No edges in subgraph, fall back to spatial
            return graph_aware_bisection(points, edge_index, indices, method="spatial")

        # Build Laplacian for subgraph
        n_sub = len(indices)
        index_map = {idx: i for i, idx in enumerate(indices)}

        # Adjacency matrix
        rows = [index_map[sub_edge_index[0, i]] for i in range(sub_edge_index.shape[1])]
        cols = [index_map[sub_edge_index[1, i]] for i in range(sub_edge_index.shape[1])]

        from scipy.sparse import csr_matrix
        A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_sub, n_sub))
        A = A + A.T  # Symmetrize
        A.data = np.clip(A.data, 0, 1)  # Remove duplicates

        # Degree matrix
        degrees = np.array(A.sum(axis=1)).flatten()
        D = csr_matrix((degrees, (range(n_sub), range(n_sub))), shape=(n_sub, n_sub))

        # Laplacian
        L = D - A

        # Get Fiedler vector (second smallest eigenvector)
        from scipy.sparse.linalg import eigsh
        try:
            eigenvalues, eigenvectors = eigsh(L.astype(float), k=2, which='SM')
            fiedler = eigenvectors[:, 1]
            left_mask = fiedler <= 0
        except:
            # Fall back to spatial if eigendecomposition fails
            return graph_aware_bisection(points, edge_index, indices, method="spatial")

    else:
        raise ValueError(f"Unknown bisection method: {method}")

    left_indices = indices[left_mask]
    right_indices = indices[~left_mask]

    return left_indices, right_indices


def generate_single_hadamard_probe(
    points: np.ndarray,
    edge_index: np.ndarray,
    max_depth: int = 6,
    split_probability: float = 0.7,
    bisection_method: str = "spatial",
    use_random_values: bool = True
) -> np.ndarray:
    """
    Generate a single Hadamard-like probe function via recursive bisection.

    The key idea: Hadamard functions are piecewise constant on hierarchical partitions.
    We create partitions that respect the manifold structure using the kNN graph.

    Args:
        points: [n_points, dim] point cloud
        edge_index: [2, n_edges] kNN graph edges
        max_depth: Maximum recursion depth (controls number of pieces)
        split_probability: Probability of splitting at each level (controls "sequency")
        bisection_method: "spatial" or "spectral"
        use_random_values: If True, assign random constant per region.
                          If False, use ±1 (classic Hadamard).

    Returns:
        signal: [n_points] piecewise constant signal
    """
    n_points = len(points)
    signal = np.zeros(n_points)

    def recursive_assign(indices: np.ndarray, depth: int, current_value: float):
        """Recursively partition and assign values."""

        # Base case: max depth or too few points
        if depth >= max_depth or len(indices) < 2:
            signal[indices] = current_value
            return

        # Random decision: split or stay constant?
        # Lower split_probability = lower "sequency" (fewer sign changes)
        # BUT: always split at depth 0 to ensure non-trivial probes
        if depth > 0 and np.random.rand() > split_probability:
            signal[indices] = current_value
            return

        # Bisect using manifold structure
        left_indices, right_indices = graph_aware_bisection(
            points, edge_index, indices, method=bisection_method
        )

        # Handle edge cases
        if len(left_indices) == 0 or len(right_indices) == 0:
            signal[indices] = current_value
            return

        # Assign values to children
        if use_random_values:
            # Random constant for each region - samples the full constraint class
            left_value = np.random.randn()
            right_value = np.random.randn()
        else:
            # Classic Hadamard: one half keeps value, other flips
            left_value = current_value
            right_value = -current_value

        recursive_assign(left_indices, depth + 1, left_value)
        recursive_assign(right_indices, depth + 1, right_value)

    # Start recursion
    initial_value = np.random.randn() if use_random_values else np.random.choice([-1.0, 1.0])
    recursive_assign(np.arange(n_points), depth=0, current_value=initial_value)

    return signal


def generate_hadamard_probes(
    points: np.ndarray,
    n_probes: int = 10,
    k_neighbors: int = 15,
    max_depth: int = 6,
    split_probability: float = 0.7,
    bisection_method: str = "spatial",
    use_random_values: bool = True,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate multiple manifold-aware Hadamard-like probe functions.

    Args:
        points: [n_points, dim] point cloud
        n_probes: Number of probe functions to generate
        k_neighbors: k for kNN graph construction
        max_depth: Maximum recursion depth
        split_probability: Probability of splitting at each level
        bisection_method: "spatial" or "spectral"
        use_random_values: If True, random constant per region. If False, ±1 only.
        seed: Random seed for reproducibility

    Returns:
        probes: [n_points, n_probes] array of probe functions
    """
    if seed is not None:
        np.random.seed(seed)

    # Build kNN graph
    print(f"Building {k_neighbors}-NN graph...")
    edge_index = build_knn_graph(points, k=k_neighbors)
    print(f"  Graph has {edge_index.shape[1]} edges")

    # Generate probes
    probes = []
    for i in range(n_probes):
        probe = generate_single_hadamard_probe(
            points=points,
            edge_index=edge_index,
            max_depth=max_depth,
            split_probability=split_probability,
            bisection_method=bisection_method,
            use_random_values=use_random_values
        )
        probes.append(probe)

        # Count unique values as a measure of complexity
        n_unique = len(np.unique(np.round(probe, decimals=6)))
        print(f"  Probe {i+1}/{n_probes}: {n_unique} distinct regions, range [{probe.min():.2f}, {probe.max():.2f}]")

    return np.stack(probes, axis=1)


# ============================================================================
# For Comparison: Laplacian-style (Gaussian smoothed) probes
# ============================================================================

def generate_gaussian_smoothed_probes(
    points: np.ndarray,
    n_probes: int = 10,
    k_neighbors: int = 15,
    n_iterations: int = 20,
    sigma: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Laplacian-style probe functions via Gaussian smoothing on kNN graph.
    (This is similar to what's done in the main codebase)

    Args:
        points: [n_points, dim] point cloud
        n_probes: Number of probe functions
        k_neighbors: k for kNN graph
        n_iterations: Number of smoothing iterations
        sigma: Gaussian kernel bandwidth
        seed: Random seed

    Returns:
        probes: [n_points, n_probes] array of smooth probe functions
    """
    if seed is not None:
        np.random.seed(seed)

    n_points = len(points)

    # Build kNN graph
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k_neighbors + 1)

    probes = []
    for i in range(n_probes):
        # Start with random noise
        signal = np.random.randn(n_points)

        # Iterative Gaussian smoothing
        for _ in range(n_iterations):
            new_signal = np.zeros(n_points)
            for j in range(n_points):
                neighbor_idx = indices[j, 1:]  # Exclude self
                neighbor_dist = distances[j, 1:]

                # Gaussian weights
                weights = np.exp(-neighbor_dist**2 / (2 * sigma**2))
                weights /= weights.sum() + 1e-10

                # Weighted average (including self)
                new_signal[j] = 0.5 * signal[j] + 0.5 * np.sum(weights * signal[neighbor_idx])

            signal = new_signal

        # Normalize
        signal = (signal - signal.mean()) / (signal.std() + 1e-10)
        probes.append(signal)
        print(f"  Gaussian probe {i+1}/{n_probes} generated")

    return np.stack(probes, axis=1)


# ============================================================================
# Mesh Loading
# ============================================================================

def load_mesh(mesh_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load mesh from file using trimesh.

    Returns:
        vertices: [n_vertices, 3]
        faces: [n_faces, 3] or None if point cloud
    """
    import trimesh

    mesh = trimesh.load(mesh_path)

    if isinstance(mesh, trimesh.PointCloud):
        return np.array(mesh.vertices), None
    else:
        return np.array(mesh.vertices), np.array(mesh.faces)


def normalize_to_unit_cube(points: np.ndarray) -> np.ndarray:
    """Normalize points to fit in [-0.5, 0.5]^3."""
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    center = (p_min + p_max) / 2
    scale = (p_max - p_min).max()
    return (points - center) / scale


# ============================================================================
# Visualization
# ============================================================================

def visualize_probes(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    hadamard_probes: np.ndarray,
    gaussian_probes: Optional[np.ndarray] = None
):
    """
    Visualize probe functions using Polyscope.

    Args:
        vertices: Mesh vertices
        faces: Mesh faces (or None for point cloud)
        hadamard_probes: [n_vertices, n_probes] Hadamard-like probes
        gaussian_probes: [n_vertices, n_probes] Gaussian probes (optional, for comparison)
    """
    # Initialize Polyscope
    ps.init()
    ps.set_program_name("Hadamard Probe Function Visualization")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color([0.0, 0.0, 0.0])

    # Always register as point cloud for clearer visualization of piecewise constant values
    geom = ps.register_point_cloud("points", vertices)
    geom.set_radius(0.008)

    # Add Hadamard probes
    n_hadamard = hadamard_probes.shape[1]
    for i in range(n_hadamard):
        probe = hadamard_probes[:, i]
        name = f"Hadamard_probe_{i+1}"
        # Use symmetric colormap centered at 0
        max_abs = np.abs(probe).max()
        geom.add_scalar_quantity(name, probe, enabled=(i == 0), cmap="jet", vminmax=(-max_abs, max_abs))

    # Add Gaussian probes if provided
    if gaussian_probes is not None:
        n_gaussian = gaussian_probes.shape[1]
        for i in range(n_gaussian):
            probe = gaussian_probes[:, i]
            name = f"Gaussian_probe_{i+1}"
            # Normalize for visualization
            probe_normalized = probe / (np.abs(probe).max() + 1e-10)
            geom.add_scalar_quantity(name, probe_normalized, enabled=False, cmap="viridis")

    # Show
    print("\n" + "="*60)
    print("VISUALIZATION CONTROLS:")
    print("="*60)
    print("- Use the left panel to toggle different probe functions")
    print("- Hadamard probes should look PIECEWISE CONSTANT (step-like)")
    print("  → Each region has a uniform color (constant value)")
    print("  → Sharp boundaries between regions")
    print("- Gaussian probes should look SMOOTH (gradual transitions)")
    print("- The colormap is 'jet' (blue=low, red=high)")
    print("="*60 + "\n")

    ps.show()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test manifold-aware Hadamard-like probe functions on 3D meshes"
    )
    parser.add_argument("--mesh_path", type=str, default=None,
                        help="Path to input mesh (OBJ, PLY, etc.)")
    parser.add_argument("--n_probes", type=int, default=300,
                        help="Number of probe functions to generate")
    parser.add_argument("--max_depth", type=int, default=6,
                        help="Maximum recursion depth for Hadamard partitions")
    parser.add_argument("--split_prob", type=float, default=0.85,
                        help="Probability of splitting at each level (lower = simpler probes)")
    parser.add_argument("--k_neighbors", type=int, default=15,
                        help="Number of neighbors for kNN graph")
    parser.add_argument("--bisection_method", type=str, default="spatial",
                        choices=["spatial", "spectral"],
                        help="Method for bisecting: 'spatial' (random hyperplane) or 'spectral' (Fiedler)")
    parser.add_argument("--binary_values", action="store_true",
                        help="Use only ±1 values (classic Hadamard). Default uses random values per region.")
    parser.add_argument("--compare_gaussian", action="store_true",
                        help="Also generate Gaussian-smoothed probes for comparison")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample to this many points (for large meshes)")

    args = parser.parse_args()

    # Load mesh
    if args.mesh_path is not None:
        print(f"Loading mesh from: {args.mesh_path}")
        vertices, faces = load_mesh(args.mesh_path)
    else:
        # Try to find a default mesh
        default_paths = [
            "data/meshes/bunny.obj",
            "data/bunny.obj",
            "meshes/bunny.obj",
            "../data/meshes/bunny.obj"
        ]

        mesh_path = None
        for path in default_paths:
            if os.path.exists(path):
                mesh_path = path
                break

        if mesh_path is None:
            # Create a simple test shape (sphere)
            print("No mesh provided and no default found. Creating a test sphere...")
            from scipy.spatial import SphericalVoronoi

            # Create sphere points
            n_points = 2000
            phi = np.random.uniform(0, 2*np.pi, n_points)
            cos_theta = np.random.uniform(-1, 1, n_points)
            theta = np.arccos(cos_theta)

            vertices = np.stack([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ], axis=1)
            faces = None
            print(f"Created sphere point cloud with {n_points} points")
        else:
            print(f"Loading default mesh from: {mesh_path}")
            vertices, faces = load_mesh(mesh_path)

    # Normalize
    vertices = normalize_to_unit_cube(vertices)
    print(f"Mesh has {len(vertices)} vertices")
    if faces is not None:
        print(f"Mesh has {len(faces)} faces")

    # Subsample if needed
    if args.subsample is not None and len(vertices) > args.subsample:
        print(f"Subsampling to {args.subsample} points...")
        indices = np.random.choice(len(vertices), args.subsample, replace=False)
        vertices = vertices[indices]
        faces = None  # Can't preserve faces after subsampling

    # Generate Hadamard probes
    print(f"\nGenerating {args.n_probes} Hadamard-like probe functions...")
    print(f"  max_depth={args.max_depth}, split_prob={args.split_prob}")
    print(f"  bisection_method={args.bisection_method}")
    print(f"  values={'binary (±1)' if args.binary_values else 'random per region'}")

    hadamard_probes = generate_hadamard_probes(
        points=vertices,
        n_probes=args.n_probes,
        k_neighbors=args.k_neighbors,
        max_depth=args.max_depth,
        split_probability=args.split_prob,
        bisection_method=args.bisection_method,
        use_random_values=not args.binary_values,
        seed=args.seed
    )

    # Optionally generate Gaussian probes for comparison
    gaussian_probes = None
    if args.compare_gaussian:
        print(f"\nGenerating {args.n_probes} Gaussian-smoothed probe functions for comparison...")
        gaussian_probes = generate_gaussian_smoothed_probes(
            points=vertices,
            n_probes=args.n_probes,
            k_neighbors=args.k_neighbors,
            n_iterations=20,
            sigma=0.1,
            seed=args.seed + 1000  # Different seed
        )

    # Visualize
    print("\nLaunching Polyscope visualization...")
    visualize_probes(vertices, faces, hadamard_probes, gaussian_probes)


if __name__ == "__main__":
    main()
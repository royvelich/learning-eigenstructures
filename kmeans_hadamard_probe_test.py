"""
Standalone script to test Hadamard-like probe functions with K-means partitioning.

The partition is computed ONCE using K-means clustering on vertex positions.
Each probe function assigns different random constant values to the same K regions.

This is conceptually cleaner than random partitions per probe:
- Fixed partition structure
- Probes sample from "all piecewise constant functions on THIS partition"
- Learned basis should be Haar-like wavelets on the K-means regions

Usage:
    python kmeans_hadamard_probe_test.py --mesh_path /path/to/mesh.obj
    python kmeans_hadamard_probe_test.py --mesh_path mesh.ply --k 16
    python kmeans_hadamard_probe_test.py --k 8 --compare_laplacian
"""

import argparse
import numpy as np
import polyscope as ps
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from typing import Tuple, Optional
import os


# ============================================================================
# K-means Partition
# ============================================================================

def compute_kmeans_partition(
        points: np.ndarray,
        k: int = 8,
        seed: Optional[int] = None
) -> np.ndarray:
    """
    Partition points into K regions using K-means clustering.

    Args:
        points: [n_points, dim] point positions
        k: Number of clusters/regions
        seed: Random seed for reproducibility

    Returns:
        region_ids: [n_points] integer region assignment (0 to K-1)
    """
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    region_ids = kmeans.fit_predict(points)
    return region_ids


# ============================================================================
# Probe Function Generation
# ============================================================================

def generate_kmeans_hadamard_probes(
        points: np.ndarray,
        n_probes: int = 8,
        k: int = 8,
        seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Hadamard-like probes on K-means partition.

    Partition is computed once, then each probe assigns random constants
    to the same K regions.

    Args:
        points: [n_points, dim] point positions
        n_probes: Number of probe functions
        k: Number of K-means clusters
        seed: Random seed

    Returns:
        probes: [n_points, n_probes] probe functions
        region_ids: [n_points] the partition used
    """
    if seed is not None:
        np.random.seed(seed)

    n_points = len(points)

    # Compute partition ONCE
    print(f"  Computing K-means partition with K={k}...")
    region_ids = compute_kmeans_partition(points, k=k, seed=seed)

    # Count points per region
    unique, counts = np.unique(region_ids, return_counts=True)
    print(f"  Partition created: {len(unique)} regions")
    print(f"  Points per region: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    # Generate probes: random constant per region per probe
    probes = np.zeros((n_points, n_probes), dtype=np.float32)

    for probe_idx in range(n_probes):
        # Random value for each region
        region_values = np.random.randn(k)

        # Assign to points
        probes[:, probe_idx] = region_values[region_ids]

    return probes, region_ids


# ============================================================================
# Comparison: Laplacian Probes
# ============================================================================

def build_knn_graph(points: np.ndarray, k: int = 15) -> np.ndarray:
    """Build k-nearest neighbor graph."""
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k + 1)

    n_points = len(points)
    sources = np.repeat(np.arange(n_points), k)
    targets = indices[:, 1:].flatten()

    return np.stack([sources, targets], axis=0)


def generate_laplacian_probes(
        points: np.ndarray,
        n_probes: int = 8,
        k_neighbors: int = 15,
        iterations: int = 30,
        sigma: float = 0.1,
        seed: Optional[int] = None
) -> np.ndarray:
    """Generate Laplacian probes (smoothed noise) for comparison."""
    if seed is not None:
        np.random.seed(seed)

    n_points = len(points)
    edge_index = build_knn_graph(points, k_neighbors)
    row, col = edge_index

    # Compute Gaussian weights
    diff = points[row] - points[col]
    distances = np.linalg.norm(diff, axis=1)
    weights = np.exp(-distances ** 2 / (2 * sigma ** 2))

    # Normalize per node
    weight_sum = np.zeros(n_points)
    np.add.at(weight_sum, row, weights)
    weights_norm = weights / (weight_sum[row] + 1e-8)

    # Start with noise
    probes = np.random.randn(n_points, n_probes).astype(np.float32)

    # Iterative smoothing
    for _ in range(iterations):
        new_probes = np.zeros_like(probes)
        np.add.at(new_probes, row, weights_norm[:, np.newaxis] * probes[col])
        probes = new_probes

    return probes


# ============================================================================
# Mesh Loading
# ============================================================================

def load_mesh(mesh_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load mesh from file."""
    import trimesh
    mesh = trimesh.load(mesh_path)

    if isinstance(mesh, trimesh.PointCloud):
        return np.array(mesh.vertices), None
    else:
        return np.array(mesh.vertices), np.array(mesh.faces)


def normalize_to_unit_cube(points: np.ndarray) -> np.ndarray:
    """Normalize points to [-0.5, 0.5]³."""
    p_min, p_max = points.min(axis=0), points.max(axis=0)
    center = (p_min + p_max) / 2
    scale = (p_max - p_min).max()
    return (points - center) / scale


def create_test_shape(shape_type: str = "sphere", n_points: int = 2000) -> np.ndarray:
    """Create test point cloud."""
    if shape_type == "sphere":
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        cos_theta = np.random.uniform(-1, 1, n_points)
        theta = np.arccos(cos_theta)

        points = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ], axis=1)

    elif shape_type == "cube":
        points = np.random.uniform(-0.5, 0.5, (n_points, 3))

    elif shape_type == "torus":
        R, r = 0.4, 0.15
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.random.uniform(0, 2 * np.pi, n_points)

        points = np.stack([
            (R + r * np.cos(phi)) * np.cos(theta),
            (R + r * np.cos(phi)) * np.sin(theta),
            r * np.sin(phi)
        ], axis=1)

    else:
        raise ValueError(f"Unknown shape: {shape_type}")

    return points.astype(np.float32)


# ============================================================================
# Visualization
# ============================================================================

def visualize_probes(
        vertices: np.ndarray,
        faces: Optional[np.ndarray],
        hadamard_probes: np.ndarray,
        region_ids: np.ndarray,
        laplacian_probes: Optional[np.ndarray] = None,
        k: int = 8
):
    """Visualize probes using Polyscope."""
    ps.init()
    ps.set_program_name("K-means Hadamard Probe Visualization")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color([0.0, 0.0, 0.0])  # Black background

    # Always use point cloud
    geom = ps.register_point_cloud("points", vertices)
    geom.set_radius(0.008)

    # Show partition (region IDs)
    geom.add_scalar_quantity(
        "Partition_regions", region_ids.astype(np.float32),
        enabled=True, cmap="jet"
    )

    # Add Hadamard probes
    n_hadamard = hadamard_probes.shape[1]
    for i in range(n_hadamard):
        probe = hadamard_probes[:, i]
        name = f"Hadamard_probe_{i + 1}"
        max_abs = np.abs(probe).max() + 1e-10
        geom.add_scalar_quantity(
            name, probe,
            enabled=(i == 0),
            cmap="jet",
            vminmax=(-max_abs, max_abs)
        )

    # Add Laplacian probes for comparison
    if laplacian_probes is not None:
        n_laplacian = laplacian_probes.shape[1]
        for i in range(n_laplacian):
            probe = laplacian_probes[:, i]
            name = f"Laplacian_probe_{i + 1}"
            max_abs = np.abs(probe).max() + 1e-10
            geom.add_scalar_quantity(
                name, probe,
                enabled=False,
                cmap="jet",
                vminmax=(-max_abs, max_abs)
            )

    print("\n" + "=" * 70)
    print("VISUALIZATION GUIDE")
    print("=" * 70)
    print(f"\nK-means partition with K={k} regions")
    print("\n  Partition_regions: Shows the K regions (each color = one region)")
    print("\n  Hadamard probes:")
    print("    - PIECEWISE CONSTANT on the partition")
    print("    - Each region has uniform color (constant value)")
    print("    - Sharp boundaries between regions")
    print("\n  Laplacian probes (if enabled):")
    print("    - SMOOTH across entire shape")
    print("    - Gradual color transitions")
    print("\nUse the left panel to toggle between visualizations.")
    print("=" * 70 + "\n")

    ps.show()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test Hadamard probes with K-means partition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kmeans_hadamard_probe_test.py --mesh_path bunny.obj
  python kmeans_hadamard_probe_test.py --k 16 --n_probes 10
  python kmeans_hadamard_probe_test.py --test_shape torus --k 8
  python kmeans_hadamard_probe_test.py --compare_laplacian
        """
    )

    parser.add_argument("--mesh_path", type=str, default=None,
                        help="Path to mesh file")
    parser.add_argument("--test_shape", type=str, default="sphere",
                        choices=["sphere", "cube", "torus"],
                        help="Test shape if no mesh provided")
    parser.add_argument("--n_probes", type=int, default=8,
                        help="Number of probe functions")
    parser.add_argument("--k", type=int, default=8,
                        help="Number of K-means clusters (regions)")
    parser.add_argument("--compare_laplacian", action="store_true",
                        help="Also generate Laplacian probes for comparison")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample to N points")

    args = parser.parse_args()

    # Load or create geometry
    if args.mesh_path is not None:
        print(f"Loading mesh: {args.mesh_path}")
        vertices, faces = load_mesh(args.mesh_path)
    else:
        print(f"Creating test shape: {args.test_shape}")
        vertices = create_test_shape(args.test_shape, 3000)
        faces = None

    vertices = normalize_to_unit_cube(vertices)
    print(f"Shape has {len(vertices)} vertices")

    # Subsample if needed
    if args.subsample and len(vertices) > args.subsample:
        print(f"Subsampling to {args.subsample} points")
        np.random.seed(args.seed)
        idx = np.random.choice(len(vertices), args.subsample, replace=False)
        vertices = vertices[idx]
        faces = None

    # Generate K-means Hadamard probes
    print(f"\nGenerating K-means Hadamard probes...")
    hadamard_probes, region_ids = generate_kmeans_hadamard_probes(
        points=vertices,
        n_probes=args.n_probes,
        k=args.k,
        seed=args.seed
    )

    # Optionally generate Laplacian probes
    laplacian_probes = None
    if args.compare_laplacian:
        print(f"\nGenerating Laplacian probes for comparison...")
        laplacian_probes = generate_laplacian_probes(
            points=vertices,
            n_probes=args.n_probes,
            k_neighbors=15,
            iterations=30,
            sigma=0.1,
            seed=args.seed + 1000
        )

    # Visualize
    print("\nLaunching visualization...")
    visualize_probes(
        vertices, faces,
        hadamard_probes, region_ids,
        laplacian_probes,
        k=args.k
    )


if __name__ == "__main__":
    main()
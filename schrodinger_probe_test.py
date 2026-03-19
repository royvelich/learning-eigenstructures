"""
Standalone script to test Schrödinger operator probe functions on 3D meshes.

Schrödinger operator: H = -Δ + V(x)
Energy functional: E[f] = ∫|∇f|² + V|f|² dx

Low-energy functions (probes) are:
1. Smooth (like Laplacian)
2. Small where potential V is large (localized in potential "wells")

Eigenfunctions will concentrate where V is LOW.

Potential choices:
- curvature: eigenfunctions avoid sharp regions, live in flat areas
- inverse_curvature: eigenfunctions concentrate at edges/corners
- center_distance: eigenfunctions concentrate at shape center
- height: eigenfunctions concentrate at bottom of shape
- random: random wells (for testing)

Usage:
    python schrodinger_probe_test.py --mesh_path /path/to/mesh.obj
    python schrodinger_probe_test.py --potential_type curvature
    python schrodinger_probe_test.py --potential_type center_distance --potential_strength 10
"""

import argparse
import numpy as np
import polyscope as ps
from scipy.spatial import cKDTree
from typing import Tuple, Optional
import os


# ============================================================================
# Schrödinger Probe Generation
# ============================================================================

def build_knn_graph(points: np.ndarray, k: int = 15) -> np.ndarray:
    """Build k-nearest neighbor graph."""
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k + 1)

    n_points = len(points)
    sources = np.repeat(np.arange(n_points), k)
    targets = indices[:, 1:].flatten()

    edge_index = np.stack([sources, targets], axis=0)
    return edge_index


def estimate_curvature(points: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """
    Estimate local curvature using PCA of local neighborhood.

    Returns values in [0, 1] where high = high curvature.
    """
    n_points = len(points)
    row, col = edge_index

    curvature = np.zeros(n_points)

    for i in range(n_points):
        # Get neighbors
        neighbor_mask = (row == i)
        neighbor_indices = col[neighbor_mask]

        if len(neighbor_indices) < 3:
            continue

        # Center neighbors
        neighbors = points[neighbor_indices]
        centered = neighbors - points[i]

        # Covariance and eigenvalues
        cov = centered.T @ centered / len(neighbor_indices)
        eigenvalues = np.linalg.eigvalsh(cov)

        # Curvature ~ ratio of smallest to largest eigenvalue
        ratio = eigenvalues[0] / (eigenvalues[-1] + 1e-8)
        curvature[i] = ratio

    # Normalize to [0, 1]
    if curvature.max() > curvature.min():
        curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min())

    return curvature


def compute_potential(
    points: np.ndarray,
    edge_index: np.ndarray,
    potential_type: str
) -> np.ndarray:
    """
    Compute potential V(x) for Schrödinger operator.

    Eigenfunctions will concentrate where V is LOW.
    """
    n_points = len(points)

    if potential_type == "random":
        V = np.random.rand(n_points)

    elif potential_type == "curvature":
        # High curvature = high potential = eigenfunctions avoid
        V = estimate_curvature(points, edge_index)

    elif potential_type == "inverse_curvature":
        # High curvature = low potential = eigenfunctions concentrate
        curv = estimate_curvature(points, edge_index)
        V = curv.max() - curv

    elif potential_type == "center_distance":
        # Far from center = high potential
        centroid = points.mean(axis=0)
        V = np.linalg.norm(points - centroid, axis=1)

    elif potential_type == "boundary_distance":
        # Approximate: distance from convex hull
        # Simple proxy: distance from centroid, inverted
        centroid = points.mean(axis=0)
        dist_from_center = np.linalg.norm(points - centroid, axis=1)
        V = dist_from_center.max() - dist_from_center

    elif potential_type == "height":
        # V = z coordinate (eigenfunctions at bottom)
        V = points[:, 2] - points[:, 2].min()

    elif potential_type == "inverse_height":
        # V = -z (eigenfunctions at top)
        V = points[:, 2].max() - points[:, 2]

    else:
        raise ValueError(f"Unknown potential type: {potential_type}")

    # Normalize to [0, 1]
    if V.max() > V.min():
        V = (V - V.min()) / (V.max() - V.min())

    return V


def gaussian_smooth_step(
    f: np.ndarray,
    points: np.ndarray,
    edge_index: np.ndarray,
    sigma: float
) -> np.ndarray:
    """One step of Gaussian smoothing."""
    n_points = len(points)
    row, col = edge_index

    # Compute Gaussian weights
    diff = points[row] - points[col]
    distances = np.linalg.norm(diff, axis=1)
    weights = np.exp(-distances**2 / (2 * sigma**2))

    # Normalize per node
    weight_sum = np.zeros(n_points)
    np.add.at(weight_sum, row, weights)
    weights_norm = weights / (weight_sum[row] + 1e-8)

    # Aggregate
    out = np.zeros_like(f)
    np.add.at(out, row, weights_norm[:, np.newaxis] * f[col])

    return out


def generate_schrodinger_probes(
    points: np.ndarray,
    n_probes: int = 8,
    k_neighbors: int = 15,
    potential_type: str = "curvature",
    potential_strength: float = 5.0,
    iterations: int = 30,
    sigma: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Schrödinger probe functions via gradient flow.

    Args:
        points: [n_points, 3] point positions
        n_probes: Number of probe functions
        k_neighbors: k for kNN graph
        potential_type: Type of potential V(x)
        potential_strength: β in exp(-βV)
        iterations: Number of diffusion-decay iterations
        sigma: Gaussian smoothing width
        seed: Random seed

    Returns:
        probes: [n_points, n_probes] probe functions
        potential: [n_points] the potential V used
    """
    if seed is not None:
        np.random.seed(seed)

    n_points = len(points)

    # Build graph
    edge_index = build_knn_graph(points, k_neighbors)

    # Compute potential
    V = compute_potential(points, edge_index, potential_type)

    # Decay factor: exp(-β·V)
    decay = np.exp(-potential_strength * V)

    # Start with random noise
    f = np.random.rand(n_points, n_probes).astype(np.float32)

    print(f"  Running gradient flow: {iterations} iterations")
    print(f"  Potential: {potential_type}, strength: {potential_strength}")

    # Gradient flow: alternate diffusion and decay
    for i in range(iterations):
        # Diffusion (smoothing)
        f = gaussian_smooth_step(f, points, edge_index, sigma)

        # Decay by potential
        f = f * decay[:, np.newaxis]

        if (i + 1) % 10 == 0:
            energy = np.mean(f**2)
            print(f"    Iteration {i+1}/{iterations}, mean energy: {energy:.4f}")

    return f, V


def generate_laplacian_probes(
    points: np.ndarray,
    n_probes: int = 8,
    k_neighbors: int = 15,
    iterations: int = 30,
    sigma: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate standard Laplacian probes (smoothed noise) for comparison."""
    if seed is not None:
        np.random.seed(seed)

    n_points = len(points)
    edge_index = build_knn_graph(points, k_neighbors)

    f = np.random.randn(n_points, n_probes).astype(np.float32)

    for _ in range(iterations):
        f = gaussian_smooth_step(f, points, edge_index, sigma)

    return f


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
        phi = np.random.uniform(0, 2*np.pi, n_points)
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
        R, r = 0.4, 0.15  # Major and minor radius
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, 2*np.pi, n_points)

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
    schrodinger_probes: np.ndarray,
    potential: np.ndarray,
    laplacian_probes: Optional[np.ndarray] = None,
    potential_type: str = "curvature"
):
    """Visualize probes using Polyscope."""
    ps.init()
    ps.set_program_name("Schrödinger Probe Visualization")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color([0.0, 0.0, 0.0])  # Black background

    # Always use point cloud for visualization
    geom = ps.register_point_cloud("points", vertices)
    geom.set_radius(0.008)

    # Show potential field
    geom.add_scalar_quantity(
        "Potential_V", potential,
        enabled=True, cmap="jet"
    )

    # Add Schrödinger probes
    n_schrodinger = schrodinger_probes.shape[1]
    for i in range(n_schrodinger):
        probe = schrodinger_probes[:, i]
        name = f"Schrodinger_probe_{i+1}"
        max_abs = np.abs(probe).max() + 1e-10
        geom.add_scalar_quantity(
            name, probe,
            enabled=(i == 0),
            cmap="jet",
            # vminmax=(-max_abs, max_abs)
        )

    # Add Laplacian probes for comparison
    if laplacian_probes is not None:
        n_laplacian = laplacian_probes.shape[1]
        for i in range(n_laplacian):
            probe = laplacian_probes[:, i]
            name = f"Laplacian_probe_{i+1}"
            max_abs = np.abs(probe).max() + 1e-10
            geom.add_scalar_quantity(
                name, probe,
                enabled=False,
                cmap="jet",
                # vminmax=(-max_abs, max_abs)
            )

    print("\n" + "="*70)
    print("VISUALIZATION GUIDE")
    print("="*70)
    print(f"\nPotential type: {potential_type}")
    print("  - Potential_V shows where V is high (yellow) vs low (purple)")
    print("  - Eigenfunctions concentrate where V is LOW (purple regions)")
    print("\nSchrödinger probes:")
    print("  - Should be SMOOTH (like Laplacian)")
    print("  - Should be LOCALIZED in low-potential regions")
    print("  - Compare to Laplacian probes which are smooth but NOT localized")
    print("\nUse the left panel to toggle between probes.")
    print("="*70 + "\n")

    ps.show()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test Schrödinger operator probe functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Potential types:
  curvature         - V = local curvature, eigenfunctions in FLAT regions
  inverse_curvature - V = -curvature, eigenfunctions at SHARP features
  center_distance   - V = dist from center, eigenfunctions at CENTER
  height            - V = z-coordinate, eigenfunctions at BOTTOM
  inverse_height    - V = -z, eigenfunctions at TOP
  random            - V = random, for testing

Examples:
  python schrodinger_probe_test.py --mesh_path bunny.obj
  python schrodinger_probe_test.py --potential_type center_distance
  python schrodinger_probe_test.py --potential_type curvature --potential_strength 10
  python schrodinger_probe_test.py --compare_laplacian
        """
    )

    parser.add_argument("--mesh_path", type=str, default=None,
                        help="Path to mesh file")
    parser.add_argument("--test_shape", type=str, default="sphere",
                        choices=["sphere", "cube", "torus"],
                        help="Test shape if no mesh provided")
    parser.add_argument("--n_probes", type=int, default=8,
                        help="Number of probes")
    parser.add_argument("--potential_type", type=str, default="curvature",
                        choices=["curvature", "inverse_curvature", "center_distance",
                                 "height", "inverse_height", "random"],
                        help="Type of potential V(x)")
    parser.add_argument("--potential_strength", type=float, default=2.0,
                        help="β in exp(-βV), controls localization strength")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Gradient flow iterations")
    parser.add_argument("--sigma", type=float, default=0.2,
                        help="Gaussian smoothing width")
    parser.add_argument("--k_neighbors", type=int, default=15,
                        help="k for kNN graph")
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
        idx = np.random.choice(len(vertices), args.subsample, replace=False)
        vertices = vertices[idx]
        faces = None

    # Generate Schrödinger probes
    print(f"\nGenerating Schrödinger probes...")
    schrodinger_probes, potential = generate_schrodinger_probes(
        points=vertices,
        n_probes=args.n_probes,
        k_neighbors=args.k_neighbors,
        potential_type=args.potential_type,
        potential_strength=args.potential_strength,
        iterations=args.iterations,
        sigma=args.sigma,
        seed=args.seed
    )

    # Optionally generate Laplacian probes
    laplacian_probes = None
    if args.compare_laplacian:
        print(f"\nGenerating Laplacian probes for comparison...")
        laplacian_probes = generate_laplacian_probes(
            points=vertices,
            n_probes=args.n_probes,
            k_neighbors=args.k_neighbors,
            iterations=args.iterations,
            sigma=args.sigma,
            seed=args.seed + 1000
        )

    # Visualize
    print("\nLaunching visualization...")
    visualize_probes(
        vertices, faces,
        schrodinger_probes, potential,
        laplacian_probes,
        args.potential_type
    )


if __name__ == "__main__":
    main()
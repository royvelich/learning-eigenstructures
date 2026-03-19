"""
Standalone script to test Morlet wavelet-like probe functions on 3D meshes.

Morlet wavelets are Gaussian-windowed sinusoids:
    ψ(x) = exp(-|x-c|² / 2σ²) · cos(ω · (x-c)·d)

where:
    - c is the center (localization point)
    - σ is the scale (Gaussian envelope width)
    - ω is the frequency (oscillation rate)
    - d is the oscillation direction

Key properties:
    - Spatially localized (Gaussian envelope)
    - Oscillatory within localization (sinusoidal)
    - Optimal time-frequency uncertainty (Heisenberg limit)

This is fundamentally different from:
    - Laplacian: global smooth oscillations
    - Hadamard: piecewise constant (no oscillation)
    - Sparse: localized but no oscillation

Usage:
    python morlet_probe_test.py --mesh_path /path/to/mesh.obj
    python morlet_probe_test.py --mesh_path /path/to/mesh.ply --n_probes 10 --frequency_range 5 20
"""

import argparse
import numpy as np
import polyscope as ps
from typing import Tuple, Optional
import os


# ============================================================================
# Morlet Wavelet Probe Function Generation
# ============================================================================

def generate_single_morlet_probe(
        points: np.ndarray,
        scale_range: Tuple[float, float] = (0.05, 0.3),
        frequency_range: Tuple[float, float] = (5.0, 30.0),
        diameter: Optional[float] = None
) -> np.ndarray:
    """
    Generate a single Morlet wavelet-like probe function.

    Morlet wavelet = Gaussian envelope × Sinusoidal oscillation

    Args:
        points: [n_points, dim] point cloud
        scale_range: (min, max) for Gaussian envelope width (relative to diameter)
        frequency_range: (min, max) for oscillation frequency
        diameter: Shape diameter (computed if not provided)

    Returns:
        signal: [n_points] Morlet wavelet probe function
    """
    n_points, dim = points.shape

    # Compute diameter if not provided
    if diameter is None:
        diameter = np.linalg.norm(points.max(axis=0) - points.min(axis=0))

    # Random center point (where the wavelet is localized)
    center_idx = np.random.randint(n_points)
    center = points[center_idx]

    # Random scale (width of Gaussian envelope)
    scale = np.random.uniform(*scale_range) * diameter

    # Random frequency
    frequency = np.random.uniform(*frequency_range)

    # Random oscillation direction
    direction = np.random.randn(dim)
    direction /= np.linalg.norm(direction) + 1e-10

    # Compute displacement from center
    displacement = points - center  # [n_points, dim]
    distances = np.linalg.norm(displacement, axis=1)

    # Gaussian envelope (spatial localization)
    envelope = np.exp(-distances ** 2 / (2 * scale ** 2))

    # Sinusoidal oscillation (frequency content)
    # Phase = projection onto oscillation direction × frequency
    phase = (displacement @ direction) * frequency
    oscillation = np.cos(phase)

    # Morlet = envelope × oscillation
    morlet = envelope * oscillation

    return morlet


def generate_morlet_probes(
        points: np.ndarray,
        n_probes: int = 10,
        scale_range: Tuple[float, float] = (0.05, 0.3),
        frequency_range: Tuple[float, float] = (5.0, 30.0),
        seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate multiple Morlet wavelet-like probe functions.

    Args:
        points: [n_points, dim] point cloud
        n_probes: Number of probe functions to generate
        scale_range: (min, max) for Gaussian envelope width (relative to diameter)
        frequency_range: (min, max) for oscillation frequency
        seed: Random seed for reproducibility

    Returns:
        probes: [n_points, n_probes] array of probe functions
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute diameter once
    diameter = np.linalg.norm(points.max(axis=0) - points.min(axis=0))

    probes = []
    for i in range(n_probes):
        probe = generate_single_morlet_probe(
            points=points,
            scale_range=scale_range,
            frequency_range=frequency_range,
            diameter=diameter
        )
        probes.append(probe)

        # Statistics
        envelope_size = np.sum(np.abs(probe) > 0.1 * np.abs(probe).max())
        n_zero_crossings = np.sum(np.abs(np.diff(np.sign(probe))) > 0)
        print(f"  Probe {i + 1}/{n_probes}: envelope covers ~{envelope_size} points, ~{n_zero_crossings} zero crossings")

    return np.stack(probes, axis=1)


# ============================================================================
# Additional Wavelet Types
# ============================================================================

def generate_single_mexican_hat_probe(
        points: np.ndarray,
        scale_range: Tuple[float, float] = (0.05, 0.3),
        diameter: Optional[float] = None
) -> np.ndarray:
    """
    Generate a Mexican Hat (Ricker/LoG) wavelet probe function.

    Mexican Hat = (1 - r²/σ²) × exp(-r²/2σ²)

    This is the negative Laplacian of a Gaussian - isotropic (no directional preference).
    Unlike Morlet, it has a central peak surrounded by a negative ring.

    Args:
        points: [n_points, dim] point cloud
        scale_range: (min, max) for wavelet width (relative to diameter)
        diameter: Shape diameter (computed if not provided)

    Returns:
        signal: [n_points] Mexican hat wavelet
    """
    n_points, dim = points.shape

    if diameter is None:
        diameter = np.linalg.norm(points.max(axis=0) - points.min(axis=0))

    # Random center
    center_idx = np.random.randint(n_points)
    center = points[center_idx]

    # Random scale
    scale = np.random.uniform(*scale_range) * diameter

    # Compute distances
    distances_sq = np.sum((points - center) ** 2, axis=1)
    normalized_dist_sq = distances_sq / (scale ** 2)

    # Mexican hat: (1 - r²/σ²) × exp(-r²/2σ²)
    mexican_hat = (1 - normalized_dist_sq) * np.exp(-normalized_dist_sq / 2)

    return mexican_hat


def generate_single_gabor_probe(
        points: np.ndarray,
        scale_range: Tuple[float, float] = (0.05, 0.3),
        frequency_range: Tuple[float, float] = (5.0, 30.0),
        aspect_ratio_range: Tuple[float, float] = (0.3, 1.0),
        diameter: Optional[float] = None
) -> np.ndarray:
    """
    Generate a Gabor filter-like probe function.

    Gabor = Anisotropic Gaussian × Sinusoid

    Similar to Morlet but with an elliptical envelope (narrower perpendicular
    to oscillation direction). This is what biological vision systems use.

    Args:
        points: [n_points, dim] point cloud
        scale_range: (min, max) for envelope width
        frequency_range: (min, max) for oscillation frequency
        aspect_ratio_range: (min, max) for envelope aspect ratio (perp/parallel)
        diameter: Shape diameter

    Returns:
        signal: [n_points] Gabor-like probe
    """
    n_points, dim = points.shape

    if diameter is None:
        diameter = np.linalg.norm(points.max(axis=0) - points.min(axis=0))

    # Random center
    center_idx = np.random.randint(n_points)
    center = points[center_idx]

    # Scales: parallel to oscillation vs perpendicular
    scale_parallel = np.random.uniform(*scale_range) * diameter
    aspect_ratio = np.random.uniform(*aspect_ratio_range)
    scale_perp = scale_parallel * aspect_ratio

    # Frequency and direction
    frequency = np.random.uniform(*frequency_range)
    direction = np.random.randn(dim)
    direction /= np.linalg.norm(direction) + 1e-10

    # Decompose displacement into parallel and perpendicular components
    displacement = points - center
    parallel_proj = (displacement @ direction)
    parallel_component = parallel_proj[:, np.newaxis] * direction
    perp_component = displacement - parallel_component

    parallel_dist_sq = parallel_proj ** 2
    perp_dist_sq = np.sum(perp_component ** 2, axis=1)

    # Anisotropic Gaussian envelope
    envelope = np.exp(
        -parallel_dist_sq / (2 * scale_parallel ** 2)
        - perp_dist_sq / (2 * scale_perp ** 2)
    )

    # Oscillation
    phase = parallel_proj * frequency
    oscillation = np.cos(phase)

    return envelope * oscillation


def generate_wavelet_probes(
        points: np.ndarray,
        n_probes: int = 10,
        wavelet_type: str = "morlet",
        scale_range: Tuple[float, float] = (0.05, 0.3),
        frequency_range: Tuple[float, float] = (5.0, 30.0),
        seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate wavelet probe functions of specified type.

    Args:
        points: [n_points, dim] point cloud
        n_probes: Number of probes
        wavelet_type: "morlet", "mexican_hat", or "gabor"
        scale_range: Envelope width range (relative to diameter)
        frequency_range: Oscillation frequency range (for morlet/gabor)
        seed: Random seed

    Returns:
        probes: [n_points, n_probes]
    """
    if seed is not None:
        np.random.seed(seed)

    diameter = np.linalg.norm(points.max(axis=0) - points.min(axis=0))

    probes = []
    for i in range(n_probes):
        if wavelet_type == "morlet":
            probe = generate_single_morlet_probe(points, scale_range, frequency_range, diameter)
        elif wavelet_type == "mexican_hat":
            probe = generate_single_mexican_hat_probe(points, scale_range, diameter)
        elif wavelet_type == "gabor":
            probe = generate_single_gabor_probe(points, scale_range, frequency_range, diameter=diameter)
        else:
            raise ValueError(f"Unknown wavelet type: {wavelet_type}")

        probes.append(probe)

        # Statistics
        envelope_size = np.sum(np.abs(probe) > 0.1 * np.abs(probe).max())
        n_zero_crossings = np.sum(np.abs(np.diff(np.sign(probe))) > 0)
        print(f"  {wavelet_type} {i + 1}/{n_probes}: ~{envelope_size} points in envelope, ~{n_zero_crossings} zero crossings")

    return np.stack(probes, axis=1)


# ============================================================================
# Fast Vectorized Version (for training)
# ============================================================================

def generate_morlet_probes_fast(
        points: np.ndarray,
        n_probes: int = 10,
        scale_range: Tuple[float, float] = (0.05, 0.3),
        frequency_range: Tuple[float, float] = (5.0, 30.0),
        seed: Optional[int] = None
) -> np.ndarray:
    """
    Fast vectorized Morlet probe generation.

    Generates all probes in parallel using numpy broadcasting.
    Much faster than the loop version for large n_probes.

    Args:
        points: [n_points, dim] point cloud
        n_probes: Number of probes
        scale_range: Envelope width range
        frequency_range: Oscillation frequency range
        seed: Random seed

    Returns:
        probes: [n_points, n_probes]
    """
    if seed is not None:
        np.random.seed(seed)

    n_points, dim = points.shape
    diameter = np.linalg.norm(points.max(axis=0) - points.min(axis=0))

    # Random centers: [n_probes, dim]
    center_indices = np.random.randint(0, n_points, n_probes)
    centers = points[center_indices]  # [n_probes, dim]

    # Random scales: [n_probes]
    scales = np.random.uniform(*scale_range, n_probes) * diameter

    # Random frequencies: [n_probes]
    frequencies = np.random.uniform(*frequency_range, n_probes)

    # Random directions: [n_probes, dim]
    directions = np.random.randn(n_probes, dim)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10

    # Compute all displacements: [n_points, n_probes, dim]
    # points: [n_points, dim] -> [n_points, 1, dim]
    # centers: [n_probes, dim] -> [1, n_probes, dim]
    displacements = points[:, np.newaxis, :] - centers[np.newaxis, :, :]

    # Distances: [n_points, n_probes]
    distances = np.linalg.norm(displacements, axis=2)

    # Envelopes: [n_points, n_probes]
    envelopes = np.exp(-distances ** 2 / (2 * scales[np.newaxis, :] ** 2))

    # Phases: project displacements onto directions
    # displacements: [n_points, n_probes, dim]
    # directions: [n_probes, dim] -> [1, n_probes, dim]
    phases = np.sum(displacements * directions[np.newaxis, :, :], axis=2)  # [n_points, n_probes]
    phases = phases * frequencies[np.newaxis, :]

    # Oscillations
    oscillations = np.cos(phases)

    # Morlet = envelope × oscillation
    probes = envelopes * oscillations

    return probes.astype(np.float32)


# ============================================================================
# For Comparison: Other Probe Types
# ============================================================================

def generate_gaussian_bump_probes(
        points: np.ndarray,
        n_probes: int = 10,
        scale_range: Tuple[float, float] = (0.05, 0.3),
        seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Gaussian bump probes (NO oscillation - just localization).
    This is like Morlet but without the sinusoidal part.
    """
    if seed is not None:
        np.random.seed(seed)

    n_points = len(points)
    diameter = np.linalg.norm(points.max(axis=0) - points.min(axis=0))

    probes = []
    for i in range(n_probes):
        # Random center
        center_idx = np.random.randint(n_points)
        center = points[center_idx]

        # Random scale
        scale = np.random.uniform(*scale_range) * diameter

        # Gaussian bump (no oscillation)
        distances = np.linalg.norm(points - center, axis=1)
        bump = np.exp(-distances ** 2 / (2 * scale ** 2))

        # Random sign
        if np.random.rand() > 0.5:
            bump = -bump

        probes.append(bump)
        print(f"  Gaussian bump {i + 1}/{n_probes} generated")

    return np.stack(probes, axis=1)


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
    """
    from scipy.spatial import cKDTree

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
                weights = np.exp(-neighbor_dist ** 2 / (2 * sigma ** 2))
                weights /= weights.sum() + 1e-10

                # Weighted average
                new_signal[j] = 0.5 * signal[j] + 0.5 * np.sum(weights * signal[neighbor_idx])

            signal = new_signal

        # Normalize
        signal = (signal - signal.mean()) / (signal.std() + 1e-10)
        probes.append(signal)
        print(f"  Gaussian smoothed probe {i + 1}/{n_probes} generated")

    return np.stack(probes, axis=1)


# ============================================================================
# Mesh Loading
# ============================================================================

def load_mesh(mesh_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load mesh from file using trimesh."""
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
        morlet_probes: np.ndarray,
        comparison_probes: Optional[np.ndarray] = None,
        comparison_name: str = "Comparison"
):
    """
    Visualize probe functions using Polyscope.
    """
    # Initialize Polyscope
    ps.init()
    ps.set_program_name("Morlet Wavelet Probe Visualization")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color([0.0, 0.0, 0.0])

    # Always register as point cloud
    geom = ps.register_point_cloud("points", vertices)
    geom.set_radius(0.008)

    # Add Morlet probes
    n_morlet = morlet_probes.shape[1]
    for i in range(n_morlet):
        probe = morlet_probes[:, i]
        name = f"Morlet_probe_{i + 1}"
        max_abs = np.abs(probe).max() + 1e-10
        geom.add_scalar_quantity(name, probe, enabled=(i == 0), cmap="jet", vminmax=(-max_abs, max_abs))

    # Add comparison probes if provided
    if comparison_probes is not None:
        n_comp = comparison_probes.shape[1]
        for i in range(n_comp):
            probe = comparison_probes[:, i]
            name = f"{comparison_name}_probe_{i + 1}"
            max_abs = np.abs(probe).max() + 1e-10
            geom.add_scalar_quantity(name, probe, enabled=False, cmap="jet", vminmax=(-max_abs, max_abs))

    # Show
    print("\n" + "=" * 60)
    print("VISUALIZATION CONTROLS:")
    print("=" * 60)
    print("- Use the left panel to toggle different probe functions")
    print("- Morlet probes should show LOCALIZED OSCILLATIONS:")
    print("  → Concentrated in a region (Gaussian envelope)")
    print("  → Oscillating within that region (red/blue ripples)")
    print("- Compare to Gaussian bumps (localized, no oscillation)")
    print("- Compare to Laplacian probes (global oscillation)")
    print("=" * 60 + "\n")

    ps.show()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test Morlet wavelet-like probe functions on 3D meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Wavelet Types:
  morlet       - Gaussian envelope × sinusoid (localized oscillation)
  mexican_hat  - Laplacian of Gaussian (isotropic, no direction preference)
  gabor        - Anisotropic Gaussian × sinusoid (biological vision)

Examples:
  python morlet_probe_test.py --mesh_path bunny.obj
  python morlet_probe_test.py --wavelet_type mexican_hat
  python morlet_probe_test.py --compare_laplacian --compare_bumps
  python morlet_probe_test.py --fast  # Use fast vectorized version
        """
    )
    parser.add_argument("--mesh_path", type=str, default=None,
                        help="Path to input mesh (OBJ, PLY, etc.)")
    parser.add_argument("--n_probes", type=int, default=8,
                        help="Number of probe functions to generate")
    parser.add_argument("--wavelet_type", type=str, default="morlet",
                        choices=["morlet", "mexican_hat", "gabor"],
                        help="Type of wavelet to generate")
    parser.add_argument("--scale_min", type=float, default=0.01,
                        help="Minimum scale (envelope width) relative to diameter")
    parser.add_argument("--scale_max", type=float, default=0.1,
                        help="Maximum scale (envelope width) relative to diameter")
    parser.add_argument("--freq_min", type=float, default=10.0,
                        help="Minimum oscillation frequency")
    parser.add_argument("--freq_max", type=float, default=30.0,
                        help="Maximum oscillation frequency")
    parser.add_argument("--compare_bumps", action="store_true",
                        help="Also generate Gaussian bump probes (no oscillation) for comparison")
    parser.add_argument("--compare_laplacian", action="store_true",
                        help="Also generate Laplacian-smoothed probes for comparison")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast vectorized generation (for Morlet only)")
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
        # Create a test shape
        print("No mesh provided. Creating a test sphere...")
        n_points = 2000
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        cos_theta = np.random.uniform(-1, 1, n_points)
        theta = np.arccos(cos_theta)

        vertices = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ], axis=1)
        faces = None
        print(f"Created sphere point cloud with {n_points} points")

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
        faces = None

    # Generate wavelet probes
    wavelet_name = args.wavelet_type.replace("_", " ").title()
    print(f"\nGenerating {args.n_probes} {wavelet_name} wavelet probe functions...")
    print(f"  scale_range=({args.scale_min}, {args.scale_max})")
    if args.wavelet_type != "mexican_hat":
        print(f"  frequency_range=({args.freq_min}, {args.freq_max})")

    if args.fast and args.wavelet_type == "morlet":
        print("  Using FAST vectorized generation")
        morlet_probes = generate_morlet_probes_fast(
            points=vertices,
            n_probes=args.n_probes,
            scale_range=(args.scale_min, args.scale_max),
            frequency_range=(args.freq_min, args.freq_max),
            seed=args.seed
        )
        print(f"  Generated {args.n_probes} probes")
    else:
        morlet_probes = generate_wavelet_probes(
            points=vertices,
            n_probes=args.n_probes,
            wavelet_type=args.wavelet_type,
            scale_range=(args.scale_min, args.scale_max),
            frequency_range=(args.freq_min, args.freq_max),
            seed=args.seed
        )

    # Optionally generate comparison probes
    comparison_probes = None
    comparison_name = ""

    if args.compare_bumps:
        print(f"\nGenerating {args.n_probes} Gaussian bump probes for comparison...")
        comparison_probes = generate_gaussian_bump_probes(
            points=vertices,
            n_probes=args.n_probes,
            scale_range=(args.scale_min, args.scale_max),
            seed=args.seed + 1000
        )
        comparison_name = "GaussianBump"

    if args.compare_laplacian:
        print(f"\nGenerating {args.n_probes} Laplacian-smoothed probes for comparison...")
        laplacian_probes = generate_gaussian_smoothed_probes(
            points=vertices,
            n_probes=args.n_probes,
            k_neighbors=15,
            n_iterations=20,
            sigma=0.1,
            seed=args.seed + 2000
        )
        if comparison_probes is not None:
            # Concatenate if we already have bumps
            comparison_probes = np.concatenate([comparison_probes, laplacian_probes], axis=1)
            comparison_name = "Comparison"
        else:
            comparison_probes = laplacian_probes
            comparison_name = "Laplacian"

    # Visualize
    print("\nLaunching Polyscope visualization...")
    visualize_probes(vertices, faces, morlet_probes, comparison_probes, comparison_name)


if __name__ == "__main__":
    main()
# standard library
import time

import omegaconf
# polyscope
import polyscope as ps

# numpy
import numpy as np

# hydra
import hydra

# omegaconf
from omegaconf import DictConfig, ListConfig
import pytorch_lightning as pl

# torch
import torch

# torch geometric
from torch_geometric.data import Dataset, Data, Batch

# matplotlib
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend to avoid conflicts
import matplotlib.pyplot as plt

# seaborn
import seaborn as sns

# sklearn
from sklearn.preprocessing import normalize

# neural-laplacian
from neural_laplacian import utils
from neural_laplacian.modules.laplacian_modules import LaplacianPredictorModule3D

# Configuration Constants
CMAP = 'jet'
POINT_RADIUS = 0.02


class VisualizationHelper:
    """Helper class for common visualization operations."""

    @staticmethod
    def normalize_to_unit_cube(points: np.ndarray) -> np.ndarray:
        """Normalize vertices/points to fit in a unit cube centered at origin."""
        p_max = points.max(axis=0)
        p_min = points.min(axis=0)
        center = (p_max + p_min) / 2
        scale = (p_max - p_min).max()
        return (points - center) / scale

    @staticmethod
    def plot_correlation(eigenvectors: np.ndarray):
        """Plot correlation matrix of eigenvectors."""
        inner_products = np.abs(np.dot(eigenvectors.T, eigenvectors))
        plt.figure(figsize=(inner_products.shape[0], inner_products.shape[1]))
        sns.heatmap(inner_products, annot=True, cmap=CMAP, fmt='.2f',
                    xticklabels=range(inner_products.shape[0]),
                    yticklabels=range(inner_products.shape[1]))
        plt.title('Covariance Matrix Heatmap')
        plt.tight_layout()
        plt.show()


class MeshVisualizer:
    """Class for visualizing meshes with Polyscope."""

    def __init__(self, data: Data, k: int, name_prefix: str = ""):
        """
        Initialize mesh visualizer.

        Args:
            data: Geometry data
            k: Number of neighbors for graph construction
            name_prefix: Optional prefix for object names
        """
        self.data = data
        self.k = k
        self.name_prefix = name_prefix
        self.geom = None
        # Always treat as point cloud for visualization
        self.is_mesh = False

    def register_geometry(self):
        """Register geometry with Polyscope."""
        points = self.data.points.cpu().numpy() if isinstance(self.data.points, torch.Tensor) else self.data.points

        if self.is_mesh:
            name = f"{self.name_prefix}geom"
            # Handle both 'face' and 'faces' attributes
            faces = self.data.faces.cpu().numpy() if isinstance(self.data.faces, torch.Tensor) else self.data.faces

            self.geom = ps.register_surface_mesh(
                name=name,
                vertices=points,
                faces=faces,
                enabled=True
            )
            self.geom.set_color((0.8, 0.8, 0.8))

            # Add decimation visualization for meshes
            if hasattr(self.data, 'faces_to_decimate'):
                decimated_faces_mask = self.data.faces_to_decimate[self.data.original_to_decimated_faces].numpy()
                self.geom.add_scalar_quantity(
                    "decimated_faces",
                    decimated_faces_mask,
                    defined_on='faces',
                    enabled=True,
                    cmap=CMAP
                )
        else:
            name = f"{self.name_prefix}000_point_cloud"
            self.geom = ps.register_point_cloud(
                name=name,
                points=points,
                enabled=True
            )
            self.geom.set_radius(POINT_RADIUS)

    def add_normals(self):
        """Add normals to the visualization if available."""
        if not hasattr(self.data, 'normals'):
            return

        normals = self.data.normals.cpu().numpy() if isinstance(self.data.normals, torch.Tensor) else self.data.normals
        self.geom.add_vector_quantity(
            "normals",
            normals,
            enabled=True,
            radius=0.001 if not self.is_mesh else None
        )

    def add_scalar_fields(self):
        """Add both original and smoothed scalar fields to the visualization."""
        # Add original scalar fields (limit to 20)
        if hasattr(self.data, 'scalar_fields'):
            scalar_fields = self.data.scalar_fields.cpu() if isinstance(self.data.scalar_fields, torch.Tensor) else self.data.scalar_fields
            num_fields = min(scalar_fields.shape[1], 20)  # Limit to 20 fields

            for field_idx in range(num_fields):
                scalar_field = scalar_fields[:, field_idx].flatten().numpy()

                # Determine global min and max values for consistent color mapping
                scalar_min = np.min(scalar_field)
                scalar_max = np.max(scalar_field)

                # Create field display name with zero-padded index
                padded_idx = str(field_idx).zfill(3)
                field_name = f"SF_{padded_idx}_original"

                # Only enable the first original field for initial viewing
                is_enabled = field_idx == 0

                self.geom.add_scalar_quantity(
                    field_name,
                    scalar_field,
                    enabled=is_enabled,
                    vminmax=(scalar_min, scalar_max),
                    cmap=CMAP
                )

        # Add smoothed scalar fields (limit to 20)
        if hasattr(self.data, 'smoothed_scalar_fields'):
            smoothed_fields = self.data.smoothed_scalar_fields.cpu() if isinstance(self.data.smoothed_scalar_fields, torch.Tensor) else self.data.smoothed_scalar_fields
            num_fields = min(smoothed_fields.shape[1], 20)  # Limit to 20 fields

            for field_idx in range(num_fields):
                scalar_field = smoothed_fields[:, field_idx].flatten().numpy()

                # Determine global min and max values for consistent color mapping
                scalar_min = np.min(scalar_field)
                scalar_max = np.max(scalar_field)

                # Create field display name with zero-padded index
                padded_idx = str(field_idx).zfill(3)
                field_name = f"SF_{padded_idx}_smoothed"

                # Don't enable smoothed fields by default to avoid UI clutter
                is_enabled = False

                self.geom.add_scalar_quantity(
                    field_name,
                    scalar_field,
                    enabled=is_enabled,
                    vminmax=(scalar_min, scalar_max),
                    cmap=CMAP
                )

    def add_eigenfunctions(self):
        """Add eigenfunctions to the visualization if available.

        Returns:
            dict: Dictionary with keys 'pred', 'GT', 'Robust' containing lists of eigenfunction names
        """
        eigenfunction_names = {'pred': [], 'GT': [], 'Robust': []}

        def _add_eigenfunction(eigenfunction, eigen_name, max_abs_val, is_enabled):
            # # Use percentile-based clipping to handle outliers for better color mapping
            # # This prevents a few extreme values from distorting the entire color scale
            # percentile_low = 1.5  # Can be adjusted (1-5% typical)
            # percentile_high = 98.5  # Can be adjusted (95-99% typical)
            #
            # eigen_min = np.percentile(eigenfunction, percentile_low)
            # eigen_max = np.percentile(eigenfunction, percentile_high)
            #
            # # Ensure we don't have degenerate range
            # if eigen_max <= eigen_min:
            #     eigen_min = np.min(eigenfunction)
            #     eigen_max = np.max(eigenfunction)

            eigen_min = np.min(eigenfunction)
            eigen_max = np.max(eigenfunction)
            vminmax = (-max_abs_val, max_abs_val) if (max_abs_val is not None) else (eigen_min, eigen_max)

            # Register on geometry
            self.geom.add_scalar_quantity(
                eigen_name,
                eigenfunction,
                enabled=is_enabled,
                vminmax=vminmax,
                cmap=CMAP
            )

        def _add_eigenfunction_set(eigenvectors, eigenvalues, prefix, enable_first=False):
            """
            Unified logic to add a set of eigenfunctions to the visualization.

            Args:
                eigenvectors: Eigenvectors tensor/array [N, K]
                eigenvalues: Eigenvalues tensor/array [K]
                prefix: String prefix for naming (e.g., 'pred', 'GT', 'Robust')
                enable_first: Whether to enable the first eigenfunction by default

            Returns:
                list: List of eigenfunction names added
            """
            names = []

            # Convert to numpy if needed
            if hasattr(eigenvectors, 'cpu'):
                eigenvectors = eigenvectors.cpu().numpy()
            if hasattr(eigenvalues, 'cpu'):
                eigenvalues = eigenvalues.cpu().numpy()

            # Check if arrays are empty
            if eigenvectors.shape[0] == 0:
                print(f"Skipping {prefix} eigenfunctions visualization - empty array")
                return names

            max_eigenfunctions = eigenvectors.shape[1]
            for i in range(max_eigenfunctions):
                # Create padded index for ordering
                padded_idx = str(i).zfill(3)

                # Get eigenfunction for this index
                eigenfunction = eigenvectors[:, i]

                # Create display name with eigenvalue
                eigenvalue = eigenvalues[i]
                eigen_name = f"{padded_idx}_{prefix}_eigenfunction (lambda={eigenvalue:.4f})"

                # Enable first predicted eigenfunction by default, others disabled
                is_enabled = enable_first and (i == 0)
                # max_abs_val = 0.04 if i > 0 else None
                max_abs_val = None

                _add_eigenfunction(eigenfunction, eigen_name, max_abs_val, is_enabled)
                names.append(eigen_name)

            return names

        # Add predicted eigenfunctions (enable first one by default)
        if hasattr(self.data, 'pred_eigenvectors'):
            eigenfunction_names['pred'] = _add_eigenfunction_set(
                self.data.pred_eigenvectors,
                self.data.pred_eigenvalues,
                'pred',
                enable_first=True
            )

        # Add ground truth eigenfunctions
        if hasattr(self.data, 'gt_eigenvectors'):
            eigenfunction_names['GT'] = _add_eigenfunction_set(
                self.data.gt_eigenvectors,
                self.data.gt_eigenvalues,
                'GT',
                enable_first=False
            )

        # Add potential field (for Schrödinger operator)
        self._add_potential_field()

        # Add polynomial basis functions for visual comparison
        self._add_polynomial_basis()

        # Add orthogonalized polynomial basis (should match learned eigenfunctions better)
        self._add_orthogonal_polynomial_basis()

        # Add robust eigenfunctions
        if hasattr(self.data, 'robust_eigenvectors'):
            eigenfunction_names['Robust'] = _add_eigenfunction_set(
                self.data.robust_eigenvectors,
                self.data.robust_eigenvalues,
                'Robust',
                enable_first=False
            )

        if hasattr(self.data, 'gt_weights') and self.data.gt_weights.shape[0] > 0:
            _add_eigenfunction(
                eigenfunction=self.data.gt_weights.cpu().numpy(),
                eigen_name='000_AREAS_GT',
                max_abs_val=None,
                is_enabled=False,
            )

        _add_eigenfunction(
            eigenfunction=self.data.pred_weights.cpu().numpy(),
            eigen_name='000_AREAS_PRED',
            max_abs_val=None,
            is_enabled=True,
        )

        _add_eigenfunction(
            eigenfunction=self.data.robust_weights.cpu().numpy(),
            eigen_name='000_AREAS_ROBUST',
            max_abs_val=None,
            is_enabled=False,
        )

        return eigenfunction_names

    def _add_potential_field(self):
        """Add potential field V(x) if available (for Schrödinger operator)."""
        if not hasattr(self.data, 'potential') or self.data.potential is None:
            return

        try:
            potential = self.data.potential
            if isinstance(potential, torch.Tensor):
                potential = potential.cpu().numpy()

            if len(potential) == 0:
                return

            # Get operator info if available
            operator_type = getattr(self.data, 'operator_type', 'schrodinger')
            potential_type = getattr(self.data, 'potential_type', 'unknown')
            potential_strength = getattr(self.data, 'potential_strength', None)

            # Create name
            if potential_strength is not None:
                name = f"Potential_V ({potential_type}, β={potential_strength})"
            else:
                name = f"Potential_V ({potential_type})"

            self.geom.add_scalar_quantity(
                name,
                potential,
                enabled=True,
                vminmax=(potential.min(), potential.max()),
                cmap=CMAP
            )
            print(f"Added potential field: {name}")

        except Exception as e:
            print(f"Error adding potential field: {e}")

    def _add_polynomial_basis(self):
        """Add first 10 polynomial basis functions for visualization.

        Adds: constant, x, y, z, x², y², z², xy, xz, yz
        Names are prepended with '00_POLY_' to appear first in the list.
        """
        if not hasattr(self.data, 'points'):
            return

        try:
            pos = self.data.points
            if isinstance(pos, torch.Tensor):
                pos = pos.cpu().numpy()

            if len(pos) == 0:
                return

            num_nodes = pos.shape[0]
            dim = pos.shape[1]

            # Center and scale positions for better visualization
            p = pos - pos.mean(axis=0, keepdims=True)
            std = p.std()
            if std > 1e-8:
                p = p / std

            # Extract coordinates
            x = p[:, 0] if dim >= 1 else np.zeros(num_nodes)
            y = p[:, 1] if dim >= 2 else np.zeros(num_nodes)
            z = p[:, 2] if dim >= 3 else np.zeros(num_nodes)

            # Define basis functions with names
            basis_functions = [
                ("00_POLY_00_constant", np.ones(num_nodes)),
                ("00_POLY_01_x", x),
                ("00_POLY_02_y", y),
                ("00_POLY_03_z", z),
                ("00_POLY_04_x²", x * x),
                ("00_POLY_05_y²", y * y),
                ("00_POLY_06_z²", z * z),
                ("00_POLY_07_xy", x * y),
                ("00_POLY_08_xz", x * z),
                ("00_POLY_09_yz", y * z),
            ]

            # Add each basis function
            for name, basis in basis_functions:
                # Normalize for better visualization
                basis_normalized = basis / (np.abs(basis).max() + 1e-8)

                self.geom.add_scalar_quantity(
                    name,
                    basis_normalized,
                    enabled=False,
                    vminmax=(-1, 1),
                    cmap=CMAP
                )

            print(f"Added 10 polynomial basis functions (00_POLY_*)")

        except Exception as e:
            print(f"Error adding polynomial basis: {e}")

    def _add_orthogonal_polynomial_basis(self):
        """Add ORTHOGONALIZED polynomial basis functions for visualization.

        Unlike _add_polynomial_basis which shows raw monomials {1, x, y, z, x², ...},
        this shows the Gram-Schmidt orthogonalized version on the actual point cloud.

        The network learns an orthogonal basis, so this should match the learned
        eigenfunctions better than raw monomials.

        Names are prepended with '00_ORTHO_POLY_' to appear first in the list.
        """
        if not hasattr(self.data, 'points'):
            return

        try:
            pos = self.data.points
            if isinstance(pos, torch.Tensor):
                pos = pos.cpu().numpy()

            if len(pos) == 0:
                return

            num_nodes = pos.shape[0]
            dim = pos.shape[1]

            # Center and scale positions
            p = pos - pos.mean(axis=0, keepdims=True)
            std = p.std()
            if std > 1e-8:
                p = p / std

            # Extract coordinates
            x = p[:, 0] if dim >= 1 else np.zeros(num_nodes)
            y = p[:, 1] if dim >= 2 else np.zeros(num_nodes)
            z = p[:, 2] if dim >= 3 else np.zeros(num_nodes)

            # Build monomial basis matrix [N, 10]
            monomials = np.column_stack([
                np.ones(num_nodes),  # 1
                x,  # x
                y,  # y
                z,  # z
                x * x,  # x²
                y * y,  # y²
                z * z,  # z²
                x * y,  # xy
                x * z,  # xz
                y * z,  # yz
            ])

            # QR decomposition for orthogonalization
            # Q contains orthonormal columns spanning the same space as monomials
            Q, R = np.linalg.qr(monomials)

            # Store Q for later verification
            self._qr_polynomial_basis = Q

            # Names for orthogonalized basis
            names = [
                "00_ORTHO_POLY_00 (from constant)",
                "00_ORTHO_POLY_01 (from x)",
                "00_ORTHO_POLY_02 (from y)",
                "00_ORTHO_POLY_03 (from z)",
                "00_ORTHO_POLY_04 (from x²)",
                "00_ORTHO_POLY_05 (from y²)",
                "00_ORTHO_POLY_06 (from z²)",
                "00_ORTHO_POLY_07 (from xy)",
                "00_ORTHO_POLY_08 (from xz)",
                "00_ORTHO_POLY_09 (from yz)",
            ]

            # Add each orthogonalized basis function
            for i, name in enumerate(names):
                basis = Q[:, i]

                # Normalize for visualization
                max_val = np.abs(basis).max()
                if max_val > 1e-8:
                    basis_normalized = basis / max_val
                else:
                    basis_normalized = basis

                self.geom.add_scalar_quantity(
                    name,
                    basis_normalized,
                    enabled=False,
                    vminmax=(-1, 1),
                    cmap=CMAP
                )

            print(f"Added 10 orthogonalized polynomial basis functions (00_ORTHO_POLY_*)")

            # Verify subspace alignment with learned basis
            self._verify_polynomial_subspace_alignment()

        except Exception as e:
            import traceback
            print(f"Error adding orthogonal polynomial basis: {e}")
            traceback.print_exc()

    def _verify_polynomial_subspace_alignment(self):
        """
        Verify that learned basis spans the same subspace as polynomial basis.

        Computes:
        1. Principal angles between subspaces (all ~1 if same subspace)
        2. Reconstruction error of polynomial basis using learned basis
        3. Random baseline comparison (to check if task is trivial)
        4. Alignment with raw (non-orthogonal) monomials
        """
        if not hasattr(self, '_qr_polynomial_basis'):
            return

        if not hasattr(self.data, 'pred_eigenvectors'):
            return

        try:
            Q_poly = self._qr_polynomial_basis  # [N, 10]
            N = Q_poly.shape[0]

            pred_eig = self.data.pred_eigenvectors
            if hasattr(pred_eig, 'cpu'):
                pred_eig = pred_eig.cpu().numpy()

            # Use first K learned eigenvectors where K = num polynomial basis
            K = Q_poly.shape[1]
            if pred_eig.shape[1] < K:
                print(f"  Warning: Only {pred_eig.shape[1]} learned eigenvectors, need {K} for full comparison")
                K = pred_eig.shape[1]

            Q_learned = pred_eig[:, :K]  # [N, K]
            Q_poly_k = Q_poly[:, :K]  # [N, K]

            # Orthonormalize learned basis (in case it's not perfectly orthonormal)
            Q_learned, _ = np.linalg.qr(Q_learned)

            # === Method 1: Principal Angles ===
            # SVD of Q_learned^T @ Q_poly gives principal angles
            M = Q_learned.T @ Q_poly_k  # [K, K]
            U, S, Vt = np.linalg.svd(M)
            principal_cosines = np.clip(S, 0, 1)  # Cosines of principal angles

            print(f"\n{'=' * 60}")
            print(f"POLYNOMIAL SUBSPACE VERIFICATION (first {K} basis functions)")
            print(f"{'=' * 60}")
            print(f"Principal angle cosines (1.0 = perfect alignment):")
            print(f"  {np.array2string(principal_cosines, precision=4, separator=', ')}")
            print(f"  Mean: {principal_cosines.mean():.4f}, Min: {principal_cosines.min():.4f}")

            # === Method 2: Reconstruction Error ===
            # Project polynomial basis onto learned basis and back
            coeffs = Q_learned.T @ Q_poly_k  # [K, K]
            reconstructed = Q_learned @ coeffs  # [N, K]

            # Per-basis-function error
            errors = np.sqrt(((Q_poly_k - reconstructed) ** 2).sum(axis=0))
            mean_error = errors.mean()

            print(f"\nReconstruction error (0.0 = perfect):")
            print(f"  Per basis: {np.array2string(errors, precision=4, separator=', ')}")
            print(f"  Mean: {mean_error:.6f}")

            # === Method 3: Explained Variance ===
            # What fraction of polynomial basis is captured by learned basis?
            total_variance = (Q_poly_k ** 2).sum()
            unexplained_variance = ((Q_poly_k - reconstructed) ** 2).sum()
            explained_ratio = 1 - unexplained_variance / total_variance

            print(f"\nExplained variance ratio: {explained_ratio:.4f} (1.0 = perfect)")

            # === Method 4: Raw Monomial Basis Comparison ===
            print(f"\n{'-' * 60}")
            print(f"RAW MONOMIAL BASIS COMPARISON (non-orthogonal)")
            print(f"{'-' * 60}")

            # Rebuild raw monomials
            pos = self.data.points
            if hasattr(pos, 'cpu'):
                pos = pos.cpu().numpy()

            p = pos - pos.mean(axis=0, keepdims=True)
            std = p.std()
            if std > 1e-8:
                p = p / std

            x = p[:, 0]
            y = p[:, 1] if p.shape[1] >= 2 else np.zeros(N)
            z = p[:, 2] if p.shape[1] >= 3 else np.zeros(N)

            raw_monomials = np.column_stack([
                np.ones(N),  # 1
                x,  # x
                y,  # y
                z,  # z
                x * x,  # x²
                y * y,  # y²
                z * z,  # z²
                x * y,  # xy
                x * z,  # xz
                y * z,  # yz
            ])[:, :K]

            monomial_names = ["1", "x", "y", "z", "x²", "y²", "z²", "xy", "xz", "yz"][:K]

            # For each raw monomial, compute how well the learned basis reconstructs it
            print(f"Per-monomial reconstruction (using learned basis):")
            monomial_explained = []
            for i in range(K):
                monomial = raw_monomials[:, i]
                monomial_norm = np.linalg.norm(monomial)
                if monomial_norm > 1e-8:
                    monomial_normalized = monomial / monomial_norm
                else:
                    monomial_normalized = monomial

                # Project onto learned basis
                coeffs_mono = Q_learned.T @ monomial_normalized
                reconstructed_mono = Q_learned @ coeffs_mono

                # Explained variance for this monomial
                residual = monomial_normalized - reconstructed_mono
                explained = 1 - (residual ** 2).sum() / (monomial_normalized ** 2).sum()
                monomial_explained.append(explained)

                print(f"  {monomial_names[i]:>3}: {explained:.4f}")

            print(f"  Mean: {np.mean(monomial_explained):.4f}")

            # === Method 5: Random Baseline Comparison ===
            print(f"\n{'-' * 60}")
            print(f"RANDOM BASELINE COMPARISON")
            print(f"{'-' * 60}")

            # Generate multiple random baselines and average
            num_trials = 10
            random_explained_ratios = []
            random_principal_cosines_min = []
            random_monomial_explained = []

            for _ in range(num_trials):
                # Random orthonormal basis
                random_matrix = np.random.randn(N, K)
                Q_random, _ = np.linalg.qr(random_matrix)

                # Explained variance for random basis (QR poly)
                coeffs_random = Q_random.T @ Q_poly_k
                reconstructed_random = Q_random @ coeffs_random
                unexplained_random = ((Q_poly_k - reconstructed_random) ** 2).sum()
                explained_random = 1 - unexplained_random / total_variance
                random_explained_ratios.append(explained_random)

                # Principal angles for random basis
                M_random = Q_random.T @ Q_poly_k
                _, S_random, _ = np.linalg.svd(M_random)
                random_principal_cosines_min.append(np.clip(S_random, 0, 1).min())

                # Raw monomial explained variance for random basis
                mono_exp_trial = []
                for i in range(K):
                    monomial = raw_monomials[:, i]
                    monomial_norm = np.linalg.norm(monomial)
                    if monomial_norm > 1e-8:
                        monomial_normalized = monomial / monomial_norm
                    else:
                        monomial_normalized = monomial
                    coeffs_mono = Q_random.T @ monomial_normalized
                    reconstructed_mono = Q_random @ coeffs_mono
                    residual = monomial_normalized - reconstructed_mono
                    explained = 1 - (residual ** 2).sum() / (monomial_normalized ** 2).sum()
                    mono_exp_trial.append(explained)
                random_monomial_explained.append(np.mean(mono_exp_trial))

            mean_random_explained = np.mean(random_explained_ratios)
            std_random_explained = np.std(random_explained_ratios)
            mean_random_cosine_min = np.mean(random_principal_cosines_min)
            mean_random_monomial = np.mean(random_monomial_explained)

            print(f"Random basis (QR poly) explained variance: {mean_random_explained:.4f} ± {std_random_explained:.4f}")
            print(f"Random basis min principal cosine: {mean_random_cosine_min:.4f}")
            print(f"Random basis (raw monomial) mean explained: {mean_random_monomial:.4f}")

            print(f"\nLearned vs Random:")
            print(f"  Learned explained variance: {explained_ratio:.4f}")
            print(f"  Random explained variance:  {mean_random_explained:.4f}")
            print(f"  Improvement: {explained_ratio - mean_random_explained:.4f}")
            print(f"\n  Learned monomial explained: {np.mean(monomial_explained):.4f}")
            print(f"  Random monomial explained:  {mean_random_monomial:.4f}")
            print(f"  Improvement: {np.mean(monomial_explained) - mean_random_monomial:.4f}")

            # === Summary ===
            is_trivial = mean_random_explained > 0.95
            is_learned_good = explained_ratio > 0.99
            improvement = explained_ratio - mean_random_explained

            print(f"\n{'-' * 60}")
            if is_trivial:
                print(f"⚠ TRIVIAL TASK: Random basis achieves {mean_random_explained:.1%} explained variance")
                print(f"  The polynomial subspace is easy to find by chance in {N}D space with {K} dimensions")
            elif is_learned_good and improvement > 0.1:
                print(f"✓ NON-TRIVIAL: Learning improved explained variance by {improvement:.1%}")
                print(f"  Random: {mean_random_explained:.1%} → Learned: {explained_ratio:.1%}")
            elif is_learned_good:
                print(f"~ LEARNED WELL, but task may be easy (small improvement over random)")
            else:
                print(f"✗ LEARNING FAILED: Did not capture polynomial subspace")

            print(f"{'=' * 60}\n")

        except Exception as e:
            import traceback
            print(f"Error in polynomial subspace verification: {e}")
            traceback.print_exc()

    def _add_ground_truth_eigenvectors(self):
        """Add ground truth eigenvectors if available."""
        if not hasattr(self.data, 'gt_eigenvectors'):
            return

        try:
            gt_eigenvectors = self.data.gt_eigenvectors.cpu() if isinstance(self.data.gt_eigenvectors, torch.Tensor) else self.data.gt_eigenvectors
            gt_eigenvalues = self.data.gt_eigenvalues.cpu() if isinstance(self.data.gt_eigenvalues, torch.Tensor) else self.data.gt_eigenvalues

            # Normalize ground truth eigenvectors
            gt_eigenvectors = normalize(gt_eigenvectors, norm='l2', axis=0)

            max_eigenfunctions = gt_eigenvectors.shape[1]
            for i in range(max_eigenfunctions):
                # Create padded index for ordering
                padded_idx = str(i).zfill(3)

                # Get the GT eigenfunctions for this index
                gt_eigenfunction = gt_eigenvectors[:, i]

                # Determine global min and max values for consistent color mapping
                eigen_min = np.min(gt_eigenfunction)
                eigen_max = np.max(gt_eigenfunction)

                # Create eigenfunction display name with eigenvalue if available
                gt_eigenvalue = gt_eigenvalues[i]
                gt_eigen_name = f"{padded_idx}_GT_eigenfunction (lambda={gt_eigenvalue:.4f})"

                # Register eigenfunctions on geometries with consistent color range
                self.geom.add_scalar_quantity(
                    gt_eigen_name,
                    gt_eigenfunction,
                    enabled=False,
                    vminmax=(eigen_min, eigen_max),
                    cmap=CMAP
                )
        except Exception as e:
            print(f"Error processing ground truth eigenvectors: {e}")

    def _add_robust_eigenvectors(self):
        """Add robust laplacian eigenvectors if available."""
        if not hasattr(self.data, 'robust_eigenvectors'):
            return

        try:
            robust_eigenvectors = self.data.robust_eigenvectors.cpu() if isinstance(self.data.robust_eigenvectors, torch.Tensor) else self.data.robust_eigenvectors
            robust_eigenvalues = self.data.robust_eigenvalues.cpu() if isinstance(self.data.robust_eigenvalues, torch.Tensor) else self.data.robust_eigenvalues

            # Normalize robust eigenvectors
            robust_eigenvectors = normalize(robust_eigenvectors, norm='l2', axis=0)

            max_eigenfunctions = robust_eigenvectors.shape[1]
            for i in range(max_eigenfunctions):
                # Create padded index for ordering
                padded_idx = str(i).zfill(3)

                # Get the robust eigenfunctions for this index
                robust_eigenfunction = robust_eigenvectors[:, i]

                # Determine global min and max values for consistent color mapping
                eigen_min = np.min(robust_eigenfunction)
                eigen_max = np.max(robust_eigenfunction)

                # Create eigenfunction display name with eigenvalue if available
                robust_eigenvalue = robust_eigenvalues[i]
                robust_eigen_name = f"{padded_idx}_Robust_eigenfunction (lambda={robust_eigenvalue:.4f})"

                # Register eigenfunctions on geometries with consistent color range
                self.geom.add_scalar_quantity(
                    robust_eigen_name,
                    robust_eigenfunction,
                    enabled=False,
                    vminmax=(eigen_min, eigen_max),
                    cmap=CMAP
                )
        except Exception as e:
            print(f"Error processing robust eigenvectors: {e}")

    def add_reconstructions(self, prefix: str):
        """Add point cloud reconstructions using eigenvectors.

        CRITICAL: Reconstructions must ALWAYS use unnormalized eigenvectors.
        This function explicitly uses the _unnormalized versions regardless of display mode.
        """
        try:
            # ALWAYS use unnormalized eigenvectors for reconstruction
            unnormalized_attr = prefix + '_eigenvectors_unnormalized'
            if hasattr(self.data, unnormalized_attr):
                # Get unnormalized eigenvectors directly (no conversion needed)
                eigenvectors_unnormalized = getattr(self.data, unnormalized_attr)
                eigenvectors_unnormalized = eigenvectors_unnormalized.cpu() if isinstance(eigenvectors_unnormalized, torch.Tensor) else eigenvectors_unnormalized

                weights = self.data[prefix + '_weights'].cpu() if isinstance(self.data[prefix + '_weights'], torch.Tensor) else self.data[prefix + '_weights']

                # Reconstruct 3D point cloud using increasing number of eigenvectors
                coord_functions = self.data.points.cpu() if isinstance(self.data.points, torch.Tensor) else self.data.points  # [N, 3]
                max_eigenvectors = eigenvectors_unnormalized.shape[1]

                # Use unnormalized eigenvectors directly - they're already in the correct form
                reconstructed_coord_functions, _ = utils.project_functions_unnormalized(
                    eigenvectors_basis=eigenvectors_unnormalized,
                    scalar_functions=coord_functions,
                    weights=weights,
                    max_eigenvectors=max_eigenvectors
                )

                # Register all reconstructions in one go using a custom vectorized registration function
                self._register_all_reconstructions(
                    all_recon_points=reconstructed_coord_functions.numpy(),
                    orig_points=coord_functions.numpy(),
                    levels=torch.arange(1, max_eigenvectors + 1).numpy(),
                    prefix=prefix
                )
        except Exception as e:
            print(f"Error adding reconstructions for {prefix}: {e}")

    def _register_all_reconstructions(self, all_recon_points, orig_points, levels, prefix):
        """Register all reconstruction levels."""
        for i, level in enumerate(levels):
            try:
                self._register_reconstruction(
                    recon_points=all_recon_points[i],
                    orig_points=orig_points,
                    num_ev=level.item(),
                    prefix=prefix
                )
            except Exception as e:
                print(f"Error registering reconstruction level {level} for {prefix}: {e}")

    def _register_reconstruction(self, recon_points, orig_points, num_ev, prefix):
        """Register a reconstruction with Polyscope."""
        try:
            padded_num_ev = str(num_ev).zfill(3)
            name = f"{padded_num_ev}_eigvecs_{prefix}"

            # Static gray color for all reconstructions
            gray_color = (140 / 255, 140 / 255, 140 / 255)

            # Register reconstructed point cloud
            if self.is_mesh:
                # Get faces
                faces = self.data.faces.cpu().numpy() if isinstance(self.data.faces, torch.Tensor) else self.data.faces

                recon_geom = ps.register_surface_mesh(
                    name=name,
                    vertices=recon_points,
                    faces=faces,
                    enabled=False
                )
                recon_geom.set_color(gray_color)
            else:
                recon_geom = ps.register_point_cloud(
                    name=name,
                    points=recon_points,
                    enabled=False
                )
                recon_geom.set_radius(POINT_RADIUS)
                recon_geom.set_color(gray_color)

            # Calculate and visualize reconstruction error
            error = np.linalg.norm(recon_points - orig_points, axis=1)
            recon_geom.add_scalar_quantity(
                name=name,
                values=error,
                enabled=False,  # Disabled by default to show gray color
                cmap=CMAP
            )
        except Exception as e:
            print(f"Error in _register_reconstruction: {e}")


class GeometryVisualizer:
    """Class for visualizing geometry data with Polyscope."""

    def __init__(self, data: Data, k: int):
        """
        Initialize geometry visualizer.

        Args:
            data: Geometry data
            k: Number of neighbors for graph construction
        """
        self.data = data
        self.k = k
        self.mesh_visualizer = MeshVisualizer(data, k=k)

    def visualize(self):
        """Visualize the geometry with all available data."""
        # Register basic geometry
        self.mesh_visualizer.register_geometry()

        # Add normals if available
        self.mesh_visualizer.add_normals()

        # Add scalar fields
        self.mesh_visualizer.add_scalar_fields()

        # Add eigenfunctions
        eigenfunction_names = self.mesh_visualizer.add_eigenfunctions()

        # Add reconstructions
        self.mesh_visualizer.add_reconstructions(prefix='gt')
        self.mesh_visualizer.add_reconstructions(prefix='pred')
        self.mesh_visualizer.add_reconstructions(prefix='robust')

        return eigenfunction_names


class DatasetController:
    """Controls dataset with direct indexing and in-place k updates."""

    def __init__(self, cfg: DictConfig, model):
        self.cfg = cfg
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create dataset once
        self.dataset = self._create_dataset()
        self.total_items = len(self.dataset)

        # Simple index tracking
        self.current_idx = 0

        # k-NN state for predicted Laplacian - initialize before creating dataset
        self.current_k = self._get_current_k()
        self.pending_k = self.current_k

        # k-NN state for robust Laplacian
        self.current_robust_k = 30  # Default value
        self.pending_robust_k = self.current_robust_k

        # Scalar field smoother iterations state
        self.current_smoother_iterations = self._get_current_smoother_iterations()
        self.pending_smoother_iterations = self.current_smoother_iterations

        # Scalar field smoother sigma state
        self.current_smoother_sigma = self._get_current_smoother_sigma()
        self.pending_smoother_sigma = self.current_smoother_sigma

        # IMPORTANT: Set model's k to match initial k before any predictions
        self.set_k(self.current_k)

    def _create_dataset(self):
        """Create dataset once - no recreation needed."""
        # print(f"Creating dataset with initial k={self.current_k}")

        # Ensure no shuffling for consistent indexing
        if hasattr(self.cfg.data_module.module, 'train_dataset_specification'):
            if hasattr(self.cfg.data_module.module.train_dataset_specification, 'shuffle'):
                self.cfg.data_module.module.train_dataset_specification.shuffle = False
                print("Disabled shuffle for train dataset to maintain consistent indexing")

        # Skip validation dataset modifications - not needed for visualization

        data_module = hydra.utils.instantiate(self.cfg.data_module.module)

        # Get the actual dataset, not the dataloader
        dataset = data_module.train_dataloader().dataset

        print(f"Dataset created with {len(dataset)} items")
        return dataset

    def set_k(self, new_k):
        """Set new k value on the model's knn_graph_config."""
        new_k = min(new_k, 100)
        try:
            if hasattr(self.model, '_knn_graph_config'):
                old_k = getattr(self.model._knn_graph_config, 'k', 'unknown')
                self.model._knn_graph_config.k = new_k
                print(f"Updated model._knn_graph_config.k: {old_k} -> {new_k}")
            else:
                print(f"Warning: Model doesn't have _knn_graph_config attribute")

            self.current_k = new_k
            self.pending_k = new_k
            print(f"Successfully updated k to {new_k}")

        except Exception as e:
            print(f"Error setting k to {new_k}: {e}")

    def _get_current_k(self):
        """Get current knn value from the model."""
        try:
            if hasattr(self.model, '_knn_graph_config'):
                knn_graph_config = self.model._knn_graph_config
                if hasattr(knn_graph_config, 'k'):
                    k = knn_graph_config.sample_k(vertices=self.get_current_item()[0].points)
                    # k = knn_graph_config.k
                    # If it's a list/tuple, return the first value for display
                    if isinstance(k, (list, tuple)):
                        return k[0] if len(k) > 0 else 1
                    else:
                        return k
            return 20  # Default value
        except Exception as e:
            print(f"Error getting smoother iterations: {e}")
            return 20

    def _get_current_smoother_iterations(self):
        """Get current smoother iterations value from the model."""
        try:
            if hasattr(self.model, '_scalar_field_smoother'):
                smoother = self.model._scalar_field_smoother
                if hasattr(smoother, 'iterations'):
                    iterations = smoother.iterations
                    # If it's a list/tuple, return the first value for display
                    if isinstance(iterations, (list, tuple)):
                        return iterations[0] if len(iterations) > 0 else 1
                    else:
                        return iterations
            return 1  # Default value
        except Exception as e:
            print(f"Error getting smoother iterations: {e}")
            return 1

    def _get_current_smoother_sigma(self):
        """Get current smoother sigma value from the model."""
        try:
            if hasattr(self.model, '_scalar_field_smoother'):
                smoother = self.model._scalar_field_smoother
                if hasattr(smoother, 'sigma'):
                    sigma = smoother.sigma
                    # If it's a list/tuple, return the first value for display
                    if isinstance(sigma, (list, tuple)):
                        return sigma[0] if len(sigma) > 0 else 0.01
                    else:
                        return sigma
            return 0.01  # Default value
        except Exception as e:
            print(f"Error getting smoother sigma: {e}")
            return 0.01

    def set_smoother_iterations(self, new_iterations):
        """Set new smoother iterations value on the model."""
        try:
            if hasattr(self.model, '_scalar_field_smoother'):
                smoother = self.model._scalar_field_smoother
                if hasattr(smoother, 'iterations'):
                    old_iterations = smoother.iterations
                    smoother.iterations = new_iterations
                    print(f"Updated model._scalar_field_smoother.iterations: {old_iterations} -> {new_iterations}")
                else:
                    print("Warning: Smoother doesn't have iterations attribute")
            else:
                print("Warning: Model doesn't have _scalar_field_smoother attribute")

            self.current_smoother_iterations = new_iterations
            self.pending_smoother_iterations = new_iterations
            print(f"Successfully updated smoother iterations to {new_iterations}")

        except Exception as e:
            print(f"Error setting smoother iterations to {new_iterations}: {e}")

    def set_smoother_sigma(self, new_sigma):
        """Set new smoother sigma value on the model."""
        try:
            if hasattr(self.model, '_scalar_field_smoother'):
                smoother = self.model._scalar_field_smoother
                if hasattr(smoother, 'sigma'):
                    old_sigma = smoother.sigma
                    smoother.sigma = new_sigma
                    print(f"Updated model._scalar_field_smoother.sigma: {old_sigma} -> {new_sigma}")
                else:
                    print("Warning: Smoother doesn't have sigma attribute")
            else:
                print("Warning: Model doesn't have _scalar_field_smoother attribute")

            self.current_smoother_sigma = new_sigma
            self.pending_smoother_sigma = new_sigma
            print(f"Successfully updated smoother sigma to {new_sigma}")

        except Exception as e:
            print(f"Error setting smoother sigma to {new_sigma}: {e}")

    def set_robust_k(self, new_robust_k):
        """Set new robust k value."""
        try:
            self.current_robust_k = new_robust_k
            self.pending_robust_k = new_robust_k
            print(f"Successfully updated robust k to {new_robust_k}")

        except Exception as e:
            print(f"Error setting robust k to {new_robust_k}: {e}")

    def apply_k_change(self):
        """Apply pending k change without changing position."""
        changed = False

        if self.pending_k != self.current_k:
            self.set_k(self.pending_k)
            changed = True

        if self.pending_robust_k != self.current_robust_k:
            self.set_robust_k(self.pending_robust_k)
            changed = True

        if self.pending_smoother_iterations != self.current_smoother_iterations:
            self.set_smoother_iterations(self.pending_smoother_iterations)
            changed = True

        if self.pending_smoother_sigma != self.current_smoother_sigma:
            self.set_smoother_sigma(self.pending_smoother_sigma)
            changed = True

        return changed

    def get_current_item(self):
        """Get current item by direct indexing."""
        if 0 <= self.current_idx < self.total_items:
            try:
                data = self.dataset[self.current_idx]
                return data.to(self.device), self.current_idx
            except Exception as e:
                print(f"Error loading item {self.current_idx}: {e}")
                return None, None
        return None, None

    def next_item(self):
        """Advance to next item."""
        if self.current_idx < self.total_items - 1:
            self.current_idx += 1
            return True
        else:
            print(f"Already at last item ({self.current_idx + 1}/{self.total_items})")
            return False

    def prev_item(self):
        """Go to previous item."""
        if self.current_idx > 0:
            self.current_idx -= 1
            return True
        else:
            print(f"Already at first item (1/{self.total_items})")
            return False

    def goto_item(self, idx):
        """Jump to specific item."""
        if 0 <= idx < self.total_items:
            self.current_idx = idx
            return True
        else:
            print(f"Invalid index {idx}. Valid range: 0-{self.total_items - 1}")
            return False

    def reset_to_first(self):
        """Reset to first item."""
        self.current_idx = 0

    def get_status(self):
        """Get current status."""
        return {
            'current_k': self.current_k,
            'pending_k': self.pending_k,
            'current_robust_k': self.current_robust_k,
            'pending_robust_k': self.pending_robust_k,
            'current_smoother_iterations': self.current_smoother_iterations,
            'pending_smoother_iterations': self.pending_smoother_iterations,
            'current_smoother_sigma': self.current_smoother_sigma,
            'pending_smoother_sigma': self.pending_smoother_sigma,
            'total_items': self.total_items,
            'current_idx': self.current_idx,
            'has_pending_changes': (self.pending_k != self.current_k or
                                    self.pending_robust_k != self.current_robust_k or
                                    self.pending_smoother_iterations != self.current_smoother_iterations or
                                    self.pending_smoother_sigma != self.current_smoother_sigma)
        }


def compute_reconstruction_metrics(pred_evecs, robust_evecs):
    """Compute quality metrics comparing predicted and robust eigenvectors."""
    try:
        # Use the utility function to compute cosine similarities
        cosine_similarities = utils.compute_eigenvector_cosine_similarities(pred_evecs, robust_evecs)

        return {
            'cosine_similarities': cosine_similarities,
            'mean_cosine_similarity': torch.mean(cosine_similarities),
            'min_cosine_similarity': torch.min(cosine_similarities)
        }
    except Exception as e:
        print(f"Error computing reconstruction metrics: {e}")
        return {
            'cosine_similarities': np.array([float('nan')]),
            'mean_cosine_similarity': float('nan'),
            'min_cosine_similarity': float('nan')
        }


def crop_transparent_background(image_path):
    """
    Crop a PNG image to the minimal bounding box of non-transparent pixels.

    Args:
        image_path: Path to the PNG image file
    """
    try:
        from PIL import Image
        import numpy as np

        # Open the image
        img = Image.open(image_path)

        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Get the alpha channel
        alpha = np.array(img)[:, :, 3]

        # Find all non-transparent pixels
        non_transparent = np.where(alpha > 0)

        # Check if there are any non-transparent pixels
        if len(non_transparent[0]) == 0:
            # Image is completely transparent, don't crop
            return

        # Get bounding box
        min_y = non_transparent[0].min()
        max_y = non_transparent[0].max()
        min_x = non_transparent[1].min()
        max_x = non_transparent[1].max()

        # Crop the image
        cropped_img = img.crop((min_x, min_y, max_x + 1, max_y + 1))

        # Save the cropped image, overwriting the original
        cropped_img.save(image_path)

    except Exception as e:
        print(f"Warning: Could not crop image {image_path}: {e}")


def create_comparison_image(data, idx, screenshot_dir="screenshots", pred_only=False, items_per_image=10):
    """
    Create a comparison image concatenating GT and Pred screenshots.

    Layout:
    - If pred_only=False: 2 columns: GT (left) | Pred (right) with cosine similarity text
    - If pred_only=True: 1 column: Pred only (no GT, no cosine similarity text)
    - Rows for eigenvectors: 0, 1, 2, 3, 4, 5, 10, 20, 30
    - Rows for reconstructions: k=10, k=20, k=30

    Args:
        data: Data object containing metrics
        idx: Index of current data item
        screenshot_dir: Directory containing screenshots
        pred_only: If True, create single-column layout with pred only
        items_per_image: If None, all items in one image. If int (e.g., 5), split into multiple images
                        with up to items_per_image items each (creates page_1, page_2, etc.)
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        from pathlib import Path

        screenshot_path = Path(screenshot_dir)

        # Define which eigenvectors to include (skip 0th eigenvector)
        eigen_indices = list(range(1, 100))
        recon_k_values = []

        # eigen_indices = [1, 2, 3, 4, 5, 6, 10, 25, 40]
        # recon_k_values = [5, 25, 45]

        # eigen_indices = [1, 2, 10, 30]
        # recon_k_values = [10, 50]

        # Fixed dimensions for consistent layout
        target_img_width = 1024  # Fixed width for each image
        target_img_height = 1024  # Fixed height for each image
        gap = 320 if not pred_only else 0  # No gap needed in single column mode

        mode_text = "pred-only" if pred_only else "comparison"
        print(f"\nCreating {mode_text} image...")
        print(f"  Target image size: {target_img_width}x{target_img_height}")
        if not pred_only:
            print(f"  Gap between columns: {gap}px")

        # Check if we have cosine similarities (only relevant for comparison mode)
        has_cosine_sims = False
        if not pred_only:
            has_cosine_sims = hasattr(data, 'metrics') and 'cosine_similarities' in data.metrics
            if has_cosine_sims:
                cosine_sims = data.metrics['cosine_similarities']
                # Convert to numpy if tensor
                if hasattr(cosine_sims, 'cpu'):
                    cosine_sims = cosine_sims.cpu().numpy()

        # Collect image pairs (or single images in pred_only mode)
        image_pairs = []

        # Eigenvector comparisons
        for eigen_idx in eigen_indices:
            pred_path = screenshot_path / f"item_{idx:04d}_pred_eigen{eigen_idx:03d}.png"

            if pred_only:
                # Pred-only mode: just collect pred images
                if pred_path.exists():
                    image_pairs.append(('eigenfunction', eigen_idx, None, pred_path, None))
                else:
                    print(f"  Warning: Pred eigenfunction {eigen_idx} screenshot not found")
            else:
                # Comparison mode: collect GT and pred pairs
                gt_path = screenshot_path / f"item_{idx:04d}_gt_eigen{eigen_idx:03d}.png"

                if gt_path.exists() and pred_path.exists():
                    cosine_sim = cosine_sims[eigen_idx] if has_cosine_sims and eigen_idx < len(cosine_sims) else None
                    image_pairs.append(('eigenfunction', eigen_idx, gt_path, pred_path, cosine_sim))
                else:
                    if not gt_path.exists():
                        print(f"  Warning: GT eigenfunction {eigen_idx} screenshot not found")
                    if not pred_path.exists():
                        print(f"  Warning: Pred eigenfunction {eigen_idx} screenshot not found")

        # Reconstruction comparisons
        for k in recon_k_values:
            pred_path = screenshot_path / f"item_{idx:04d}_pred_recon_k{k:03d}.png"

            if pred_only:
                # Pred-only mode: just collect pred images
                if pred_path.exists():
                    image_pairs.append(('reconstruction', k, None, pred_path, None))
                else:
                    print(f"  Warning: Pred reconstruction k={k} screenshot not found")
            else:
                # Comparison mode: collect GT and pred pairs
                gt_path = screenshot_path / f"item_{idx:04d}_gt_recon_k{k:03d}.png"

                if gt_path.exists() and pred_path.exists():
                    image_pairs.append(('reconstruction', k, gt_path, pred_path, None))
                else:
                    if not gt_path.exists():
                        print(f"  Warning: GT reconstruction k={k} screenshot not found")
                    if not pred_path.exists():
                        print(f"  Warning: Pred reconstruction k={k} screenshot not found")

        if len(image_pairs) == 0:
            print("  No image pairs found for comparison")
            return

        print(f"  Found {len(image_pairs)} image pairs to concatenate")

        # Load and process images
        rows = []
        max_width = 0

        for img_type, index, gt_path, pred_path, cosine_sim in image_pairs:
            # Load images
            if pred_only:
                # Pred-only mode: only load pred image
                pred_img = Image.open(pred_path).convert('RGBA')
                gt_img = None
            else:
                # Comparison mode: load both GT and pred
                gt_img = Image.open(gt_path).convert('RGBA')
                pred_img = Image.open(pred_path).convert('RGBA')

            # Resize images to fixed dimensions while maintaining aspect ratio
            def resize_to_fixed(img, target_width, target_height):
                """Resize image to fit within target dimensions while maintaining aspect ratio, then pad to exact size."""
                # Calculate scaling factor to fit within target dimensions
                width_ratio = target_width / img.width
                height_ratio = target_height / img.height
                scale_factor = min(width_ratio, height_ratio)

                # Calculate new dimensions
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)

                # Resize image
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)

                # Create canvas with target dimensions and center the resized image
                canvas = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 0))
                paste_x = (target_width - new_width) // 2
                paste_y = (target_height - new_height) // 2
                canvas.paste(img_resized, (paste_x, paste_y))

                return canvas

            def crop_to_content(img):
                """Crop image to its non-transparent content, but preserve full height for landscape images."""
                bbox = img.getbbox()
                if bbox:
                    left, top, right, bottom = bbox

                    # Always crop horizontally (left and right)
                    # For vertical cropping, check if image is landscape (width > height after initial resize)
                    # If landscape, keep the full vertical extent (top=0, bottom=img.height) to maintain consistent row heights

                    # Determine if this is a landscape-oriented content
                    content_width = right - left
                    content_height = bottom - top

                    if content_width > content_height:
                        # Landscape: keep full vertical extent, only crop horizontal
                        return img.crop((left, 0, right, img.height))
                    else:
                        # Portrait or square: crop normally on all sides
                        return img.crop(bbox)
                return img

            # Process pred image
            pred_img = resize_to_fixed(pred_img, target_img_width, target_img_height)
            pred_img = crop_to_content(pred_img)

            if pred_only:
                # Single-column mode: just use pred image as the row
                row_img = pred_img
                row_width = pred_img.width
                row_height = pred_img.height
            else:
                # Comparison mode: process GT image and create two-column layout
                gt_img = resize_to_fixed(gt_img, target_img_width, target_img_height)
                gt_img = crop_to_content(gt_img)

                # All images now have variable dimensions based on their content
                # Make them the same height for consistent rows
                max_height = max(gt_img.height, pred_img.height)
                if gt_img.height < max_height:
                    new_gt = Image.new('RGBA', (gt_img.width, max_height), (255, 255, 255, 0))
                    new_gt.paste(gt_img, (0, (max_height - gt_img.height) // 2))
                    gt_img = new_gt
                if pred_img.height < max_height:
                    new_pred = Image.new('RGBA', (pred_img.width, max_height), (255, 255, 255, 0))
                    new_pred.paste(pred_img, (0, (max_height - pred_img.height) // 2))
                    pred_img = new_pred

                # Create row with actual gap between content
                row_width = gt_img.width + gap + pred_img.width
                row_height = max_height
                row_img = Image.new('RGBA', (row_width, row_height), (255, 255, 255, 0))  # Transparent background
                row_img.paste(gt_img, (0, 0))
                row_img.paste(pred_img, (gt_img.width + gap, 0))

                # Add text overlay in the gap between columns if cosine similarity available
                if cosine_sim is not None:
                    draw = ImageDraw.Draw(row_img)

                    # Try to use a nice font, fall back to default if not available
                    try:
                        font = ImageFont.truetype('arialbd.ttf', 140)
                    except:
                        # Use default font but try to scale it
                        font = ImageFont.load_default(size=140)

                    # Format cosine similarity value - just the number, 2 decimal places
                    cosine_value = cosine_sim.item() if hasattr(cosine_sim, 'item') else float(cosine_sim)
                    text = f"{cosine_value:.2f}"

                    # Get text bounding box
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Position text in the center of the gap (between gt_img and pred_img)
                    text_x = gt_img.width + (gap - text_width) // 2
                    text_y = (row_height - text_height) // 2  # Vertically centered in the row

                    # Draw text directly with no background - black color
                    print(f"    Drawing text '{text}' at ({text_x}, {text_y}) in the gap between columns")
                    draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font=font)

            rows.append(row_img)
            max_width = max(max_width, row_width)

        # Split rows into pages if items_per_image is specified
        if items_per_image is None:
            # Original behavior: all rows in one image
            row_pages = [rows]
        else:
            # Split rows into chunks of items_per_image
            row_pages = [rows[i:i + items_per_image] for i in range(0, len(rows), items_per_image)]

        num_pages = len(row_pages)
        print(f"  Creating {num_pages} comparison image(s) (items_per_image={items_per_image})")

        # Process each page
        for page_num, page_rows in enumerate(row_pages, start=1):
            # Concatenate rows in this page vertically with larger gap
            row_gap = 60
            total_height = sum(img.height for img in page_rows) + row_gap * (len(page_rows) - 1)
            final_img = Image.new('RGBA', (max_width, total_height), (255, 255, 255, 0))  # Transparent background

            y_offset = 0
            for row_img in page_rows:
                # Center the row if it's narrower than max_width
                x_offset = (max_width - row_img.width) // 2
                final_img.paste(row_img, (x_offset, y_offset), row_img)
                y_offset += row_img.height + row_gap

            # Crop left and right transparent space
            # Find the bounding box of non-transparent pixels
            bbox = final_img.getbbox()
            if bbox:
                # Crop only left and right, keep top and bottom as is
                left, top, right, bottom = bbox
                final_img = final_img.crop((left, 0, right, total_height))
                print(f"  Page {page_num}: Cropped transparent space: left={left}px, right={max_width - right}px")

            # Save the comparison image with appropriate filename
            if num_pages == 1:
                # Single page: use original naming
                comparison_path = screenshot_path / f"item_{idx:04d}_comparison.png"
            else:
                # Multiple pages: add page number
                comparison_path = screenshot_path / f"item_{idx:04d}_comparison_page_{page_num}.png"

            final_img.save(comparison_path)
            print(f"  Saved comparison image (page {page_num}/{num_pages}) to: {comparison_path}")


    except Exception as e:
        print(f"Warning: Could not create comparison image: {e}")
        import traceback
        traceback.print_exc()


def save_eigenvalues_and_metrics_csv(data, save_path="eigenvalues_and_metrics.csv"):
    """
    Save eigenvalues and cosine similarities to a CSV file for analysis.

    CSV Structure:
    - Column 1: Eigenfunction index
    - Columns 2-4: Eigenvalues (GT, Pred, Robust)
    - Columns 5-6: Cosine similarities (Normalized, Unnormalized)

    Args:
        data: Data object containing eigenvalues and metrics
        save_path: Path to save the CSV file
    """
    try:
        import csv

        # Extract eigenvalues
        gt_eigenvalues = None
        pred_eigenvalues = None
        robust_eigenvalues = None

        if hasattr(data, 'gt_eigenvalues') and data.gt_eigenvalues is not None:
            gt_eigenvalues = data.gt_eigenvalues.cpu().numpy() if isinstance(data.gt_eigenvalues, torch.Tensor) else data.gt_eigenvalues
            # Check if empty
            if isinstance(gt_eigenvalues, np.ndarray) and gt_eigenvalues.shape[0] == 0:
                gt_eigenvalues = None

        if hasattr(data, 'pred_eigenvalues') and data.pred_eigenvalues is not None:
            pred_eigenvalues = data.pred_eigenvalues.cpu().numpy() if isinstance(data.pred_eigenvalues, torch.Tensor) else data.pred_eigenvalues

        if hasattr(data, 'robust_eigenvalues') and data.robust_eigenvalues is not None:
            robust_eigenvalues = data.robust_eigenvalues.cpu().numpy() if isinstance(data.robust_eigenvalues, torch.Tensor) else data.robust_eigenvalues

        # Extract cosine similarities
        cosine_sim_normalized = None
        cosine_sim_unnormalized = None

        if hasattr(data, 'metrics_normalized') and 'cosine_similarities' in data.metrics_normalized:
            cosine_sim_normalized = data.metrics_normalized['cosine_similarities']
            if hasattr(cosine_sim_normalized, 'cpu'):
                cosine_sim_normalized = cosine_sim_normalized.cpu().numpy()

        if hasattr(data, 'metrics_unnormalized') and 'cosine_similarities' in data.metrics_unnormalized:
            cosine_sim_unnormalized = data.metrics_unnormalized['cosine_similarities']
            if hasattr(cosine_sim_unnormalized, 'cpu'):
                cosine_sim_unnormalized = cosine_sim_unnormalized.cpu().numpy()

        # Determine the maximum length (number of eigenfunctions)
        max_length = 0
        if gt_eigenvalues is not None:
            max_length = max(max_length, len(gt_eigenvalues))
        if pred_eigenvalues is not None:
            max_length = max(max_length, len(pred_eigenvalues))
        if robust_eigenvalues is not None:
            max_length = max(max_length, len(robust_eigenvalues))
        if cosine_sim_normalized is not None:
            max_length = max(max_length, len(cosine_sim_normalized))
        if cosine_sim_unnormalized is not None:
            max_length = max(max_length, len(cosine_sim_unnormalized))

        if max_length == 0:
            print("No data available to save to CSV")
            return

        # Write CSV file
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            header = ['Index', 'Eigenvalue_GT', 'Eigenvalue_Pred', 'Eigenvalue_Robust',
                      'Cosine_Similarity_Normalized', 'Cosine_Similarity_Unnormalized']
            writer.writerow(header)

            # Write data rows
            for i in range(max_length):
                row = [i]

                # Add eigenvalues
                row.append(gt_eigenvalues[i] if gt_eigenvalues is not None and i < len(gt_eigenvalues) else '')
                row.append(pred_eigenvalues[i] if pred_eigenvalues is not None and i < len(pred_eigenvalues) else '')
                row.append(robust_eigenvalues[i] if robust_eigenvalues is not None and i < len(robust_eigenvalues) else '')

                # Add cosine similarities
                row.append(cosine_sim_normalized[i] if cosine_sim_normalized is not None and i < len(cosine_sim_normalized) else '')
                row.append(cosine_sim_unnormalized[i] if cosine_sim_unnormalized is not None and i < len(cosine_sim_unnormalized) else '')

                writer.writerow(row)

        print(f"Eigenvalues and metrics saved to CSV: {save_path}")

    except Exception as e:
        print(f"Warning: Could not save eigenvalues and metrics to CSV: {e}")
        import traceback
        traceback.print_exc()


def save_eigenvalue_plot(data, save_path="eigenvalue_comparison.png"):
    """
    Save the eigenvalue comparison plot to a file.

    Args:
        data: Data object containing eigenvalues
        save_path: Path to save the plot image
    """
    try:
        # Create a new figure for this plot with extra width for legend
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Check if we have eigenvalues to plot
        has_gt = hasattr(data, 'gt_eigenvalues') and data.gt_eigenvalues is not None
        has_pred = hasattr(data, 'pred_eigenvalues') and data.pred_eigenvalues is not None
        # has_robust = hasattr(data, 'robust_eigenvalues') and data.robust_eigenvalues is not None
        has_robust = False

        if not (has_gt or has_pred or has_robust):
            ax.text(0.5, 0.5, 'No eigenvalues available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return

        # Plot each set of eigenvalues if available (skip first eigenvalue which is 0)
        if has_gt:
            gt_eigenvalues = data.gt_eigenvalues.cpu().numpy() if isinstance(data.gt_eigenvalues, torch.Tensor) else data.gt_eigenvalues
            # Skip the first eigenvalue (index 0)
            if len(gt_eigenvalues) > 1:
                indices_gt = np.arange(1, len(gt_eigenvalues))
                ax.plot(indices_gt, gt_eigenvalues[1:], 'b-o', label='Cot. Lap.',
                        markersize=6, linewidth=2.5, alpha=0.8)

        if has_pred:
            pred_eigenvalues = data.pred_eigenvalues.cpu().numpy() if isinstance(data.pred_eigenvalues, torch.Tensor) else data.pred_eigenvalues
            # Skip the first eigenvalue (index 0)
            if len(pred_eigenvalues) > 1:
                indices_pred = np.arange(1, len(pred_eigenvalues))
                ax.plot(indices_pred, pred_eigenvalues[1:], 'r-s', label='Ours',
                        markersize=6, linewidth=2.5, alpha=0.8)

        if has_robust:
            robust_eigenvalues = data.robust_eigenvalues.cpu().numpy() if isinstance(data.robust_eigenvalues, torch.Tensor) else data.robust_eigenvalues
            # Skip the first eigenvalue (index 0)
            if len(robust_eigenvalues) > 1:
                indices_robust = np.arange(1, len(robust_eigenvalues))
                ax.plot(indices_robust, robust_eigenvalues[1:], 'g-^', label='Robust',
                        markersize=6, linewidth=2.5, alpha=0.8)

        # Set logarithmic scale for y-axis (eigenvalues typically decay exponentially)
        ax.set_yscale('log')

        # Labels and title with larger fonts
        ax.set_xlabel('Eigenvalue Index', fontsize=19, fontweight='bold')
        ax.set_ylabel('Eigenvalue (log scale)', fontsize=19, fontweight='bold')
        ax.set_title('Eigenvalue Comparison: GT vs Predicted vs Robust', fontsize=18, fontweight='bold')

        # Increase tick label font size
        ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=10)
        ax.tick_params(axis='both', which='minor', labelsize=16, width=1.5, length=8)

        # Thicker spines (edges)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # Grid with thicker lines
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=3)

        # Legend positioned outside the plot area to the right
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                  framealpha=0.95, fontsize=14, edgecolor='black',
                  fancybox=False, shadow=False, frameon=True)

        # Adjust layout to prevent legend cutoff
        fig.tight_layout()

        # Save the figure with extra space for legend
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Eigenvalue comparison plot saved to: {save_path}")

        # Close the figure to free memory
        plt.close(fig)

    except Exception as e:
        print(f"Warning: Could not save eigenvalue plot: {e}")


def save_eigenfunction_screenshots(data, idx, eigenfunction_names, geom_name="000_point_cloud", screenshot_dir="screenshots", num_eigenvectors=100, pred_only=False):
    """
    Save screenshots of the first N predicted and GT eigenfunctions.

    Args:
        data: Data object containing eigenvectors
        idx: Index of current data item
        eigenfunction_names: Dictionary of eigenfunction names from add_eigenfunctions()
        geom_name: Name of the registered point cloud geometry
        screenshot_dir: Directory to save screenshots
        num_eigenvectors: Number of eigenvectors to capture (default: 30)
        pred_only: If True, capture only predicted eigenfunctions (skip GT and comparison images)
    """
    try:
        import os
        from pathlib import Path

        # Create screenshot directory if it doesn't exist
        screenshot_path = Path(screenshot_dir)
        screenshot_path.mkdir(parents=True, exist_ok=True)

        # Get the point cloud geometry
        try:
            geom = ps.get_point_cloud(geom_name)
        except:
            print(f"Could not find point cloud with name '{geom_name}', skipping screenshots")
            return

        # Capture screenshot of original point cloud with grey color
        print(f"Capturing screenshot of original point cloud...")

        # Get the point positions
        points_array = data.points.cpu().numpy() if hasattr(data.points, 'cpu') else data.points

        # Hide the original geometry temporarily
        original_geom_visible = geom.is_enabled()
        geom.set_enabled(False)

        # Register a temporary point cloud with grey color
        gray_color = (140 / 255, 140 / 255, 140 / 255)
        temp_geom = ps.register_point_cloud(
            name="_temp_original_grey_",
            points=points_array,
            enabled=True
        )
        temp_geom.set_radius(POINT_RADIUS)
        temp_geom.set_color(gray_color)

        # Take screenshot
        original_screenshot_path = screenshot_path / f"item_{idx:04d}_original.png"
        ps.screenshot(str(original_screenshot_path))

        # Crop to minimal bounding box
        crop_transparent_background(original_screenshot_path)
        print(f"  Saved original point cloud: {original_screenshot_path}")

        # Remove the temporary geometry and restore original
        ps.remove_point_cloud("_temp_original_grey_")
        geom.set_enabled(original_geom_visible)

        # Capture area weights screenshots
        print(f"\nCapturing area weights screenshots...")
        area_weights_captured = 0

        # Capture predicted area weights
        if hasattr(data, 'pred_weights') and data.pred_weights is not None:
            pred_weights_data = data.pred_weights.cpu().numpy() if hasattr(data.pred_weights, 'cpu') else data.pred_weights
            geom.add_scalar_quantity('000_AREAS_PRED', pred_weights_data, enabled=True, cmap=CMAP)

            pred_weights_path = screenshot_path / f"item_{idx:04d}_areas_pred.png"
            ps.screenshot(str(pred_weights_path))
            crop_transparent_background(pred_weights_path)
            print(f"  Saved predicted area weights: {pred_weights_path}")
            area_weights_captured += 1

        # Capture GT area weights (if available and not empty)
        if hasattr(data, 'gt_weights') and data.gt_weights is not None:
            gt_weights = data.gt_weights.cpu().numpy() if hasattr(data.gt_weights, 'cpu') else data.gt_weights
            if isinstance(gt_weights, np.ndarray) and gt_weights.shape[0] > 0:
                geom.add_scalar_quantity('000_AREAS_GT', gt_weights, enabled=True, cmap=CMAP)

                gt_weights_path = screenshot_path / f"item_{idx:04d}_areas_gt.png"
                ps.screenshot(str(gt_weights_path))
                crop_transparent_background(gt_weights_path)
                print(f"  Saved GT area weights: {gt_weights_path}")
                area_weights_captured += 1

        # Capture robust area weights
        if hasattr(data, 'robust_weights') and data.robust_weights is not None:
            robust_weights_data = data.robust_weights.cpu().numpy() if hasattr(data.robust_weights, 'cpu') else data.robust_weights
            geom.add_scalar_quantity('000_AREAS_ROBUST', robust_weights_data, enabled=True, cmap=CMAP)

            robust_weights_path = screenshot_path / f"item_{idx:04d}_areas_robust.png"
            ps.screenshot(str(robust_weights_path))
            crop_transparent_background(robust_weights_path)
            print(f"  Saved robust area weights: {robust_weights_path}")
            area_weights_captured += 1

        # Capture potential field screenshot (for Schrödinger operator)
        if hasattr(data, 'potential') and data.potential is not None:
            potential = data.potential
            if isinstance(potential, torch.Tensor):
                potential = potential.cpu().numpy()

            if len(potential) > 0:
                potential_type = getattr(data, 'potential_type', 'unknown')
                potential_strength = getattr(data, 'potential_strength', None)

                # Create name
                if potential_strength is not None:
                    potential_name = f"Potential_V ({potential_type}, β={potential_strength})"
                else:
                    potential_name = f"Potential_V ({potential_type})"

                geom.add_scalar_quantity(
                    potential_name,
                    potential,
                    enabled=True,
                    vminmax=(potential.min(), potential.max()),
                    cmap=CMAP
                )

                potential_path = screenshot_path / f"item_{idx:04d}_potential.png"
                ps.screenshot(str(potential_path))
                crop_transparent_background(potential_path)
                print(f"  Saved potential field: {potential_path}")

        # Check if we have eigenfunction data
        has_pred = hasattr(data, 'pred_eigenvectors') and data.pred_eigenvectors is not None
        has_gt = hasattr(data, 'gt_eigenvectors') and data.gt_eigenvectors is not None

        if not has_pred:
            print("No predicted eigenfunctions available for screenshot")
            return

        # Determine how many eigenvectors we can actually save
        pred_eigenvectors = data.pred_eigenvectors.cpu().numpy() if hasattr(data.pred_eigenvectors, 'cpu') else data.pred_eigenvectors
        num_pred_available = pred_eigenvectors.shape[1]
        num_pred_to_save = min(num_eigenvectors, num_pred_available)

        if has_gt:
            gt_eigenvectors = data.gt_eigenvectors
            if isinstance(gt_eigenvectors, np.ndarray):
                # Check if empty
                if gt_eigenvectors.shape[0] == 0:
                    has_gt = False
                    num_gt_to_save = 0
                else:
                    num_gt_available = gt_eigenvectors.shape[1]
                    num_gt_to_save = min(num_eigenvectors, num_gt_available)
            else:
                gt_eigenvectors = gt_eigenvectors.cpu().numpy()
                num_gt_available = gt_eigenvectors.shape[1]
                num_gt_to_save = min(num_eigenvectors, num_gt_available)
        else:
            num_gt_to_save = 0

        print(f"Capturing screenshots for {num_pred_to_save} predicted eigenfunctions...")

        # Screenshot predicted eigenfunctions
        for i in range(num_pred_to_save):
            pred_data = pred_eigenvectors[:, i]
            pred_name = eigenfunction_names['pred'][i] if i < len(eigenfunction_names['pred']) else f"{i:03d}_pred_eigenfunction"

            # Re-add the quantity with enabled=True
            geom.add_scalar_quantity(pred_name, pred_data, enabled=True, cmap=CMAP)

            pred_screenshot_path = screenshot_path / f"item_{idx:04d}_pred_eigen{i:03d}.png"
            ps.screenshot(str(pred_screenshot_path))

            # Crop to minimal bounding box
            crop_transparent_background(pred_screenshot_path)

            if i == 0 or i == num_pred_to_save - 1 or (i + 1) % 10 == 0:
                print(f"  Saved pred eigenfunction {i}: {pred_screenshot_path}")

        # Screenshot GT eigenfunctions if available (skip if pred_only mode)
        if not pred_only and has_gt and num_gt_to_save > 0:
            print(f"Capturing screenshots for {num_gt_to_save} GT eigenfunctions...")

            for i in range(num_gt_to_save):
                gt_data = gt_eigenvectors[:, i]
                gt_name = eigenfunction_names['GT'][i] if i < len(eigenfunction_names['GT']) else f"{i:03d}_GT_eigenfunction"

                # Re-add the quantity with enabled=True
                geom.add_scalar_quantity(gt_name, gt_data, enabled=True, cmap=CMAP)

                gt_screenshot_path = screenshot_path / f"item_{idx:04d}_gt_eigen{i:03d}.png"
                ps.screenshot(str(gt_screenshot_path))

                # Crop to minimal bounding box
                crop_transparent_background(gt_screenshot_path)

                if i == 0 or i == num_gt_to_save - 1 or (i + 1) % 10 == 0:
                    print(f"  Saved GT eigenfunction {i}: {gt_screenshot_path}")
        elif pred_only:
            print("Skipping GT eigenfunction screenshots (pred-only mode)")
        else:
            print("No GT eigenfunctions available for screenshot")

        # Capture reconstruction screenshots at specific k values
        reconstruction_k_values = []
        print(f"\nCapturing reconstruction screenshots at k={reconstruction_k_values}...")

        # Helper function to capture reconstruction screenshots
        def capture_reconstruction_screenshots(prefix, eigenvectors, weights, k_values):
            """Capture reconstruction screenshots for specific k values."""
            # Convert to appropriate format
            if hasattr(eigenvectors, 'cpu'):
                eigenvectors = eigenvectors.cpu()
            if hasattr(weights, 'cpu'):
                weights = weights.cpu()

            # Get coordinate functions
            coord_functions = data.points.cpu() if isinstance(data.points, torch.Tensor) else data.points

            captured_count = 0
            for k in k_values:
                if k > eigenvectors.shape[1]:
                    print(f"  Skipping {prefix} reconstruction k={k} (only {eigenvectors.shape[1]} eigenvectors available)")
                    continue

                # Compute reconstruction using k eigenvectors
                eigenvectors_weighted = utils.scale_by_half_inv(scalar_functions=eigenvectors, weights=weights)
                reconstructed, _ = utils.project_functions_unnormalized(
                    eigenvectors_basis=eigenvectors_weighted,
                    scalar_functions=coord_functions,
                    weights=weights,
                    max_eigenvectors=k
                )

                # Get the reconstructed points for this k
                recon_points = reconstructed[k - 1].numpy()  # Index k-1 because project_functions returns all up to k

                # Create the reconstruction name that matches what's registered
                padded_k = str(k).zfill(3)
                recon_name = f"{padded_k}_eigvecs_{prefix}"

                try:
                    # Get or register the reconstruction point cloud
                    try:
                        recon_geom = ps.get_point_cloud(recon_name)
                    except:
                        # If not registered, register it now with the gray color
                        recon_geom = ps.register_point_cloud(
                            name=recon_name,
                            points=recon_points,
                            enabled=False
                        )
                        recon_geom.set_radius(POINT_RADIUS)

                    # Set the static gray color: RGB(140, 140, 140) normalized to [0,1]
                    gray_color = (140 / 255, 140 / 255, 140 / 255)
                    recon_geom.set_color(gray_color)

                    # Hide the original point cloud so only reconstruction is visible
                    original_geom_visible = geom.is_enabled()
                    geom.set_enabled(False)

                    # Enable this reconstruction
                    recon_geom.set_enabled(True)

                    # Take screenshot
                    recon_screenshot_path = screenshot_path / f"item_{idx:04d}_{prefix}_recon_k{k:03d}.png"
                    ps.screenshot(str(recon_screenshot_path))

                    # Crop to minimal bounding box
                    crop_transparent_background(recon_screenshot_path)

                    # Restore original point cloud visibility and disable reconstruction
                    geom.set_enabled(original_geom_visible)
                    recon_geom.set_enabled(False)

                    captured_count += 1
                    print(f"  Saved {prefix} reconstruction k={k}: {recon_screenshot_path}")

                except Exception as e:
                    print(f"  Warning: Could not capture {prefix} reconstruction k={k}: {e}")

            return captured_count

        # Capture predicted reconstructions
        pred_recon_count = 0
        if has_pred:
            pred_recon_count = capture_reconstruction_screenshots(
                'pred',
                data.pred_eigenvectors,
                data.pred_weights,
                reconstruction_k_values
            )

        # Capture GT reconstructions (skip if pred_only mode)
        gt_recon_count = 0
        if not pred_only and has_gt and num_gt_to_save > 0:
            gt_recon_count = capture_reconstruction_screenshots(
                'gt',
                data.gt_eigenvectors,
                data.gt_weights,
                reconstruction_k_values
            )
        elif pred_only:
            print("Skipping GT reconstruction screenshots (pred-only mode)")

        # Capture Robust reconstructions (skip if pred_only mode)
        robust_recon_count = 0
        has_robust = hasattr(data, 'robust_eigenvectors') and data.robust_eigenvectors is not None
        if not pred_only and has_robust:
            # Check if robust eigenvectors array is not empty
            robust_eigenvectors = data.robust_eigenvectors
            if isinstance(robust_eigenvectors, np.ndarray):
                if robust_eigenvectors.shape[0] > 0:
                    robust_recon_count = capture_reconstruction_screenshots(
                        'robust',
                        data.robust_eigenvectors,
                        data.robust_weights,
                        reconstruction_k_values
                    )
            else:
                robust_recon_count = capture_reconstruction_screenshots(
                    'robust',
                    data.robust_eigenvectors,
                    data.robust_weights,
                    reconstruction_k_values
                )
        elif pred_only:
            print("Skipping Robust reconstruction screenshots (pred-only mode)")

        # Save cosine similarities to text file
        if hasattr(data, 'metrics') and 'cosine_similarities' in data.metrics:
            print(f"\nSaving cosine similarities to text file...")
            cosine_sim_path = screenshot_path / f"item_{idx:04d}_cosine_similarities.txt"

            try:
                with open(cosine_sim_path, 'w') as f:
                    f.write("Cosine Similarities between Predicted and Robust Eigenvectors\n")
                    f.write("=" * 70 + "\n")
                    f.write(f"Item Index: {idx}\n")
                    f.write(f"Mean Cosine Similarity: {data.metrics['mean_cosine_similarity']:.6f}\n")
                    f.write(f"Min Cosine Similarity: {data.metrics['min_cosine_similarity']:.6f}\n")
                    f.write("\n")
                    f.write("Per-Eigenvector Cosine Similarities:\n")
                    f.write("-" * 70 + "\n")

                    cosine_sims = data.metrics['cosine_similarities']
                    for i, sim in enumerate(cosine_sims):
                        # Convert to float if it's a tensor
                        sim_value = sim.item() if hasattr(sim, 'item') else float(sim)
                        f.write(f"Eigenvector {i:3d}: {sim_value:.6f}\n")

                print(f"  Saved cosine similarities to: {cosine_sim_path}")
            except Exception as e:
                print(f"  Warning: Could not save cosine similarities: {e}")
        else:
            print(f"\nWarning: No cosine similarity metrics available to save")

        # Capture scalar field (probe function) screenshots
        print(f"\nCapturing screenshots for scalar fields (probe functions)...")
        num_scalar_fields = 5  # First 5 scalar fields
        scalar_field_count = 0

        # Capture original scalar fields
        if hasattr(data, 'scalar_fields') and data.scalar_fields is not None:
            scalar_fields = data.scalar_fields.cpu().numpy() if hasattr(data.scalar_fields, 'cpu') else data.scalar_fields
            num_available = min(num_scalar_fields, scalar_fields.shape[1])

            print(f"  Capturing {num_available} original scalar fields...")
            for i in range(num_available):
                scalar_field_data = scalar_fields[:, i]
                field_name = f"SF_{i:03d}_original"

                # Add the scalar field with enabled=True
                geom.add_scalar_quantity(field_name, scalar_field_data, enabled=True, cmap=CMAP)

                # Take screenshot
                sf_screenshot_path = screenshot_path / f"item_{idx:04d}_scalar_field_original_{i:03d}.png"
                ps.screenshot(str(sf_screenshot_path))

                # Crop to minimal bounding box
                crop_transparent_background(sf_screenshot_path)

                scalar_field_count += 1
                if i == 0 or i == num_available - 1:
                    print(f"    Saved original scalar field {i}: {sf_screenshot_path}")

        # Capture smoothed scalar fields
        if hasattr(data, 'smoothed_scalar_fields') and data.smoothed_scalar_fields is not None:
            smoothed_fields = data.smoothed_scalar_fields.cpu().numpy() if hasattr(data.smoothed_scalar_fields, 'cpu') else data.smoothed_scalar_fields
            num_available = min(num_scalar_fields, smoothed_fields.shape[1])

            print(f"  Capturing {num_available} smoothed scalar fields...")
            for i in range(num_available):
                scalar_field_data = smoothed_fields[:, i]
                field_name = f"SF_{i:03d}_smoothed"

                # Add the scalar field with enabled=True
                geom.add_scalar_quantity(field_name, scalar_field_data, enabled=True, cmap=CMAP)

                # Take screenshot
                sf_screenshot_path = screenshot_path / f"item_{idx:04d}_scalar_field_smoothed_{i:03d}.png"
                ps.screenshot(str(sf_screenshot_path))

                # Crop to minimal bounding box
                crop_transparent_background(sf_screenshot_path)

                scalar_field_count += 1
                if i == 0 or i == num_available - 1:
                    print(f"    Saved smoothed scalar field {i}: {sf_screenshot_path}")

        # Reset to first predicted eigenfunction enabled (original state)
        pred_data = pred_eigenvectors[:, 0]
        pred_name = eigenfunction_names['pred'][0] if len(eigenfunction_names['pred']) > 0 else "000_pred_eigenfunction"
        geom.add_scalar_quantity(pred_name, pred_data, enabled=True, cmap=CMAP)

        # Create comparison image (always create, but layout depends on mode)
        print("\nCreating comparison image...")
        create_comparison_image(data, idx, screenshot_dir=str(screenshot_path), pred_only=pred_only)

        print(f"\n[SUCCESS] Screenshot capture complete:")
        print(f"  - {area_weights_captured} area weights visualizations")
        print(f"  - {num_pred_to_save} predicted eigenfunctions")
        if not pred_only:
            print(f"  - {num_gt_to_save} GT eigenfunctions")
        print(f"  - {pred_recon_count} predicted reconstructions")
        if not pred_only:
            print(f"  - {gt_recon_count} GT reconstructions")
        print(f"  - {robust_recon_count} robust reconstructions")
        print(f"  - {scalar_field_count} scalar fields (probe functions)")
        if hasattr(data, 'metrics') and 'cosine_similarities' in data.metrics:
            print(f"  - Cosine similarities saved to text file")
        print(f"  - Comparison image created")

    except Exception as e:
        print(f"Warning: Could not save eigenfunction screenshots: {e}")
        import traceback
        traceback.print_exc()


@hydra.main(version_base="1.2", config_path="config/training")
def main(cfg: DictConfig):
    """Main function for visualization with interactive k-NN control."""
    # Initialize Polyscope
    ps.init()

    # Set anti-aliasing to 4x SSAA for high quality rendering
    ps.set_SSAA_factor(2)

    # Disable automatic scene extents computation and set to unit cube
    ps.set_automatically_compute_scene_extents(False)
    ps.set_length_scale(1.0)
    low = np.array((-1.0, -1.0, -1.0))
    high = np.array((1.0, 1.0, 1.0))
    ps.set_bounding_box(low, high)

    ps.set_up_dir("z_up")
    ps.look_at(camera_location=[0.0, 2.0, -4.0], target=[0, 0, 0])

    # Set ground plane to shadow-only mode with custom settings
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height(-0.375)
    ps.set_shadow_blur_iters(35)
    ps.set_shadow_darkness(0.6)

    ps.set_background_color((0.0, 0.0, 0.0, 0.0))

    # Seed for reproducibility
    pl.seed_everything(cfg.globals.seed)

    # Check if checkpoint path is provided
    if not hasattr(cfg, 'checkpoint_path') or cfg.checkpoint_path is None:
        raise ValueError("Please provide a checkpoint path using ++checkpoint_path=<path>")

    # Load the model from checkpoint
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model = LaplacianPredictorModule3D.load_from_checkpoint(cfg.checkpoint_path)
    model.eval()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize dataset controller
    dataset_controller = DatasetController(cfg, model)

    # Get operator_config from the dataset (if present)
    operator_config = getattr(dataset_controller.dataset, '_operator_config', None)
    if operator_config is not None:
        print(f"Operator config (from dataset): {operator_config}")
    else:
        print("Operator config: None (using Laplacian GT)")

    # Current visualization state
    current_data = None
    eigenfunction_names_cache = None  # Store eigenfunction names for manual screenshot capture
    last_processed_idx = -1
    last_processed_k = -1
    last_processed_robust_k = -1  # Add tracking for robust k-NN
    last_processed_smoother_iterations = -1  # Add tracking for smoother iterations
    last_processed_smoother_sigma = -1  # Add tracking for smoother sigma
    normalization_mode = "normalized"  # "normalized" or "unnormalized"
    pred_only_screenshots = False  # Flag to control whether to capture only pred screenshots (no GT, no comparison)

    def apply_normalization_transformation(data, old_mode, new_mode):
        """
        Switch between pre-computed normalized and unnormalized eigenvector representations.
        No computation needed - just switches which version is active.

        Args:
            data: Data object containing both normalized and unnormalized eigenvectors
            old_mode: Previous normalization mode ("normalized" or "unnormalized")
            new_mode: New normalization mode ("normalized" or "unnormalized")
        """
        if old_mode == new_mode:
            return  # No transformation needed

        print(f"Switching eigenvector display from {old_mode} to {new_mode}...")

        # Switch predicted eigenvectors
        if hasattr(data, 'pred_eigenvectors_normalized') and hasattr(data, 'pred_eigenvectors_unnormalized'):
            if new_mode == "unnormalized":
                data.pred_eigenvectors = data.pred_eigenvectors_unnormalized
                print("  Switched to pred eigenvectors (unnormalized)")
            else:
                data.pred_eigenvectors = data.pred_eigenvectors_normalized
                print("  Switched to pred eigenvectors (normalized)")

        # Switch GT eigenvectors
        if hasattr(data, 'gt_eigenvectors_normalized') and hasattr(data, 'gt_eigenvectors_unnormalized'):
            gt_eigenvectors_np = data.gt_eigenvectors_normalized.cpu().numpy() if isinstance(data.gt_eigenvectors_normalized, torch.Tensor) else data.gt_eigenvectors_normalized
            if gt_eigenvectors_np.shape[0] > 0:  # Check if not empty
                if new_mode == "unnormalized":
                    data.gt_eigenvectors = data.gt_eigenvectors_unnormalized
                    print("  Switched to GT eigenvectors (unnormalized)")
                else:
                    data.gt_eigenvectors = data.gt_eigenvectors_normalized
                    print("  Switched to GT eigenvectors (normalized)")

        # Switch robust eigenvectors
        if hasattr(data, 'robust_eigenvectors_normalized') and hasattr(data, 'robust_eigenvectors_unnormalized'):
            if new_mode == "unnormalized":
                data.robust_eigenvectors = data.robust_eigenvectors_unnormalized
                print("  Switched to robust eigenvectors (unnormalized)")
            else:
                data.robust_eigenvectors = data.robust_eigenvectors_normalized
                print("  Switched to robust eigenvectors (normalized)")

        # Switch active metrics
        if hasattr(data, 'metrics_normalized') and hasattr(data, 'metrics_unnormalized'):
            if new_mode == "unnormalized":
                data.metrics = data.metrics_unnormalized
                print("  Switched to metrics (unnormalized)")
            else:
                data.metrics = data.metrics_normalized
                print("  Switched to metrics (normalized)")

    def gui_callback():
        """GUI callback for dataset controls and status."""
        import polyscope.imgui as psim

        nonlocal dataset_controller, current_data, eigenfunction_names_cache, last_processed_idx, last_processed_k, last_processed_robust_k, last_processed_smoother_iterations, last_processed_smoother_sigma, normalization_mode

        # Set the tree node to be open by default
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Dataset Controls"):
            status = dataset_controller.get_status()

            # Predicted k-NN input control
            changed, new_k = psim.InputInt("Predicted k-NN neighbors", status['pending_k'], step=1, step_fast=5)
            if changed and new_k > 0:  # Ensure k is positive
                dataset_controller.pending_k = new_k

            # Robust k-NN input control
            changed_robust, new_robust_k = psim.InputInt("Robust k-NN neighbors", status['pending_robust_k'], step=1, step_fast=5)
            if changed_robust and new_robust_k > 0:  # Ensure k is positive
                dataset_controller.pending_robust_k = new_robust_k

            # Smoother iterations input control
            changed_smoother, new_smoother_iterations = psim.InputInt("Smoother iterations", status['pending_smoother_iterations'][0] if isinstance(status['pending_smoother_iterations'], ListConfig) else status['pending_smoother_iterations'], step=1, step_fast=2)
            if changed_smoother and new_smoother_iterations > 0:  # Ensure iterations is positive
                dataset_controller.pending_smoother_iterations = new_smoother_iterations

            # Smoother sigma input control
            changed_sigma, new_sigma = psim.InputFloat("Smoother sigma", status['pending_smoother_sigma'], step=0.001, step_fast=0.01)
            if changed_sigma and new_sigma > 0:  # Ensure sigma is positive
                dataset_controller.pending_smoother_sigma = new_sigma

            # Normalization mode dropdown
            psim.Separator()
            psim.TextUnformatted("Eigenvector Display Mode:")
            mode_items = ["normalized", "unnormalized"]
            current_mode_idx = 0 if normalization_mode == "normalized" else 1
            changed_mode, new_mode_idx = psim.Combo("##normalization_mode", current_mode_idx, mode_items)
            if changed_mode:
                old_mode = normalization_mode
                normalization_mode = mode_items[new_mode_idx]
                print(f"Normalization mode changed: {old_mode} -> {normalization_mode}")
                # Apply transformation to current data if available
                if current_data is not None:
                    apply_normalization_transformation(current_data, old_mode, normalization_mode)
                    # Force re-visualization without reprocessing
                    ps.remove_all_structures()
                    create_visualization(current_data, status['current_k'])

            # Flip sign button
            psim.Separator()
            if psim.Button("Flip All Eigenvector Signs"):
                if current_data is not None:
                    print("Flipping signs of all eigenvectors (pred, GT, robust)...")

                    # Flip predicted eigenvectors (both normalized and unnormalized)
                    if hasattr(current_data, 'pred_eigenvectors_normalized'):
                        current_data.pred_eigenvectors_normalized = -current_data.pred_eigenvectors_normalized
                    if hasattr(current_data, 'pred_eigenvectors_unnormalized'):
                        current_data.pred_eigenvectors_unnormalized = -current_data.pred_eigenvectors_unnormalized

                    # Flip GT eigenvectors (both normalized and unnormalized)
                    if hasattr(current_data, 'gt_eigenvectors_normalized'):
                        if isinstance(current_data.gt_eigenvectors_normalized, np.ndarray):
                            if current_data.gt_eigenvectors_normalized.shape[0] > 0:
                                current_data.gt_eigenvectors_normalized = -current_data.gt_eigenvectors_normalized
                        else:
                            current_data.gt_eigenvectors_normalized = -current_data.gt_eigenvectors_normalized
                    if hasattr(current_data, 'gt_eigenvectors_unnormalized'):
                        if isinstance(current_data.gt_eigenvectors_unnormalized, np.ndarray):
                            if current_data.gt_eigenvectors_unnormalized.shape[0] > 0:
                                current_data.gt_eigenvectors_unnormalized = -current_data.gt_eigenvectors_unnormalized
                        else:
                            current_data.gt_eigenvectors_unnormalized = -current_data.gt_eigenvectors_unnormalized

                    # Flip robust eigenvectors (both normalized and unnormalized)
                    if hasattr(current_data, 'robust_eigenvectors_normalized'):
                        current_data.robust_eigenvectors_normalized = -current_data.robust_eigenvectors_normalized
                    if hasattr(current_data, 'robust_eigenvectors_unnormalized'):
                        current_data.robust_eigenvectors_unnormalized = -current_data.robust_eigenvectors_unnormalized

                    # Update the active eigenvectors based on current mode
                    if normalization_mode == "normalized":
                        current_data.pred_eigenvectors = current_data.pred_eigenvectors_normalized
                        current_data.gt_eigenvectors = current_data.gt_eigenvectors_normalized
                        current_data.robust_eigenvectors = current_data.robust_eigenvectors_normalized
                    else:
                        current_data.pred_eigenvectors = current_data.pred_eigenvectors_unnormalized
                        current_data.gt_eigenvectors = current_data.gt_eigenvectors_unnormalized
                        current_data.robust_eigenvectors = current_data.robust_eigenvectors_unnormalized

                    print("Signs flipped successfully!")

                    # Force re-visualization without reprocessing
                    ps.remove_all_structures()
                    eigenfunction_names_cache = create_visualization(current_data, status['current_k'])
                else:
                    print("No data available to flip signs")

            # Apply k change button
            if status['has_pending_changes']:
                changes_text = []
                if status['current_k'] != status['pending_k']:
                    changes_text.append(f"Predicted: {status['current_k']} -> {status['pending_k']}")
                if status['current_robust_k'] != status['pending_robust_k']:
                    changes_text.append(f"Robust: {status['current_robust_k']} -> {status['pending_robust_k']}")
                if status['current_smoother_iterations'] != status['pending_smoother_iterations']:
                    changes_text.append(f"Smoother: {status['current_smoother_iterations']} -> {status['pending_smoother_iterations']}")
                if status['current_smoother_sigma'] != status['pending_smoother_sigma']:
                    changes_text.append(f"Sigma: {status['current_smoother_sigma']:.4f} -> {status['pending_smoother_sigma']:.4f}")

                psim.TextColored((1.0, 1.0, 0.0, 1.0), f"Pending changes: {', '.join(changes_text)}")
                if psim.Button("Apply Changes"):
                    if dataset_controller.apply_k_change():
                        print(f"Applied changes - Predicted: {dataset_controller.current_k}, Robust: {dataset_controller.current_robust_k}, Smoother: {dataset_controller.current_smoother_iterations}, Sigma: {dataset_controller.current_smoother_sigma}")
                        # Force reprocessing of current item with new values
                        last_processed_k = -1  # This will trigger reprocessing
                        last_processed_robust_k = -1
                        last_processed_smoother_sigma = -1

            # Navigation controls
            psim.Separator()
            if psim.Button("Previous Item"):
                if dataset_controller.prev_item():
                    last_processed_idx = -1  # Force reprocessing

            psim.SameLine()
            if psim.Button("Next Item"):
                if dataset_controller.next_item():
                    last_processed_idx = -1  # Force reprocessing

            # Direct index input
            changed, new_idx = psim.InputInt("Item Index", status['current_idx'], step=1, step_fast=10)
            if changed:
                if dataset_controller.goto_item(new_idx):
                    last_processed_idx = -1  # Force reprocessing

            psim.SameLine()
            if psim.Button("Reset to First"):
                dataset_controller.reset_to_first()
                last_processed_idx = -1  # Force reprocessing

            # Status information
            psim.Separator()
            psim.TextUnformatted(f"Predicted k-NN: {status['current_k']}")
            if status['current_k'] != status['pending_k']:
                psim.SameLine()
                psim.TextColored((1.0, 0.5, 0.0, 1.0), f"(pending: {status['pending_k']})")

            psim.TextUnformatted(f"Robust k-NN: {status['current_robust_k']}")
            if status['current_robust_k'] != status['pending_robust_k']:
                psim.SameLine()
                psim.TextColored((1.0, 0.5, 0.0, 1.0), f"(pending: {status['pending_robust_k']})")

            psim.TextUnformatted(f"Smoother iterations: {status['current_smoother_iterations']}")
            if status['current_smoother_iterations'] != status['pending_smoother_iterations']:
                psim.SameLine()
                psim.TextColored((1.0, 0.5, 0.0, 1.0), f"(pending: {status['pending_smoother_iterations']})")

            psim.TextUnformatted(f"Smoother sigma: {status['current_smoother_sigma']:.4f}")
            if status['current_smoother_sigma'] != status['pending_smoother_sigma']:
                psim.SameLine()
                psim.TextColored((1.0, 0.5, 0.0, 1.0), f"(pending: {status['pending_smoother_sigma']:.4f})")

            psim.TextUnformatted(f"Display Mode: {normalization_mode}")

            psim.TextUnformatted(f"Item: {status['current_idx'] + 1} / {status['total_items']}")

            # Display current processing status
            if (last_processed_idx == status['current_idx'] and
                    last_processed_k == status['current_k'] and
                    last_processed_robust_k == status['current_robust_k'] and
                    last_processed_smoother_iterations == status['current_smoother_iterations'] and
                    last_processed_smoother_sigma == status['current_smoother_sigma']):
                psim.TextColored((0.0, 1.0, 0.0, 1.0), "[OK] Current item processed")
            else:
                psim.TextColored((1.0, 1.0, 0.0, 1.0), "[!] Processing needed...")

            psim.TreePop()

        # Metrics display
        if current_data is not None and hasattr(current_data, 'metrics'):
            # Set the tree node to be open by default
            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
            mode_suffix = f" ({normalization_mode})"
            if psim.TreeNode(f"Quality Metrics (Pred vs Robust){mode_suffix}"):
                metrics = current_data.metrics

                # Show which mode's metrics are displayed
                psim.TextColored((0.0, 1.0, 1.0, 1.0), f"Displaying metrics for: {normalization_mode} mode")
                psim.Separator()

                psim.TextUnformatted(f"Mean Cosine Similarity: {metrics['mean_cosine_similarity']:.6f}")
                psim.TextUnformatted(f"Min Cosine Similarity: {metrics['min_cosine_similarity']:.6f}")

                # Display individual eigenvector similarities
                cosine_sims = metrics['cosine_similarities']
                if len(cosine_sims) > 0:
                    psim.TextUnformatted("Per-eigenvector similarities:")
                    for i, sim in enumerate(cosine_sims[:30]):  # Show first 30
                        psim.TextUnformatted(f"  Eigenvector {i}: {sim:.6f}")

                psim.TreePop()

        # Screenshot capture button
        if current_data is not None and eigenfunction_names_cache is not None:
            psim.Separator()
            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
            if psim.TreeNode("Screenshot Capture"):
                psim.TextUnformatted("Position the camera as desired, then click:")

                # Checkbox for pred-only screenshots
                nonlocal pred_only_screenshots
                changed_checkbox, pred_only_screenshots = psim.Checkbox("Capture Pred Only (no GT comparison)", pred_only_screenshots)
                if changed_checkbox:
                    mode_text = "pred-only mode" if pred_only_screenshots else "full comparison mode"
                    print(f"Screenshot mode changed to: {mode_text}")

                if psim.Button("Capture All Eigenfunction Screenshots"):
                    status = dataset_controller.get_status()
                    print("\n" + "=" * 60)
                    print("MANUAL SCREENSHOT CAPTURE INITIATED")
                    if pred_only_screenshots:
                        print("MODE: Pred-only (no GT, no comparison images)")
                    else:
                        print("MODE: Full comparison (GT + Pred with cosine similarities)")
                    print("=" * 60)
                    save_eigenfunction_screenshots(current_data, status['current_idx'], eigenfunction_names_cache, pred_only=pred_only_screenshots)
                    print("=" * 60)
                    print("SCREENSHOT CAPTURE COMPLETE")
                    print("=" * 60 + "\n")

                psim.TextColored((0.7, 0.7, 0.7, 1.0), "Screenshots will be saved to ./screenshots/")

                psim.TreePop()

    # Set GUI callback
    ps.set_user_callback(gui_callback)

    def time_operation(operation_name: str, operation_func):
        """Helper function to time operations and log execution time."""
        start_time = time.time()
        result = operation_func()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"--------------------- {operation_name} Execution time: {execution_time:.4f} seconds")
        return result

    def compute_model_predictions(data, status):
        """Compute neural network predictions for Laplacian eigenvectors."""

        def predict_operation():
            batch = Batch.from_data_list([data])
            with torch.no_grad():
                laplacian_prediction, processed_batch = model.predict_step(batch, batch_idx=0)

            # Extract predictions from the prediction object (these are in normalized form)
            pred_eigenvectors_normalized = -laplacian_prediction.eigenvectors_list[0].cpu()
            pred_weights = laplacian_prediction.weights_list[0].cpu()

            # Compute unnormalized version
            pred_eigenvectors_unnormalized = utils.scale_by_half_inv(
                scalar_functions=pred_eigenvectors_normalized,
                weights=pred_weights
            )

            # Store both versions
            data.pred_eigenvectors_normalized = pred_eigenvectors_normalized
            data.pred_eigenvectors_unnormalized = pred_eigenvectors_unnormalized
            data.pred_weights = pred_weights

            # Set active eigenvectors based on current mode (for backward compatibility)
            data.pred_eigenvectors = pred_eigenvectors_normalized

            # Prepend zero eigenvalue to predicted eigenvalues
            original_eigenvalues = laplacian_prediction.unweighted_eigenvalues_list[0].cpu()
            # original_eigenvalues = original_eigenvalues / torch.sqrt(torch.sum(data.pred_weights))
            zero_eigenvalue = torch.tensor([0.0], dtype=original_eigenvalues.dtype)
            data.pred_eigenvalues = torch.cat([zero_eigenvalue, original_eigenvalues], dim=0)
            data.pred_eigenvalues = data.pred_eigenvalues[:-1]

            # Extract smoothed scalar fields from the processed batch
            processed_data = processed_batch.to_data_list()[0]
            if hasattr(processed_data, 'smoothed_scalar_fields'):
                data.smoothed_scalar_fields = processed_data.smoothed_scalar_fields.cpu()

            return pred_eigenvectors_normalized.shape[1]  # Return number of eigenvectors

        n_eig = time_operation("PRED", predict_operation)
        print(f"Model predicts {n_eig} eigenvectors with k={status['current_k']}")
        return n_eig

    def compute_ground_truth_laplacian(data, n_eig):
        """Compute ground truth Laplacian eigenfunctions (mesh-based, uses PyFM)."""
        print(f"Computing ground truth with {n_eig} eigenvectors...")

        # Check if we have valid faces for PyFM computation
        has_faces = hasattr(data, 'faces') and data.faces is not None
        if has_faces:
            faces_array = data.faces.cpu().numpy() if isinstance(data.faces, torch.Tensor) else data.faces
            has_valid_faces = faces_array.shape[0] > 0
        else:
            has_valid_faces = False

        if not has_valid_faces:
            print("Skipping GT computation - no face connectivity available (point cloud only)")
            # Set empty arrays for GT data so the attributes exist but are empty
            data.gt_eigenvectors_normalized = np.array([])
            data.gt_eigenvectors_unnormalized = np.array([])
            data.gt_eigenvectors = np.array([])
            data.gt_eigenvalues = np.array([])
            data.gt_weights = np.array([])
            return

        def gt_operation():
            return utils.compute_pyfm_normalized_laplacian_eigenfunctions(
                vertices=data.points.cpu().numpy(),
                faces=faces_array,
                num_eigenfunctions=n_eig
            )

        gt_eigenvectors_normalized, gt_eigenvalues, gt_weights = time_operation("GT", gt_operation)

        # Compute unnormalized version
        gt_eigenvectors_unnormalized = utils.scale_by_half_inv(
            scalar_functions=gt_eigenvectors_normalized,
            weights=gt_weights
        )

        # Store both versions
        data.gt_eigenvectors_normalized = gt_eigenvectors_normalized
        data.gt_eigenvectors_unnormalized = gt_eigenvectors_unnormalized
        data.gt_eigenvalues = gt_eigenvalues
        data.gt_weights = gt_weights

        # Set active eigenvectors (for backward compatibility)
        data.gt_eigenvectors = gt_eigenvectors_normalized

        # Set operator metadata
        data.operator_type = "laplacian"
        data.potential_type = None
        data.potential_strength = None
        data.potential = None

    def compute_ground_truth(data, n_eig, n_neighbors, operator_config=None):
        """
        Unified GT computation that routes based on operator_config.

        Args:
            data: Data object with point cloud
            n_eig: Number of eigenfunctions to compute
            n_neighbors: Number of neighbors for k-NN (used for Schrödinger and robust Laplacian)
            operator_config: Optional OperatorConfig. If None or type="laplacian", computes
                           Laplacian GT. If type="schrodinger", computes Schrödinger GT.
        """
        if operator_config is None or operator_config.type == "laplacian":
            # Use mesh-based Laplacian GT if faces available, otherwise skip
            compute_ground_truth_laplacian(data, n_eig)
        elif operator_config.type == "schrodinger":
            # Compute Schrödinger GT
            compute_schrodinger(
                data,
                n_eig,
                n_neighbors,
                operator_config.potential_type,
                operator_config.potential_strength
            )
        else:
            raise ValueError(f"Unknown operator type: {operator_config.type}")

    def compute_robust_laplacian(data, n_eig, n_neighbors):
        """Compute robust Laplacian eigenvectors with configurable k-NN."""
        print(f"Computing robust Laplacian with {n_eig} eigenvectors and {n_neighbors} neighbors...")

        def robust_operation():
            return utils.compute_robust_normalized_laplacian_eigenvectors(
                vertices=data.points.cpu().numpy(),
                num_eigenfunctions=n_eig,
                n_neighbors=n_neighbors
            )

        robust_eigenvectors_normalized, robust_eigenvalues, robust_weights = time_operation("ROBUST", robust_operation)

        # Compute unnormalized version
        robust_eigenvectors_unnormalized = utils.scale_by_half_inv(
            scalar_functions=robust_eigenvectors_normalized,
            weights=robust_weights
        )

        # Store both versions
        data.robust_eigenvectors_normalized = robust_eigenvectors_normalized
        data.robust_eigenvectors_unnormalized = robust_eigenvectors_unnormalized
        data.robust_eigenvalues = robust_eigenvalues
        data.robust_weights = robust_weights

        # Set active eigenvectors (for backward compatibility)
        data.robust_eigenvectors = robust_eigenvectors_normalized

    def compute_schrodinger(data, n_eig, n_neighbors, potential_type, potential_strength):
        """Compute Schrödinger operator eigenvectors: H = -Δ + V(x).

        Args:
            data: Data object with point cloud
            n_eig: Number of eigenfunctions to compute
            n_neighbors: Number of neighbors for k-NN
            potential_type: Type of potential ('curvature', 'center_distance', 'height', etc.)
            potential_strength: β value (strength of potential term)
        """
        print(f"Computing Schrödinger eigenvectors (potential={potential_type}, β={potential_strength})...")

        def schrodinger_operation():
            return utils.compute_robust_schrodinger_eigenvectors(
                vertices=data.points.cpu().numpy(),
                num_eigenfunctions=n_eig,
                n_neighbors=n_neighbors,
                potential_type=potential_type,
                potential_strength=potential_strength
            )

        schrodinger_eigenvectors_normalized, schrodinger_eigenvalues, schrodinger_weights, potential = time_operation(
            "SCHRODINGER", schrodinger_operation
        )

        # Compute unnormalized version
        schrodinger_eigenvectors_unnormalized = utils.scale_by_half_inv(
            scalar_functions=schrodinger_eigenvectors_normalized,
            weights=schrodinger_weights
        )

        # Store Schrödinger results as GT (since this IS the ground truth for this operator)
        data.gt_eigenvectors_normalized = schrodinger_eigenvectors_normalized
        data.gt_eigenvectors_unnormalized = schrodinger_eigenvectors_unnormalized
        data.gt_eigenvalues = schrodinger_eigenvalues
        data.gt_weights = schrodinger_weights
        data.gt_eigenvectors = schrodinger_eigenvectors_normalized

        # Store potential field and operator metadata
        data.potential = potential
        data.operator_type = 'schrodinger'
        data.potential_type = potential_type
        data.potential_strength = potential_strength

        print(f"Schrödinger eigendecomposition complete. Potential stored for visualization.")

    def apply_sign_alignment(data):
        """Apply sign alignment using GT eigenvectors as reference (fallback to robust if GT unavailable).

        This function aligns BOTH normalized and unnormalized versions of eigenvectors to ensure
        consistency when switching between display modes.
        """

        # Check if GT is available and not empty
        has_valid_gt = False
        if hasattr(data, 'gt_eigenvectors_normalized'):
            gt_eigenvectors_np = data.gt_eigenvectors_normalized.cpu().numpy() if isinstance(data.gt_eigenvectors_normalized, torch.Tensor) else data.gt_eigenvectors_normalized
            has_valid_gt = gt_eigenvectors_np.shape[0] > 0

        if has_valid_gt:
            # Use GT as reference
            print("Applying sign alignment using GT eigenvectors as reference...")

            # Align predicted eigenvectors (both normalized and unnormalized) to GT reference
            if hasattr(data, 'pred_eigenvectors_normalized') and hasattr(data, 'pred_eigenvectors_unnormalized'):
                # Align normalized versions
                data.pred_eigenvectors_normalized = utils.align_eigenvector_signs(
                    data.pred_eigenvectors_normalized, data.gt_eigenvectors_normalized
                )
                # Align unnormalized versions
                data.pred_eigenvectors_unnormalized = utils.align_eigenvector_signs(
                    data.pred_eigenvectors_unnormalized, data.gt_eigenvectors_unnormalized
                )
                print("Aligned predicted eigenvectors (both normalized and unnormalized) to GT reference")

            # Align robust eigenvectors (both normalized and unnormalized) to GT reference
            if hasattr(data, 'robust_eigenvectors_normalized') and hasattr(data, 'robust_eigenvectors_unnormalized'):
                # Align normalized versions
                data.robust_eigenvectors_normalized = utils.align_eigenvector_signs(
                    data.robust_eigenvectors_normalized, data.gt_eigenvectors_normalized
                )
                # Align unnormalized versions
                data.robust_eigenvectors_unnormalized = utils.align_eigenvector_signs(
                    data.robust_eigenvectors_unnormalized, data.gt_eigenvectors_unnormalized
                )
                print("Aligned robust eigenvectors (both normalized and unnormalized) to GT reference")

            # GT eigenvectors remain unchanged (they are the reference)
        else:
            # Fallback to robust as reference
            print("GT not available - falling back to robust eigenvectors as reference...")

            # Align predicted eigenvectors (both normalized and unnormalized) to robust reference
            if hasattr(data, 'pred_eigenvectors_normalized') and hasattr(data, 'pred_eigenvectors_unnormalized'):
                if hasattr(data, 'robust_eigenvectors_normalized') and hasattr(data, 'robust_eigenvectors_unnormalized'):
                    # Align normalized versions
                    data.pred_eigenvectors_normalized = utils.align_eigenvector_signs(
                        data.pred_eigenvectors_normalized, data.robust_eigenvectors_normalized
                    )
                    # Align unnormalized versions
                    data.pred_eigenvectors_unnormalized = utils.align_eigenvector_signs(
                        data.pred_eigenvectors_unnormalized, data.robust_eigenvectors_unnormalized
                    )
                    print("Aligned predicted eigenvectors (both normalized and unnormalized) to robust reference")

            # GT alignment skipped since GT is empty
            if hasattr(data, 'gt_eigenvectors_normalized'):
                print("Skipping GT alignment - GT eigenvectors array is empty")

            # Robust eigenvectors remain unchanged (they are the reference)

    def compute_quality_metrics(data):
        """Compute quality metrics comparing predicted and GT eigenvectors in both normalized and unnormalized modes."""

        # Check if GT is available and not empty
        has_valid_gt = False
        if hasattr(data, 'gt_eigenvectors_normalized'):
            gt_eigenvectors_np = data.gt_eigenvectors_normalized.cpu().numpy() if isinstance(data.gt_eigenvectors_normalized, torch.Tensor) else data.gt_eigenvectors_normalized
            has_valid_gt = gt_eigenvectors_np.shape[0] > 0

        if has_valid_gt:
            # Compare pred vs GT in both modes
            print("Computing quality metrics (pred vs GT) in both normalized and unnormalized modes...")

            # Normalized mode metrics
            data.metrics_normalized = compute_reconstruction_metrics(
                data.pred_eigenvectors_normalized,
                data.gt_eigenvectors_normalized
            )
            print(f"  Normalized mode: Mean cosine similarity = {data.metrics_normalized['mean_cosine_similarity']:.6f}")

            # Unnormalized mode metrics
            data.metrics_unnormalized = compute_reconstruction_metrics(
                data.pred_eigenvectors_unnormalized,
                data.gt_eigenvectors_unnormalized
            )
            print(f"  Unnormalized mode: Mean cosine similarity = {data.metrics_unnormalized['mean_cosine_similarity']:.6f}")

        else:
            # Fallback to comparing pred vs robust
            print("Computing quality metrics (pred vs robust - GT unavailable) in both modes...")

            # Normalized mode metrics
            data.metrics_normalized = compute_reconstruction_metrics(
                data.pred_eigenvectors_normalized,
                data.robust_eigenvectors_normalized
            )
            print(f"  Normalized mode: Mean cosine similarity = {data.metrics_normalized['mean_cosine_similarity']:.6f}")

            # Unnormalized mode metrics
            data.metrics_unnormalized = compute_reconstruction_metrics(
                data.pred_eigenvectors_unnormalized,
                data.robust_eigenvectors_unnormalized
            )
            print(f"  Unnormalized mode: Mean cosine similarity = {data.metrics_unnormalized['mean_cosine_similarity']:.6f}")

        # Set active metrics based on current mode (for backward compatibility)
        data.metrics = data.metrics_normalized

    def create_visualization(data, k):
        """Create and display the visualization."""
        visualizer = GeometryVisualizer(data=data, k=k)
        eigenfunction_names = visualizer.visualize()
        return eigenfunction_names

    def should_reprocess(status):
        """Check if reprocessing is needed based on current state."""
        return not (last_processed_idx == status['current_idx'] and
                    last_processed_k == status['current_k'] and
                    last_processed_robust_k == status['current_robust_k'] and
                    last_processed_smoother_iterations == status['current_smoother_iterations'] and
                    last_processed_smoother_sigma == status['current_smoother_sigma'] and
                    current_data is not None)

    def update_processing_state(status):
        """Update the processing state variables."""
        nonlocal current_data, last_processed_idx, last_processed_k, last_processed_robust_k, last_processed_smoother_iterations, last_processed_smoother_sigma
        last_processed_idx = status['current_idx']
        last_processed_k = status['current_k']
        last_processed_robust_k = status['current_robust_k']
        last_processed_smoother_iterations = status['current_smoother_iterations']
        last_processed_smoother_sigma = status['current_smoother_sigma']

    def process_data_item():
        """Process the current data item with model predictions and ground truth."""
        nonlocal current_data, eigenfunction_names_cache, last_processed_idx, last_processed_k, last_processed_robust_k

        status = dataset_controller.get_status()

        # Check if we need to process
        if not should_reprocess(status):
            return  # Already processed

        print(f"\nProcessing item {status['current_idx'] + 1}/{status['total_items']} with predicted k={status['current_k']}, robust k={status['current_robust_k']}, smoother iterations={status['current_smoother_iterations']}, sigma={status['current_smoother_sigma']:.4f}")

        # Clear previous visualization
        ps.remove_all_structures()

        # Get current data
        data, idx = dataset_controller.get_current_item()
        if data is None:
            print("No data available")
            return

        try:
            # Step 1: Compute model predictions
            n_eig = compute_model_predictions(data, status)

            # Step 2: Compute ground truth (Laplacian or Schrödinger based on operator_config)
            compute_ground_truth(data, n_eig, status['current_robust_k'], operator_config)

            # Step 3: Compute robust Laplacian
            compute_robust_laplacian(data, n_eig, status['current_robust_k'])

            # Step 4: Apply sign alignment using robust as reference
            apply_sign_alignment(data)

            # Step 5: Compute quality metrics (now with aligned eigenvectors)
            # This computes metrics for BOTH normalized and unnormalized modes
            compute_quality_metrics(data)

            # Step 5.5: Set active eigenvectors and metrics based on current mode
            # All eigenvectors are pre-computed in both forms, just switch which is active
            if normalization_mode == "unnormalized":
                print("Setting active eigenvectors to unnormalized mode...")
                data.pred_eigenvectors = data.pred_eigenvectors_unnormalized
                data.gt_eigenvectors = data.gt_eigenvectors_unnormalized
                data.robust_eigenvectors = data.robust_eigenvectors_unnormalized
                data.metrics = data.metrics_unnormalized
            else:
                print("Setting active eigenvectors to normalized mode...")
                data.pred_eigenvectors = data.pred_eigenvectors_normalized
                data.gt_eigenvectors = data.gt_eigenvectors_normalized
                data.robust_eigenvectors = data.robust_eigenvectors_normalized
                data.metrics = data.metrics_normalized

            # Step 6: Create visualization
            eigenfunction_names = create_visualization(data, status['current_k'])

            # Step 7: Save eigenvalue comparison plot to file
            save_eigenvalue_plot(data, save_path="eigenvalue_comparison.svg")

            # Step 7.5: Save eigenvalues and metrics to CSV file
            save_eigenvalues_and_metrics_csv(data, save_path="eigenvalues_and_metrics.csv")

            # Step 8: Update state
            current_data = data
            eigenfunction_names_cache = eigenfunction_names  # Store for manual screenshot capture
            update_processing_state(status)

            print(f"Visualization ready for item {status['current_idx'] + 1}/{status['total_items']} with predicted k={status['current_k']}, robust k={status['current_robust_k']}, smoother iterations={status['current_smoother_iterations']}, sigma={status['current_smoother_sigma']:.4f}")
            print("Check 'eigenvalue_comparison.svg' for eigenvalue plot")
            print("Check 'eigenvalues_and_metrics.csv' for detailed eigenvalues and cosine similarities")

        except Exception as e:
            print(f"Error processing data item: {e}")
            import traceback
            traceback.print_exc()

    # Main visualization loop
    print("Starting interactive visualization...")
    print("Use the GUI controls to:")
    print("- Change predicted k-NN parameter and see its effect on neural network predictions")
    print("- Change robust k-NN parameter and see its effect on robust Laplacian baseline")
    print("- Navigate through dataset items")
    print("- Jump to specific items by index")

    # Process initial item
    process_data_item()

    # Show polyscope with callback
    def main_callback():
        gui_callback()
        process_data_item()

    ps.set_user_callback(main_callback)
    ps.show()


if __name__ == "__main__":
    main()
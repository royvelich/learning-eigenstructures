import polyscope as ps
import numpy as np
import torch
from neural_laplacian.datasets import ValidationDataset
from typing import Optional
import hydra
from omegaconf import DictConfig


def compute_heat_kernel(eigenvectors: torch.Tensor,
                        eigenvalues: torch.Tensor,
                        vertex_areas: torch.Tensor,
                        t: float,
                        source_idx: int,
                        max_eigenvectors: Optional[int] = None,
                        is_normalized_laplacian: bool = True) -> torch.Tensor:
    """
    Compute heat kernel using spectral decomposition.

    k_t,x(y) = Σ_i exp(-λ_i * t) * ψ_i(x) * ψ_i(y) * area_y

    Args:
        eigenvectors: Eigenvectors [N, K] - normalized Laplacian eigenfunctions if is_normalized_laplacian=True
        eigenvalues: Eigenvalues [K] - always for normalized Laplacian
        vertex_areas: Vertex areas [N]
        t: Time parameter
        source_idx: Index of source vertex
        max_eigenvectors: Maximum number of eigenvectors to use
        is_normalized_laplacian: Whether input eigenvectors are from normalized Laplacian

    Returns:
        Heat kernel values [N] from source to all vertices
    """
    if max_eigenvectors is None:
        max_eigenvectors = eigenvectors.shape[1]
    else:
        max_eigenvectors = min(max_eigenvectors, eigenvectors.shape[1])

    # Use subset of eigenvectors/eigenvalues
    eigenvectors_subset = eigenvectors[:, :max_eigenvectors]
    eigenvalues_subset = eigenvalues[:max_eigenvectors]

    # Convert normalized Laplacian eigenfunctions to non-normalized if needed
    if is_normalized_laplacian:
        # ψ_i = M^(-1/2) φ_i, where φ_i are normalized Laplacian eigenfunctions
        # and ψ_i are non-normalized Laplacian eigenfunctions
        M_inv_sqrt = torch.diag(1.0 / torch.sqrt(vertex_areas))
        psi = M_inv_sqrt @ eigenvectors_subset  # Convert to non-normalized eigenfunctions
    else:
        psi = eigenvectors_subset

    # For non-normalized Laplacian, the eigenvalues need to be scaled
    # If λ_norm are normalized Laplacian eigenvalues, then λ = λ_norm for the heat kernel
    lambda_values = eigenvalues_subset

    # Compute exponential decay factors
    exp_factors = torch.exp(-lambda_values * t)  # [K]

    # Get source eigenvector values (non-normalized eigenfunctions)
    source_values = psi[source_idx, :]  # [K]

    # Compute heat kernel: Σ_i exp(-λ_i * t) * ψ_i(source) * ψ_i(y)
    # Broadcasting: [K] * [K] * [N, K] -> [N, K] -> [N]
    heat_kernel = torch.sum(exp_factors[None, :] * source_values[None, :] * psi, dim=1)

    # Weight by vertex areas (mass matrix)
    heat_kernel = heat_kernel * vertex_areas

    return heat_kernel


def apply_varadhan_formula(heat_kernel: torch.Tensor, t: float, epsilon: float = 1e-12) -> torch.Tensor:
    """
    Apply Varadhan's formula to compute distances from heat kernel.

    φ(x,y) = -4t * log(k_t,x(y))

    Args:
        heat_kernel: Heat kernel values [N]
        t: Time parameter
        epsilon: Small value to avoid log(0)

    Returns:
        Distance approximation [N]
    """
    # Clamp heat kernel to avoid numerical issues
    clamped_kernel = torch.clamp(heat_kernel, min=epsilon)

    # Apply Varadhan's formula
    distances = -4 * t * torch.log(clamped_kernel)

    return distances


def compute_varadhan_distances(eigenvectors: torch.Tensor,
                               eigenvalues: torch.Tensor,
                               vertex_areas: torch.Tensor,
                               source_idx: int,
                               t: float,
                               max_eigenvectors: Optional[int] = None,
                               is_normalized_laplacian: bool = True) -> torch.Tensor:
    """
    Complete pipeline: compute heat kernel and apply Varadhan's formula.

    Args:
        eigenvectors: Eigenvectors [N, K]
        eigenvalues: Eigenvalues [K]
        vertex_areas: Vertex areas [N]
        source_idx: Index of source vertex
        t: Time parameter
        max_eigenvectors: Maximum number of eigenvectors to use
        is_normalized_laplacian: Whether input eigenvectors are from normalized Laplacian

    Returns:
        Distance approximation [N]
    """
    # Compute heat kernel
    heat_kernel = compute_heat_kernel(
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        vertex_areas=vertex_areas,
        t=t,
        source_idx=source_idx,
        max_eigenvectors=max_eigenvectors,
        is_normalized_laplacian=is_normalized_laplacian
    )

    # Apply Varadhan's formula
    distances = apply_varadhan_formula(heat_kernel=heat_kernel, t=t)

    # Set source distance to zero
    distances[source_idx] = 0.0

    return distances


class SimpleVaradhanController:
    """Simple controller for Varadhan formula evaluation."""

    def __init__(self, data, cfg: DictConfig):
        self.data = data
        self.cfg = cfg

        # Extract data
        self.vertices = data.pos.cpu().numpy()
        self.pred_eigenvectors = data.pred_eigenvectors.cpu()
        self.pred_eigenvalues = data.pred_eigenvalues.cpu()
        self.pred_vertex_areas = data.pred_vertex_areas.cpu()

        # Ground truth data
        self.gt_eigenvectors = data.gt_eigenvectors.cpu()
        self.gt_eigenvalues = data.gt_eigenvalues.cpu()
        self.gt_vertex_areas = data.gt_vertex_areas.cpu()

        # Generate random baseline eigenvectors
        self._generate_random_baseline()

        # Parameters from config
        self.t_pred = cfg.varadhan.predicted.initial_t
        self.min_t_pred = cfg.varadhan.predicted.min_t
        self.max_t_pred = cfg.varadhan.predicted.max_t
        self.max_eigenvectors_pred = min(cfg.varadhan.predicted.max_eigenvectors, self.pred_eigenvectors.shape[1])

        self.t_gt = cfg.varadhan.gt.initial_t
        self.min_t_gt = cfg.varadhan.gt.min_t
        self.max_t_gt = cfg.varadhan.gt.max_t
        self.max_eigenvectors_gt = min(cfg.varadhan.gt.max_eigenvectors, self.gt_eigenvectors.shape[1])

        self.t_random = cfg.varadhan.random.initial_t
        self.min_t_random = cfg.varadhan.random.min_t
        self.max_t_random = cfg.varadhan.random.max_t
        self.max_eigenvectors_random = min(cfg.varadhan.random.max_eigenvectors, self.random_eigenvectors.shape[1])

        self.source_idx = cfg.varadhan.source_idx

        # UI state
        self.show_predicted = False  # Checkbox state for predicted distances
        self.show_gt = True  # Checkbox state for GT distances
        self.show_random = False  # Checkbox state for random baseline distances

        # Register geometry
        self.ps_mesh = ps.register_point_cloud("point_cloud", self.vertices)
        self.ps_mesh.set_radius(cfg.visualization.point_radius)

        # Register source point
        source_vertex = self.vertices[self.source_idx].reshape(1, -1)
        ps.register_point_cloud("Source", source_vertex, color=(1.0, 0.0, 0.0))
        ps.get_point_cloud("Source").set_radius(cfg.visualization.source_radius)

        # Compute and display initial distances
        self._update_distances()

        print(f"Varadhan Formula Evaluation")
        print(f"Point cloud: {len(self.vertices)} vertices")
        print(f"Source vertex: {self.source_idx}")
        print(f"Predicted eigenvectors: {self.pred_eigenvectors.shape[1]}")
        print(f"Ground truth eigenvectors: {self.gt_eigenvectors.shape[1]}")
        print(f"Random baseline eigenvectors: {self.random_eigenvectors.shape[1]}")

    def _generate_random_baseline(self):
        """Generate random orthogonal eigenvectors as baseline."""
        # Use same shape as predicted eigenvectors
        n_vertices, n_eigenvectors = self.pred_eigenvectors.shape

        # Generate random matrix
        random_matrix = torch.randn(n_vertices, n_eigenvectors, dtype=torch.float32)

        # Apply QR decomposition to get orthogonal vectors
        self.random_eigenvectors, _ = torch.linalg.qr(random_matrix)

        # Use predicted eigenvalues and vertex areas for the baseline
        self.random_eigenvalues = self.pred_eigenvalues.clone()
        self.random_vertex_areas = self.pred_vertex_areas.clone()

        print(f"Generated random baseline with {n_eigenvectors} orthogonal eigenvectors")

    def _update_distances(self):
        """Compute and update distance visualizations."""

        # Compute predicted distances
        pred_distances = compute_varadhan_distances(
            eigenvectors=self.pred_eigenvectors,
            eigenvalues=self.pred_eigenvalues,
            vertex_areas=self.pred_vertex_areas,
            source_idx=self.source_idx,
            t=self.t_pred,
            max_eigenvectors=self.max_eigenvectors_pred,
            is_normalized_laplacian=True
        ).numpy()

        # Add predicted distances to mesh
        # Exclude source point from color limit calculation (it's always 0)
        pred_distances_no_source = np.delete(pred_distances, self.source_idx)
        pred_min, pred_max = pred_distances_no_source.min(), pred_distances_no_source.max()
        self.ps_mesh.add_scalar_quantity(
            "Predicted Distances",
            pred_distances,
            enabled=self.show_predicted,  # Use checkbox state
            cmap="coolwarm",
            vminmax=(pred_min, pred_max)
        )

        # Compute ground truth distances
        gt_distances = compute_varadhan_distances(
            eigenvectors=self.gt_eigenvectors,
            eigenvalues=self.gt_eigenvalues,
            vertex_areas=self.gt_vertex_areas,
            source_idx=self.source_idx,
            t=self.t_gt,
            max_eigenvectors=self.max_eigenvectors_gt,
            is_normalized_laplacian=False
        ).numpy()

        # Add ground truth distances to mesh
        # Exclude source point from color limit calculation (it's always 0)
        gt_distances_no_source = np.delete(gt_distances, self.source_idx)
        gt_min, gt_max = gt_distances_no_source.min(), gt_distances_no_source.max()
        self.ps_mesh.add_scalar_quantity(
            "Ground Truth Distances",
            gt_distances,
            enabled=self.show_gt,  # Use checkbox state
            cmap="coolwarm",
            vminmax=(gt_min, gt_max)
        )

        # Compute random baseline distances
        random_distances = compute_varadhan_distances(
            eigenvectors=self.random_eigenvectors,
            eigenvalues=self.random_eigenvalues,
            vertex_areas=self.random_vertex_areas,
            source_idx=self.source_idx,
            t=self.t_random,
            max_eigenvectors=self.max_eigenvectors_random,
            is_normalized_laplacian=True  # Random baseline uses same setup as predicted
        ).numpy()

        # Add random baseline distances to mesh
        # Exclude source point from color limit calculation (it's always 0)
        random_distances_no_source = np.delete(random_distances, self.source_idx)
        random_min, random_max = random_distances_no_source.min(), random_distances_no_source.max()
        self.ps_mesh.add_scalar_quantity(
            "Random Baseline Distances",
            random_distances,
            enabled=self.show_random,  # Use checkbox state
            cmap="coolwarm",
            vminmax=(random_min, random_max)
        )

        print(f"Updated distances:")
        print(f"  Predicted: t={self.t_pred:.6f}, max_ev={self.max_eigenvectors_pred}, range=[{pred_min:.4f}, {pred_max:.4f}] (enabled: {self.show_predicted})")
        print(f"  Ground Truth: t={self.t_gt:.6f}, max_ev={self.max_eigenvectors_gt}, range=[{gt_min:.4f}, {gt_max:.4f}] (enabled: {self.show_gt})")
        print(f"  Random Baseline: t={self.t_random:.6f}, max_ev={self.max_eigenvectors_random}, range=[{random_min:.4f}, {random_max:.4f}] (enabled: {self.show_random})")

    def gui_callback(self):
        """Simple GUI with separate controls for predicted and GT."""
        import polyscope.imgui as psim

        if psim.TreeNode("Varadhan Controls"):

            # Initialize change flags
            changed_t_pred = False
            changed_ev_pred = False
            changed_t_gt = False
            changed_ev_gt = False
            changed_t_random = False
            changed_ev_random = False

            # Predicted parameters section
            if psim.TreeNode("Predicted Parameters"):
                # Predicted time parameter slider
                changed_t_pred, self.t_pred = psim.SliderFloat(
                    "Predicted Time t",
                    self.t_pred,
                    self.min_t_pred,
                    self.max_t_pred,
                    "%.6f"
                )

                # Predicted max eigenvectors slider
                changed_ev_pred, self.max_eigenvectors_pred = psim.SliderInt(
                    "Predicted Max Eigenvectors",
                    self.max_eigenvectors_pred,
                    1,
                    self.pred_eigenvectors.shape[1]
                )

                psim.TreePop()

            # Ground Truth parameters section
            if psim.TreeNode("Ground Truth Parameters"):
                # GT time parameter slider
                changed_t_gt, self.t_gt = psim.SliderFloat(
                    "GT Time t",
                    self.t_gt,
                    self.min_t_gt,
                    self.max_t_gt,
                    "%.6f"
                )

                # GT max eigenvectors slider
                changed_ev_gt, self.max_eigenvectors_gt = psim.SliderInt(
                    "GT Max Eigenvectors",
                    self.max_eigenvectors_gt,
                    1,
                    self.gt_eigenvectors.shape[1]
                )

                psim.TreePop()

            # Random Baseline parameters section
            if psim.TreeNode("Random Baseline Parameters"):
                # Random time parameter slider
                changed_t_random, self.t_random = psim.SliderFloat(
                    "Random Time t",
                    self.t_random,
                    self.min_t_random,
                    self.max_t_random,
                    "%.6f"
                )

                # Random max eigenvectors slider
                changed_ev_random, self.max_eigenvectors_random = psim.SliderInt(
                    "Random Max Eigenvectors",
                    self.max_eigenvectors_random,
                    1,
                    self.random_eigenvectors.shape[1]
                )

                psim.TreePop()

            psim.Separator()

            # Checkboxes for showing predicted, GT, and random distances
            changed_pred, self.show_predicted = psim.Checkbox("Show Predicted", self.show_predicted)
            changed_gt, self.show_gt = psim.Checkbox("Show Ground Truth", self.show_gt)
            changed_random, self.show_random = psim.Checkbox("Show Random Baseline", self.show_random)

            # Update if any parameter changed
            if (changed_t_pred or changed_ev_pred or changed_t_gt or changed_ev_gt or
                    changed_t_random or changed_ev_random or changed_pred or changed_gt or changed_random):
                self._update_distances()

            psim.Separator()

            # Display current values
            psim.Text(f"Predicted: t={self.t_pred:.6f}, max_ev={self.max_eigenvectors_pred}")
            psim.Text(f"GT: t={self.t_gt:.6f}, max_ev={self.max_eigenvectors_gt}")
            psim.Text(f"Random: t={self.t_random:.6f}, max_ev={self.max_eigenvectors_random}")
            psim.Text(f"Source vertex: {self.source_idx}")

            psim.TreePop()


def run_simple_varadhan(data, cfg: DictConfig):
    """Run simple Varadhan evaluation."""

    # Create controller
    controller = SimpleVaradhanController(data, cfg)

    # Set callback
    ps.set_user_callback(controller.gui_callback)

    print("\nStarting Varadhan evaluation...")
    print("Use the GUI sliders to adjust t and max eigenvectors")
    print("Use Polyscope's default UI to show/hide predicted and ground truth distances")

    # Show visualization
    ps.show()


@hydra.main(version_base="1.2", config_path="./config/playgrounds")
def main(cfg: DictConfig):
    # Initialize Polyscope
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color((0.1, 0.1, 0.1, 1.0))

    # Load validation dataset
    dataset = ValidationDataset(root_dir=cfg.data.path)
    print(f"Loaded validation dataset with {len(dataset)} items")

    for idx in range(len(dataset)):
        print(f"\nProcessing validation item {idx + 1}/{len(dataset)}")

        # Get data item
        data = dataset.get(idx)

        # Clear previous visualization
        ps.remove_all_structures()

        # Run Varadhan evaluation
        run_simple_varadhan(data, cfg)

        # Ask user if they want to continue to next item
        user_input = input(f"Item {idx + 1}/{len(dataset)} complete. Press Enter to continue, 'q' to quit: ")
        if user_input.lower() == 'q':
            break


if __name__ == "__main__":
    main()
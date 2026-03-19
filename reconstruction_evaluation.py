# standard library
import time
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

# numpy
import numpy as np

# hydra
import hydra

# omegaconf
from omegaconf import DictConfig
import pytorch_lightning as pl

# torch
import torch

# torch geometric
from torch_geometric.data import Data, Batch

# point-cloud-utils
import point_cloud_utils as pcu

# neural-laplacian
from neural_laplacian import utils
from neural_laplacian.modules.laplacian_modules import LaplacianPredictorModule3D


@dataclass
class ReconstructionMetrics:
    """Container for reconstruction quality metrics."""
    mesh_idx: int
    k: int
    method: str  # 'pred' or 'robust'
    chamfer_distance: float
    hausdorff_distance: float


@dataclass
class AggregatedMetrics:
    """Container for aggregated statistics across all meshes."""
    k: int
    method: str
    chamfer_mean: float
    chamfer_min: float
    chamfer_max: float
    chamfer_std: float
    hausdorff_mean: float
    hausdorff_min: float
    hausdorff_max: float
    hausdorff_std: float
    num_samples: int


class ReconstructionEvaluator:
    """Evaluates reconstruction quality using distance metrics."""

    def __init__(self, cfg: DictConfig, model: LaplacianPredictorModule3D, k_values: List[int]):
        """
        Initialize the evaluator.

        Args:
            cfg: Configuration object
            model: Trained model for predictions
            k_values: List of k values to evaluate
        """
        self.cfg = cfg
        self.model = model
        self.k_values = k_values
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize dataset
        self.dataset = self._create_dataset()

        # Storage for all metrics
        self.all_metrics: List[ReconstructionMetrics] = []

    def _create_dataset(self):
        """Create dataset from config."""
        data_module = hydra.utils.instantiate(self.cfg.data_module.module)
        dataset = data_module.train_dataloader().dataset
        print(f"Dataset created with {len(dataset)} items")
        return dataset

    def _compute_model_predictions(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute neural network predictions for Laplacian eigenvectors.

        Returns:
            pred_eigenvectors, pred_weights, n_eig
        """
        batch = Batch.from_data_list([data])
        with torch.no_grad():
            laplacian_prediction, processed_batch = self.model.predict_step(batch, batch_idx=0)

        # Extract predictions
        pred_eigenvectors = -laplacian_prediction.eigenvectors_list[0].cpu()
        pred_weights = laplacian_prediction.weights_list[0].cpu()
        n_eig = pred_eigenvectors.shape[1]

        return pred_eigenvectors, pred_weights, n_eig

    def _compute_ground_truth_laplacian(self, data: Data, n_eig: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ground truth Laplacian eigenfunctions.

        Returns:
            gt_eigenvectors, gt_eigenvalues, gt_weights
        """
        # Check if we have valid faces
        has_faces = hasattr(data, 'faces') and data.faces is not None
        if has_faces:
            faces_array = data.faces.cpu().numpy() if isinstance(data.faces, torch.Tensor) else data.faces
            has_valid_faces = faces_array.shape[0] > 0
        else:
            has_valid_faces = False

        if not has_valid_faces:
            print("Warning: No face connectivity - skipping GT computation")
            return np.array([]), np.array([]), np.array([])

        gt_eigenvectors, gt_eigenvalues, gt_weights = utils.compute_pyfm_normalized_laplacian_eigenfunctions(
            vertices=data.points.cpu().numpy(),
            faces=faces_array,
            num_eigenfunctions=n_eig
        )

        return gt_eigenvectors, gt_eigenvalues, gt_weights

    def _compute_robust_laplacian(self, data: Data, n_eig: int, n_neighbors: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute robust Laplacian eigenvectors.

        Returns:
            robust_eigenvectors, robust_eigenvalues, robust_weights
        """
        robust_eigenvectors, robust_eigenvalues, robust_weights = utils.compute_robust_normalized_laplacian_eigenvectors(
            vertices=data.points.cpu().numpy(),
            num_eigenfunctions=n_eig,
            n_neighbors=n_neighbors
        )

        return robust_eigenvectors, robust_eigenvalues, robust_weights

    def _reconstruct_points(self, eigenvectors: np.ndarray, weights: np.ndarray,
                            original_points: np.ndarray, k: int) -> np.ndarray:
        """
        Reconstruct point cloud using first k eigenvectors.

        Args:
            eigenvectors: Eigenvector basis [N, num_eigenvectors]
            weights: Vertex area weights [N]
            original_points: Original point coordinates [N, 3]
            k: Number of eigenvectors to use

        Returns:
            Reconstructed points [N, 3]
        """
        # Convert to torch tensors if needed
        if isinstance(eigenvectors, np.ndarray):
            eigenvectors = torch.from_numpy(eigenvectors).float()
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float()
        if isinstance(original_points, np.ndarray):
            original_points = torch.from_numpy(original_points).float()

        # Scale eigenvectors by weights (following visualization.py logic)
        eigenvectors_weighted = utils.scale_by_half_inv(scalar_functions=eigenvectors, weights=weights)

        # Project and reconstruct using first k eigenvectors
        reconstructed = utils.project_functions_unnormalized(
            eigenvectors_basis=eigenvectors_weighted,
            scalar_functions=original_points,
            weights=weights,
            max_eigenvectors=k
        )

        # Get the reconstruction for k eigenvectors (it returns all up to k)
        # Shape should be [k, N, 3], we want the last one (index k-1)
        if reconstructed.ndim == 3:
            reconstructed = reconstructed[k - 1]  # Get reconstruction with exactly k eigenvectors

        result = reconstructed.numpy()

        # Ensure output is [N, 3]
        if result.ndim != 2 or result.shape[1] != 3:
            raise ValueError(f"Reconstruction has unexpected shape: {result.shape}, expected [N, 3]")

        return result

    def _compute_distances(self, points_a: np.ndarray, points_b: np.ndarray) -> Tuple[float, float]:
        """
        Compute Chamfer and Hausdorff distances between two point clouds.

        Args:
            points_a: First point cloud [N, 3]
            points_b: Second point cloud [N, 3]

        Returns:
            chamfer_distance, hausdorff_distance
        """
        # Ensure arrays are 2D with shape [N, 3]
        if points_a.ndim != 2 or points_a.shape[1] != 3:
            print(f"Warning: points_a has unexpected shape {points_a.shape}, reshaping...")
            points_a = points_a.reshape(-1, 3)

        if points_b.ndim != 2 or points_b.shape[1] != 3:
            print(f"Warning: points_b has unexpected shape {points_b.shape}, reshaping...")
            points_b = points_b.reshape(-1, 3)

        # Ensure contiguous arrays (pcu might require this)
        points_a = np.ascontiguousarray(points_a, dtype=np.float64)
        points_b = np.ascontiguousarray(points_b, dtype=np.float64)

        # Compute Chamfer distance
        chamfer_dist = pcu.chamfer_distance(points_a, points_b)

        # Compute Hausdorff distance
        hausdorff_dist = pcu.hausdorff_distance(points_a, points_b)

        return float(chamfer_dist), float(hausdorff_dist)

    def evaluate_mesh(self, mesh_idx: int) -> List[ReconstructionMetrics]:
        """
        Evaluate reconstruction quality for a single mesh.

        Args:
            mesh_idx: Index of mesh in dataset

        Returns:
            List of metrics for all k values and methods
        """
        print(f"\n{'=' * 80}")
        print(f"Processing mesh {mesh_idx + 1}/{len(self.dataset)}")
        print(f"{'=' * 80}")

        # Get data
        data = self.dataset[mesh_idx].to(self.device)
        original_points = data.points.cpu().numpy()

        metrics_list = []

        try:
            # Step 1: Compute predictions
            print("Computing model predictions...")
            start_time = time.time()
            pred_eigenvectors, pred_weights, n_eig = self._compute_model_predictions(data)
            print(f"  Prediction time: {time.time() - start_time:.4f}s")
            print(f"  Predicted {n_eig} eigenvectors")

            # Step 2: Compute ground truth
            print("Computing ground truth...")
            start_time = time.time()
            gt_eigenvectors, gt_eigenvalues, gt_weights = self._compute_ground_truth_laplacian(data, n_eig)
            print(f"  GT computation time: {time.time() - start_time:.4f}s")

            # Check if GT is valid
            if gt_eigenvectors.shape[0] == 0:
                print("  Skipping mesh - no GT available (point cloud only)")
                return metrics_list

            # Step 3: Compute robust Laplacian
            print("Computing robust Laplacian...")
            start_time = time.time()
            robust_eigenvectors, robust_eigenvalues, robust_weights = self._compute_robust_laplacian(data, n_eig)
            print(f"  Robust computation time: {time.time() - start_time:.4f}s")

            # Step 4: Align signs (using robust as reference, like visualization.py)
            print("Aligning eigenvector signs...")
            pred_eigenvectors = utils.align_eigenvector_signs(pred_eigenvectors, robust_eigenvectors)
            gt_eigenvectors = utils.align_eigenvector_signs(gt_eigenvectors, robust_eigenvectors)

            # Step 5: Evaluate for each k value
            for k in self.k_values:
                if k > n_eig:
                    print(f"  Skipping k={k} (exceeds available eigenvectors: {n_eig})")
                    continue

                print(f"\n  Evaluating k={k}...")

                # Reconstruct using GT eigenvectors
                gt_recon = self._reconstruct_points(gt_eigenvectors, gt_weights, original_points, k)

                # Reconstruct using predicted eigenvectors
                pred_recon = self._reconstruct_points(pred_eigenvectors.numpy(), pred_weights.numpy(),
                                                      original_points, k)

                # Reconstruct using robust eigenvectors
                robust_recon = self._reconstruct_points(robust_eigenvectors, robust_weights,
                                                        original_points, k)

                # Compute distances: Pred vs GT
                chamfer_pred, hausdorff_pred = self._compute_distances(pred_recon, gt_recon)
                print(f"    Pred vs GT - Chamfer: {chamfer_pred:.6f}, Hausdorff: {hausdorff_pred:.6f}")

                metrics_list.append(ReconstructionMetrics(
                    mesh_idx=mesh_idx,
                    k=k,
                    method='pred',
                    chamfer_distance=chamfer_pred,
                    hausdorff_distance=hausdorff_pred
                ))

                # Compute distances: Robust vs GT
                chamfer_robust, hausdorff_robust = self._compute_distances(robust_recon, gt_recon)
                print(f"    Robust vs GT - Chamfer: {chamfer_robust:.6f}, Hausdorff: {hausdorff_robust:.6f}")

                metrics_list.append(ReconstructionMetrics(
                    mesh_idx=mesh_idx,
                    k=k,
                    method='robust',
                    chamfer_distance=chamfer_robust,
                    hausdorff_distance=hausdorff_robust
                ))

        except Exception as e:
            print(f"Error processing mesh {mesh_idx}: {e}")
            import traceback
            traceback.print_exc()

        return metrics_list

    def evaluate_all(self) -> List[ReconstructionMetrics]:
        """
        Evaluate all meshes in the dataset.

        Returns:
            List of all metrics
        """
        print(f"\n{'#' * 80}")
        print(f"Starting evaluation on {len(self.dataset)} meshes")
        print(f"K values: {self.k_values}")
        print(f"{'#' * 80}\n")

        total_start_time = time.time()

        for mesh_idx in range(len(self.dataset)):
            mesh_metrics = self.evaluate_mesh(mesh_idx)
            self.all_metrics.extend(mesh_metrics)

        total_time = time.time() - total_start_time
        print(f"\n{'#' * 80}")
        print(f"Evaluation complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per mesh: {total_time / len(self.dataset):.2f}s")
        print(f"{'#' * 80}\n")

        return self.all_metrics

    def compute_aggregated_statistics(self) -> List[AggregatedMetrics]:
        """
        Compute aggregated statistics for each k and method.

        Returns:
            List of aggregated metrics
        """
        aggregated = []

        for k in self.k_values:
            for method in ['pred', 'robust']:
                # Filter metrics for this k and method
                filtered = [m for m in self.all_metrics if m.k == k and m.method == method]

                if not filtered:
                    continue

                # Extract distances
                chamfer_dists = [m.chamfer_distance for m in filtered]
                hausdorff_dists = [m.hausdorff_distance for m in filtered]

                # Compute statistics
                aggregated.append(AggregatedMetrics(
                    k=k,
                    method=method,
                    chamfer_mean=float(np.mean(chamfer_dists)),
                    chamfer_min=float(np.min(chamfer_dists)),
                    chamfer_max=float(np.max(chamfer_dists)),
                    chamfer_std=float(np.std(chamfer_dists)),
                    hausdorff_mean=float(np.mean(hausdorff_dists)),
                    hausdorff_min=float(np.min(hausdorff_dists)),
                    hausdorff_max=float(np.max(hausdorff_dists)),
                    hausdorff_std=float(np.std(hausdorff_dists)),
                    num_samples=len(filtered)
                ))

        return aggregated

    def print_results(self, aggregated_metrics: List[AggregatedMetrics]):
        """Print aggregated results to console."""
        print("\n" + "=" * 100)
        print("AGGREGATED RESULTS")
        print("=" * 100)

        for k in self.k_values:
            print(f"\n{'─' * 100}")
            print(f"K = {k}")
            print(f"{'─' * 100}")

            for method in ['pred', 'robust']:
                metrics = [m for m in aggregated_metrics if m.k == k and m.method == method]
                if not metrics:
                    continue

                m = metrics[0]
                method_name = "Predicted" if method == 'pred' else "Robust"

                print(f"\n{method_name} vs Ground Truth (n={m.num_samples}):")
                print(f"  Chamfer Distance:")
                print(f"    Mean: {m.chamfer_mean:.6f}")
                print(f"    Std:  {m.chamfer_std:.6f}")
                print(f"    Min:  {m.chamfer_min:.6f}")
                print(f"    Max:  {m.chamfer_max:.6f}")
                print(f"  Hausdorff Distance:")
                print(f"    Mean: {m.hausdorff_mean:.6f}")
                print(f"    Std:  {m.hausdorff_std:.6f}")
                print(f"    Min:  {m.hausdorff_min:.6f}")
                print(f"    Max:  {m.hausdorff_max:.6f}")

        print("\n" + "=" * 100 + "\n")

    def save_results_to_csv(self, aggregated_metrics: List[AggregatedMetrics],
                            output_dir: str = "evaluation_results"):
        """
        Save results to CSV files.

        Args:
            aggregated_metrics: List of aggregated metrics
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save per-mesh metrics
        per_mesh_file = output_path / "per_mesh_metrics.csv"
        with open(per_mesh_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mesh_idx', 'k', 'method', 'chamfer_distance', 'hausdorff_distance'])
            for m in self.all_metrics:
                writer.writerow([m.mesh_idx, m.k, m.method, m.chamfer_distance, m.hausdorff_distance])

        print(f"Saved per-mesh metrics to: {per_mesh_file}")

        # Save aggregated statistics
        aggregated_file = output_path / "aggregated_statistics.csv"
        with open(aggregated_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'k', 'method', 'num_samples',
                'chamfer_mean', 'chamfer_std', 'chamfer_min', 'chamfer_max',
                'hausdorff_mean', 'hausdorff_std', 'hausdorff_min', 'hausdorff_max'
            ])
            for m in aggregated_metrics:
                writer.writerow([
                    m.k, m.method, m.num_samples,
                    m.chamfer_mean, m.chamfer_std, m.chamfer_min, m.chamfer_max,
                    m.hausdorff_mean, m.hausdorff_std, m.hausdorff_min, m.hausdorff_max
                ])

        print(f"Saved aggregated statistics to: {aggregated_file}")


@hydra.main(version_base="1.2", config_path="config/training")
def main(cfg: DictConfig):
    """Main function for reconstruction evaluation."""

    # Seed for reproducibility
    pl.seed_everything(cfg.globals.seed)

    # Check if checkpoint path is provided
    if not hasattr(cfg, 'checkpoint_path') or cfg.checkpoint_path is None:
        raise ValueError("Please provide a checkpoint path using ++checkpoint_path=<path>")

    # Load the model from checkpoint
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model = LaplacianPredictorModule3D.load_from_checkpoint(cfg.checkpoint_path)
    model.eval()

    # K values to evaluate
    k_values = [5, 10, 20, 50]

    # Create evaluator
    evaluator = ReconstructionEvaluator(cfg, model, k_values)

    # Run evaluation
    all_metrics = evaluator.evaluate_all()

    # Compute aggregated statistics
    aggregated_metrics = evaluator.compute_aggregated_statistics()

    # Print results to console
    evaluator.print_results(aggregated_metrics)

    # Save results to CSV
    evaluator.save_results_to_csv(aggregated_metrics)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
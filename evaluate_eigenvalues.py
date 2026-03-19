"""
Batch Eigenvalue Evaluation Script

This script iterates over checkpoint-shape pairs and evaluates eigenvalues
across different parameter combinations (k-NN, smoothing iterations, sigma).

Usage:
    python batch_eigenvalue_evaluation.py \
        ++checkpoints_dir=/path/to/checkpoints \
        ++shapes_dir=/path/to/shapes \
        ++output_dir=/path/to/results

Note: This script uses Hydra and needs to be run from the project directory
      with access to config/training directory.
"""

# standard library
import csv
from pathlib import Path
from typing import List, Tuple, Dict

# numpy
import numpy as np

# torch
import torch
from torch_geometric.data import Batch

# pytorch lightning
import pytorch_lightning as pl

# hydra
import hydra
from omegaconf import DictConfig, OmegaConf

# neural-laplacian
from neural_laplacian import utils
from neural_laplacian.modules.laplacian_modules import LaplacianPredictorModule3D


class BatchEigenvalueEvaluator:
    """Evaluates eigenvalues for checkpoint-shape pairs across parameter combinations."""

    def __init__(self,
                 cfg: DictConfig,
                 checkpoints_dir: Path,
                 shapes_dir: Path,
                 output_dir: Path,
                 param_combinations: List[Tuple[int, int, float]]):
        """
        Initialize the batch evaluator.

        Args:
            cfg: Hydra configuration (used for dataset creation)
            checkpoints_dir: Directory containing .ckpt files
            shapes_dir: Directory containing subdirectories (named after checkpoints) with shapes
            output_dir: Directory to save results
            param_combinations: List of (knn, smoothing_iter, sigma) tuples
        """
        self.cfg = cfg
        self.checkpoints_dir = Path(checkpoints_dir)
        self.shapes_dir = Path(shapes_dir)
        self.output_dir = Path(output_dir)
        self.param_combinations = param_combinations

        # Create output directories
        self.per_shape_dir = self.output_dir / "per_shape"
        self.per_param_dir = self.output_dir / "per_param_combination"
        self.per_shape_dir.mkdir(parents=True, exist_ok=True)
        self.per_param_dir.mkdir(parents=True, exist_ok=True)

        # Storage for cross-shape aggregation
        # Structure: {(knn, iter, sigma): [shape1_eigenvalues, shape2_eigenvalues, ...]}
        self.results_by_param = {combo: [] for combo in param_combinations}

        # Storage for per-shape best combinations
        # Structure: {shape_name: {'best_combo': (k, iter, sigma), 'best_error': float}}
        self.per_shape_best = {}

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Set seed from config
        pl.seed_everything(cfg.globals.seed)

    def find_checkpoint_shape_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find matching checkpoint-shape directory pairs.

        Returns:
            List of (checkpoint_path, shape_dir) tuples
        """
        pairs = []

        # Find all checkpoint files
        checkpoint_files = sorted(self.checkpoints_dir.glob("*.ckpt"))

        for ckpt_path in checkpoint_files:
            # Get checkpoint name without extension
            ckpt_name = ckpt_path.stem

            # Find corresponding shape directory
            shape_dir = self.shapes_dir / ckpt_name

            if shape_dir.exists() and shape_dir.is_dir():
                pairs.append((ckpt_path, shape_dir))
                print(f"Found pair: {ckpt_name}")
            else:
                print(f"Warning: No matching shape directory for checkpoint {ckpt_name}")

        print(f"\nFound {len(pairs)} checkpoint-shape pairs")
        return pairs

    def load_model(self, checkpoint_path: Path) -> LaplacianPredictorModule3D:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded model
        """
        print(f"\nLoading model from: {checkpoint_path}")

        # Load model
        model = LaplacianPredictorModule3D.load_from_checkpoint(str(checkpoint_path))
        model.eval()
        model = model.to(self.device)

        return model

    def create_dataset_with_override(self, shape_dir: Path):
        """
        Create dataset by overriding root_dirs in the config.

        Args:
            shape_dir: Path to shape directory to use

        Returns:
            Dataset object
        """
        # Clone the config to avoid modifying original
        cfg_copy = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))

        # Override root_dirs in the dataset specification
        # The structure is: data_module.module.train_dataset_specification.dataset.root_dirs
        if hasattr(cfg_copy, 'data_module') and hasattr(cfg_copy.data_module, 'module'):
            if hasattr(cfg_copy.data_module.module, 'train_dataset_specification'):
                train_spec = cfg_copy.data_module.module.train_dataset_specification

                # Access the dataset config
                if hasattr(train_spec, 'dataset'):
                    # Override root_dirs with proper structure for pathlib.Path
                    train_spec.dataset.root_dirs = [
                        {
                            '_target_': 'pathlib.Path',
                            '_args_': [str(shape_dir)]
                        }
                    ]
                    print(f"Overriding root_dirs to: {shape_dir}")

                # Disable shuffle for consistent behavior
                if hasattr(train_spec, 'shuffle'):
                    train_spec.shuffle = False

        # Instantiate data module
        data_module = hydra.utils.instantiate(cfg_copy.data_module.module)
        dataset = data_module.train_dataloader().dataset

        print(f"Dataset created with {len(dataset)} items")
        return dataset

    def compute_eigenvalues_for_params(self,
                                       model: LaplacianPredictorModule3D,
                                       data,
                                       knn: int,
                                       smoothing_iter: int,
                                       sigma: float) -> Dict[str, np.ndarray]:
        """
        Compute eigenvalues for a single parameter combination.

        Args:
            model: Trained model
            data: Data object
            knn: k-NN parameter for predicted Laplacian
            smoothing_iter: Smoothing iterations
            sigma: Smoothing sigma

        Returns:
            Dictionary with keys: 'pred', 'gt' containing eigenvalue arrays (all as numpy arrays)
        """
        print(f"  Computing with knn={knn}, smoothing_iter={smoothing_iter}, sigma={sigma:.4f}")

        # Set model parameters
        if hasattr(model, '_knn_graph_config'):
            model._knn_graph_config.k = knn

        if hasattr(model, '_scalar_field_smoother'):
            model._scalar_field_smoother.iterations = smoothing_iter
            model._scalar_field_smoother.sigma = sigma

        results = {}

        # 1. Compute predicted eigenvalues
        try:
            batch = Batch.from_data_list([data])
            with torch.no_grad():
                laplacian_prediction, processed_batch = model.predict_step(batch, batch_idx=0)

            # Get predicted eigenvalues (prepend zero eigenvalue)
            pred_eigenvalues = laplacian_prediction.unweighted_eigenvalues_list[0].cpu().numpy()
            zero_eigenvalue = np.array([0.0], dtype=pred_eigenvalues.dtype)
            pred_eigenvalues = np.concatenate([zero_eigenvalue, pred_eigenvalues])
            pred_eigenvalues = pred_eigenvalues[:-1]  # Remove last to match original behavior

            results['pred'] = pred_eigenvalues
            print(f"    Predicted: {len(pred_eigenvalues)} eigenvalues")

        except Exception as e:
            print(f"    Error computing predicted eigenvalues: {e}")
            results['pred'] = np.array([])

        # 2. Compute ground truth eigenvalues (if mesh has faces)
        try:
            has_faces = hasattr(data, 'faces') and data.faces is not None
            if has_faces:
                faces_array = data.faces.cpu().numpy() if isinstance(data.faces, torch.Tensor) else data.faces
                has_valid_faces = faces_array.shape[0] > 0
            else:
                has_valid_faces = False

            if has_valid_faces:
                n_eig = len(results['pred'])
                vertices = data.points.cpu().numpy() if isinstance(data.points, torch.Tensor) else data.points

                gt_eigenvectors, gt_eigenvalues, gt_weights = utils.compute_pyfm_normalized_laplacian_eigenfunctions(
                    vertices=vertices,
                    faces=faces_array,
                    num_eigenfunctions=n_eig
                )

                # Ensure gt_eigenvalues is numpy array
                if isinstance(gt_eigenvalues, torch.Tensor):
                    gt_eigenvalues = gt_eigenvalues.cpu().numpy()

                results['gt'] = gt_eigenvalues
                print(f"    Ground truth: {len(gt_eigenvalues)} eigenvalues")
            else:
                print(f"    Skipping GT - no face connectivity")
                results['gt'] = np.array([])

        except Exception as e:
            print(f"    Error computing GT eigenvalues: {e}")
            results['gt'] = np.array([])

        return results

    def save_eigenvalues_csv(self,
                            eigenvalues_dict: Dict[str, np.ndarray],
                            output_path: Path,
                            knn: int,
                            smoothing_iter: int,
                            sigma: float):
        """
        Save eigenvalues to CSV file.

        Args:
            eigenvalues_dict: Dictionary with 'pred', 'gt' eigenvalue arrays
            output_path: Path to save CSV file
            knn: k-NN parameter used
            smoothing_iter: Smoothing iterations used
            sigma: Sigma parameter used
        """
        # Determine the maximum number of eigenvalues
        max_len = max(
            len(eigenvalues_dict.get('pred', [])),
            len(eigenvalues_dict.get('gt', []))
        )

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header with parameter info
            writer.writerow(['# Parameters:', f'knn={knn}', f'smoothing_iter={smoothing_iter}', f'sigma={sigma}'])
            writer.writerow([])  # Empty row

            # Write column headers
            writer.writerow(['eigenvalue_index', 'pred_eigenvalue', 'gt_eigenvalue'])

            # Write eigenvalues
            for i in range(max_len):
                pred_val = eigenvalues_dict['pred'][i] if i < len(eigenvalues_dict['pred']) else ''
                gt_val = eigenvalues_dict['gt'][i] if i < len(eigenvalues_dict['gt']) else ''

                writer.writerow([i, pred_val, gt_val])

        print(f"    Saved eigenvalues to: {output_path}")

        # Return the dictionary for cross-shape aggregation
        return eigenvalues_dict

    def aggregate_across_shapes_for_param(self, param_combo: Tuple[int, int, float]):
        """
        Compute aggregate statistics across all shapes for a specific parameter combination.

        Args:
            param_combo: (knn, smoothing_iter, sigma) tuple
        """
        knn, smoothing_iter, sigma = param_combo
        print(f"\n  Aggregating across shapes for knn={knn}, iter={smoothing_iter}, sigma={sigma:.4f}")

        results_list = self.results_by_param[param_combo]

        if not results_list:
            print(f"    No results to aggregate for this parameter combination")
            return

        # Collect all eigenvalues by type
        all_pred = []
        all_gt = []
        all_abs_errors = []  # |pred - gt|
        all_rel_errors = []  # |pred - gt| / |gt| * 100

        for eigenvalues_dict in results_list:
            if len(eigenvalues_dict['pred']) > 0:
                all_pred.append(eigenvalues_dict['pred'])
            if len(eigenvalues_dict['gt']) > 0:
                all_gt.append(eigenvalues_dict['gt'])

            # Compute errors if GT is available
            if len(eigenvalues_dict['gt']) > 0 and len(eigenvalues_dict['pred']) > 0:
                min_len = min(len(eigenvalues_dict['gt']), len(eigenvalues_dict['pred']))

                # Absolute error
                abs_error = np.abs(eigenvalues_dict['pred'][:min_len] - eigenvalues_dict['gt'][:min_len])
                all_abs_errors.append(abs_error)

                # Relative error (percentage)
                gt_nonzero = eigenvalues_dict['gt'][:min_len] != 0
                rel_error = np.zeros(min_len)
                if np.any(gt_nonzero):
                    rel_error[gt_nonzero] = np.abs(eigenvalues_dict['pred'][:min_len][gt_nonzero] - eigenvalues_dict['gt'][:min_len][gt_nonzero]) / np.abs(eigenvalues_dict['gt'][:min_len][gt_nonzero]) * 100
                all_rel_errors.append(rel_error)

        # Compute statistics
        def compute_stats_with_errors(eigenvalue_lists, abs_error_lists, rel_error_lists):
            """Compute mean, std, min, max for eigenvalues and errors."""
            if not eigenvalue_lists:
                return None

            max_len = max(len(e) for e in eigenvalue_lists)

            stats = {
                'mean': [],
                'std': [],
                'min': [],
                'max': [],
                'abs_error_mean': [],
                'abs_error_std': [],
                'abs_error_min': [],
                'abs_error_max': [],
                'rel_error_mean': [],
                'rel_error_std': [],
                'rel_error_min': [],
                'rel_error_max': []
            }

            for i in range(max_len):
                # Eigenvalue statistics
                values = [e[i] for e in eigenvalue_lists if i < len(e)]
                if values:
                    stats['mean'].append(np.mean(values))
                    stats['std'].append(np.std(values))
                    stats['min'].append(np.min(values))
                    stats['max'].append(np.max(values))
                else:
                    stats['mean'].append(None)
                    stats['std'].append(None)
                    stats['min'].append(None)
                    stats['max'].append(None)

                # Absolute error statistics
                if abs_error_lists:
                    abs_errors = [e[i] for e in abs_error_lists if i < len(e)]
                    if abs_errors:
                        stats['abs_error_mean'].append(np.mean(abs_errors))
                        stats['abs_error_std'].append(np.std(abs_errors))
                        stats['abs_error_min'].append(np.min(abs_errors))
                        stats['abs_error_max'].append(np.max(abs_errors))
                    else:
                        stats['abs_error_mean'].append(None)
                        stats['abs_error_std'].append(None)
                        stats['abs_error_min'].append(None)
                        stats['abs_error_max'].append(None)

                # Relative error statistics
                if rel_error_lists:
                    rel_errors = [e[i] for e in rel_error_lists if i < len(e) and e[i] != 0]  # Skip zeros
                    if rel_errors:
                        stats['rel_error_mean'].append(np.mean(rel_errors))
                        stats['rel_error_std'].append(np.std(rel_errors))
                        stats['rel_error_min'].append(np.min(rel_errors))
                        stats['rel_error_max'].append(np.max(rel_errors))
                    else:
                        stats['rel_error_mean'].append(None)
                        stats['rel_error_std'].append(None)
                        stats['rel_error_min'].append(None)
                        stats['rel_error_max'].append(None)

            return stats

        pred_stats = compute_stats_with_errors(all_pred, all_abs_errors, all_rel_errors)
        gt_stats = compute_stats_with_errors(all_gt, [], [])

        # Save to CSV with shape name appended
        csv_filename = f"aggregate_knn{knn}_iter{smoothing_iter}_sigma{sigma:.4f}.csv"
        output_path = self.per_param_dir / csv_filename

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['# Aggregate Statistics across all shapes'])
            writer.writerow(['# Parameters:', f'knn={knn}', f'smoothing_iter={smoothing_iter}', f'sigma={sigma}'])
            writer.writerow(['# Number of shapes:', len(results_list)])
            writer.writerow([])

            # Column headers
            headers = ['eigenvalue_index']
            if pred_stats:
                headers.extend(['pred_mean', 'pred_std', 'pred_min', 'pred_max'])
                if all_abs_errors:
                    headers.extend(['abs_error_mean', 'abs_error_std', 'abs_error_min', 'abs_error_max'])
                if all_rel_errors:
                    headers.extend(['rel_error_mean_%', 'rel_error_std_%', 'rel_error_min_%', 'rel_error_max_%'])
            if gt_stats:
                headers.extend(['gt_mean', 'gt_std', 'gt_min', 'gt_max'])

            writer.writerow(headers)

            # Determine max length
            max_len = 0
            if pred_stats:
                max_len = max(max_len, len(pred_stats['mean']))
            if gt_stats:
                max_len = max(max_len, len(gt_stats['mean']))

            # Write statistics
            for i in range(max_len):
                row = [i]

                if pred_stats:
                    row.extend([
                        pred_stats['mean'][i] if i < len(pred_stats['mean']) else '',
                        pred_stats['std'][i] if i < len(pred_stats['std']) else '',
                        pred_stats['min'][i] if i < len(pred_stats['min']) else '',
                        pred_stats['max'][i] if i < len(pred_stats['max']) else ''
                    ])
                    if all_abs_errors:
                        row.extend([
                            pred_stats['abs_error_mean'][i] if i < len(pred_stats['abs_error_mean']) else '',
                            pred_stats['abs_error_std'][i] if i < len(pred_stats['abs_error_std']) else '',
                            pred_stats['abs_error_min'][i] if i < len(pred_stats['abs_error_min']) else '',
                            pred_stats['abs_error_max'][i] if i < len(pred_stats['abs_error_max']) else ''
                        ])
                    if all_rel_errors:
                        row.extend([
                            pred_stats['rel_error_mean'][i] if i < len(pred_stats['rel_error_mean']) else '',
                            pred_stats['rel_error_std'][i] if i < len(pred_stats['rel_error_std']) else '',
                            pred_stats['rel_error_min'][i] if i < len(pred_stats['rel_error_min']) else '',
                            pred_stats['rel_error_max'][i] if i < len(pred_stats['rel_error_max']) else ''
                        ])

                if gt_stats:
                    row.extend([
                        gt_stats['mean'][i] if i < len(gt_stats['mean']) else '',
                        gt_stats['std'][i] if i < len(gt_stats['std']) else '',
                        gt_stats['min'][i] if i < len(gt_stats['min']) else '',
                        gt_stats['max'][i] if i < len(gt_stats['max']) else ''
                    ])

                writer.writerow(row)

        print(f"    Saved aggregate to: {output_path}")

        # Return overall error metrics for summary (both absolute and relative)
        overall_metrics = {}
        if all_abs_errors:
            # Compute overall mean absolute error across all eigenvalues and all shapes
            all_errors_flat = np.concatenate(all_abs_errors)
            overall_metrics['abs_mean_error'] = np.mean(all_errors_flat)
            overall_metrics['abs_std_error'] = np.std(all_errors_flat)

        if all_rel_errors:
            # Compute overall mean relative error
            all_rel_flat = np.concatenate(all_rel_errors)
            # Filter out zeros
            all_rel_flat = all_rel_flat[all_rel_flat != 0]
            if len(all_rel_flat) > 0:
                overall_metrics['rel_mean_error'] = np.mean(all_rel_flat)
                overall_metrics['rel_std_error'] = np.std(all_rel_flat)

        return overall_metrics

    def create_summary_comparison(self):
        """
        Create a summary CSV comparing all parameter combinations.
        """
        print(f"\n{'='*80}")
        print("Creating summary comparison across all parameter combinations")
        print(f"{'='*80}")

        summary_data = []

        for param_combo in self.param_combinations:
            knn, smoothing_iter, sigma = param_combo

            # Aggregate for this parameter combination
            metrics = self.aggregate_across_shapes_for_param(param_combo)

            if metrics:
                summary_data.append({
                    'knn': knn,
                    'smoothing_iter': smoothing_iter,
                    'sigma': sigma,
                    'abs_mean_error': metrics.get('abs_mean_error', np.nan),
                    'abs_std_error': metrics.get('abs_std_error', np.nan),
                    'rel_mean_error': metrics.get('rel_mean_error', np.nan),
                    'rel_std_error': metrics.get('rel_std_error', np.nan)
                })

        if not summary_data:
            print("No data to create summary")
            return

        # Save summary CSV
        summary_path = self.output_dir / "summary.csv"

        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['# Summary: Parameter Combination Comparison'])
            writer.writerow(['# Overall mean error across all eigenvalues and all shapes'])
            writer.writerow([])

            writer.writerow([
                'knn', 'smoothing_iter', 'sigma',
                'abs_mean_error', 'abs_std_error',
                'rel_mean_error_%', 'rel_std_error_%'
            ])

            for row_data in summary_data:
                writer.writerow([
                    row_data['knn'],
                    row_data['smoothing_iter'],
                    row_data['sigma'],
                    row_data['abs_mean_error'],
                    row_data['abs_std_error'],
                    row_data['rel_mean_error'],
                    row_data['rel_std_error']
                ])

        print(f"\nSaved summary to: {summary_path}")

        # Print best parameter combinations for each metric
        if summary_data:
            valid_abs_data = [d for d in summary_data if not np.isnan(d['abs_mean_error'])]
            valid_rel_data = [d for d in summary_data if not np.isnan(d['rel_mean_error'])]

            if valid_abs_data:
                best_abs = min(valid_abs_data, key=lambda x: x['abs_mean_error'])
                print(f"\n{'='*80}")
                print("BEST PARAMETER COMBINATION (lowest absolute mean error):")
                print(f"  knn={best_abs['knn']}, smoothing_iter={best_abs['smoothing_iter']}, sigma={best_abs['sigma']}")
                print(f"  Absolute mean error: {best_abs['abs_mean_error']:.6f} ± {best_abs['abs_std_error']:.6f}")
                print(f"{'='*80}")

            if valid_rel_data:
                best_rel = min(valid_rel_data, key=lambda x: x['rel_mean_error'])
                print(f"\n{'='*80}")
                print("BEST PARAMETER COMBINATION (lowest relative mean error %):")
                print(f"  knn={best_rel['knn']}, smoothing_iter={best_rel['smoothing_iter']}, sigma={best_rel['sigma']}")
                print(f"  Relative mean error: {best_rel['rel_mean_error']:.2f}% ± {best_rel['rel_std_error']:.2f}%")
                print(f"{'='*80}")

    def compute_and_save_cosine_similarities(self,
                                             model: LaplacianPredictorModule3D,
                                             data,
                                             ckpt_name: str,
                                             output_subdir: Path):
        """
        Compute cosine similarities for normalized and unnormalized eigenvectors once.

        Args:
            model: Trained model
            data: Data object
            ckpt_name: Checkpoint name
            output_subdir: Output directory for this shape
        """
        print(f"\n  Computing cosine similarities (normalized and unnormalized)...")

        try:
            # Run prediction once
            batch = Batch.from_data_list([data])
            with torch.no_grad():
                laplacian_prediction, processed_batch = model.predict_step(batch, batch_idx=0)

            # Get predicted eigenvectors (normalized)
            pred_eigenvectors_normalized = laplacian_prediction.eigenvectors_list[0].cpu()
            pred_weights = laplacian_prediction.weights_list[0].cpu()

            # Compute unnormalized version
            pred_eigenvectors_unnormalized = utils.scale_by_half_inv(
                scalar_functions=pred_eigenvectors_normalized,
                weights=pred_weights
            )

            # Get GT eigenvectors if available
            has_faces = hasattr(data, 'faces') and data.faces is not None
            if has_faces:
                faces_array = data.faces.cpu().numpy() if isinstance(data.faces, torch.Tensor) else data.faces
                has_valid_faces = faces_array.shape[0] > 0
            else:
                has_valid_faces = False

            if not has_valid_faces:
                print("    Skipping cosine similarities - no face connectivity for GT")
                return

            # Compute GT eigenvectors
            n_eig = pred_eigenvectors_normalized.shape[1]
            vertices = data.points.cpu().numpy() if isinstance(data.points, torch.Tensor) else data.points

            gt_eigenvectors_normalized, gt_eigenvalues, gt_weights = utils.compute_pyfm_normalized_laplacian_eigenfunctions(
                vertices=vertices,
                faces=faces_array,
                num_eigenfunctions=n_eig
            )

            # Compute GT unnormalized
            gt_eigenvectors_unnormalized = utils.scale_by_half_inv(
                scalar_functions=gt_eigenvectors_normalized,
                weights=gt_weights
            )

            # Compute cosine similarities for normalized
            cosine_sims_normalized = utils.compute_eigenvector_cosine_similarities(
                pred_eigenvectors_normalized,
                gt_eigenvectors_normalized
            )

            # Compute cosine similarities for unnormalized
            cosine_sims_unnormalized = utils.compute_eigenvector_cosine_similarities(
                pred_eigenvectors_unnormalized,
                gt_eigenvectors_unnormalized
            )

            # Convert to numpy if tensors
            if hasattr(cosine_sims_normalized, 'cpu'):
                cosine_sims_normalized = cosine_sims_normalized.cpu().numpy()
            if hasattr(cosine_sims_unnormalized, 'cpu'):
                cosine_sims_unnormalized = cosine_sims_unnormalized.cpu().numpy()

            # Save to CSV
            cosine_csv_path = output_subdir / "cosine_similarities.csv"

            with open(cosine_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                writer.writerow(['# Cosine Similarities between Predicted and GT Eigenvectors'])
                writer.writerow(['# Shape:', ckpt_name])
                writer.writerow([])

                writer.writerow(['eigenfunction_index', 'normalized', 'unnormalized'])

                max_len = max(len(cosine_sims_normalized), len(cosine_sims_unnormalized))
                for i in range(max_len):
                    norm_val = cosine_sims_normalized[i] if i < len(cosine_sims_normalized) else ''
                    unnorm_val = cosine_sims_unnormalized[i] if i < len(cosine_sims_unnormalized) else ''
                    writer.writerow([i, norm_val, unnorm_val])

            print(f"    Saved cosine similarities to: {cosine_csv_path}")
            print(f"    Mean normalized: {np.mean(cosine_sims_normalized):.6f}")
            print(f"    Mean unnormalized: {np.mean(cosine_sims_unnormalized):.6f}")

        except Exception as e:
            print(f"    Error computing cosine similarities: {e}")
            import traceback
            traceback.print_exc()

    def process_checkpoint_shape_pair(self, checkpoint_path: Path, shape_dir: Path):
        """
        Process a single checkpoint-shape pair across all parameter combinations.

        Args:
            checkpoint_path: Path to checkpoint file
            shape_dir: Path to shape directory
        """
        ckpt_name = checkpoint_path.stem
        print(f"\n{'='*80}")
        print(f"Processing: {ckpt_name}")
        print(f"{'='*80}")

        # Create output directory for this checkpoint (under per_shape)
        output_subdir = self.per_shape_dir / ckpt_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        try:
            # Load model
            model = self.load_model(checkpoint_path)

            # Create dataset with overridden root_dirs
            dataset = self.create_dataset_with_override(shape_dir)

            if len(dataset) == 0:
                print(f"Error: Dataset is empty for {shape_dir}")
                return

            # Get the first (and likely only) data item
            data = dataset[0]
            data = data.to(self.device)

            # Compute cosine similarities once at the beginning
            self.compute_and_save_cosine_similarities(model, data, ckpt_name, output_subdir)

            # Track best combinations for this shape (absolute and relative errors)
            best_abs_error = float('inf')
            best_abs_combo = None
            best_rel_error = float('inf')
            best_rel_combo = None

            # Iterate over parameter combinations
            for knn, smoothing_iter, sigma in self.param_combinations:
                print(f"\nParameter combination: knn={knn}, smoothing_iter={smoothing_iter}, sigma={sigma:.4f}")

                # Compute eigenvalues
                eigenvalues_dict = self.compute_eigenvalues_for_params(
                    model=model,
                    data=data,
                    knn=knn,
                    smoothing_iter=smoothing_iter,
                    sigma=sigma
                )

                # Save to CSV (per shape) with shape name appended
                csv_filename = f"{ckpt_name}_knn{knn}_iter{smoothing_iter}_sigma{sigma:.4f}.csv"
                csv_path = output_subdir / csv_filename

                eigenvalues_dict = self.save_eigenvalues_csv(
                    eigenvalues_dict=eigenvalues_dict,
                    output_path=csv_path,
                    knn=knn,
                    smoothing_iter=smoothing_iter,
                    sigma=sigma
                )

                # Store for cross-shape aggregation
                param_combo = (knn, smoothing_iter, sigma)
                self.results_by_param[param_combo].append(eigenvalues_dict)

                # Track best combinations for this shape
                if len(eigenvalues_dict['gt']) > 0 and len(eigenvalues_dict['pred']) > 0:
                    min_len = min(len(eigenvalues_dict['gt']), len(eigenvalues_dict['pred']))

                    # Absolute error
                    abs_error = np.mean(np.abs(eigenvalues_dict['pred'][:min_len] - eigenvalues_dict['gt'][:min_len]))

                    # Relative error (percentage): |pred - gt| / |gt| * 100
                    # Avoid division by zero - only compute for non-zero GT eigenvalues
                    gt_nonzero = eigenvalues_dict['gt'][:min_len] != 0
                    if np.any(gt_nonzero):
                        rel_errors = np.abs(eigenvalues_dict['pred'][:min_len][gt_nonzero] - eigenvalues_dict['gt'][:min_len][gt_nonzero]) / np.abs(eigenvalues_dict['gt'][:min_len][gt_nonzero]) * 100
                        rel_error = np.mean(rel_errors)
                    else:
                        rel_error = float('inf')

                    if abs_error < best_abs_error:
                        best_abs_error = abs_error
                        best_abs_combo = param_combo

                    if rel_error < best_rel_error:
                        best_rel_error = rel_error
                        best_rel_combo = param_combo

            # Store best combinations for this shape
            if best_abs_combo is not None:
                self.per_shape_best[ckpt_name] = {
                    'best_abs_combo': best_abs_combo,
                    'best_abs_error': best_abs_error,
                    'best_rel_combo': best_rel_combo,
                    'best_rel_error': best_rel_error
                }
                print(f"\n  Best absolute error for {ckpt_name}: knn={best_abs_combo[0]}, iter={best_abs_combo[1]}, sigma={best_abs_combo[2]:.4f} (error={best_abs_error:.6f})")
                print(f"  Best relative error for {ckpt_name}: knn={best_rel_combo[0]}, iter={best_rel_combo[1]}, sigma={best_rel_combo[2]:.4f} (error={best_rel_error:.2f}%)")

            print(f"\n[SUCCESS] Completed processing {ckpt_name}")

        except Exception as e:
            print(f"\n[ERROR] Failed to process {ckpt_name}: {e}")
            import traceback
            traceback.print_exc()

    def save_per_shape_best_combinations(self):
        """
        Save a CSV with the best parameter combination for each shape (for both absolute and relative errors).
        """
        if not self.per_shape_best:
            print("\nNo per-shape best combinations to save")
            return

        best_path = self.output_dir / "per_shape_best_combinations.csv"

        with open(best_path, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['# Best Parameter Combinations for Each Shape'])
            writer.writerow([])

            writer.writerow([
                'shape_name',
                'best_abs_knn', 'best_abs_iter', 'best_abs_sigma', 'abs_mean_error',
                'best_rel_knn', 'best_rel_iter', 'best_rel_sigma', 'rel_mean_error_%'
            ])

            for shape_name, info in sorted(self.per_shape_best.items()):
                abs_knn, abs_iter, abs_sigma = info['best_abs_combo']
                abs_error = info['best_abs_error']

                rel_knn, rel_iter, rel_sigma = info['best_rel_combo']
                rel_error = info['best_rel_error']

                writer.writerow([
                    shape_name,
                    abs_knn, abs_iter, abs_sigma, abs_error,
                    rel_knn, rel_iter, rel_sigma, rel_error
                ])

        print(f"\nSaved per-shape best combinations to: {best_path}")

    def run(self):
        """Run the batch evaluation."""
        print("="*80)
        print("BATCH EIGENVALUE EVALUATION")
        print("="*80)
        print(f"Checkpoints directory: {self.checkpoints_dir}")
        print(f"Shapes directory: {self.shapes_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Parameter combinations: {len(self.param_combinations)}")
        print("="*80)

        # Find checkpoint-shape pairs
        pairs = self.find_checkpoint_shape_pairs()

        if not pairs:
            print("No checkpoint-shape pairs found!")
            return

        # Process each pair
        for i, (checkpoint_path, shape_dir) in enumerate(pairs, 1):
            print(f"\n\nProcessing pair {i}/{len(pairs)}")
            self.process_checkpoint_shape_pair(checkpoint_path, shape_dir)

        # After all shapes are processed, create aggregations and summary
        print("\n\n" + "="*80)
        print("CREATING CROSS-SHAPE AGGREGATIONS")
        print("="*80)

        self.create_summary_comparison()
        self.save_per_shape_best_combinations()

        print("\n" + "="*80)
        print("BATCH EVALUATION COMPLETE")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  Per-shape results: {self.per_shape_dir}")
        print(f"  Per-shape cosine similarities: {self.per_shape_dir}/<shape>/cosine_similarities.csv")
        print(f"  Per-shape best combinations: {self.output_dir / 'per_shape_best_combinations.csv'}")
        print(f"  Per-parameter aggregates: {self.per_param_dir}")
        print(f"  Summary comparison: {self.output_dir / 'summary.csv'}")


@hydra.main(version_base="1.2", config_path="config/training")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration."""

    # Check required parameters
    if not hasattr(cfg, 'checkpoints_dir'):
        raise ValueError("Please provide ++checkpoints_dir=/path/to/checkpoints")
    if not hasattr(cfg, 'shapes_dir'):
        raise ValueError("Please provide ++shapes_dir=/path/to/shapes")
    if not hasattr(cfg, 'output_dir'):
        raise ValueError("Please provide ++output_dir=/path/to/output")

    # Grid search parameters with defaults
    min_k = cfg.get('min_k', 20)
    max_k = cfg.get('max_k', 30)
    num_k_points = cfg.get('num_k_points', 2)

    min_iter = cfg.get('min_iter', 1)
    max_iter = cfg.get('max_iter', 2)
    num_iter_points = cfg.get('num_iter_points', 2)

    min_sigma = cfg.get('min_sigma', 0.01)
    max_sigma = cfg.get('max_sigma', 0.02)
    num_sigma_points = cfg.get('num_sigma_points', 2)

    # Generate grid of parameter values
    import numpy as np

    # For integers (k, iter), use linspace and round
    if num_k_points == 1:
        k_values = [min_k]
    else:
        k_values = np.linspace(min_k, max_k, num_k_points).astype(int).tolist()

    if num_iter_points == 1:
        iter_values = [min_iter]
    else:
        iter_values = np.linspace(min_iter, max_iter, num_iter_points).astype(int).tolist()

    # For floats (sigma), use linspace directly
    if num_sigma_points == 1:
        sigma_values = [min_sigma]
    else:
        sigma_values = np.linspace(min_sigma, max_sigma, num_sigma_points).tolist()

    # Create all combinations (Cartesian product)
    from itertools import product
    param_combinations = list(product(k_values, iter_values, sigma_values))

    print("="*80)
    print("PARAMETER GRID SEARCH CONFIGURATION")
    print("="*80)
    print(f"k values: {k_values}")
    print(f"iteration values: {iter_values}")
    print(f"sigma values: {sigma_values}")
    print(f"Total combinations: {len(param_combinations)}")
    print("="*80)
    print("\nGenerated combinations:")
    for i, (k, it, sig) in enumerate(param_combinations, 1):
        print(f"  {i}. knn={k}, iter={it}, sigma={sig:.4f}")
    print("="*80)

    # Create evaluator
    evaluator = BatchEigenvalueEvaluator(
        cfg=cfg,
        checkpoints_dir=cfg.checkpoints_dir,
        shapes_dir=cfg.shapes_dir,
        output_dir=cfg.output_dir,
        param_combinations=param_combinations
    )

    # Run evaluation
    evaluator.run()


if __name__ == "__main__":
    main()
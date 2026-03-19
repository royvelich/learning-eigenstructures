"""
Laplacian prediction modules for neural network-based spectral decomposition.
This is the CURRENT version from the project files - NO MODIFICATIONS.
"""

# standard library
import os
import pickle
import zipfile
from pathlib import Path
from typing import List, Type, Callable, Optional, Dict, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

# neural laplacian
from neural_laplacian import utils
from neural_laplacian.modules.architectures import ConfigurableGNNBase, ConfigurableMLP, ConfigurablePooling
from neural_laplacian.losses import LaplacianLoss, ReconstructionLoss
from neural_laplacian.activations import ParameterizedSigmoid
from neural_laplacian.configs import KnnGraphConfig, LaplacianLossConfig
from neural_laplacian.convolutions import ScalarFieldSmoother

# wandb
import wandb

# omegaconf
from omegaconf import DictConfig, OmegaConf

# torch
import torch
import torch.nn.functional as F

# numpy
import numpy as np

# lightning
import lightning
import pytorch_lightning as pl
from lightning.pytorch.callbacks import Callback

# torch_geometric
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

# neural laplacian
from neural_laplacian.utils import split_results_by_nodes, split_results_by_graphs
from neural_laplacian.features import FourierFeatureExtractor, FeatureExtractor


class PosType(Enum):
    POINTS = "POINTS"
    FEATURES = "FEATURES"
    NONE = "NONE"


@dataclass
class LaplacianPrediction:
    eigenvectors_list: List[torch.Tensor]
    weights_list: List[torch.Tensor]
    unweighted_eigenvalues_list: Optional[List[torch.Tensor]]
    weighted_eigenvalues_list: Optional[List[torch.Tensor]]
    cosine_similarities_list: Optional[List[torch.Tensor]] = None  # Cosine similarities vs GT per shape


class PointFeaturesModule(torch.nn.Module):
    """
    Vertex feature extraction module that uses any torch.nn.Module for core feature computation.

    This class handles preprocessing, postprocessing, and pipeline logic while delegating
    the core feature computation to a configurable torch module (e.g., GNN, Transformer, etc.).
    """

    def __init__(self,
                 core_modules: List[torch.nn.Module],
                 preprocess_mlp: Optional[ConfigurableMLP],
                 postprocess_mlp: Optional[ConfigurableMLP],
                 feature_extractors: Optional[List[FeatureExtractor]],
                 pos_type: PosType):
        """
        Initialize the vertex features module.

        Args:
            core_module: The main module for computing features (GNN, Transformer, etc.)
            preprocess_mlp: Optional MLP for preprocessing raw input features
            postprocess_mlp: Optional MLP for postprocessing computed features
            feature_extractors: Optional Fourier feature extractor for positional encoding
        """
        super().__init__()
        self._core_modules = torch.nn.ModuleList(core_modules)
        self._preprocess_mlp = preprocess_mlp
        self._postprocess_mlp = postprocess_mlp
        self._feature_extractors = torch.nn.ModuleList(feature_extractors)
        self._pos_type = pos_type

    def _pre_preprocess_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare input tensor for preprocessing MLP by flattening if necessary.

        Args:
            x: Input tensor of shape [..., feature_dim]

        Returns:
            Flattened tensor of shape [batch_size, feature_dim]
        """
        if len(x.shape) > 2:
            x = x.reshape(-1, x.shape[-1])
        return x

    def _pre_core_computation(self, x: torch.Tensor, preprocess_mlp_out: torch.Tensor) -> torch.Tensor:
        """
        Prepare features for core computation by handling dimensionality and aggregation.

        Args:
            x: Original input tensor
            preprocess_mlp_out: Output from preprocessing MLP

        Returns:
            Tensor ready for core feature computation
        """
        if len(x.shape) > 2:
            # Reshape back to original structure and apply max pooling
            x = preprocess_mlp_out.reshape(x.shape[0], x.shape[1], -1)
            x, _ = torch.max(x, dim=1)
        else:
            x = preprocess_mlp_out
        return x

    def pre_forward(self, batch: Batch) -> Batch:
        """
        Complete forward pass through the vertex features module.

        Args:
            batch: Input batch containing raw features in batch.raw_x

        Returns:
            Final vertex features tensor
        """
        features_list = []
        if len(self._feature_extractors) > 0:
            for feature_extractor in self._feature_extractors:
                features = feature_extractor.extract_features(points=batch.points)
                features_list.append(features)
        else:
            features_list.append(batch.points)

        x = torch.cat(features_list, dim=1)
        batch = utils.rebuild_batch_from_tensor(batch=batch, property_name='x', property_tensor=x)

        if self._pos_type == PosType.FEATURES:
            batch = utils.rebuild_batch_from_tensor(batch=batch, property_name='pos', property_tensor=x)
        elif self._pos_type == PosType.POINTS:
            batch = utils.rebuild_batch_from_tensor(batch=batch, property_name='pos', property_tensor=batch.points)

        return batch

    def forward(self, batch: Batch) -> Batch:
        """
        Complete forward pass through the vertex features module.

        Args:
            batch: Input batch containing raw features in batch.raw_x

        Returns:
            Final vertex features tensor
        """
        # features_list = []
        # if len(self._feature_extractors) > 0:
        #     for feature_extractor in self._feature_extractors:
        #         features = feature_extractor.extract_features(points=batch.points)
        #         features_list.append(features)
        # else:
        #     features_list.append(batch.points)
        #
        # x = torch.cat(features_list, dim=1)
        #
        # if self._pos_type == PosType.FEATURES:
        #     batch = utils.rebuild_batch_from_tensor(batch=batch, property_name='pos', property_tensor=x)
        # elif self._pos_type == PosType.POINTS:
        #     batch = utils.rebuild_batch_from_tensor(batch=batch, property_name='pos', property_tensor=batch.points)

        preprocess_mlp_in = self._pre_preprocess_mlp(x=batch.x)
        if self._preprocess_mlp is not None:
            preprocess_mlp_out = self._preprocess_mlp(x=preprocess_mlp_in)
        else:
            preprocess_mlp_out = preprocess_mlp_in

        x = self._pre_core_computation(x=batch.x, preprocess_mlp_out=preprocess_mlp_out)
        batch = utils.rebuild_batch_from_tensor(batch=batch, property_name='x', property_tensor=x)

        for core_module in self._core_modules:
            x = core_module(batch)
            batch = utils.rebuild_batch_from_tensor(batch=batch, property_name='x', property_tensor=x)

        if not isinstance(x, list):
            x = [x]

        x = torch.cat(x, dim=1)
        if self._postprocess_mlp is not None:
            x = self._postprocess_mlp(x)

        batch = utils.rebuild_batch_from_tensor(batch=batch, property_name='point_features', property_tensor=x)
        return batch


class LaplacianPredictorModuleBase(ABC, lightning.pytorch.LightningModule):
    def __init__(self,
                 point_features_module: PointFeaturesModule,
                 reconstruction_loss: ReconstructionLoss,
                 optimizer_cfg: DictConfig,
                 knn_graph_config: KnnGraphConfig,
                 scalar_field_smoother: ScalarFieldSmoother,
                 eigenvectors_mlp: ConfigurableMLP,
                 pooling: ConfigurablePooling,
                 scheduler_cfg: Optional[DictConfig],
                 metric_activation: Optional[str],
                 compute_metrics_on_predict: bool = False,  # Whether to compute metrics during predict_step
                 **kwargs
                 ):
        super().__init__()
        self._point_features_module = point_features_module
        self._optimizer_cfg = optimizer_cfg
        self._scheduler_cfg = scheduler_cfg
        self._reconstruction_loss = reconstruction_loss
        self._validation_pairs = []
        self._total_val_batches = 0
        self._batches_to_sample = []
        self._knn_graph_config = knn_graph_config
        self._scalar_field_smoother = scalar_field_smoother
        self._eigenvectors_mlp = eigenvectors_mlp
        self._pooling = pooling
        self._metric_activation = metric_activation
        self._compute_metrics_on_predict = compute_metrics_on_predict

    def setup(self, stage):
        def exclude_fn(path: str):
            if 'lightning_logs' in path:
                return True
            if 'outputs' in path:
                return True
            if 'neural-laplacian-venv' in path:
                return True
            if 'brezis_1d' in path:
                return True
            if 'local_laplacian' in path:
                return True
            if 'wandb' in path:
                return True
            if '.git' in path:
                return True
            return False

        def include_fn(path: str):
            return True if path.endswith('.py') or path.endswith('.yml') or path.endswith('.yaml') else False

        if self.trainer.global_rank == 0 and wandb.run is not None:
            self.logger.experiment.log_code(root=".", exclude_fn=exclude_fn, include_fn=include_fn)
            dict_cfg = OmegaConf.to_container(self.trainer.cfg, resolve=True)
            self.logger.experiment.config.update(dict_cfg, allow_val_change=True)

    def _compute_edge_index(self, batch: Batch) -> Batch:
        """
        Compute edge_index for each data object in the batch using kNN.
        """
        data_list = batch.to_data_list()
        for data in data_list:
            k = self._knn_graph_config.sample_k(data.pos)
            data.edge_index = knn_graph(
                x=data.pos,
                k=k,
                cosine=self._knn_graph_config.cosine,
                loop=self._knn_graph_config.loop
            )
        return Batch.from_data_list(data_list)

    def _smooth_scalar_fields(self, batch: Batch) -> Batch:
        """
        Apply smoothing operations to scalar fields using the computed edge_index.
        """
        data_list = batch.to_data_list()

        for data in data_list:
            smoothed_scalar_fields = data.scalar_fields
            smoothed_scalar_fields = self._scalar_field_smoother(
                x=smoothed_scalar_fields,
                edge_index=data.edge_index,
                pos=data.pos
            )

            data.smoothed_scalar_fields = smoothed_scalar_fields

        return Batch.from_data_list(data_list)

    def _compute_predictions_and_losses(self, batch: Batch) -> Tuple[LaplacianPrediction, torch.Tensor, Batch]:
        """
        First part of shared step: compute predictions, losses, and eigenvalues.

        Args:
            batch: Input batch

        Returns:
            Tuple of (laplacian_prediction, loss_components)
        """
        # Get predictions (compute vertex features with edge_index already available)
        batch = self._point_features_module.pre_forward(batch=batch)
        batch = self._compute_edge_index(batch)
        batch = self._point_features_module(batch=batch)

        laplacian_prediction = self.forward(batch)

        # Apply scalar field smoothing after vertex features and edge_index are computed
        batch = self._smooth_scalar_fields(batch)

        # Continue with existing logic
        batch = utils.rebuild_batch_from_list(batch=batch, property_name='eigenvectors', property_tensor_list=laplacian_prediction.eigenvectors_list)
        batch = utils.rebuild_batch_from_list(batch=batch, property_name='weights', property_tensor_list=laplacian_prediction.weights_list)

        data_list = batch.to_data_list()
        unweighted_eigenvalues_list = []
        weighted_eigenvalues_list = []
        loss_values = []

        # Evaluate ReconstructionLoss for each data item
        for data in data_list:
            reconstruction_loss_result = self._reconstruction_loss(
                scalar_fields=data.smoothed_scalar_fields,
                eigenvectors=data.eigenvectors,
                eigenvalues=None,
                vertices=data.pos,
                normals=None,
                weights=data.weights,
                edge_index=data.edge_index,
                reconstructed_laplacian=None
            )

            # Store eigenvalues computed by the loss
            unweighted_eigenvalues_list.append(reconstruction_loss_result.unweighted_eigenvalues)
            weighted_eigenvalues_list.append(reconstruction_loss_result.weighted_eigenvalues)
            loss_values.append(reconstruction_loss_result.loss)

        # Assign unweighted_eigenvalues_list to the laplacian_prediction object
        laplacian_prediction.unweighted_eigenvalues_list = unweighted_eigenvalues_list
        laplacian_prediction.weighted_eigenvalues_list = weighted_eigenvalues_list

        # Optionally compute cosine similarities if flag is set
        if self._compute_metrics_on_predict:
            cosine_similarities_list = []
            for data_idx, (data, pred_eigenvectors) in enumerate(zip(data_list, laplacian_prediction.eigenvectors_list)):
                if hasattr(data, 'gt_eigenvectors') and data.gt_eigenvectors.shape[0] > 0:
                    pred_eigenvectors_cpu = pred_eigenvectors.detach().cpu()
                    gt_eigenvectors = torch.from_numpy(data.gt_eigenvectors).detach().cpu()

                    # Compute cosine similarities
                    cosine_sims = utils.compute_eigenvector_cosine_similarities(
                        pred_eigenvectors_cpu, gt_eigenvectors
                    )
                    cosine_similarities_list.append(cosine_sims)
                else:
                    # No GT available, append None or empty tensor
                    cosine_similarities_list.append(None)

            laplacian_prediction.cosine_similarities_list = cosine_similarities_list

        # Compute mean loss across batch
        total_loss = torch.mean(torch.stack(loss_values))

        return laplacian_prediction, total_loss, batch

    def forward(self, batch: Batch) -> LaplacianPrediction:
        """
        Shared forward pass: compute eigenvectors and weights using QR decomposition.

        Args:
            batch: Input batch

        Returns:
            LaplacianPrediction with eigenvectors and weights
        """
        data_list = batch.to_data_list()
        eigenvectors_list = []
        weights_list = []

        for data in data_list:
            raw_eigenvectors = self._eigenvectors_mlp(x=data.point_features)

            if self._metric_activation is not None:
                raw_eigenvectors_new = raw_eigenvectors.clone()
                # raw_eigenvectors_new[:, 0] = F.softmax(raw_eigenvectors[:, 0]) * torch.norm(raw_eigenvectors[:, 0])
                method = getattr(F, self._metric_activation)
                raw_eigenvectors_new[:, 0] = method(raw_eigenvectors[:, 0])
                raw_eigenvectors = raw_eigenvectors_new

            # QR decomposition
            Q, R = torch.linalg.qr(raw_eigenvectors)

            # R[0,0] contains the norm of the first column before normalization
            # (up to sign, which is why we use abs)
            # first_column_scale = torch.abs(R[0, 0])

            # Use the first column's original scale for weights
            weights = (Q[:, 0] ** 2)

            eigenvectors_list.append(Q)  # Already orthonormal
            weights_list.append(weights)

        return LaplacianPrediction(
            weights_list=weights_list,
            eigenvectors_list=eigenvectors_list,
            unweighted_eigenvalues_list=None,
            weighted_eigenvalues_list=None
        )

    def enable_metrics_on_predict(self):
        """Enable computation of metrics during predict_step."""
        self._compute_metrics_on_predict = True

    def disable_metrics_on_predict(self):
        """Disable computation of metrics during predict_step."""
        self._compute_metrics_on_predict = False

    def _log_loss(self, loss_value: torch.Tensor, batch: Batch, on_step: bool, stage: str, name: str):
        """Log reconstruction loss."""
        self.log(
            f'{stage}/ReconstructionLoss',
            loss_value,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(batch)
        )

    def _compute_and_log_metrics(self,
                                 laplacian_prediction: LaplacianPrediction,
                                 loss_value: torch.Tensor,
                                 batch: Batch,
                                 stage: str,
                                 name: str) -> Dict[str, torch.Tensor]:
        """
        Compute metrics and logging for all model types.
        Logs basic loss, then calls abstract method for additional metrics.

        Args:
            laplacian_prediction: Prediction results
            loss_value: Computed loss value
            batch: Input batch
            stage: 'train' or 'val'

        Returns:
            Final loss dictionary
        """
        # Log basic reconstruction loss (common to all models)
        on_step = stage == 'train'
        self._log_loss(loss_value=loss_value, batch=batch, on_step=on_step, stage=stage, name=name)

        # Call abstract method for additional metrics specific to each model type
        self._compute_and_log_additional_metrics(
            laplacian_prediction=laplacian_prediction,
            loss_value=loss_value,
            batch=batch,
            stage=stage,
            on_step=on_step,
            name=name
        )

        return {f'{name}_loss' if stage == 'val' else 'loss': loss_value}

    @abstractmethod
    def _compute_and_log_additional_metrics(self, laplacian_prediction: LaplacianPrediction,
                                            loss_value: torch.Tensor,
                                            batch: Batch,
                                            stage: str,
                                            on_step: bool,
                                            name: str) -> None:
        """
        Abstract method for computing and logging additional metrics.
        Must be implemented by concrete subclasses.

        Args:
            laplacian_prediction: Prediction results
            loss_value: Computed loss value
            batch: Input batch
            stage: 'train' or 'val'
            on_step: Whether to log on step (for training)
        """
        pass

    def predict_step(self, batch: Batch, batch_idx: int) -> Tuple[LaplacianPrediction, Batch]:
        """Prediction step logic for inference."""
        laplacian_prediction, _, batch = self._compute_predictions_and_losses(batch)
        return laplacian_prediction, batch

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step logic."""
        laplacian_prediction, loss_value, batch = self._compute_predictions_and_losses(batch)
        return self._compute_and_log_metrics(laplacian_prediction=laplacian_prediction, loss_value=loss_value, batch=batch, stage='train', name='train')

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """Validation step logic."""
        laplacian_prediction, loss_value, batch = self._compute_predictions_and_losses(batch)
        dataset_name = self.trainer.val_dataloaders[dataloader_idx].dataset.name
        return self._compute_and_log_metrics(laplacian_prediction=laplacian_prediction, loss_value=loss_value, batch=batch, stage='val', name=f'val/{dataset_name}')

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        # Create optimizer
        optimizer = self._optimizer_cfg(params=self.parameters())

        # If no scheduler config, return just the optimizer
        if self._scheduler_cfg is None:
            return optimizer

        # Create scheduler
        scheduler = self._scheduler_cfg(optimizer=optimizer)

        # Return optimizer and scheduler configuration
        scheduler_config = {
            "scheduler": scheduler,
            "interval": 'epoch'
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }


class LaplacianPredictorModule3D(LaplacianPredictorModuleBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def _log_eigenvalues(self, eigenvalues_list: List[torch.Tensor], batch: Batch, on_step: bool, stage: str, name: str, prefix: str):
        eigenvalues_mean = torch.mean(torch.stack(eigenvalues_list), dim=0)
        for i, eigenvalue in enumerate(eigenvalues_mean):
            self.log(
                name=f'{name}/{prefix}_eigenvalue{i + 1}',
                value=eigenvalue.item(),
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=len(batch)
            )

    def _log_avg_cosine_similarities_per_vector(self, similarities_tensor: torch.Tensor, num_eigenvectors: int, batch: Batch, on_step: bool, stage: str, name: str):
        avg_similarities = torch.mean(similarities_tensor, dim=0)
        for i in range(num_eigenvectors):
            self.log(
                name=f'{name}/eigenvector_{i:02d}_cosine_sim',
                value=avg_similarities[i].item(),
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=len(batch)
            )

    def _log_overall_avg_cosine_similarities(self, similarities_tensor: torch.Tensor, batch: Batch, on_step: bool, stage: str, name: str):
        overall_avg_similarity = torch.mean(similarities_tensor)
        self.log(
            f'{name}/eigenvectors_avg_cosine_sim',
            overall_avg_similarity.item(),
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch)
        )

    def _compute_and_log_additional_metrics(self, laplacian_prediction: LaplacianPrediction,
                                            loss_value: torch.Tensor,
                                            batch: Batch,
                                            stage: str,
                                            on_step: bool,
                                            name: str) -> None:
        """
        Compute and log additional metrics for 3D geometry.
        Computes cosine similarities with ground truth eigenvectors and logs eigenvalues.

        Args:
            laplacian_prediction: Prediction results
            loss_value: Computed loss value
            batch: Input batch
            stage: 'train' or 'val'
            on_step: Whether to log on step (for training)
        """
        data_list = batch.to_data_list()

        # Compute eigenvector cosine similarities against ground truth
        all_cosine_sims = []

        for data_idx, (data, pred_eigenvectors) in enumerate(zip(data_list, laplacian_prediction.eigenvectors_list)):
            if data.gt_eigenvectors.shape[0] > 0:
                pred_eigenvectors = pred_eigenvectors.detach().cpu()
                gt_eigenvectors = torch.from_numpy(data.gt_eigenvectors).detach().cpu()

                # Use the utility function to compute similarities
                cosine_sims = utils.compute_eigenvector_cosine_similarities(
                    pred_eigenvectors, gt_eigenvectors
                )
                all_cosine_sims.append(cosine_sims)

        # Log eigenvalues
        self._log_eigenvalues(eigenvalues_list=laplacian_prediction.unweighted_eigenvalues_list, batch=batch, on_step=on_step, stage=stage, name=name, prefix='unweighted')
        self._log_eigenvalues(eigenvalues_list=laplacian_prediction.weighted_eigenvalues_list, batch=batch, on_step=on_step, stage=stage, name=name, prefix='weighted')

        # Log cosine similarities if ground truth is available
        if len(all_cosine_sims) > 0:
            similarities_tensor = torch.stack(all_cosine_sims, dim=0)
            num_eigenvectors = similarities_tensor.shape[1]
            self._log_avg_cosine_similarities_per_vector(similarities_tensor=similarities_tensor, num_eigenvectors=num_eigenvectors, batch=batch, on_step=on_step, stage=stage, name=name)
            self._log_overall_avg_cosine_similarities(similarities_tensor=similarities_tensor, batch=batch, on_step=on_step, stage=stage, name=name)


class LaplacianPredictorModuleImageManifold(LaplacianPredictorModuleBase):
    def __init__(self,
                 spectral_clustering_k_values: List[int],
                 manifold_learning_methods: Optional[List[str]],
                 eigenmaps_methods: Optional[List[str]],
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self._spectral_clustering_k_values = spectral_clustering_k_values

        # Default to all methods if not specified
        if manifold_learning_methods is None:
            self._manifold_learning_methods = ['pca', 'umap', 'tsne', 'isomap']
        else:
            # Validate methods
            valid_methods = {'pca', 'umap', 'tsne', 'isomap'}
            for method in manifold_learning_methods:
                if method not in valid_methods:
                    raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")
            self._manifold_learning_methods = manifold_learning_methods

        print(f"Manifold learning methods enabled: {self._manifold_learning_methods}")

        # Handle eigenmaps methods
        if eigenmaps_methods is None:
            self._eigenmaps_methods = ['learned']
        else:
            # Validate eigenmaps methods
            valid_eigenmaps = {'learned', 'graph', 'random'}
            for method in eigenmaps_methods:
                if method not in valid_eigenmaps:
                    raise ValueError(f"Invalid eigenmaps method '{method}'. Must be one of {valid_eigenmaps}")
            self._eigenmaps_methods = eigenmaps_methods

        print(f"Eigenmaps methods enabled: {self._eigenmaps_methods}")

    def _compute_manifold_embeddings(self,
                                    raw_features: np.ndarray,
                                    n_components: int,
                                    k_neighbors: int) -> Dict[str, np.ndarray]:
        """
        Apply selected classical manifold learning methods to raw features.

        Args:
            raw_features: Raw DINO/CLIP features [n_samples, feature_dim]
            n_components: Target dimensionality
            k_neighbors: Number of neighbors (from knn_graph_config)

        Returns:
            Dict with keys from self._manifold_learning_methods
            Values are embeddings [n_samples, n_components] or None if failed
        """
        # Use the utility function from utils.py
        embeddings = utils.compute_manifold_embeddings(
            features=raw_features,
            n_components=n_components,
            k_neighbors=k_neighbors,
            methods=self._manifold_learning_methods,
            use_cosine=self._knn_graph_config.cosine,
            min_dist=0.1,
            random_state=None  # No random_state to enable parallelism for UMAP
        )

        return embeddings

    def _compute_graph_laplacian_eigenvectors(self,
                                              edge_index: torch.Tensor,
                                              pos: torch.Tensor,
                                              n_eigenvectors: int,
                                              use_cosine: bool) -> torch.Tensor:
        """
        Compute normalized graph Laplacian eigenvectors with Gaussian edge weights.
        Follows the Latent Functional Maps paper methodology.

        Args:
            edge_index: Edge connectivity [2, n_edges]
            pos: Node features [n_nodes, feature_dim]
            n_eigenvectors: Number of eigenvectors to compute
            use_cosine: Whether to use cosine distance (from k-NN graph config)

        Returns:
            Eigenvectors [n_nodes, n_eigenvectors]
        """
        # Use the utility function from utils.py
        return utils.compute_graph_laplacian_eigenvectors(
            edge_index=edge_index,
            pos=pos,
            n_eigenvectors=n_eigenvectors,
            use_cosine=use_cosine
        )

    def _generate_random_orthogonal_basis(self, n_points: int, n_eigenvectors: int, device: torch.device) -> torch.Tensor:
        """
        Generate random orthogonal basis using QR decomposition.

        Args:
            n_points: Number of points
            n_eigenvectors: Number of vectors in basis
            device: Device to create tensor on

        Returns:
            Random orthogonal matrix [n_points, n_eigenvectors]
        """
        # Use the utility function from utils.py
        return utils.generate_random_orthogonal_basis(
            n_points=n_points,
            n_eigenvectors=n_eigenvectors,
            device=device,
            random_state=42  # Use fixed seed for reproducibility
        )

    def _compute_clustering_metrics(self, embeddings: torch.Tensor, class_ids: np.ndarray, n_clusters: int) -> Tuple[float, float, float, float, float, float, float]:
        """
        Run k-means clustering and compute all 7 clustering metrics.

        Args:
            embeddings: Point embeddings [n_points, embedding_dim]
            class_ids: Ground truth class labels [n_points]
            n_clusters: Number of clusters for k-means

        Returns:
            Tuple of (NMI, ARI, Completeness, AMI, Homogeneity, V-Measure, Fowlkes-Mallows)
        """
        # Convert to numpy if needed
        embeddings_np = embeddings.detach().cpu().numpy()

        # Use utility functions from utils.py
        predicted_labels = utils.compute_kmeans_clustering(
            embeddings=embeddings_np,
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        nmi, ari, completeness, ami, homogeneity, v_measure, fmi = utils.compute_clustering_metrics(
            predicted_labels=predicted_labels,
            true_labels=class_ids
        )

        return nmi, ari, completeness, ami, homogeneity, v_measure, fmi

    def _compute_and_log_additional_metrics(self,
                                            laplacian_prediction: LaplacianPrediction,
                                            loss_value: torch.Tensor,
                                            batch: Batch,
                                            stage: str,
                                            on_step: bool,
                                            name: str) -> None:
        """
        Compute and log spectral clustering metrics for image manifold validation.
        Compares learned eigenvectors against graph Laplacian and random baselines.
        Also compares against classical manifold learning methods (UMAP, t-SNE, Isomap, PCA).

        Args:
            laplacian_prediction: Prediction results
            loss_value: Computed loss value
            batch: Input batch
            stage: 'train' or 'val'
            on_step: Whether to log on step (for training)
        """
        data_list = batch.to_data_list()

        # ========================================
        # PART 1: Spectral Clustering (EXISTING)
        # ========================================
        # Accumulate metrics across all data items for each k
        metric_names = ['nmi', 'ari', 'completeness', 'ami', 'homogeneity', 'v_measure', 'fmi']
        metrics_per_k = {k: {
            f'{metric}_{method}': []
            for metric in metric_names
            for method in self._eigenmaps_methods
        } for k in self._spectral_clustering_k_values}

        for data in data_list:
            # Check if we have class labels
            if not hasattr(data, 'class_ids') or len(data.class_ids) == 0:
                continue

            class_ids = data.class_ids
            n_clusters = len(np.unique(class_ids))

            # Get learned eigenvectors
            idx = data_list.index(data)
            learned_eigenvectors = laplacian_prediction.eigenvectors_list[idx]  # [n_images, n_eig]

            # Determine max k for baselines
            max_k = max(self._spectral_clustering_k_values)

            # Conditionally compute graph Laplacian eigenvectors only if needed
            if 'graph' in self._eigenmaps_methods:
                use_cosine = self._knn_graph_config.cosine
                graph_laplacian_eigenvectors = self._compute_graph_laplacian_eigenvectors(
                    edge_index=data.edge_index,
                    pos=data.pos,
                    n_eigenvectors=min(max_k, learned_eigenvectors.shape[1]),
                    use_cosine=use_cosine
                )
            else:
                graph_laplacian_eigenvectors = None

            # Conditionally generate random orthogonal basis only if needed
            if 'random' in self._eigenmaps_methods:
                random_eigenvectors = self._generate_random_orthogonal_basis(
                    n_points=learned_eigenvectors.shape[0],
                    n_eigenvectors=min(max_k, learned_eigenvectors.shape[1]),
                    device=learned_eigenvectors.device
                )
            else:
                random_eigenvectors = None

            # For each k value, compute clustering metrics and accumulate
            for k in self._spectral_clustering_k_values:
                if k > learned_eigenvectors.shape[1]:
                    continue

                # Learned eigenvectors (only if 'learned' in methods)
                if 'learned' in self._eigenmaps_methods:
                    nmi_learned, ari_learned, comp_learned, ami_learned, homo_learned, vm_learned, fmi_learned = self._compute_clustering_metrics(
                        learned_eigenvectors[:, 1:(k+1)], class_ids, n_clusters
                    )
                    metrics_per_k[k]['nmi_learned'].append(nmi_learned)
                    metrics_per_k[k]['ari_learned'].append(ari_learned)
                    metrics_per_k[k]['completeness_learned'].append(comp_learned)
                    metrics_per_k[k]['ami_learned'].append(ami_learned)
                    metrics_per_k[k]['homogeneity_learned'].append(homo_learned)
                    metrics_per_k[k]['v_measure_learned'].append(vm_learned)
                    metrics_per_k[k]['fmi_learned'].append(fmi_learned)

                # Graph Laplacian (only if 'graph' in methods)
                if 'graph' in self._eigenmaps_methods:
                    nmi_graph, ari_graph, comp_graph, ami_graph, homo_graph, vm_graph, fmi_graph = self._compute_clustering_metrics(
                        graph_laplacian_eigenvectors[:, :k], class_ids, n_clusters
                    )
                    metrics_per_k[k]['nmi_graph'].append(nmi_graph)
                    metrics_per_k[k]['ari_graph'].append(ari_graph)
                    metrics_per_k[k]['completeness_graph'].append(comp_graph)
                    metrics_per_k[k]['ami_graph'].append(ami_graph)
                    metrics_per_k[k]['homogeneity_graph'].append(homo_graph)
                    metrics_per_k[k]['v_measure_graph'].append(vm_graph)
                    metrics_per_k[k]['fmi_graph'].append(fmi_graph)

                # Random baseline (only if 'random' in methods)
                if 'random' in self._eigenmaps_methods:
                    nmi_random, ari_random, comp_random, ami_random, homo_random, vm_random, fmi_random = self._compute_clustering_metrics(
                        random_eigenvectors[:, :k], class_ids, n_clusters
                    )
                    metrics_per_k[k]['nmi_random'].append(nmi_random)
                    metrics_per_k[k]['ari_random'].append(ari_random)
                    metrics_per_k[k]['completeness_random'].append(comp_random)
                    metrics_per_k[k]['ami_random'].append(ami_random)
                    metrics_per_k[k]['homogeneity_random'].append(homo_random)
                    metrics_per_k[k]['v_measure_random'].append(vm_random)
                    metrics_per_k[k]['fmi_random'].append(fmi_random)

        # Compute averages and log spectral clustering metrics
        for k in self._spectral_clustering_k_values:
            # Check if any method has data for this k
            has_data = any(
                len(metrics_per_k[k].get(f'nmi_{method}', [])) > 0
                for method in self._eigenmaps_methods
            )
            if not has_data:
                continue

            # Log NMI and ARI for each enabled method
            if 'learned' in self._eigenmaps_methods and len(metrics_per_k[k]['nmi_learned']) > 0:
                nmi_learned_mean = np.mean(metrics_per_k[k]['nmi_learned'])
                ari_learned_mean = np.mean(metrics_per_k[k]['ari_learned'])
                self.log(f'{name}/spectral_avg_nmi_k{k}_learned', nmi_learned_mean,
                         on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=len(batch))
                self.log(f'{name}/spectral_avg_ari_k{k}_learned', ari_learned_mean,
                         on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=len(batch))

            if 'graph' in self._eigenmaps_methods and len(metrics_per_k[k]['nmi_graph']) > 0:
                nmi_graph_mean = np.mean(metrics_per_k[k]['nmi_graph'])
                ari_graph_mean = np.mean(metrics_per_k[k]['ari_graph'])
                self.log(f'{name}/spectral_avg_nmi_k{k}_graph_laplacian', nmi_graph_mean,
                         on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=len(batch))
                self.log(f'{name}/spectral_avg_ari_k{k}_graph_laplacian', ari_graph_mean,
                         on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=len(batch))

            if 'random' in self._eigenmaps_methods and len(metrics_per_k[k]['nmi_random']) > 0:
                nmi_random_mean = np.mean(metrics_per_k[k]['nmi_random'])
                ari_random_mean = np.mean(metrics_per_k[k]['ari_random'])
                self.log(f'{name}/spectral_avg_nmi_k{k}_random', nmi_random_mean,
                         on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=len(batch))
                self.log(f'{name}/spectral_avg_ari_k{k}_random', ari_random_mean,
                         on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=len(batch))

            # Log additional metrics (Completeness, AMI, Homogeneity, V-Measure, FMI)
            for metric_name in ['completeness', 'ami', 'homogeneity', 'v_measure', 'fmi']:
                for method in self._eigenmaps_methods:
                    metric_key = f'{metric_name}_{method}'
                    if metric_key in metrics_per_k[k] and len(metrics_per_k[k][metric_key]) > 0:
                        metric_mean = np.mean(metrics_per_k[k][metric_key])
                        log_name = f'{name}/spectral_avg_{metric_name}_k{k}_{method if method != "graph" else "graph_laplacian"}'
                        self.log(log_name, metric_mean,
                                on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=len(batch))

        # ========================================
        # PART 2: Manifold Learning Comparison (NEW)
        # ========================================
        if stage == 'val' and len(self._manifold_learning_methods) > 0:  # Only if methods enabled
            # Accumulate manifold learning metrics for enabled methods only
            manifold_metrics_per_k = {k: {} for k in self._spectral_clustering_k_values}
            for k in self._spectral_clustering_k_values:
                for method in self._manifold_learning_methods:
                    for metric_name in ['nmi', 'ari', 'completeness', 'ami', 'homogeneity', 'v_measure', 'fmi']:
                        manifold_metrics_per_k[k][f'{metric_name}_{method}'] = []

            for data in data_list:
                # Check if we have class labels
                if not hasattr(data, 'class_ids') or len(data.class_ids) == 0:
                    continue

                class_ids = data.class_ids
                n_clusters = len(np.unique(class_ids))

                # Get raw features (DINO/CLIP embeddings)
                raw_features = data.pos.cpu().numpy()  # [n_images, feature_dim]

                # Sample k_neighbors from knn_graph_config (same as used for graph construction)
                k_neighbors = self._knn_graph_config.sample_k(data.pos)

                # For each k value (dimensionality)
                for k in self._spectral_clustering_k_values:
                    # Skip if k is too large
                    if k > raw_features.shape[0] or k > raw_features.shape[1]:
                        continue

                    # Apply classical manifold learning methods to raw features
                    print(f"Computing manifold embeddings for k={k}, k_neighbors={k_neighbors}...")
                    manifold_embeddings = self._compute_manifold_embeddings(
                        raw_features=raw_features,
                        n_components=k,
                        k_neighbors=k_neighbors
                    )

                    # Run k-means on each manifold learning method
                    for method_name, embedding in manifold_embeddings.items():
                        if embedding is not None:
                            try:
                                nmi, ari, completeness, ami, homogeneity, v_measure, fmi = self._compute_clustering_metrics(
                                    torch.from_numpy(embedding).float(), class_ids, n_clusters
                                )
                                manifold_metrics_per_k[k][f'nmi_{method_name}'].append(nmi)
                                manifold_metrics_per_k[k][f'ari_{method_name}'].append(ari)
                                manifold_metrics_per_k[k][f'completeness_{method_name}'].append(completeness)
                                manifold_metrics_per_k[k][f'ami_{method_name}'].append(ami)
                                manifold_metrics_per_k[k][f'homogeneity_{method_name}'].append(homogeneity)
                                manifold_metrics_per_k[k][f'v_measure_{method_name}'].append(v_measure)
                                manifold_metrics_per_k[k][f'fmi_{method_name}'].append(fmi)
                                print(f"  {method_name} k={k}: NMI={nmi:.4f}, ARI={ari:.4f}, Comp={completeness:.4f}, AMI={ami:.4f}, Homo={homogeneity:.4f}, VM={v_measure:.4f}, FMI={fmi:.4f}")
                            except Exception as e:
                                print(f"Clustering failed for {method_name} at k={k}: {e}")

            # Log manifold learning metrics for enabled methods only
            for k in self._spectral_clustering_k_values:
                for method_name in self._manifold_learning_methods:
                    # Check if method has data
                    nmi_key = f'nmi_{method_name}'
                    if nmi_key in manifold_metrics_per_k[k] and len(manifold_metrics_per_k[k][nmi_key]) > 0:
                        # Log all 7 metrics
                        for metric_name in ['nmi', 'ari', 'completeness', 'ami', 'homogeneity', 'v_measure', 'fmi']:
                            metric_key = f'{metric_name}_{method_name}'
                            metric_mean = np.mean(manifold_metrics_per_k[k][metric_key])
                            self.log(f'{name}/manifold_{metric_name}_k{k}_{method_name}', metric_mean,
                                    on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=len(batch))
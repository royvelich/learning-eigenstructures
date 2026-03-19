from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from neural_laplacian import utils
from neural_laplacian.utils import ProjectionMethod
import numpy as np


class LaplacianLoss(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self,
                eigenvectors: torch.Tensor,
                eigenvalues: torch.Tensor,
                vertices: torch.Tensor,
                normals: torch.Tensor,
                weights: torch.Tensor,
                scalar_fields: torch.Tensor,
                edge_index: torch.Tensor,
                reconstructed_laplacian: torch.Tensor
                ) -> torch.Tensor:
        pass


@dataclass
class ReconstructionLossResult:
    loss: torch.Tensor
    unweighted_eigenvalues: torch.Tensor
    weighted_eigenvalues: torch.Tensor


class ReconstructionLoss(LaplacianLoss):
    def __init__(self, projection_method: ProjectionMethod, use_weighted_norm: bool, max_eigenvectors: Optional[int] = None):
        """
        Initialize the ReconstructionLoss.

        Args:
            projection_method: Which projection method to use for reconstruction error computation.
                - UNNORMALIZED: M-orthogonal eigenvectors, simple coefficient computation
                - NORMALIZED: M-weighted projection, solves Gc=b for each truncation level
                - WHITENED: Pure Euclidean projection in whitened domain (Best Bases theorem)
            use_weighted_norm: Whether to use M-weighted norm for the loss computation.
                              Note: For WHITENED, this is ignored since Euclidean norm is correct.
            max_eigenvectors: Number of eigenvectors to use for reconstruction
        """
        super().__init__()
        self._projection_method = projection_method
        self._use_weighted_norm = use_weighted_norm
        self._max_eigenvectors = max_eigenvectors

    @property
    def projection_method(self) -> ProjectionMethod:
        return self._projection_method

    @property
    def use_weighted_norm(self) -> bool:
        return self._use_weighted_norm

    def forward(self,
                eigenvectors: torch.Tensor,
                eigenvalues: torch.Tensor,
                vertices: torch.Tensor,
                normals: torch.Tensor,
                weights: torch.Tensor,
                scalar_fields: torch.Tensor,
                edge_index: torch.Tensor,
                reconstructed_laplacian: torch.Tensor
                ) -> ReconstructionLossResult:
        """
        Compute the reconstruction loss for scalar fields.

        Args:
            eigenvectors: Eigenvectors matrix [num_vertices, num_eigenvectors]
            eigenvalues: Eigenvalues tensor [num_eigenvectors]
            vertices: Vertex positions [num_vertices, 3]
            normals: Vertex normals [num_vertices, 3]
            weights: Vertex area weights [num_vertices]
            scalar_fields: Scalar fields [num_vertices, num_fields]
            edge_index: Graph edge indices [2, num_edges]
            reconstructed_laplacian: Precomputed normalized Laplacian [num_vertices, num_vertices]

        Returns:
            Reconstruction loss value
        """

        max_eigenvectors = eigenvectors.shape[1] if self._max_eigenvectors is None else self._max_eigenvectors
        loss, unweighted_eigenvalues, weighted_eigenvalues = utils.reconstruction_error(
            eigenvectors=eigenvectors,
            scalar_functions=scalar_fields,
            weights=weights,
            max_eigenvectors=max_eigenvectors,
            # projection_method=self._projection_method,
            projection_method=ProjectionMethod.NORMALIZED,
            use_weighted_norm=self._use_weighted_norm
        )

        loss = loss / np.sqrt(max_eigenvectors)

        return ReconstructionLossResult(loss=loss, unweighted_eigenvalues=unweighted_eigenvalues, weighted_eigenvalues=weighted_eigenvalues)
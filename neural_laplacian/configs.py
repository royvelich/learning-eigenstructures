# standard library
from typing import List, Union, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from pathlib import Path
import pickle

# omegaconf
import omegaconf.listconfig

# numpy
import numpy as np
import scipy.sparse

# torch
import torch

# neural_laplacian
from neural_laplacian.generators import ScalarFieldGenerator
from neural_laplacian.losses import LaplacianLoss


@dataclass
class LaplacianLossConfig:
    loss: LaplacianLoss
    weight: float


@dataclass
class OperatorConfig:
    """
    Configuration for differential operator used in eigendecomposition.

    Supports:
    - Laplacian (default): Standard graph Laplacian
    - Schrödinger: H = L + β·diag(V) where V is a potential function

    The operator config is used to:
    1. Generate correct cache key for eigendecomposition
    2. Load correct GT eigenfunctions from cache
    3. Ensure consistency between probe generation and GT evaluation

    IMPORTANT: n_neighbors must be consistent across:
    - GT eigendecomposition (caching)
    - Probe generation (SchrodingerFieldProcessor)
    - Visualization (compute_schrodinger)
    """
    type: str = "laplacian"  # "laplacian" or "schrodinger"
    potential_type: Optional[str] = None  # For Schrödinger: "curvature", "center_distance", "height", etc.
    potential_strength: Optional[float] = None  # β value for Schrödinger
    n_neighbors: int = 30  # Number of neighbors for k-NN (used in curvature estimation and Laplacian)

    def __post_init__(self):
        """Validate configuration."""
        valid_types = ["laplacian", "schrodinger"]
        if self.type not in valid_types:
            raise ValueError(f"operator type must be one of {valid_types}, got '{self.type}'")

        if self.type == "schrodinger":
            if self.potential_type is None:
                raise ValueError("potential_type is required for Schrödinger operator")

            valid_potentials = ["curvature", "inverse_curvature", "center_distance",
                                "height", "inverse_height", "random"]
            if self.potential_type not in valid_potentials:
                raise ValueError(f"potential_type must be one of {valid_potentials}, got '{self.potential_type}'")

            if self.potential_strength is None:
                self.potential_strength = 5.0  # Default β value

    def __repr__(self):
        if self.type == "laplacian":
            return f"OperatorConfig(type='laplacian', n_neighbors={self.n_neighbors})"
        else:
            return f"OperatorConfig(type='schrodinger', potential='{self.potential_type}', β={self.potential_strength}, n_neighbors={self.n_neighbors})"


@dataclass
class ScalarFieldConfig:
    """Configuration for scalar field generation."""
    generators: List[ScalarFieldGenerator]  # Generator to create scalar fields

    def generate_scalar_fields(self, num_points: int) -> torch.Tensor:
        # Use the configured generator to create scalar fields
        scalar_fields_list = []
        for generator in self.generators:
            scalar_fields_list += generator.generate(num_points=num_points)

        scalar_fields_list = random.sample(scalar_fields_list, len(scalar_fields_list))

        return torch.cat(scalar_fields_list, dim=1)


@dataclass
class GaussianNoiseConfig:
    """Configuration for adding Gaussian noise to vertex positions."""
    std: Union[float, Tuple[float, float]]  # Standard deviation (single value or range)
    mean: float = 0.0  # Mean of the Gaussian distribution
    deterministic: bool = True  # Whether to use deterministic noise per item
    seed_offset: int = 0  # Global seed offset for reproducibility (only used if deterministic=True)

    def sample_std(self) -> float:
        """
        Sample a standard deviation value based on the configuration.

        Returns:
            float: If std is a float, returns it directly.
                   If std is a tuple, returns a random value from the range.
        """
        if isinstance(self.std, (tuple, list, omegaconf.ListConfig)):
            if len(self.std) != 2:
                raise ValueError(f"std must be either a float or a tuple/list of exactly 2 floats, got {self.std}")
            return np.random.uniform(self.std[0], self.std[1])
        else:
            return self.std

    def apply_noise(self, positions: torch.Tensor, item_idx: int) -> torch.Tensor:
        """
        Apply Gaussian noise to the given positions.

        Args:
            positions: Tensor of shape [N, 3] containing 3D positions
            item_idx: Index of the dataset item (used for deterministic seeding if enabled)

        Returns:
            Tensor of shape [N, 3] with noise added
        """
        if self.deterministic:
            # Save current RNG states
            torch_state = torch.get_rng_state()
            np_state = np.random.get_state()

            # Set deterministic seed based on item index and global offset
            seed = item_idx + self.seed_offset
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Sample std (this will also be deterministic for the same item_idx)
            std = self.sample_std()

            # Generate noise
            noise = torch.randn_like(positions) * std + self.mean
            noisy_positions = positions + noise

            # Restore original RNG states
            torch.set_rng_state(torch_state)
            np.random.set_state(np_state)

            return noisy_positions
        else:
            # Non-deterministic: just apply random noise
            std = self.sample_std()
            noise = torch.randn_like(positions) * std + self.mean
            return positions + noise


@dataclass
class SamplingConfig(ABC):
    """Base configuration for point cloud sampling strategies."""
    # samples: Optional[Union[int, float, Tuple[int, int], Tuple[float, float]]]
    samples: Any

    def get_number_of_samples(self, points: Union[int, np.ndarray]) -> int:
        """
        Get the number of samples based on configuration.

        Args:
            points: Either total number of points (int) or points array (np.ndarray)

        Returns:
            Number of samples to select
        """
        # Get num_points from input
        if isinstance(points, np.ndarray):
            num_points = points.shape[0]
        else:
            num_points = points

        if self.samples is None:
            # If not specified, return all points
            return num_points

        # Check if it's a tuple/list (range)
        if isinstance(self.samples, (tuple, list, omegaconf.ListConfig)):
            min_val, max_val = self.samples

            # Check if it's int range or float range
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer range - absolute number of samples
                return np.random.randint(min_val, max_val + 1)
            else:
                # Float range - percentage of samples
                target_perc = np.random.uniform(min_val, max_val)
                return int(num_points * target_perc)
        else:
            # Single value
            if isinstance(self.samples, int):
                # Absolute number of samples
                return self.samples
            else:
                # Float - percentage of samples
                return int(num_points * self.samples)

    @abstractmethod
    def sample_indices(self, points: Union[int, np.ndarray], rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        Generate sampling indices according to the specific strategy.

        Args:
            points: Either total number of points (int) or points array (np.ndarray)
            rng: Optional RandomState for deterministic sampling. If None, uses global np.random.

        Returns:
            Array of sampled indices
        """
        pass


@dataclass
class FarthestPointSamplingConfig(SamplingConfig):
    """Configuration for farthest point sampling."""
    random_start: bool = True

    def sample_indices(self, points: Union[int, np.ndarray], rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        Generate FPS sampling indices.

        Args:
            points: Points array (np.ndarray) - FPS requires actual point positions
            rng: Optional RandomState for deterministic sampling. If None, uses global np.random.

        Returns:
            Array of sampled indices using FPS algorithm
        """
        if not isinstance(points, np.ndarray):
            raise TypeError("FarthestPointSamplingConfig requires points to be np.ndarray, not int")

        num_samples = self.get_number_of_samples(points)

        if num_samples >= points.shape[0]:
            return np.arange(points.shape[0])

        # Use the existing FPS implementation from utils
        # Note: You may need to modify farthest_point_sampling to accept rng parameter
        from neural_laplacian.utils import farthest_point_sampling
        return farthest_point_sampling(
            vertices=points,
            num_samples=num_samples,
            random_start=self.random_start
        )


@dataclass
class UniformSamplingConfig(SamplingConfig):
    """Configuration for uniform random sampling of point clouds."""

    def sample_indices(self, points: Union[int, np.ndarray], rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        Generate uniform random sampling indices.

        Args:
            points: Either total number of points (int) or points array (np.ndarray)
            rng: Optional RandomState for deterministic sampling. If None, uses global np.random.

        Returns:
            Array of uniformly sampled indices
        """
        # Use provided rng or default to global np.random
        random_source = rng if rng is not None else np.random

        num_samples = self.get_number_of_samples(points)

        # Get num_points
        if isinstance(points, np.ndarray):
            num_points = points.shape[0]
        else:
            num_points = points

        if num_samples >= num_points:
            return np.arange(num_points)

        # Simple random sampling using provided rng
        sampled_indices = random_source.choice(num_points, num_samples, replace=False)

        return np.sort(sampled_indices)


@dataclass
class ClassWeights(ABC):
    """Base class for class weighting strategies."""

    @abstractmethod
    def get_weights(self, num_classes: int, rng: Optional[np.random.RandomState] = None) -> List[float]:
        """
        Get class weights.

        Args:
            num_classes: Number of classes
            rng: Optional RandomState for deterministic sampling (used by RandomWeights)

        Returns:
            List of weights that sum to 1.0
        """
        pass


@dataclass
class FixedWeights(ClassWeights):
    """Fixed class weights."""
    weights: List[float]

    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError(f"weights must sum to 1.0, got {sum(self.weights)}")

    def get_weights(self, num_classes: int, rng: Optional[np.random.RandomState] = None) -> List[float]:
        """Return the fixed weights (rng parameter ignored for fixed weights)."""
        if len(self.weights) != num_classes:
            raise ValueError(f"Expected {num_classes} weights, got {len(self.weights)}")
        return self.weights


@dataclass
class RandomWeights(ClassWeights):
    """Random class weights with controllable entropy."""
    entropy_range: Tuple[float, float]

    def get_weights(self, num_classes: int, rng: Optional[np.random.RandomState] = None) -> List[float]:
        """
        Generate random weights with random entropy.

        Args:
            num_classes: Number of classes
            rng: Optional RandomState for deterministic sampling. If None, uses global np.random.

        Returns:
            List of random weights that sum to 1.0
        """
        # Use provided rng or default to global np.random
        random_source = rng if rng is not None else np.random

        # Sample random entropy from range
        min_entropy, max_entropy = self.entropy_range
        entropy = random_source.uniform(min_entropy, max_entropy)

        # Sample from Dirichlet distribution
        # Lower alpha = more concentrated weights (low entropy)
        # Higher alpha = more uniform weights (high entropy)
        alpha = entropy * 10  # Scale entropy to a reasonable alpha range
        weights = random_source.dirichlet([alpha] * num_classes)

        return weights.tolist()


@dataclass
class ClassWeightedSamplingConfig(SamplingConfig):
    labels_pkl_path: Path
    class_weights: ClassWeights

    def __post_init__(self):
        with open(self.labels_pkl_path, 'rb') as f:
            self._labels_dict = pickle.load(f)

        # Build arrays
        self._all_indices = np.arange(len(self._labels_dict))
        self._class_ids = np.array([
            info['class_id']
            for info in self._labels_dict.values()
        ])

    @property
    def labels_dict(self) -> dict:
        return self._labels_dict

    def sample_indices(self, points: Union[int, np.ndarray], rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        if isinstance(points, np.ndarray):
            raise TypeError("ClassWeightedSamplingConfig expects points to be int")

        random_source = rng if rng is not None else np.random
        num_samples = self.get_number_of_samples(len(self._labels_dict))
        num_classes = len(np.unique(self._class_ids))

        # Get class weights
        if isinstance(self.class_weights, RandomWeights):
            class_weights = self.class_weights.get_weights(num_classes, rng=rng)
        else:
            class_weights = self.class_weights.get_weights(num_classes)

        # Map class weights to per-sample weights
        sample_weights = np.array([class_weights[cid] for cid in self._class_ids])
        sample_weights /= sample_weights.sum()  # Normalize

        # Single weighted choice - guarantees exactly num_samples
        sampled_indices = random_source.choice(
            self._all_indices,
            size=num_samples,
            replace=False,
            p=sample_weights
        )

        return sampled_indices


@dataclass
class DecimationConfig:
    """Configuration for mesh decimation using quadric edge collapse."""
    target_perc_range: Tuple[float, float]
    quality_thr_range: Tuple[float, float]
    target_face_num_range: Tuple[int, int]
    preserve_boundary: bool = True
    preserve_normal: bool = True
    preserve_topology: bool = True
    optimal_placement: bool = True
    planar_quadric: bool = True


@dataclass
class RemeshingConfig:
    """Configuration for isotropic explicit remeshing."""
    iterations: int = 10
    adaptive: bool = False
    targetlen: float = 0.01  # 1% as default
    featuredeg: float = 30
    checksurfdist: bool = True
    maxsurfdist: float = 0.01  # 1% as default
    splitflag: bool = True
    collapseflag: bool = True
    swapflag: bool = True
    smoothflag: bool = True
    reprojectflag: bool = True


@dataclass
class KnnGraphConfig:
    k: Optional[Union[int, Tuple[int, int]]]
    loop: bool
    cosine: bool

    def sample_k(self, vertices: torch.Tensor) -> int:
        """
        Sample a k value based on the configuration.

        Returns:
            int: If k is an int, returns it directly.
                 If k is a tuple/list, returns a random value from the range.
        """
        if self.k is not None:
            if isinstance(self.k, (tuple, list, omegaconf.ListConfig)):
                if len(self.k) != 2:
                    raise ValueError(f"k must be either an int or a tuple/list of exactly 2 integers, got {self.k}")
                return np.random.randint(low=self.k[0], high=self.k[1] + 1)  # +1 because randint is exclusive on upper bound
            else:
                return self.k
        else:
            k = max(int(vertices.shape[0] / 100), 10)
            return k


@dataclass
class AnchorPointsConfig:
    ratio_range: Tuple[float, float]


@dataclass
class NormalEstimationConfig:
    """Configuration for point cloud normal estimation."""
    k_neighbors: int = 10
    k_orient: int = 10
    lambda_param: float = 0.0
    cos_alpha_tol: float = 1.0
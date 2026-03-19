import omegaconf
import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import get_laplacian, add_self_loops
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch_scatter import scatter
import numpy as np
from neural_laplacian import utils


class ScalarFieldSmoother(MessagePassing):
    def __init__(self, iterations: Union[int, omegaconf.ListConfig], cosine: bool, sigma: float, normalize_weights: bool, aggr: str) -> None:
        """
        Initialize the scalar field smoothing operator.

        Args:
            aggr (str): Aggregation method ('add', 'mean', 'max'). Default: 'add'
            sigma (float): Sigma parameter for Gaussian kernel. Controls smoothing strength.
                           Higher values lead to more influence from distant points.
            normalize_weights (bool): Whether to normalize weights to sum to 1 for each node.
            iterations (int): Number of iterations to apply the smoothing. Default: 1
        """
        super(ScalarFieldSmoother, self).__init__(aggr=aggr)
        self._sigma = sigma
        self._cosine = cosine
        self._normalize_weights = normalize_weights
        self._iterations = iterations

    @property
    def sigma(self) -> float:
        if isinstance(self._sigma, omegaconf.ListConfig):
            min_sigma, max_sigma = self._sigma
            return np.random.uniform(min_sigma, max_sigma)
        else:
            return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    @property
    def iterations(self) -> int:
        if isinstance(self._iterations, omegaconf.ListConfig):
            min_iterations, max_iterations = self._iterations
            return np.random.randint(min_iterations, max_iterations + 1)
        else:
            return self._iterations

    @iterations.setter
    def iterations(self, iterations):
        self._iterations = iterations

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.LongTensor,
            edge_weight: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Scalar values at each point [num_nodes, num_fields]
            edge_index (torch.LongTensor): Graph connectivity [2, num_edges]
            edge_weight (torch.Tensor, optional): Edge weights based on distance [num_edges]
            pos (torch.Tensor, optional): Point positions [num_nodes, 3]

        Returns:
            torch.Tensor: Smoothed scalar field [num_nodes, num_fields]
        """
        # Apply the smoothing multiple times based on iterations parameter
        smoothed_x = x

        for _ in range(self.iterations):
            # If edge weights aren't provided, compute them based on distances
            curr_edge_weight = edge_weight
            if curr_edge_weight is None and pos is not None:
                distances = utils.compute_distances(pos=pos, edge_index=edge_index, use_cosine=self._cosine)
                sigma = self.sigma
                # sigma = torch.median(distances)

                # Get source and target node indices
                # row, col = edge_index

                # Calculate Euclidean distances between connected nodes
                # delta = pos[row] - pos[col]
                # curr_edge_weight = torch.norm(delta, dim=1)

                # Convert distances to weights using Gaussian kernel: exp(-dÂ²/ÏƒÂ²)
                # Using the sigma parameter to control the kernel width
                curr_edge_weight = torch.exp(-(distances ** 2) / (sigma ** 2))

            # Normalize weights so they sum to 1 for each node
            norm: Optional[torch.Tensor] = None
            if curr_edge_weight is not None and self._normalize_weights:
                row, col = edge_index
                out = scatter(curr_edge_weight, row, dim=0, reduce="sum")
                norm = curr_edge_weight / out[row]
            else:
                norm = curr_edge_weight

            # Perform the weighted message passing for this iteration
            # For each field, smoothed_x will be of shape [num_nodes, num_fields]
            smoothed_x = self.propagate(edge_index, x=smoothed_x, norm=norm)

        return smoothed_x

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        # Weight the neighbor values
        # x_j is of shape [num_edges, num_fields]
        # norm is of shape [num_edges]
        # We need to expand norm to match x_j's shape for proper multiplication
        return norm.view(-1, 1) * x_j

    def edge_update(self) -> Tensor:
        raise NotImplementedError

    def message_and_aggregate(self, edge_index: Adj) -> Tensor:
        raise NotImplementedError


class HadamardFieldProcessor(torch.nn.Module):
    """
    Generates piecewise constant probe functions via hierarchical bisection.

    This is the Hadamard analogue of ScalarFieldSmoother. Instead of smoothing
    random noise to create Laplacian-like probe functions, it partitions the
    manifold hierarchically and assigns constant values per region to create
    Hadamard-like probe functions.

    Key differences from Laplacian probes:
    - Laplacian probes: smooth, continuous gradients (low Dirichlet energy)
    - Hadamard probes: piecewise constant, sharp boundaries (low sequency)

    The learned basis will be optimal for reconstructing piecewise constant
    functions on hierarchical partitions of the manifold.

    This implementation uses vectorized operations for efficiency.
    """

    def __init__(
            self,
            max_depth: Union[int, omegaconf.ListConfig] = 6,
            split_probability: Union[float, omegaconf.ListConfig] = 0.75,
            **kwargs  # Accept extra args for compatibility with ScalarFieldSmoother interface
    ) -> None:
        """
        Initialize the Hadamard field processor.

        Args:
            max_depth: Maximum recursion depth for hierarchical bisection.
                      Controls maximum number of regions (up to 2^max_depth).
                      Can be int or [min, max] range.
            split_probability: Probability of splitting at each level after depth 0.
                              Lower values = simpler probes (fewer regions).
                              Can be float or [min, max] range.
        """
        super().__init__()
        self._max_depth = max_depth
        self._split_probability = split_probability

    @property
    def max_depth(self) -> int:
        if isinstance(self._max_depth, omegaconf.ListConfig):
            min_depth, max_depth = self._max_depth
            return np.random.randint(min_depth, max_depth + 1)
        else:
            return self._max_depth

    @property
    def split_probability(self) -> float:
        if isinstance(self._split_probability, omegaconf.ListConfig):
            min_prob, max_prob = self._split_probability
            return np.random.uniform(min_prob, max_prob)
        else:
            return self._split_probability

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.LongTensor,
            edge_weight: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transform random noise into piecewise constant probe functions.

        Args:
            x (torch.Tensor): Random scalar values [num_nodes, num_fields]
                             Used as seeds for random values per region.
            edge_index (torch.LongTensor): Graph connectivity [2, num_edges]
                                          (not directly used, kept for interface compatibility)
            edge_weight (torch.Tensor, optional): Edge weights (not used)
            pos (torch.Tensor): Point positions [num_nodes, dim] - REQUIRED

        Returns:
            torch.Tensor: Piecewise constant fields [num_nodes, num_fields]
        """
        if pos is None:
            raise ValueError("HadamardFieldProcessor requires point positions (pos)")

        num_nodes, num_fields = x.shape
        device = x.device
        dtype = x.dtype

        # Convert positions to numpy for efficient processing
        pos_np = pos.detach().cpu().numpy()

        # Get parameters (sampled once per forward pass)
        max_depth = self.max_depth
        split_prob = self.split_probability

        # Generate all fields using vectorized approach
        output = self._generate_piecewise_constant_fields_vectorized(
            pos_np=pos_np,
            num_fields=num_fields,
            max_depth=max_depth,
            split_prob=split_prob
        )

        return torch.from_numpy(output).to(device=device, dtype=dtype)

    def _generate_piecewise_constant_fields_vectorized(
            self,
            pos_np: np.ndarray,
            num_fields: int,
            max_depth: int,
            split_prob: float
    ) -> np.ndarray:
        """
        Generate multiple piecewise constant fields using vectorized operations.

        Instead of recursion, we:
        1. Pre-generate random hyperplanes for all levels and fields
        2. Compute binary partition codes for all points (vectorized)
        3. Apply split_probability to merge some regions
        4. Assign random values to final regions

        Args:
            pos_np: Point positions [num_nodes, dim]
            num_fields: Number of fields to generate
            max_depth: Maximum recursion depth
            split_prob: Probability of splitting at each level

        Returns:
            Piecewise constant signals [num_nodes, num_fields]
        """
        num_nodes, dim = pos_np.shape

        # Output array
        output = np.zeros((num_nodes, num_fields), dtype=np.float32)

        for field_idx in range(num_fields):
            # Generate random hyperplanes for this field: one per level
            # Each hyperplane is defined by a direction vector
            directions = np.random.randn(max_depth, dim)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10

            # Decide which levels actually split (based on split_probability)
            # Level 0 always splits
            do_split = np.ones(max_depth, dtype=bool)
            do_split[1:] = np.random.rand(max_depth - 1) < split_prob

            # Compute which side of each hyperplane each point falls on
            # We need to track region assignments level by level

            # Start: all points in region 0
            region_ids = np.zeros(num_nodes, dtype=np.int32)

            for level in range(max_depth):
                if not do_split[level]:
                    # Don't split at this level - keep current regions
                    continue

                # Get unique regions at this level
                unique_regions = np.unique(region_ids)

                # For each region, compute a split
                new_region_ids = region_ids.copy()

                for region_id in unique_regions:
                    mask = (region_ids == region_id)
                    region_points = pos_np[mask]

                    if len(region_points) < 2:
                        continue

                    # Compute centroid of this region
                    centroid = region_points.mean(axis=0)

                    # Project onto this level's direction
                    projections = (region_points - centroid) @ directions[level]

                    # Split at median
                    median_proj = np.median(projections)

                    # Points above median go to new region (region_id * 2 + 1)
                    # Points at or below stay (region_id * 2)
                    split_mask = projections > median_proj

                    # Update region IDs
                    region_indices = np.where(mask)[0]
                    new_region_ids[region_indices[~split_mask]] = region_id * 2
                    new_region_ids[region_indices[split_mask]] = region_id * 2 + 1

                region_ids = new_region_ids

            # Assign random value to each final region
            unique_final_regions = np.unique(region_ids)
            region_values = {r: np.random.randn() for r in unique_final_regions}

            # Vectorized assignment
            for region_id, value in region_values.items():
                output[region_ids == region_id, field_idx] = value

        return output


class HadamardFieldProcessorFast(torch.nn.Module):
    """
    Ultra-fast version of HadamardFieldProcessor using fully vectorized operations.

    This version generates partitions by:
    1. Pre-computing ALL random hyperplane projections at once [num_nodes, max_depth, num_fields]
    2. Converting to binary codes [num_nodes, num_fields]
    3. Mapping codes to random values

    Much faster but uses more memory. Good for moderate num_nodes and num_fields.
    """

    def __init__(
            self,
            max_depth: Union[int, omegaconf.ListConfig] = 6,
            split_probability: Union[float, omegaconf.ListConfig] = 0.75,
            **kwargs
    ) -> None:
        super().__init__()
        self._max_depth = max_depth
        self._split_probability = split_probability

    @property
    def max_depth(self) -> int:
        if isinstance(self._max_depth, omegaconf.ListConfig):
            min_depth, max_depth = self._max_depth
            return np.random.randint(min_depth, max_depth + 1)
        else:
            return self._max_depth

    @property
    def split_probability(self) -> float:
        if isinstance(self._split_probability, omegaconf.ListConfig):
            min_prob, max_prob = self._split_probability
            return np.random.uniform(min_prob, max_prob)
        else:
            return self._split_probability

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.LongTensor,
            edge_weight: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if pos is None:
            raise ValueError("HadamardFieldProcessorFast requires point positions (pos)")

        num_nodes, num_fields = x.shape
        device = x.device
        dtype = x.dtype

        pos_np = pos.detach().cpu().numpy()
        max_depth = self.max_depth
        split_prob = self.split_probability

        output = self._generate_fields_fully_vectorized(
            pos_np=pos_np,
            num_fields=num_fields,
            max_depth=max_depth,
            split_prob=split_prob
        )

        return torch.from_numpy(output).to(device=device, dtype=dtype)

    def _generate_fields_fully_vectorized(
            self,
            pos_np: np.ndarray,
            num_fields: int,
            max_depth: int,
            split_prob: float
    ) -> np.ndarray:
        """
        Fully vectorized generation across all fields simultaneously.
        """
        num_nodes, dim = pos_np.shape

        # Generate random directions for all levels and all fields at once
        # Shape: [max_depth, num_fields, dim]
        directions = np.random.randn(max_depth, num_fields, dim)
        directions /= np.linalg.norm(directions, axis=2, keepdims=True) + 1e-10

        # Generate split decisions for all levels and fields
        # Shape: [max_depth, num_fields]
        # Level 0 always splits
        do_split = np.random.rand(max_depth, num_fields) < split_prob
        do_split[0, :] = True  # Always split at level 0

        # Center the points (use global centroid for simplicity in fully vectorized version)
        # For more accurate results, HadamardFieldProcessor computes per-region centroids
        centroid = pos_np.mean(axis=0)
        centered_pos = pos_np - centroid

        # Compute projections for all points, all levels, all fields at once
        # pos: [num_nodes, dim]
        # directions: [max_depth, num_fields, dim]
        # Result: [num_nodes, max_depth, num_fields]
        projections = np.einsum('nd,lfd->nlf', centered_pos, directions)

        # Convert projections to binary (above/below median per level per field)
        # Compute medians: [max_depth, num_fields]
        medians = np.median(projections, axis=0)

        # Binary codes: [num_nodes, max_depth, num_fields]
        binary_codes = (projections > medians).astype(np.int32)

        # Apply split probability: where do_split is False, set that level's bit to 0
        # This effectively merges regions that would have been split
        binary_codes *= do_split[np.newaxis, :, :]

        # Convert binary codes to region IDs
        # Each column of binary_codes[:, :, f] is a binary number
        powers_of_2 = 2 ** np.arange(max_depth)  # [1, 2, 4, 8, ...]
        region_ids = np.einsum('nlf,l->nf', binary_codes, powers_of_2)

        # Now assign random values to each unique region for each field
        output = np.zeros((num_nodes, num_fields), dtype=np.float32)

        for field_idx in range(num_fields):
            field_regions = region_ids[:, field_idx]
            unique_regions = np.unique(field_regions)

            # Generate random value for each region
            region_values = np.random.randn(len(unique_regions))

            # Create mapping and apply
            region_to_value = dict(zip(unique_regions, region_values))
            output[:, field_idx] = np.array([region_to_value[r] for r in field_regions])

        return output


class MorletFieldProcessor(torch.nn.Module):
    """
    Generates Morlet wavelet-like probe functions.

    Morlet wavelet = Gaussian envelope × Sinusoidal oscillation
    ψ(x) = exp(-|x-c|²/2σ²) × cos(ω·(x-c)·d)

    This induces a basis optimal for localized oscillatory signals,
    combining spatial localization with frequency structure.

    Key properties:
    - Spatially localized (Gaussian envelope)
    - Oscillatory within localization (sinusoidal)
    - Optimal time-frequency uncertainty

    Different from:
    - Laplacian: global smooth oscillations
    - Hadamard: piecewise constant (no oscillation)
    - Sparse: localized but no oscillation
    """

    def __init__(
            self,
            scale_range: Union[Tuple[float, float], omegaconf.ListConfig] = (0.05, 0.3),
            frequency_range: Union[Tuple[float, float], omegaconf.ListConfig] = (5.0, 30.0),
            **kwargs
    ) -> None:
        """
        Initialize the Morlet field processor.

        Args:
            scale_range: (min, max) for Gaussian envelope width (relative to shape diameter)
            frequency_range: (min, max) for oscillation frequency
        """
        super().__init__()
        self._scale_range = scale_range
        self._frequency_range = frequency_range

    @property
    def scale_range(self) -> Tuple[float, float]:
        if isinstance(self._scale_range, omegaconf.ListConfig):
            return tuple(self._scale_range)
        return self._scale_range

    @property
    def frequency_range(self) -> Tuple[float, float]:
        if isinstance(self._frequency_range, omegaconf.ListConfig):
            return tuple(self._frequency_range)
        return self._frequency_range

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.LongTensor,
            edge_weight: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transform random noise into Morlet wavelet probe functions.

        Args:
            x (torch.Tensor): Random scalar values [num_nodes, num_fields]
                             (used only for shape, actual values ignored)
            edge_index (torch.LongTensor): Graph connectivity (not used)
            edge_weight (torch.Tensor, optional): Edge weights (not used)
            pos (torch.Tensor): Point positions [num_nodes, dim] - REQUIRED

        Returns:
            torch.Tensor: Morlet wavelet fields [num_nodes, num_fields]
        """
        if pos is None:
            raise ValueError("MorletFieldProcessor requires point positions (pos)")

        num_nodes, num_fields = x.shape
        device = x.device
        dtype = x.dtype

        pos_np = pos.detach().cpu().numpy()

        output = self._generate_morlet_fields_vectorized(
            pos_np=pos_np,
            num_fields=num_fields
        )

        return torch.from_numpy(output).to(device=device, dtype=dtype)

    def _generate_morlet_fields_vectorized(
            self,
            pos_np: np.ndarray,
            num_fields: int
    ) -> np.ndarray:
        """
        Fast vectorized Morlet probe generation.
        """
        num_nodes, dim = pos_np.shape

        # Compute diameter for scale normalization
        diameter = np.linalg.norm(pos_np.max(axis=0) - pos_np.min(axis=0))

        scale_range = self.scale_range
        frequency_range = self.frequency_range

        # Random centers: [num_fields]
        center_indices = np.random.randint(0, num_nodes, num_fields)
        centers = pos_np[center_indices]  # [num_fields, dim]

        # Random scales: [num_fields]
        scales = np.random.uniform(*scale_range, num_fields) * diameter

        # Random frequencies: [num_fields]
        frequencies = np.random.uniform(*frequency_range, num_fields)

        # Random directions: [num_fields, dim]
        directions = np.random.randn(num_fields, dim)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10

        # Compute displacements: [num_nodes, num_fields, dim]
        displacements = pos_np[:, np.newaxis, :] - centers[np.newaxis, :, :]

        # Distances: [num_nodes, num_fields]
        distances = np.linalg.norm(displacements, axis=2)

        # Gaussian envelopes: [num_nodes, num_fields]
        envelopes = np.exp(-distances ** 2 / (2 * scales[np.newaxis, :] ** 2))

        # Phases: project onto directions × frequency
        phases = np.sum(displacements * directions[np.newaxis, :, :], axis=2)
        phases = phases * frequencies[np.newaxis, :]

        # Oscillations
        oscillations = np.cos(phases)

        # Morlet = envelope × oscillation
        output = envelopes * oscillations

        return output.astype(np.float32)


class MexicanHatFieldProcessor(torch.nn.Module):
    """
    Generates Mexican Hat (Laplacian of Gaussian) wavelet probe functions.

    Mexican Hat = (1 - r²/σ²) × exp(-r²/2σ²)

    This is isotropic (no directional preference) with a central peak
    surrounded by a negative ring. Also known as Ricker wavelet.

    Different from Morlet: no oscillation direction, radially symmetric.
    """

    def __init__(
            self,
            scale_range: Union[Tuple[float, float], omegaconf.ListConfig] = (0.05, 0.3),
            **kwargs
    ) -> None:
        """
        Args:
            scale_range: (min, max) for wavelet width (relative to diameter)
        """
        super().__init__()
        self._scale_range = scale_range

    @property
    def scale_range(self) -> Tuple[float, float]:
        if isinstance(self._scale_range, omegaconf.ListConfig):
            return tuple(self._scale_range)
        return self._scale_range

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.LongTensor,
            edge_weight: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if pos is None:
            raise ValueError("MexicanHatFieldProcessor requires point positions (pos)")

        num_nodes, num_fields = x.shape
        device = x.device
        dtype = x.dtype

        pos_np = pos.detach().cpu().numpy()
        output = self._generate_mexican_hat_fields(pos_np, num_fields)

        return torch.from_numpy(output).to(device=device, dtype=dtype)

    def _generate_mexican_hat_fields(
            self,
            pos_np: np.ndarray,
            num_fields: int
    ) -> np.ndarray:
        """Vectorized Mexican Hat generation."""
        num_nodes, dim = pos_np.shape
        diameter = np.linalg.norm(pos_np.max(axis=0) - pos_np.min(axis=0))

        scale_range = self.scale_range

        # Random centers
        center_indices = np.random.randint(0, num_nodes, num_fields)
        centers = pos_np[center_indices]

        # Random scales
        scales = np.random.uniform(*scale_range, num_fields) * diameter

        # Displacements: [num_nodes, num_fields, dim]
        displacements = pos_np[:, np.newaxis, :] - centers[np.newaxis, :, :]

        # Squared distances: [num_nodes, num_fields]
        distances_sq = np.sum(displacements ** 2, axis=2)

        # Normalized squared distances
        normalized_dist_sq = distances_sq / (scales[np.newaxis, :] ** 2)

        # Mexican hat: (1 - r²/σ²) × exp(-r²/2σ²)
        output = (1 - normalized_dist_sq) * np.exp(-normalized_dist_sq / 2)

        return output.astype(np.float32)


class SparseFieldProcessor(torch.nn.Module):
    """
    Generates sparse probe functions (few non-zero values).

    This induces a basis optimal for sparse signals, related to
    compressed sensing and dictionary learning.

    The learned basis should be localized (delta-like functions
    concentrated on few points).
    """

    def __init__(
            self,
            sparsity: Union[float, omegaconf.ListConfig] = 0.05,
            **kwargs
    ) -> None:
        """
        Args:
            sparsity: Fraction of non-zero entries (0.05 = 5% non-zero)
                     Can be float or [min, max] range.
        """
        super().__init__()
        self._sparsity = sparsity

    @property
    def sparsity(self) -> float:
        if isinstance(self._sparsity, omegaconf.ListConfig):
            return np.random.uniform(*self._sparsity)
        return self._sparsity

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.LongTensor,
            edge_weight: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transform random noise into sparse signals.

        Args:
            x: Random values [num_nodes, num_fields] - used as source values
            edge_index, edge_weight, pos: Not used (interface compatibility)

        Returns:
            Sparse signals [num_nodes, num_fields]
        """
        sparsity = self.sparsity

        # Create sparse mask
        mask = torch.rand_like(x) < sparsity

        # Apply mask: non-zero values where mask is True
        output = torch.where(mask, x, torch.zeros_like(x))

        return output


class SchrodingerFieldProcessor(MessagePassing):
    """
    Generate probe functions for Schrödinger operator: H = L + β·V(x)

    The eigenfunctions of H satisfy: H·φ = λ·φ

    Low-energy eigenfunctions are:
    1. Smooth (low Laplacian energy)
    2. Small where V is large (low potential energy)

    Two approaches available (controlled by `use_multiplicative_decay`):

    1. SUBTRACTIVE (default): Gradient flow on Rayleigh quotient
        f^{t+1} = f^t - dt·(L·f^t + β·V·f^t)

    2. MULTIPLICATIVE: Alternating diffusion + exponential decay
        f^{t+1} = exp(-β·V) · smooth(f^t)

    IMPORTANT: Uses utils.compute_potential() for V to ensure exact match with GT.

    Inherits from MessagePassing for efficient Laplacian computation.
    """

    def __init__(
            self,
            potential_type: str = "curvature",
            potential_strength: Union[float, omegaconf.ListConfig] = 5.0,
            iterations: Union[int, omegaconf.ListConfig] = 20,
            sigma: Union[float, omegaconf.ListConfig] = 0.1,
            dt: float = 0.5,
            n_neighbors: int = 30,
            use_multiplicative_decay: bool = False,
            normalize_per_iteration: bool = False,
            aggr: str = "add",
            **kwargs
    ):
        """
        Args:
            potential_type: Type of potential function V(x)
            potential_strength: β coefficient for potential term
            iterations: Number of gradient flow iterations
            sigma: Gaussian kernel width (used if not using adaptive sigma)
            dt: Time step for gradient descent (subtractive mode only)
            n_neighbors: Number of neighbors for curvature estimation.
                        MUST MATCH the n_neighbors used in GT eigendecomposition!
            use_multiplicative_decay: If True, use alternating diffusion + exp(-βV) decay.
                                     If False, use subtractive gradient descent.
            normalize_per_iteration: Whether to normalize f after each iteration
            aggr: Aggregation method for message passing
        """
        super().__init__(aggr=aggr)
        self.potential_type = potential_type
        self._potential_strength = potential_strength
        self._iterations = iterations
        self._sigma = sigma
        self.dt = dt
        self.n_neighbors = n_neighbors
        self.use_multiplicative_decay = use_multiplicative_decay
        self.normalize_per_iteration = normalize_per_iteration

    @property
    def potential_strength(self) -> float:
        if isinstance(self._potential_strength, omegaconf.ListConfig):
            return np.random.uniform(*self._potential_strength)
        return self._potential_strength

    @property
    def iterations(self) -> int:
        if isinstance(self._iterations, omegaconf.ListConfig):
            return np.random.randint(self._iterations[0], self._iterations[1] + 1)
        return self._iterations

    @property
    def sigma(self) -> float:
        if isinstance(self._sigma, omegaconf.ListConfig):
            return np.random.uniform(*self._sigma)
        return self._sigma

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Message function for Laplacian: weighted neighbor values."""
        return norm.view(-1, 1) * x_j

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate Schrödinger probe functions via gradient flow.
        """
        if pos is None:
            raise ValueError("SchrodingerFieldProcessor requires point positions (pos)")

        # Get parameters (potentially randomized)
        iterations = self.iterations
        beta = self.potential_strength

        # Compute potential field using SAME function as GT eigendecomposition
        V_numpy = utils.compute_potential(
            vertices=pos.detach().cpu().numpy(),
            potential_type=self.potential_type,
            n_neighbors=self.n_neighbors,
            normalize=True
        )
        V = torch.from_numpy(V_numpy).to(pos.device, pos.dtype)

        # Compute normalized edge weights for Laplacian
        row, col = edge_index
        num_nodes = pos.shape[0]

        distances = torch.norm(pos[row] - pos[col], dim=1)
        sigma = self.sigma
        edge_weights = torch.exp(-distances ** 2 / (2 * sigma ** 2))

        # Compute degree and normalized weights
        degree = scatter(edge_weights, row, dim=0, dim_size=num_nodes, reduce="sum")
        degree_inv = 1.0 / (degree + 1e-8)
        norm = edge_weights * degree_inv[row]

        # Start with input noise
        f = x

        if self.use_multiplicative_decay:
            # ===== MULTIPLICATIVE APPROACH =====
            # Alternating: diffusion then exp(-βV) decay
            decay = torch.exp(-beta * V)  # [num_nodes]

            for _ in range(iterations):
                # Step 1: Diffusion (weighted average of neighbors)
                f = self.propagate(edge_index, x=f, norm=norm)

                # Step 2: Multiplicative decay by potential
                f = f * decay.unsqueeze(1)

                if self.normalize_per_iteration:
                    f = F.normalize(f, p=2, dim=0)
        else:
            # ===== SUBTRACTIVE APPROACH =====
            # Gradient descent: f^{t+1} = f^t - dt·(L·f + β·V·f)
            dt = self.dt

            for _ in range(iterations):
                # Compute W·f (weighted average of neighbors)
                Wf = self.propagate(edge_index, x=f, norm=norm)

                # Compute L·f = f - W·f (random walk Laplacian: L = I - D^{-1}A)
                Lf = f - Wf

                # Compute H·f = L·f + β·V·f
                Hf = Lf + beta * V.unsqueeze(1) * f

                # Gradient descent step
                f = f - dt * Hf

                if self.normalize_per_iteration:
                    f = F.normalize(f, p=2, dim=0)

        return f


class SimpleSchrodingerFieldProcessor(MessagePassing):
    """
    Simple Schrödinger probe generator: potential modulation + Gaussian smoothing.

    This is a simpler alternative to SchrodingerFieldProcessor that separates
    the two effects:

    1. LOCALIZATION: Modulate initial noise by exp(-βV) to concentrate in low-V regions
    2. SMOOTHING: Apply regular Gaussian smoothing (like ScalarFieldSmoother)

    The idea is that Schrödinger eigenfunctions are:
    - Smooth (handled by Gaussian smoothing)
    - Localized where V is low (handled by initial modulation)

    By applying potential modulation ONCE at the start (not interleaved),
    we get a cleaner separation of effects and use the same smoothing
    as standard Laplacian probes.

    IMPORTANT: Uses utils.compute_potential() for V to match GT computation.
    """

    def __init__(
            self,
            potential_type: str = "curvature",
            potential_strength: Union[float, omegaconf.ListConfig] = 5.0,
            iterations: Union[int, omegaconf.ListConfig] = 20,
            sigma: Union[float, omegaconf.ListConfig] = 0.1,
            n_neighbors: int = 30,
            normalize_weights: bool = True,
            cosine: bool = False,
            aggr: str = "add",
            **kwargs
    ):
        """
        Args:
            potential_type: Type of potential function V(x)
                - "curvature": V = local curvature estimate
                - "inverse_curvature": V = max_curv - curvature
                - "center_distance": V = distance from centroid
                - "random": V = random values per point
                - "height": V = z-coordinate
            potential_strength: β in exp(-βV), controls localization strength
            iterations: Number of Gaussian smoothing iterations
            sigma: Gaussian kernel width for smoothing
            n_neighbors: Number of neighbors for curvature estimation.
                        MUST MATCH the n_neighbors used in GT eigendecomposition!
            normalize_weights: Whether to normalize edge weights to sum to 1
            cosine: Whether to use cosine distance (like ScalarFieldSmoother)
            aggr: Aggregation method for message passing
        """
        super().__init__(aggr=aggr)
        self.potential_type = potential_type
        self._potential_strength = potential_strength
        self._iterations = iterations
        self._sigma = sigma
        self.n_neighbors = n_neighbors
        self.normalize_weights = normalize_weights
        self.cosine = cosine

    @property
    def potential_strength(self) -> float:
        if isinstance(self._potential_strength, omegaconf.ListConfig):
            return np.random.uniform(*self._potential_strength)
        return self._potential_strength

    @property
    def iterations(self) -> int:
        if isinstance(self._iterations, omegaconf.ListConfig):
            return np.random.randint(self._iterations[0], self._iterations[1] + 1)
        return self._iterations

    @property
    def sigma(self) -> float:
        if isinstance(self._sigma, omegaconf.ListConfig):
            return np.random.uniform(*self._sigma)
        return self._sigma

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Message function: weighted neighbor values."""
        return norm.view(-1, 1) * x_j

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate Schrödinger probe functions.

        Steps:
            1. Compute potential V(x) using same method as GT
            2. Modulate input noise: x_mod = x * exp(-β·V)
            3. Apply Gaussian smoothing (like ScalarFieldSmoother)

        Args:
            x: Input noise [num_nodes, num_fields]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional precomputed edge weights
            pos: Point positions [num_nodes, dim] - REQUIRED

        Returns:
            Probe functions [num_nodes, num_fields]
        """
        if pos is None:
            raise ValueError("SimpleSchrodingerFieldProcessor requires point positions (pos)")

        # Get parameters (potentially randomized)
        iterations = self.iterations
        sigma = self.sigma
        beta = self.potential_strength

        # ===== STEP 1: Compute potential =====
        V_numpy = utils.compute_potential(
            vertices=pos.detach().cpu().numpy(),
            potential_type=self.potential_type,
            n_neighbors=self.n_neighbors,
            normalize=True
        )
        V = torch.from_numpy(V_numpy).to(pos.device, pos.dtype)

        # ===== STEP 2: Modulate noise by potential =====
        # exp(-βV) is large where V is small (low potential regions)
        # This localizes the initial noise to where eigenfunctions should concentrate
        decay = torch.exp(-beta * V)  # [num_nodes]
        f = x * decay.unsqueeze(1)  # [num_nodes, num_fields]

        # ===== STEP 3: Gaussian smoothing (same as ScalarFieldSmoother) =====
        for _ in range(iterations):
            # Compute edge weights from distances
            distances = utils.compute_distances(pos=pos, edge_index=edge_index, use_cosine=self.cosine)
            curr_edge_weight = torch.exp(-(distances ** 2) / (sigma ** 2))

            # Normalize weights
            if self.normalize_weights:
                row, col = edge_index
                weight_sum = scatter(curr_edge_weight, row, dim=0, reduce="sum")
                norm = curr_edge_weight / (weight_sum[row] + 1e-8)
            else:
                norm = curr_edge_weight

            # Message passing (weighted average of neighbors)
            f = self.propagate(edge_index, x=f, norm=norm)

        return f


class FixedPartitionHadamardProcessor(torch.nn.Module):
    """
    Hadamard probes with FIXED partition structure using K-means clustering.

    Unlike HadamardFieldProcessor which generates a new random partition for
    each probe, this class computes the partition ONCE via K-means and caches it.
    Each probe then assigns different random constant values to the same fixed regions.

    This is cleaner for analysis:
    - Fixed partition structure (K spatial regions)
    - Probes sample from "all piecewise constant functions on THIS partition"
    - Only the constant values per region vary between probes

    The learned basis should be indicator functions on the K regions.
    """

    def __init__(
            self,
            k: Union[int, omegaconf.ListConfig] = 8,
            seed: Optional[int] = None,
            **kwargs
    ):
        """
        Args:
            k: Number of K-means clusters (regions).
               If ListConfig [min, max], sampled once when partition is created.
            seed: Random seed for reproducible partition. If None, random.
        """
        super().__init__()
        self._k_config = k
        self._seed = seed

        # Cached partition (computed on first forward)
        self._region_ids: Optional[torch.Tensor] = None
        self._num_regions: Optional[int] = None

    def _sample_k(self) -> int:
        """Sample k from config."""
        if isinstance(self._k_config, omegaconf.ListConfig):
            return np.random.randint(self._k_config[0], self._k_config[1] + 1)
        return self._k_config

    def _compute_partition(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Compute K-means partition.

        Args:
            pos: Point positions [num_nodes, dim]

        Returns:
            region_ids: [num_nodes] integer region assignment (0 to K-1)
        """
        from sklearn.cluster import KMeans

        pos_np = pos.detach().cpu().numpy()

        # Sample k (gets fixed for this partition)
        k = self._sample_k()
        self._num_regions = k

        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=self._seed, n_init=10)
        region_ids = kmeans.fit_predict(pos_np)

        return torch.from_numpy(region_ids.astype(np.int64))

    def reset_partition(self):
        """Force recomputation of partition on next forward call."""
        self._region_ids = None
        self._num_regions = None

    @property
    def num_regions(self) -> Optional[int]:
        """Number of regions in the current partition (None if not computed yet)."""
        return self._num_regions

    @property
    def region_ids(self) -> Optional[torch.Tensor]:
        """Current partition region IDs (None if not computed yet)."""
        return self._region_ids

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate Hadamard probe functions on fixed K-means partition.

        Args:
            x: Input tensor [num_nodes, num_fields] (shape used, values ignored)
            edge_index: Graph connectivity (not used, for interface compatibility)
            edge_weight: Edge weights (not used)
            pos: Point positions [num_nodes, dim] - REQUIRED

        Returns:
            Piecewise constant probe functions [num_nodes, num_fields]
        """
        if pos is None:
            raise ValueError("FixedPartitionHadamardProcessor requires point positions (pos)")

        num_nodes, num_fields = x.shape
        device = x.device
        dtype = x.dtype

        # Compute partition on first call (then cache)
        if self._region_ids is None:
            self._region_ids = self._compute_partition(pos)
            print(f"FixedPartitionHadamardProcessor: Created K-means partition with "
                  f"{self._num_regions} regions")

        region_ids = self._region_ids.to(device)

        # Generate random values for each region for each field
        # Shape: [num_regions, num_fields]
        region_values = torch.randn(self._num_regions, num_fields, device=device, dtype=dtype)

        # Assign values to points based on their region
        # output[i, j] = region_values[region_ids[i], j]
        output = region_values[region_ids]

        return output


class SmoothedPartitionHadamardProcessor(MessagePassing):
    """
    Hadamard probes with K-means partition + Gaussian smoothing.

    This is a variant of FixedPartitionHadamardProcessor that applies
    Gaussian smoothing AFTER creating the piecewise constant functions.

    Steps:
        1. Create K-means partition (cached)
        2. Assign random constant values to each region → piecewise constant
        3. Apply Gaussian smoothing → smooth functions that "remember" partition structure

    The result is smooth functions that are still influenced by the partition
    structure, but without hard discontinuities.

    This is analogous to what SimpleSchrodingerFieldProcessor does:
    - SimpleSchrodinger: modulate by potential, then smooth
    - SmoothedPartition: assign by region, then smooth
    """

    def __init__(
            self,
            k: Union[int, omegaconf.ListConfig] = 8,
            iterations: Union[int, omegaconf.ListConfig] = 20,
            sigma: Union[float, omegaconf.ListConfig] = 0.1,
            normalize_weights: bool = True,
            cosine: bool = False,
            seed: Optional[int] = None,
            aggr: str = "add",
            **kwargs
    ):
        """
        Args:
            k: Number of K-means clusters (regions).
               If ListConfig [min, max], sampled once when partition is created.
            iterations: Number of Gaussian smoothing iterations.
            sigma: Gaussian kernel width for smoothing.
            normalize_weights: Whether to normalize edge weights to sum to 1.
            cosine: Whether to use cosine distance.
            seed: Random seed for reproducible partition. If None, random.
            aggr: Aggregation method for message passing.
        """
        super().__init__(aggr=aggr)
        self._k_config = k
        self._iterations = iterations
        self._sigma = sigma
        self.normalize_weights = normalize_weights
        self.cosine = cosine
        self._seed = seed

        # Cached partition (computed on first forward)
        self._region_ids: Optional[torch.Tensor] = None
        self._num_regions: Optional[int] = None

    @property
    def iterations(self) -> int:
        if isinstance(self._iterations, omegaconf.ListConfig):
            return np.random.randint(self._iterations[0], self._iterations[1] + 1)
        return self._iterations

    @property
    def sigma(self) -> float:
        if isinstance(self._sigma, omegaconf.ListConfig):
            return np.random.uniform(*self._sigma)
        return self._sigma

    def _sample_k(self) -> int:
        """Sample k from config."""
        if isinstance(self._k_config, omegaconf.ListConfig):
            return np.random.randint(self._k_config[0], self._k_config[1] + 1)
        return self._k_config

    def _compute_partition(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Compute K-means partition.

        Args:
            pos: Point positions [num_nodes, dim]

        Returns:
            region_ids: [num_nodes] integer region assignment (0 to K-1)
        """
        from sklearn.cluster import KMeans

        pos_np = pos.detach().cpu().numpy()

        # Sample k (gets fixed for this partition)
        k = self._sample_k()
        self._num_regions = k

        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=self._seed, n_init=10)
        region_ids = kmeans.fit_predict(pos_np)

        return torch.from_numpy(region_ids.astype(np.int64))

    def reset_partition(self):
        """Force recomputation of partition on next forward call."""
        self._region_ids = None
        self._num_regions = None

    @property
    def num_regions(self) -> Optional[int]:
        """Number of regions in the current partition (None if not computed yet)."""
        return self._num_regions

    @property
    def region_ids(self) -> Optional[torch.Tensor]:
        """Current partition region IDs (None if not computed yet)."""
        return self._region_ids

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Message function: weighted neighbor values."""
        return norm.view(-1, 1) * x_j

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate smoothed Hadamard probe functions.

        Steps:
            1. Create K-means partition (cached)
            2. Assign random values per region → piecewise constant
            3. Apply Gaussian smoothing → smooth

        Args:
            x: Input tensor [num_nodes, num_fields] (shape used, values ignored)
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional precomputed edge weights
            pos: Point positions [num_nodes, dim] - REQUIRED

        Returns:
            Smooth probe functions [num_nodes, num_fields]
        """
        if pos is None:
            raise ValueError("SmoothedPartitionHadamardProcessor requires point positions (pos)")

        num_nodes, num_fields = x.shape
        device = x.device
        dtype = x.dtype

        # ===== STEP 1: Compute partition (cached) =====
        if self._region_ids is None:
            self._region_ids = self._compute_partition(pos)
            print(f"SmoothedPartitionHadamardProcessor: Created K-means partition with "
                  f"{self._num_regions} regions")

        region_ids = self._region_ids.to(device)

        # ===== STEP 2: Piecewise constant functions =====
        # Generate random values for each region for each field
        region_values = torch.randn(self._num_regions, num_fields, device=device, dtype=dtype)

        # Assign values to points based on their region
        f = region_values[region_ids]  # [num_nodes, num_fields]

        # ===== STEP 3: Gaussian smoothing =====
        iterations = self.iterations
        sigma = self.sigma

        for _ in range(iterations):
            # Compute edge weights from distances
            distances = utils.compute_distances(pos=pos, edge_index=edge_index, use_cosine=self.cosine)
            curr_edge_weight = torch.exp(-(distances ** 2) / (sigma ** 2))

            # Normalize weights
            if self.normalize_weights:
                row, col = edge_index
                weight_sum = scatter(curr_edge_weight, row, dim=0, reduce="sum")
                norm = curr_edge_weight / (weight_sum[row] + 1e-8)
            else:
                norm = curr_edge_weight

            # Message passing (weighted average of neighbors)
            f = self.propagate(edge_index, x=f, norm=norm)

        return f


class PolynomialProbeProcessor(torch.nn.Module):
    """
    Generate polynomial probe functions.

    Probes are random polynomials of degree ≤ max_degree evaluated at point positions.

    These are eigenfunctions of the polynomial kernel operator:
        k(x, y) = (1 + x·y)^d

    Unlike Laplacian probes (local smoothness), polynomial probes capture
    global geometric structure through polynomial basis functions.

    Basis functions by degree:
        - Degree 0: {1} — 1 function (constant)
        - Degree 1: {1, x, y, z} — 4 functions
        - Degree 2: {1, x, y, z, x², y², z², xy, xz, yz} — 10 functions
        - Degree 3: 20 functions
        - ...

    The learned basis should converge to orthogonal polynomials on the shape.
    """

    def __init__(
            self,
            max_degree: Union[int, omegaconf.ListConfig] = 2,
            normalize_basis: bool = True,
            center_positions: bool = True,
            scale_positions: bool = True,
            **kwargs
    ):
        """
        Args:
            max_degree: Maximum polynomial degree.
                       If ListConfig [min, max], randomly sampled per forward.
            normalize_basis: Whether to normalize each basis function to unit norm.
            center_positions: Whether to center positions (subtract mean).
            scale_positions: Whether to scale positions (divide by std).
        """
        super().__init__()
        self._max_degree = max_degree
        self.normalize_basis = normalize_basis
        self.center_positions = center_positions
        self.scale_positions = scale_positions

    @property
    def max_degree(self) -> int:
        if isinstance(self._max_degree, omegaconf.ListConfig):
            return np.random.randint(self._max_degree[0], self._max_degree[1] + 1)
        return self._max_degree

    def _build_polynomial_basis(self, pos: torch.Tensor, degree: int) -> torch.Tensor:
        """
        Build polynomial basis up to given degree.

        Args:
            pos: Point positions [num_nodes, dim]
            degree: Maximum polynomial degree

        Returns:
            basis: [num_nodes, num_basis_functions]
        """
        num_nodes, dim = pos.shape
        device = pos.device
        dtype = pos.dtype

        # Optionally preprocess positions
        p = pos.clone()
        if self.center_positions:
            p = p - p.mean(dim=0, keepdim=True)
        if self.scale_positions:
            std = p.std()
            if std > 1e-8:
                p = p / std

        # For 3D: x, y, z
        if dim >= 1:
            x = p[:, 0]
        if dim >= 2:
            y = p[:, 1]
        if dim >= 3:
            z = p[:, 2]

        basis_functions = []

        # Degree 0: constant
        basis_functions.append(torch.ones(num_nodes, device=device, dtype=dtype))

        if degree >= 1 and dim >= 1:
            # Degree 1: x, y, z
            basis_functions.append(x)
            if dim >= 2:
                basis_functions.append(y)
            if dim >= 3:
                basis_functions.append(z)

        if degree >= 2 and dim >= 1:
            # Degree 2: x², y², z², xy, xz, yz
            basis_functions.append(x * x)
            if dim >= 2:
                basis_functions.append(y * y)
                basis_functions.append(x * y)
            if dim >= 3:
                basis_functions.append(z * z)
                basis_functions.append(x * z)
                basis_functions.append(y * z)

        if degree >= 3 and dim >= 1:
            # Degree 3: x³, y³, z³, x²y, x²z, xy², y²z, xz², yz², xyz
            basis_functions.append(x * x * x)
            if dim >= 2:
                basis_functions.append(y * y * y)
                basis_functions.append(x * x * y)
                basis_functions.append(x * y * y)
            if dim >= 3:
                basis_functions.append(z * z * z)
                basis_functions.append(x * x * z)
                basis_functions.append(y * y * z)
                basis_functions.append(x * z * z)
                basis_functions.append(y * z * z)
                basis_functions.append(x * y * z)

        if degree >= 4 and dim >= 1:
            # Degree 4 (partial - main terms)
            basis_functions.append(x ** 4)
            if dim >= 2:
                basis_functions.append(y ** 4)
                basis_functions.append((x ** 2) * (y ** 2))
                basis_functions.append((x ** 3) * y)
                basis_functions.append(x * (y ** 3))
            if dim >= 3:
                basis_functions.append(z ** 4)
                basis_functions.append((x ** 2) * (z ** 2))
                basis_functions.append((y ** 2) * (z ** 2))
                basis_functions.append((x ** 3) * z)
                basis_functions.append((y ** 3) * z)
                basis_functions.append(x * (z ** 3))
                basis_functions.append(y * (z ** 3))
                basis_functions.append((x ** 2) * y * z)
                basis_functions.append(x * (y ** 2) * z)
                basis_functions.append(x * y * (z ** 2))

        # Stack into matrix
        basis = torch.stack(basis_functions, dim=1)  # [num_nodes, num_basis]

        # Optionally normalize each basis function
        if self.normalize_basis:
            norms = torch.norm(basis, dim=0, keepdim=True) + 1e-8
            basis = basis / norms

        return basis

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate polynomial probe functions.

        Args:
            x: Input tensor [num_nodes, num_fields] (shape used for num_fields)
            edge_index: Graph connectivity (not used, for interface compatibility)
            edge_weight: Edge weights (not used)
            pos: Point positions [num_nodes, dim] - REQUIRED

        Returns:
            Polynomial probe functions [num_nodes, num_fields]
        """
        if pos is None:
            raise ValueError("PolynomialProbeProcessor requires point positions (pos)")

        num_nodes, num_fields = x.shape
        device = x.device
        dtype = x.dtype

        # Build polynomial basis
        degree = self.max_degree
        basis = self._build_polynomial_basis(pos, degree)
        num_basis = basis.shape[1]

        # Random coefficients for each field
        # Each field is a random linear combination of basis functions
        coeffs = torch.randn(num_basis, num_fields, device=device, dtype=dtype)

        # Evaluate polynomials: f = basis @ coeffs
        f = basis @ coeffs  # [num_nodes, num_fields]

        return f

    def get_num_basis_functions(self, dim: int = 3) -> dict:
        """
        Get the number of basis functions for each degree.

        Returns dict mapping degree -> cumulative number of basis functions.
        """
        # For 3D
        counts = {
            0: 1,  # constant
            1: 4,  # + x, y, z
            2: 10,  # + x², y², z², xy, xz, yz
            3: 20,  # + x³, y³, z³, x²y, x²z, xy², y²z, xz², yz², xyz
            4: 35,  # + degree 4 terms
        }
        return counts
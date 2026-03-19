# standard library
from abc import ABC, abstractmethod
from PIL import Image
from typing import Optional, List
from itertools import chain

# numpy
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

# transformers
from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor


class FeatureExtractor(nn.Module, ABC):
    """Abstract base class for feature extractors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def extract_features(self, points: torch.Tensor) -> torch.Tensor:
        """
        Extract features from points and normals.

        Args:
            points: Point coordinates [N, 3]
            normals: Normal vectors [N, 3]

        Returns:
            Extracted features [N, features_dim]
        """
        pass

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.extract_features(points)


class PolynomialEncodings(FeatureExtractor):
    """
    NeRF-style positional encoding feature extractor with optional polynomial features.

    This extractor implements the positional encoding described in the NeRF paper:
    γ(p) = (sin(2^0π*p), cos(2^0π*p), ..., sin(2^(L-1)π*p), cos(2^(L-1)π*p))

    Where p is the input coordinate and L is the number of frequency levels.

    Additionally supports polynomial features of any degree:
    - polynomial_degree=1: [x, y, z] (original coordinates)
    - polynomial_degree=2: [x, y, z, x², xy, xz, y², yz, z²]
    - polynomial_degree=3: [x, y, z, x², xy, xz, y², yz, z², x³, x²y, x²z, xy², xyz, xz², y³, y²z, yz², z³]
    """

    def __init__(self,
                 polynomial_degree: int):
        """
        Initialize the NeRF positional encoding extractor.

        Args:
            output_dim: Desired output dimension for NeRF encoding. Must be divisible by 2*input_dim.
                       For 3D inputs: 60 (L=10) or 24 (L=4) are common choices.
            log_sampling: If True, use log-spaced frequencies (2^i). If False, use linear spacing.
            include_polynomial_features: Whether to include polynomial features of input coordinates.
            polynomial_degree: Maximum degree of polynomial features (1=linear, 2=quadratic, etc.).
                              Only used if include_polynomial_features=True.
        """
        super().__init__()
        self.polynomial_degree = polynomial_degree
        self._polynomial_combinations = None

    def _generate_polynomial_combinations(self, degree: int, num_dims: int = 3):
        """
        Generate all polynomial combinations from degree 1 up to given degree.

        Args:
            degree: Maximum polynomial degree
            num_dims: Number of input dimensions

        Returns:
            List of tuples representing exponent combinations
        """
        import itertools
        combinations = []

        for d in range(1, degree + 1):  # Start from degree 1, no constant term
            for combo in itertools.combinations_with_replacement(range(num_dims), d):
                # Convert to exponent tuple (how many times each dimension appears)
                exponents = [0] * num_dims
                for dim in combo:
                    exponents[dim] += 1
                combinations.append(tuple(exponents))

        return combinations

    def _compute_polynomial_features(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute polynomial features for given points.

        Args:
            points: Input points of shape [N, D]

        Returns:
            Polynomial features of shape [N, num_polynomial_features]
        """
        N, D = points.shape

        # Generate polynomial combinations if not cached
        if self._polynomial_combinations is None:
            self._polynomial_combinations = self._generate_polynomial_combinations(
                self.polynomial_degree, D
            )

        if not self._polynomial_combinations:
            return torch.empty(N, 0, device=points.device)

        poly_features = []

        for exponents in self._polynomial_combinations:
            # Compute polynomial term: x^a * y^b * z^c * ...
            term = torch.ones(N, device=points.device, dtype=points.dtype)
            for dim, exp in enumerate(exponents):
                if exp > 0:
                    term = term * (points[:, dim] ** exp)

            poly_features.append(term)

        return torch.stack(poly_features, dim=1)

    def _get_polynomial_feature_count(self, input_dim: int, degree: int) -> int:
        """
        Calculate number of polynomial features for given input dimension and degree.

        Args:
            input_dim: Number of input dimensions
            degree: Maximum polynomial degree

        Returns:
            Number of polynomial features
        """
        from math import comb

        total_features = 0
        for d in range(1, degree + 1):  # No constant term
            # Number of ways to distribute d identical items into input_dim bins
            total_features += comb(d + input_dim - 1, input_dim - 1)

        return total_features

    def extract_features(self, points: torch.Tensor) -> torch.Tensor:
        """
        Extract NeRF-style positional encoding features from points, optionally with polynomial features.

        Args:
            points: Point coordinates [N, 3] or [N, D] for D-dimensional inputs
            normals: Normal vectors (not used in NeRF encoding)

        Returns:
            Combined features [N, total_feature_dim]
        """
        return self._compute_polynomial_features(points)


class NeRFPositionalEncodings(FeatureExtractor):
    """
    NeRF-style positional encoding feature extractor with optional polynomial features.

    This extractor implements the positional encoding described in the NeRF paper:
    γ(p) = (sin(2^0π*p), cos(2^0π*p), ..., sin(2^(L-1)π*p), cos(2^(L-1)π*p))

    Where p is the input coordinate and L is the number of frequency levels.

    Additionally supports polynomial features of any degree:
    - polynomial_degree=1: [x, y, z] (original coordinates)
    - polynomial_degree=2: [x, y, z, x², xy, xz, y², yz, z²]
    - polynomial_degree=3: [x, y, z, x², xy, xz, y², yz, z², x³, x²y, x²z, xy², xyz, xz², y³, y²z, yz², z³]
    """

    def __init__(self,
                 output_dim: int = 60,
                 log_sampling: bool = True,
                 **kwargs):
        """
        Initialize the NeRF positional encoding extractor.

        Args:
            output_dim: Desired output dimension for NeRF encoding. Must be divisible by 2*input_dim.
                       For 3D inputs: 60 (L=10) or 24 (L=4) are common choices.
            log_sampling: If True, use log-spaced frequencies (2^i). If False, use linear spacing.
            include_polynomial_features: Whether to include polynomial features of input coordinates.
            polynomial_degree: Maximum degree of polynomial features (1=linear, 2=quadratic, etc.).
                              Only used if include_polynomial_features=True.
        """
        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.log_sampling = log_sampling

        # We'll compute num_freq_bands when we know the input dimension
        self.freq_bands = None
        self._num_freq_bands = None

    def _initialize_freq_bands(self, input_dim: int, device: torch.device):
        """
        Initialize frequency bands based on input dimension and desired output dimension.

        Args:
            input_dim: Dimension of input coordinates
            device: Device to place the frequency bands tensor
        """
        # Calculate number of frequency bands needed
        # output_dim = input_dim * num_freq_bands * 2 (for sin and cos)
        # Therefore: num_freq_bands = output_dim / (input_dim * 2)

        if self.output_dim % (input_dim * 2) != 0:
            raise ValueError(
                f"Output dimension {self.output_dim} must be divisible by 2*input_dim={2 * input_dim}. "
                f"Common values for 3D inputs are 60 (L=10) or 24 (L=4)."
            )

        self._num_freq_bands = self.output_dim // (input_dim * 2)

        # Pre-compute frequency bands
        if self.log_sampling:
            # Use powers of 2 as in the NeRF paper: 2^0, 2^1, ..., 2^(L-1)
            self.freq_bands = 2.0 ** torch.arange(self._num_freq_bands, device=device)
        else:
            # Linear spacing alternative
            self.freq_bands = torch.linspace(1.0, 2.0 ** (self._num_freq_bands - 1),
                                             self._num_freq_bands, device=device)

    def extract_features(self, points: torch.Tensor) -> torch.Tensor:
        """
        Extract NeRF-style positional encoding features from points, optionally with polynomial features.

        Args:
            points: Point coordinates [N, 3] or [N, D] for D-dimensional inputs
            normals: Normal vectors (not used in NeRF encoding)

        Returns:
            Combined features [N, total_feature_dim]
        """
        # Initialize frequency bands on first use
        if self.freq_bands is None:
            self._initialize_freq_bands(points.shape[1], points.device)

        # Ensure freq_bands is on the same device as input
        if self.freq_bands.device != points.device:
            self.freq_bands = self.freq_bands.to(points.device)

        features_to_concat = []

        # Add NeRF positional encoding
        encoded = self._encode(points)
        features_to_concat.append(encoded)

        return torch.cat(features_to_concat, dim=1)

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input coordinates.

        Args:
            inputs: Input tensor of shape [N, D]

        Returns:
            Encoded tensor of shape [N, D * num_freq_bands * 2]
        """
        N, D = inputs.shape

        # Collect encoded features
        encoded_features = []

        # Apply sin and cos at each frequency band
        for freq in self.freq_bands:
            # Apply encoding: sin(2^i * π * input) and cos(2^i * π * input)
            encoded_features.append(torch.sin(2.0 * np.pi * inputs * freq))
            encoded_features.append(torch.cos(2.0 * np.pi * inputs * freq))

        # Concatenate all features
        return torch.cat(encoded_features, dim=-1)

    def get_output_dim(self, input_dim: int = None) -> int:
        """
        Get the total output dimension including polynomial features and NeRF encoding.

        Args:
            input_dim: Dimension of input coordinates (optional, for validation)

        Returns:
            Total output dimension after encoding
        """
        if input_dim is not None:
            # Validate that output_dim is compatible with input_dim for NeRF encoding
            if self.output_dim % (input_dim * 2) != 0:
                raise ValueError(
                    f"Output dimension {self.output_dim} must be divisible by 2*input_dim={2 * input_dim}"
                )

        total_dim = self.output_dim  # NeRF encoding dimension

        return total_dim


class FourierFeatureExtractor(FeatureExtractor):
    """
    Feature extractor that applies Fourier feature transformation to point coordinates.

    This extractor takes point coordinates and applies random Fourier feature mapping
    to create high-dimensional feature representations.
    """

    def __init__(self,
                 fourier_scale: float = 10.0,
                 num_fourier_features: int = 256,
                 distribution: str = 'gaussian',
                 seed: Optional[int] = None,
                 **kwargs):
        """
        Initialize the Fourier feature extractor.

        Args:
            fourier_scale: Standard deviation of the distribution used to sample frequencies
            num_fourier_features: Total number of output Fourier features (must be even)
            distribution: Distribution to sample frequencies from ('gaussian', 'uniform', 'laplacian')
            seed: Random seed for reproducible frequency matrix initialization (None for random)
        """
        super().__init__(**kwargs)

        if num_fourier_features % 2 != 0:
            raise ValueError("num_fourier_features must be even (since we use both cos and sin)")

        self._fourier_scale = fourier_scale
        self._num_fourier_features = num_fourier_features
        self._num_frequency_components = num_fourier_features // 2  # Half for cos, half for sin
        self._distribution = distribution
        self._seed = seed
        self._B = None  # Frequency matrix, initialized when needed

    def extract_features(self, points: torch.Tensor) -> torch.Tensor:
        """
        Extract Fourier features from point coordinates.

        Args:
            points: Point coordinates [N, 3]
            normals: Normal vectors [N, 3] (not used)

        Returns:
            Fourier features [N, num_fourier_features]
        """
        # Apply Fourier transformation directly to point coordinates
        fourier_features = self._apply_fourier_transform(points.float())
        return fourier_features

    def _initialize_fourier_matrix(self, input_dim: int, device: torch.device):
        """Initialize the matrix of frequencies B with the specified seed for reproducibility."""
        # Set random seed for reproducibility if provided
        if self._seed is not None:
            # Save current random states
            torch_state = torch.get_rng_state()
            np_state = np.random.get_state()

            # Set seeds
            torch.manual_seed(self._seed)
            np.random.seed(self._seed)

        # Generate frequency matrix based on distribution
        if self._distribution == 'gaussian':
            self._B = torch.randn((input_dim, self._num_frequency_components), device=device) * self._fourier_scale
        elif self._distribution == 'uniform':
            self._B = (torch.rand((input_dim, self._num_frequency_components), device=device) * 2 - 1) * self._fourier_scale
        elif self._distribution == 'laplacian':
            # Laplacian distribution via exponential + sign flip
            exponential_samples = torch.tensor(
                np.random.exponential(scale=self._fourier_scale, size=(input_dim, self._num_frequency_components)),
                device=device, dtype=torch.float32
            )
            signs = torch.sign(torch.rand((input_dim, self._num_frequency_components), device=device) - 0.5)
            self._B = signs * exponential_samples
        else:
            raise ValueError(f"Unsupported distribution: {self._distribution}")

        # Restore random states if seed was set
        if self._seed is not None:
            torch.set_rng_state(torch_state)
            np.random.set_state(np_state)

    def _apply_fourier_transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature transform to features.

        Args:
            features: Input features tensor of shape [N, D]

        Returns:
            Transformed features with exactly num_fourier_features dimensions [N, num_fourier_features]
        """
        # If this is our first call, initialize B on the same device as input
        if self._B is None:
            self._initialize_fourier_matrix(features.shape[1], features.device)

        # Move B to the same device as features if needed
        if self._B.device != features.device:
            self._B = self._B.to(features.device)

        # Apply Fourier feature mapping: [cos(2πBx), sin(2πBx)]
        x_proj = 2 * np.pi * torch.matmul(features, self._B)
        fourier_features = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)

        # Output shape: [N, num_fourier_features] where num_fourier_features = 2 * num_frequency_components
        return fourier_features


class DINOFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using DINO/DINOv2 from HuggingFace.
    Based on official HuggingFace documentation.
    """

    def __init__(self,
                 model_name: str,
                 normalize_features: bool,
                 use_cls_token: bool,
                 **kwargs):
        """
        Initialize DINO feature extractor.

        Args:
            model_name: HuggingFace model identifier
                - 'facebook/dinov2-small'  (384 dim)
                - 'facebook/dinov2-base'   (768 dim)
                - 'facebook/dinov2-large'  (1024 dim)
                - 'facebook/dinov2-giant'  (1536 dim)
            normalize_features: Whether to L2 normalize
            use_cls_token: If True use CLS token, else mean pool spatial tokens
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.normalize_features = normalize_features
        self.use_cls_token = use_cls_token

        # Load model and processor using AutoModel and AutoImageProcessor
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.eval()

        # Get feature dimension from config
        self.feature_dim = self.model.config.hidden_size

    def extract_features(self, points: List[List[Image]]) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            points: Images as tensor [B, C, H, W] or PIL images
            normals: Not used (interface compatibility)

        Returns:
            Features tensor [B, feature_dim]
        """
        images = list(chain(*points))

        # Process images and run model
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get last hidden states which contain CLS + patch tokens
            last_hidden_states = outputs.last_hidden_state

            # Extract features based on pooling strategy
            if self.use_cls_token:
                # CLS token is at position 0
                features = last_hidden_states[:, 0, :]
            else:
                # Mean pool spatial tokens (exclude CLS at position 0)
                features = last_hidden_states[:, 1:, :].mean(dim=1)

        # Normalize if requested
        if self.normalize_features:
            features = F.normalize(features, p=2, dim=-1)

        return features

    def get_feature_dim(self) -> int:
        return self.feature_dim


class CLIPFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using CLIP from HuggingFace.
    Based on official HuggingFace documentation.
    """

    def __init__(self,
                 model_name: str = 'openai/clip-vit-base-patch32',
                 normalize_features: bool = False,
                **kwargs):  # CLIP already normalizes
        """
        Initialize CLIP feature extractor.

        Args:
            model_name: HuggingFace model identifier
                - 'openai/clip-vit-base-patch32'  (512 dim)
                - 'openai/clip-vit-base-patch16'  (512 dim)
                - 'openai/clip-vit-large-patch14' (768 dim)
            device: Device for computation
            normalize_features: Whether to normalize (CLIP does this by default)
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.normalize_features = normalize_features

        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # Get vision feature dimension
        if hasattr(self.model.config, 'projection_dim'):
            self.feature_dim = self.model.config.projection_dim
        else:
            # Fallback to vision config hidden size
            self.feature_dim = self.model.config.vision_config.hidden_size

    def extract_features(self, points: List[List[Image]]) -> torch.Tensor:
        """
        Extract CLIP visual features.

        Args:
            points: Images as tensor or PIL images
            normals: Not used

        Returns:
            Features tensor [B, feature_dim]
        """
        images = list(chain(*points))


        # # DEBUG: Check what we're actually getting
        # print(f"Number of images: {len(images)}")
        # print(f"Type of first image: {type(images[0])}")
        # if hasattr(images[0], 'size'):
        #     print(f"First image size: {images[0].size}")
        #
        # # Try processing each image to find the problematic one
        # for i, img in enumerate(images):
        #     try:
        #         test_input = self.processor(images=img, return_tensors="pt")
        #         print(f"Image {i}: OK - {type(img)}")
        #     except Exception as e:
        #         print(f"Image {i}: FAILED - {type(img)} - {e}")


        # Process images - CLIP processor handles normalization
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        # Move only pixel_values to device for vision encoding
        pixel_values = inputs['pixel_values'].to(self.model.device)

        # Extract image features using CLIP's vision encoder
        with torch.no_grad():
            # Use get_image_features which returns normalized embeddings
            features = self.model.get_image_features(pixel_values=pixel_values)

        # Additional normalization only if explicitly requested
        # (CLIP already normalizes by default)
        if self.normalize_features:
            features = F.normalize(features, p=2, dim=-1)

        return features

    def get_feature_dim(self) -> int:
        return self.feature_dim


class ResNet50FeatureExtractor(FeatureExtractor):
    """
    Feature extractor using ResNet-50 from HuggingFace.
    Based on official HuggingFace transformers documentation.
    """

    def __init__(self,
                 model_name: str = 'microsoft/resnet-50',
                 normalize_features: bool = False,
                 pooling_type: str = 'avg',
                 **kwargs):
        """
        Initialize ResNet-50 feature extractor.

        Args:
            model_name: HuggingFace model identifier
                - 'microsoft/resnet-50' (2048 dim from last layer)
            normalize_features: Whether to L2 normalize the features
            pooling_type: Type of pooling to apply. Options:
                - 'avg': Global average pooling (default ResNet behavior)
                - 'max': Global max pooling
                - 'both': Concatenate both avg and max pooling (4096 dim output)
                - 'flatten': Flatten spatial dimensions (C*H*W dim output)
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.normalize_features = normalize_features
        self.pooling_type = pooling_type.lower()

        if self.pooling_type not in ['avg', 'max', 'both', 'flatten']:
            raise ValueError(f"pooling_type must be one of ['avg', 'max', 'both', 'flatten'], got '{pooling_type}'")

        # Load ResNet-50 model and image processor using Auto classes
        from transformers import AutoModel, AutoImageProcessor

        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.eval()

        # Get base feature dimension from config
        # For ResNet-50, the last hidden state has shape [B, 2048, H, W]
        base_dim = self.model.config.hidden_sizes[-1]  # 2048 for ResNet-50

        # Compute actual feature dimension based on pooling type
        if self.pooling_type == 'both':
            self.feature_dim = base_dim * 2  # 4096 for avg+max concatenation
        elif self.pooling_type == 'flatten':
            # Will be computed dynamically based on input size
            self.feature_dim = None
        else:
            self.feature_dim = base_dim  # 2048 for avg or max

    def extract_features(self, points: List[List[Image]]) -> torch.Tensor:
        """
        Extract ResNet-50 features from images.

        Args:
            points: Images as List[List[PIL.Image]] (nested list structure for batching)
            normals: Not used (interface compatibility)

        Returns:
            Features tensor [B, feature_dim] where feature_dim=2048
        """
        # Flatten the nested list structure
        images = list(chain(*points))

        # Process images with the image processor
        # The processor handles resizing, normalization, etc.
        inputs = self.processor(images=images, return_tensors="pt")

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Get the last hidden state
            # Shape: [B, C, H, W] where C=2048, H and W depend on input size
            last_hidden_state = outputs.last_hidden_state

            if self.pooling_type == 'avg':
                # Apply global average pooling across spatial dimensions
                # [B, C, H, W] -> [B, C]
                features = last_hidden_state.mean(dim=[2, 3])
            elif self.pooling_type == 'max':
                # Apply global max pooling across spatial dimensions
                # [B, C, H, W] -> [B, C]
                B, C, H, W = last_hidden_state.shape
                features = last_hidden_state.view(B, C, -1).max(dim=2)[0]
            elif self.pooling_type == 'both':
                # Concatenate both avg and max pooling
                # [B, C, H, W] -> [B, 2*C]
                B, C, H, W = last_hidden_state.shape
                avg_pool = last_hidden_state.mean(dim=[2, 3])
                max_pool = last_hidden_state.view(B, C, -1).max(dim=2)[0]
                features = torch.cat([avg_pool, max_pool], dim=1)
            else:  # flatten
                # Flatten spatial dimensions
                # [B, C, H, W] -> [B, C*H*W]
                B, C, H, W = last_hidden_state.shape
                features = last_hidden_state.view(B, -1)
                # Update feature_dim if it wasn't set
                if self.feature_dim is None:
                    self.feature_dim = features.shape[1]

        # Normalize if requested
        if self.normalize_features:
            features = F.normalize(features, p=2, dim=-1)

        return features

    def get_feature_dim(self) -> int:
        """
        Get the output feature dimension.

        Returns:
            - 2048 for ResNet-50 with 'avg' or 'max' pooling
            - 4096 for ResNet-50 with 'both' pooling
            - Variable for 'flatten' (depends on input image size)
        """
        if self.feature_dim is None:
            raise RuntimeError(
                "Feature dimension not yet determined. "
                "Run extract_features() at least once to compute the dimension for 'flatten' mode."
            )
        return self.feature_dim
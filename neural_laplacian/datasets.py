# standard library
from pathlib import Path
from typing import List, Union, Tuple, Optional
from abc import ABC, abstractmethod
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import pickle

# omegaconf
import omegaconf.listconfig

# numpy
import numpy as np

# torch
import torch

# torch geometric
from torch_geometric.data import Dataset, Data

# pymeshlab
import pymeshlab

# neural-laplacian
from neural_laplacian import utils
from neural_laplacian.configs import (
    ScalarFieldConfig,
    DecimationConfig,
    FarthestPointSamplingConfig,
    UniformSamplingConfig,
    RemeshingConfig,
    GaussianNoiseConfig,
    ClassWeightedSamplingConfig,
    OperatorConfig
)

# open-cv
import cv2


class PointCloudDatasetBase(ABC, Dataset):
    def __init__(
            self,
            root_dirs: List[Path],
            file_size: Optional[Tuple[Optional[float], Optional[float]]],
            scalar_field_config: ScalarFieldConfig,
            max_items: Optional[Union[int, float]],
            file_extensions: List[str],
            name: Optional[str],
            files_path_cache: Optional[str]
    ):
        """
        Initialize the point cloud dataset base.

        Args:
            root_dirs: List of directories containing files
            min_file_size: Minimum file size in MB to consider
            max_file_size: Maximum file size in MB to consider
            max_items: Maximum number of items to include
            scalar_field_config: Configuration for scalar field generation
            file_extensions: List of file extensions to scan for  # ADD THIS
        """
        super().__init__()
        self._root_dirs = root_dirs
        self._file_size = file_size
        self._max_items = max_items
        self._scalar_field_config = scalar_field_config
        if name is not None:
            self._name = name
        else:
            self._name = "_".join(p.name for p in root_dirs)

        # ADD THIS: Scan for files
        if files_path_cache is None:
            self._file_paths = utils.scan_files(
                root_dirs=root_dirs,
                file_size=file_size,
                max_items=max_items,
                file_extensions=file_extensions,
            )
        else:
            self._file_paths = utils.load_cached_file_list(root_dirs=root_dirs, cache_basename=files_path_cache)

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def len(self) -> int:
        pass

    @abstractmethod
    def get(self, idx: int) -> Data:
        pass


class PointCloudDataset(PointCloudDatasetBase):
    """PyTorch Geometric dataset for point cloud generation and sampling."""

    def __init__(
            self,
            num_eigenfunctions: int,
            decimation_config: Optional[DecimationConfig],
            farthest_point_sampling_config: Optional[FarthestPointSamplingConfig],
            remeshing_config: Optional[RemeshingConfig],
            gaussian_noise_config: Optional[GaussianNoiseConfig],
            cache_dir: str,
            include_faces: bool = False,
            cache_eigendecomposition: bool = True,
            operator_config: Optional[OperatorConfig] = None,
            **kwargs
    ):
        """
        Initialize the point cloud dataset.

        Args:
            root_dirs: List of directories containing mesh files
            min_file_size: Minimum file size in MB to consider
            max_file_size: Maximum file size in MB to consider
            max_items: Maximum number of items to include
            scalar_field_config: Configuration for scalar field generation and smoothing
            decimation_config: Configuration for mesh decimation (optional)
            farthest_point_sampling_config: Configuration for FPS sampling (optional)
            num_eigenfunctions: Number of eigenfunctions to compute
            remeshing_config: Configuration for the remeshing process (not used anymore)
            include_faces: Whether to include faces in the output
            gaussian_noise_config: Configuration for Gaussian noise (optional)
            cache_eigendecomposition: Whether to cache eigendecompositions to disk
            cache_dir: Directory to store eigendecomposition cache
            operator_config: Configuration for differential operator (Laplacian or Schrödinger).
                           If None, uses standard Laplacian (backward compatible).
        """
        super().__init__(
            file_extensions=['*.obj', '*.ply', '*.off', '*.stl'],
            **kwargs)

        self._num_eigenfunctions = num_eigenfunctions
        self._decimation_config = decimation_config
        self._farthest_point_sampling_config = farthest_point_sampling_config
        self._gaussian_noise_config = gaussian_noise_config

        # Point cloud specific configs (some may not be used)
        self._remeshing_config = remeshing_config  # Kept for compatibility but not used
        self._include_faces = False if farthest_point_sampling_config is not None else include_faces

        # Operator configuration (None = use standard Laplacian for backward compatibility)
        self._operator_config = operator_config

        if operator_config is not None:
            print(f"Dataset configured with operator: {operator_config}")

        # Eigendecomposition caching - now using shared cache manager
        self._eigen_cache_manager = utils.create_eigen_cache_manager(
            cache_dir=cache_dir,
            enabled=cache_eigendecomposition
        )

    def len(self) -> int:
        """Return the number of meshes in the dataset."""
        return len(self._file_paths)

    def _load_mesh(self, file_path: Path) -> pymeshlab.MeshSet:
        """Load mesh from file using PyMeshLab."""
        ms: pymeshlab.MeshSet = pymeshlab.MeshSet()
        ms.load_new_mesh(str(file_path))
        return ms

    def _load_geometry(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load geometry from file."""
        ms_original = self._load_mesh(file_path)
        vertices = ms_original.current_mesh().vertex_matrix()
        faces = ms_original.current_mesh().face_matrix()
        return vertices, faces

    def _decimate_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly decimate a 3D mesh using quadric edge collapse with random quality threshold
        and random face preservation.

        Args:
            vertices: Original mesh vertices (N, 3)
            faces: Original mesh faces (M, 3)

        Returns:
            Tuple of (decimated_vertices, decimated_faces)
        """
        # Create a MeshSet for decimation
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertices, faces))

        if self._decimation_config.target_face_num_range is not None:
            if isinstance(self._decimation_config.target_face_num_range, omegaconf.ListConfig):
                min_target_face_num, max_target_face_num = self._decimation_config.target_face_num_range
                # Use randint for integer face counts instead of uniform
                target_face_num = np.random.randint(min_target_face_num, max_target_face_num + 1)
            else:
                target_face_num = self._decimation_config.target_face_num_range

            target_perc = target_face_num / faces.shape[0]
        else:
            # Draw a random target percentage from the configured range
            if isinstance(self._decimation_config.target_perc_range, omegaconf.ListConfig):
                min_perc, max_perc = self._decimation_config.target_perc_range
                target_perc = np.random.uniform(min_perc, max_perc)
            else:
                target_perc = self._decimation_config.target_perc_range

        # Select a random quality threshold using config range
        if isinstance(self._decimation_config.quality_thr_range, omegaconf.ListConfig):
            min_quality, max_quality = self._decimation_config.quality_thr_range
            quality_thr = np.random.uniform(min_quality, max_quality)
        else:
            quality_thr = self._decimation_config.quality_thr_range

        # Apply decimation only on selected faces (those not marked for preservation)
        ms.meshing_decimation_quadric_edge_collapse(
            targetperc=target_perc,
            qualitythr=quality_thr,
            preserveboundary=self._decimation_config.preserve_boundary,
            preservenormal=self._decimation_config.preserve_normal,
            preservetopology=self._decimation_config.preserve_topology,
            optimalplacement=self._decimation_config.optimal_placement,
            planarquadric=self._decimation_config.planar_quadric,
            selected=False
        )

        # Get decimated mesh data
        decimated_mesh = ms.current_mesh()
        decimated_vertices = decimated_mesh.vertex_matrix()
        decimated_faces = decimated_mesh.face_matrix()

        return decimated_vertices, decimated_faces

    def _apply_decimation_if_configured(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply mesh decimation if configured."""
        if self._decimation_config is not None:
            return self._decimate_mesh(vertices=vertices, faces=faces)
        return vertices, faces

    def _apply_fps_sampling_if_configured(self, vertices: np.ndarray, scalar_fields: torch.Tensor,
                                          gt_eigenvectors: torch.Tensor, gt_vertex_areas: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
        """Apply farthest point sampling if configured."""
        if self._farthest_point_sampling_config is not None:
            sampled_indices = self._farthest_point_sampling_config.sample_indices(points=vertices)

            # Apply sampling to all relevant data
            vertices_sampled = vertices[sampled_indices]
            scalar_fields_sampled = scalar_fields[sampled_indices]

            if gt_eigenvectors is not None:
                gt_eigenvectors_sampled = gt_eigenvectors[sampled_indices]
                gt_vertex_areas_sampled = gt_vertex_areas[sampled_indices]
            else:
                gt_eigenvectors_sampled = None
                gt_vertex_areas_sampled = None

            return vertices_sampled, scalar_fields_sampled, gt_eigenvectors_sampled, gt_vertex_areas_sampled, sampled_indices

        return vertices, scalar_fields, gt_eigenvectors, gt_vertex_areas, None

    def _apply_noise_if_configured(self, vertices: torch.Tensor, idx: int) -> torch.Tensor:
        """Apply Gaussian noise if configured."""
        if self._gaussian_noise_config is not None:
            return self._gaussian_noise_config.apply_noise(vertices, item_idx=idx)
        return vertices

    def _create_data_object(self, vertices: torch.Tensor, faces: Optional[torch.Tensor], scalar_fields: torch.Tensor) -> Data:
        """Create the final Data object."""
        data = Data(
            points=vertices,
            scalar_fields=scalar_fields,
            num_nodes=torch.tensor(vertices.shape[0])
        )

        if self._include_faces and faces is not None:
            data['faces'] = faces

        return data

    def _add_ground_truth_data(
            self,
            data: Data,
            gt_eigenvectors: torch.Tensor,
            gt_eigenvalues: torch.Tensor,
            gt_vertex_areas: torch.Tensor,
            potential: Optional[torch.Tensor] = None
    ) -> Data:
        """Add ground truth eigendecomposition data to the Data object."""
        # Convert back to numpy to avoid GPU loading
        data.gt_eigenvectors = np.array([]) if gt_eigenvectors is None else gt_eigenvectors.cpu().numpy()
        data.gt_eigenvalues = np.array([]) if gt_eigenvalues is None else gt_eigenvalues.cpu().numpy()
        data.gt_vertex_areas = np.array([]) if gt_vertex_areas is None else gt_vertex_areas.cpu().numpy()

        # Add potential and operator metadata (for Schrödinger)
        if potential is not None and len(potential) > 0:
            data.potential = potential.cpu().numpy() if isinstance(potential, torch.Tensor) else potential
        else:
            data.potential = None

        # Store operator config info
        if self._operator_config is not None:
            data.operator_type = self._operator_config.type
            data.potential_type = self._operator_config.potential_type
            data.potential_strength = self._operator_config.potential_strength
        else:
            data.operator_type = "laplacian"
            data.potential_type = None
            data.potential_strength = None

        return data

    def get(self, idx: int) -> Data:
        """Get a processed geometry (point cloud)."""
        file_path: Path = self._file_paths[idx]

        # Use utils function with optional decimation
        vertices, faces = utils.prepare_geometry(
            file_path=file_path,
            decimation_config=self._decimation_config
        )

        # Step 3: Load eigendecomposition from cache
        potential = None

        if self._operator_config is None:
            # Backward compatible: use old Laplacian-only method
            gt_eigenvectors, gt_eigenvalues, gt_vertex_areas = self._eigen_cache_manager.load_eigendecomposition(
                file_path=file_path,
                vertices=vertices,
                faces=faces,
                num_eigenfunctions=self._num_eigenfunctions,
                decimation_config=self._decimation_config
            )
        else:
            # New flow: use operator-aware method (supports Laplacian and Schrödinger)
            gt_eigenvectors, gt_eigenvalues, gt_vertex_areas, potential = self._eigen_cache_manager.load_operator_eigendecomposition(
                file_path=file_path,
                vertices=vertices,
                faces=faces,
                num_eigenfunctions=self._num_eigenfunctions,
                operator_type=self._operator_config.type,
                potential_type=self._operator_config.potential_type,
                potential_strength=self._operator_config.potential_strength,
                n_neighbors=self._operator_config.n_neighbors,
                decimation_config=self._decimation_config
            )

        # Step 4: Generate scalar fields
        scalar_fields = self._scalar_field_config.generate_scalar_fields(num_points=vertices.shape[0])

        # Step 5: Apply FPS sampling if configured
        vertices, scalar_fields, gt_eigenvectors, gt_vertex_areas, sampled_indices = self._apply_fps_sampling_if_configured(
            vertices=vertices, scalar_fields=scalar_fields, gt_eigenvectors=gt_eigenvectors, gt_vertex_areas=gt_vertex_areas
        )

        # Also sample potential if FPS was applied
        if sampled_indices is not None and potential is not None:
            potential = potential[sampled_indices]

        # Step 6: Convert to tensors and apply noise
        vertices_tensor = torch.tensor(vertices, dtype=torch.float)
        vertices_tensor = self._apply_noise_if_configured(vertices=vertices_tensor, idx=idx)

        faces_tensor = torch.tensor(faces, dtype=torch.long) if faces is not None else None
        scalar_fields_tensor = scalar_fields.float()

        # Step 8: Create Data object
        data = self._create_data_object(
            vertices=vertices_tensor,
            faces=faces_tensor,
            scalar_fields=scalar_fields_tensor,
        )

        # Step 9: Add ground truth data (including potential for Schrödinger)
        data = self._add_ground_truth_data(
            data=data,
            gt_eigenvectors=gt_eigenvectors,
            gt_eigenvalues=gt_eigenvalues,
            gt_vertex_areas=gt_vertex_areas,
            potential=potential
        )

        return data


# datasets.py - ImageManifoldDataset section

class ImageManifoldDataset(PointCloudDatasetBase):
    """
    Dataset for treating images as points on a manifold.

    Each image represents a single point on the manifold (not pixels).
    Supports both UniformSamplingConfig (no labels) and ClassWeightedSamplingConfig (with labels).
    Images are stored as PIL Images in the 'points' field of the Data object.
    """

    def __init__(
            self,
            sampling_config: Union[UniformSamplingConfig, ClassWeightedSamplingConfig],
            num_workers: int,
            batches: int,
            deterministic_sampling: bool,
            labels_pkl_path: Optional[Path],
            **kwargs
    ):
        """
        Initialize the image manifold dataset.

        Args:
            root_dirs: List of directories containing image files
            file_size: Tuple of (min_file_size, max_file_size) in MB
            max_items: Maximum number of images to include (before sampling)
            scalar_field_config: Configuration for scalar field generation
            sampling_config: Either UniformSamplingConfig or ClassWeightedSamplingConfig
            num_workers: Number of parallel workers for image loading
            batches: Number of batches in the dataset
            deterministic_sampling: If True, use deterministic sampling based on idx
            labels_pkl_path: Optional path to pickle file with image labels. If provided, class_ids will be added to Data objects
        """
        # Call parent init which scans for files
        super().__init__(
            file_extensions=['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp', '*.tiff', '*.webp'],
            **kwargs
        )
        self._sampling_config = sampling_config
        self._num_workers = num_workers
        self._batches = batches
        self._deterministic_sampling = deterministic_sampling

        # Load labels dictionary if path is provided
        if labels_pkl_path is not None:
            with open(labels_pkl_path, 'rb') as f:
                self._labels_dict = pickle.load(f)

            # Build a lookup map from filename to full path for efficient access
            self._filename_to_path = {path.name: path for path in self._file_paths}
        else:
            self._labels_dict = None
            self._filename_to_path = None

    def _load_image_opencv(self, img_path: Path) -> Optional[Tuple[Image.Image, Path]]:
        """
        Load a single image using OpenCV and convert to PIL.

        Args:
            img_path: Path to the image file

        Returns:
            Tuple of (PIL Image, path) or None if loading failed
        """
        try:
            # Load with OpenCV (faster than PIL)
            img_array = cv2.imread(str(img_path))

            if img_array is None:
                raise IOError(f"OpenCV couldn't load {img_path}")

            # Convert BGR to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            img = Image.fromarray(img_array)

            return (img, img_path)

        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            return None

    def _load_images_parallel(self, image_paths: List[Path]) -> Tuple[List[Image.Image], List[Path]]:
        """
        Load multiple images in parallel using ThreadPoolExecutor.

        Args:
            image_paths: List of paths to load

        Returns:
            Tuple of (list of PIL Images, list of valid paths)
        """
        images = []
        valid_paths = []

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            # Submit all loading tasks
            future_to_path = {
                executor.submit(self._load_image_opencv, path): path
                for path in image_paths
            }

            # Collect results as they complete
            for future in future_to_path:
                result = future.result()
                if result is not None:
                    img, path = result
                    images.append(img)
                    valid_paths.append(path)

        return images, valid_paths

    def len(self) -> int:
        """Return number of batches in the dataset."""
        return self._batches

    def get(self, idx: int) -> Data:
        """
        Get a dataset item containing sampled images as points.

        Args:
            idx: Dataset index (used as seed for deterministic sampling)

        Returns:
            Data object with:
                - points: List of PIL Images
                - scalar_fields: Generated scalar fields
                - num_nodes: Number of images
                - class_ids: Numpy array of class IDs (only if using ClassWeightedSamplingConfig)
        """
        if self._deterministic_sampling:
            # Reproducible - same idx always gives same sample
            rng = np.random.RandomState(seed=idx)
        else:
            # Random - different samples each time
            rng = np.random.RandomState()

        # Handle sampling differently based on config type
        if isinstance(self._sampling_config, ClassWeightedSamplingConfig):
            # CLASS-WEIGHTED SAMPLING: Sample from labels_dict
            sampled_indices = self._sampling_config.sample_indices(
                len(self._sampling_config.labels_dict),
                rng=rng
            )

            # Convert indices to filenames from labels_dict
            all_filenames = list(self._sampling_config.labels_dict.keys())
            sampled_filenames = [all_filenames[i] for i in sampled_indices]

            # Build full paths by finding each filename in our scanned file_paths
            # Skip any filenames that weren't found in the scanned directories
            image_paths_subset = []
            for fn in sampled_filenames:
                if fn in self._filename_to_path:
                    image_paths_subset.append(self._filename_to_path[fn])
        else:
            # UNIFORM SAMPLING: Sample directly from file_paths
            sampled_indices = self._sampling_config.sample_indices(
                len(self._file_paths),
                rng=rng
            )
            image_paths_subset = [self._file_paths[i] for i in sampled_indices]

        # Load images in parallel using OpenCV
        images, valid_paths = self._load_images_parallel(image_paths_subset)

        num_images = len(images)

        # Generate scalar fields
        scalar_fields = self._scalar_field_config.generate_scalar_fields(num_points=num_images)

        # Create Data object
        data = Data(
            points=images,  # List of PIL Images
            scalar_fields=scalar_fields.float(),
            num_nodes=torch.tensor(num_images)
        )

        # Add class_ids from labels if available
        if self._labels_dict is not None:
            class_ids = np.array([
                self._labels_dict[path.name]['class_id']
                for path in valid_paths
            ])
            data.class_ids = class_ids
        else:
            data.class_ids = np.array([])

        return data
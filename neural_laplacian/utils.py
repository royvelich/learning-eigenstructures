# standard library
import importlib
from typing import Type, Callable, List, Tuple, Optional, Union, Literal, Dict
from enum import Enum
import inspect
import hashlib
import pickle
import time
from pathlib import Path
import json

# torch
import torch
import torch.nn.functional as F

# numpy
import numpy as np
from scipy import sparse

# scipy
from scipy.spatial import cKDTree
import scipy.sparse.linalg as sla

# open3d
import open3d as o3d
from sklearn.cluster import mean_shift

# python-shot
# import handcrafted_descriptor as hd

# sklearn
from sklearn.neighbors import NearestNeighbors

# igl
import igl

# robust laplacian
import robust_laplacian

# pyfm
from pyFM.mesh import TriMesh

# torch geometric
from torch_geometric.data import Batch, Data


class ProjectionMethod(Enum):
    """
    Enum for selecting the projection method in reconstruction error computation.

    UNNORMALIZED: M-orthogonal eigenvectors, simple coefficient computation.
                  Coefficients = (váµ¢áµ€Mf) / (váµ¢áµ€Mváµ¢)

    NORMALIZED: M-weighted projection for normalized Laplacian eigenvectors.
                Solves Gc = b where G = Aáµ€MA for each truncation level k.

    WHITENED: Pure Euclidean projection in whitened domain (Best Bases theorem).
              Whitens signals: g = M^(1/2) f
              Simple projection: Ä = Q Qáµ€ g
              Euclidean error: ||g - Ä||â‚‚Â²
              This is the mathematically "pure" application of the Best Bases theorem
              for the Symmetric Normalized Laplacian L_sym.

    EUCLIDEAN: Pure Euclidean projection without whitening.
               Uses raw signals f directly (no M^(1/2) scaling).
               Simple projection: fÌ‚ = Q Qáµ€ f
               Euclidean error: ||f - fÌ‚||â‚‚Â²
    """
    UNNORMALIZED = "unnormalized"
    NORMALIZED = "normalized"
    WHITENED = "whitened"
    EUCLIDEAN = "euclidean"


class EigenCacheManager:
    """
    Manages eigendecomposition caching with consistent key generation and storage.

    This class provides shared cache logic that can be used by both the dataset
    classes and standalone cache computation tools.
    """

    def __init__(self, cache_dir: Optional[str] = None, enabled: bool = True):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files. If None, uses "./eigen_cache"
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        if self.enabled:
            self.cache_dir = Path(cache_dir) if cache_dir is not None else Path("./eigen_cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def generate_cache_key(self,
                           file_path: Path,
                           vertices: np.ndarray,
                           faces: Optional[np.ndarray],
                           decimation_config=None) -> str:
        """
        Generate a unique cache key for eigendecomposition.

        Args:
            file_path: Path to the mesh file
            vertices: Processed vertices (after decimation if applicable)
            faces: Processed faces (after decimation if applicable)
            decimation_config: Decimation configuration (optional)

        Returns:
            Unique cache key string
        """
        # Create a hash based on:
        # 1. File path and modification time
        # 2. Vertices and faces content (to handle decimation variations)
        # 3. Decimation config (if applicable)
        # NOTE: num_eigenfunctions is intentionally excluded to allow cache reuse

        file_stat = file_path.stat()
        hash_components = [
            str(file_path),
            str(file_stat.st_mtime),  # File modification time
            str(file_stat.st_size),  # File size
            str(vertices.shape),
            str(faces.shape if faces is not None else "None"),
        ]

        # Add decimation config to hash if applicable
        if decimation_config is not None:
            hash_components.extend([
                str(decimation_config.preserve_boundary),
                str(decimation_config.preserve_normal),
                str(decimation_config.preserve_topology),
                str(decimation_config.optimal_placement),
                str(decimation_config.planar_quadric),
            ])

        # Add a hash of the actual vertex/face data to handle different decimation outcomes
        vertices_hash = hashlib.md5(vertices.tobytes()).hexdigest()[:8]
        faces_hash = hashlib.md5(faces.tobytes()).hexdigest()[:8] if faces is not None else "none"
        hash_components.extend([vertices_hash, faces_hash])

        # Create final hash
        combined_string = "_".join(hash_components)
        cache_key = hashlib.md5(combined_string.encode()).hexdigest()

        return cache_key

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        if not self.enabled:
            raise ValueError("Cache is not enabled")
        return self.cache_dir / f"eigendecomposition_{cache_key}.pkl"

    def load_from_cache(self, cache_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Load eigendecomposition from cache.

        Args:
            cache_key: Unique cache key

        Returns:
            Tuple of (eigenvectors, eigenvalues, vertex_areas) if found, None otherwise
        """
        if not self.enabled:
            return None

        cache_path = self.get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                # Validate cached data structure
                if (isinstance(cached_data, dict) and
                        'eigenvectors' in cached_data and
                        'eigenvalues' in cached_data and
                        'vertex_areas' in cached_data and
                        'num_eigenfunctions' in cached_data):
                    return (
                        cached_data['eigenvectors'],
                        cached_data['eigenvalues'],
                        cached_data['vertex_areas']
                    )

            except Exception as e:
                # Silent failure - cache will be recomputed
                pass

        return None

    def save_to_cache(self,
                      cache_key: str,
                      eigenvectors: torch.Tensor,
                      eigenvalues: torch.Tensor,
                      vertex_areas: torch.Tensor) -> None:
        """
        Save eigendecomposition to cache.

        Args:
            cache_key: Unique cache key
            eigenvectors: Computed eigenvectors
            eigenvalues: Computed eigenvalues
            vertex_areas: Computed vertex areas
        """
        if not self.enabled:
            return

        cache_path = self.get_cache_path(cache_key)

        try:
            cache_data = {
                'eigenvectors': eigenvectors,
                'eigenvalues': eigenvalues,
                'vertex_areas': vertex_areas,
                'num_eigenfunctions': eigenvectors.shape[1],  # Save actual number computed
                'timestamp': time.time()
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

        except Exception as e:
            # Silent failure - computation will proceed without caching
            pass

    def load_eigendecomposition(self,
                                file_path: Path,
                                vertices: np.ndarray,
                                faces: Optional[np.ndarray],
                                num_eigenfunctions: int,
                                decimation_config=None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Load eigendecomposition from cache if available.

        Args:
            file_path: Path to the original mesh file
            vertices: Processed vertices (after decimation if applicable)
            faces: Processed faces (after decimation if applicable)
            num_eigenfunctions: Number of eigenfunctions needed
            decimation_config: Decimation configuration (optional)

        Returns:
            Tuple of (eigenvectors, eigenvalues, vertex_areas) if found in cache with sufficient eigenfunctions,
            (None, None, None) otherwise
        """
        if not self.enabled:
            return None, None, None

        # Generate cache key
        cache_key = self.generate_cache_key(file_path, vertices, faces, decimation_config)

        # Try to load from cache
        cached_result = self.load_from_cache(cache_key)

        if cached_result is not None:
            cached_eigenvectors, cached_eigenvalues, cached_vertex_areas = cached_result
            cached_num_eigenvectors = cached_eigenvectors.shape[1]

            if cached_num_eigenvectors >= num_eigenfunctions:
                # We have enough eigenvectors, return what we need
                return (
                    cached_eigenvectors[:, :num_eigenfunctions],
                    cached_eigenvalues[:num_eigenfunctions],
                    cached_vertex_areas
                )

        # Not in cache or insufficient eigenvectors
        return None, None, None

    def compute_eigendecomposition(self,
                                   file_path: Path,
                                   vertices: np.ndarray,
                                   faces: Optional[np.ndarray],
                                   num_eigenfunctions: int,
                                   decimation_config=None,
                                   verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute eigendecomposition and save to cache.

        Args:
            file_path: Path to the original mesh file
            vertices: Processed vertices (after decimation if applicable)
            faces: Processed faces (after decimation if applicable)
            num_eigenfunctions: Number of eigenfunctions to compute
            decimation_config: Decimation configuration (optional)
            verbose: Whether to print progress messages

        Returns:
            Tuple of (eigenvectors, eigenvalues, vertex_areas)

        Raises:
            Exception: If eigendecomposition computation fails
        """
        try:
            if verbose:
                print(f"Computing eigendecomposition for {file_path.name}...")

            # Compute eigendecomposition with requested number of eigenvectors
            gt_eigenvectors_torch, gt_eigenvalues_torch, gt_vertex_areas_torch = compute_robust_normalized_laplacian_eigenvectors(
                vertices,
                num_eigenfunctions
            )

            # Save to cache if enabled
            if self.enabled:
                cache_key = self.generate_cache_key(file_path, vertices, faces, decimation_config)
                self.save_to_cache(
                    cache_key,
                    gt_eigenvectors_torch,
                    gt_eigenvalues_torch,
                    gt_vertex_areas_torch
                )

                if verbose:
                    print(f"Saved eigendecomposition for {file_path.name} to cache")

            return gt_eigenvectors_torch, gt_eigenvalues_torch, gt_vertex_areas_torch

        except Exception as e:
            if verbose:
                print(f"ERROR computing eigendecomposition for {file_path.name}: {e}")
            raise e

    # =========================================================================
    # Schrödinger Operator Support
    # =========================================================================

    def generate_operator_cache_key(
            self,
            file_path: Path,
            vertices: np.ndarray,
            faces: Optional[np.ndarray],
            operator_type: str = "laplacian",
            potential_type: Optional[str] = None,
            potential_strength: Optional[float] = None,
            n_neighbors: int = 30,
            decimation_config=None
    ) -> str:
        """
        Generate cache key that includes operator configuration.

        Args:
            file_path: Path to the mesh file
            vertices: Processed vertices
            faces: Processed faces
            operator_type: "laplacian" or "schrodinger"
            potential_type: Type of potential (for Schrödinger)
            potential_strength: β value (for Schrödinger)
            n_neighbors: Number of neighbors for k-NN graph and curvature
            decimation_config: Decimation configuration

        Returns:
            Unique cache key string
        """
        # Start with base hash components
        file_stat = file_path.stat()
        hash_components = [
            str(file_path),
            str(file_stat.st_mtime),
            str(file_stat.st_size),
            str(vertices.shape),
            str(faces.shape if faces is not None else "None"),
            operator_type,
            str(n_neighbors),  # Include n_neighbors in cache key
        ]

        # Add operator-specific components
        if operator_type == "schrodinger":
            hash_components.extend([
                str(potential_type),
                f"{potential_strength:.6f}" if potential_strength is not None else "None",
            ])

        # Add decimation config
        if decimation_config is not None:
            hash_components.extend([
                str(decimation_config.preserve_boundary),
                str(decimation_config.preserve_normal),
                str(decimation_config.preserve_topology),
                str(decimation_config.optimal_placement),
                str(decimation_config.planar_quadric),
            ])

        # Add vertex/face data hash
        vertices_hash = hashlib.md5(vertices.tobytes()).hexdigest()[:8]
        faces_hash = hashlib.md5(faces.tobytes()).hexdigest()[:8] if faces is not None else "none"
        hash_components.extend([vertices_hash, faces_hash])

        # Create final hash
        combined_string = "_".join(hash_components)
        cache_key = hashlib.md5(combined_string.encode()).hexdigest()

        return cache_key

    def save_schrodinger_to_cache(
            self,
            cache_key: str,
            eigenvectors: torch.Tensor,
            eigenvalues: torch.Tensor,
            vertex_areas: torch.Tensor,
            potential: torch.Tensor,
            operator_type: str,
            potential_type: Optional[str],
            potential_strength: Optional[float]
    ) -> None:
        """
        Save Schrödinger eigendecomposition to cache (includes potential).

        Args:
            cache_key: Unique cache key
            eigenvectors: Computed eigenvectors
            eigenvalues: Computed eigenvalues
            vertex_areas: Vertex areas
            potential: Potential function V(x)
            operator_type: "laplacian" or "schrodinger"
            potential_type: Type of potential
            potential_strength: β value
        """
        if not self.enabled:
            return

        cache_path = self.get_cache_path(cache_key)

        try:
            cache_data = {
                'eigenvectors': eigenvectors,
                'eigenvalues': eigenvalues,
                'vertex_areas': vertex_areas,
                'potential': potential,
                'num_eigenfunctions': eigenvectors.shape[1],
                'operator_type': operator_type,
                'potential_type': potential_type,
                'potential_strength': potential_strength,
                'timestamp': time.time()
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

        except Exception as e:
            pass  # Silent failure

    def load_schrodinger_from_cache(
            self,
            cache_key: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]]:
        """
        Load Schrödinger eigendecomposition from cache.

        Args:
            cache_key: Unique cache key

        Returns:
            Tuple of (eigenvectors, eigenvalues, vertex_areas, potential, metadata) or None
        """
        if not self.enabled:
            return None

        cache_path = self.get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                # Validate structure
                required_keys = ['eigenvectors', 'eigenvalues', 'vertex_areas', 'num_eigenfunctions']
                if all(k in cached_data for k in required_keys):
                    metadata = {
                        'operator_type': cached_data.get('operator_type', 'laplacian'),
                        'potential_type': cached_data.get('potential_type'),
                        'potential_strength': cached_data.get('potential_strength'),
                    }
                    return (
                        cached_data['eigenvectors'],
                        cached_data['eigenvalues'],
                        cached_data['vertex_areas'],
                        cached_data.get('potential'),  # May be None for Laplacian
                        metadata
                    )
            except Exception as e:
                pass

        return None

    def compute_operator_eigendecomposition(
            self,
            file_path: Path,
            vertices: np.ndarray,
            faces: Optional[np.ndarray],
            num_eigenfunctions: int,
            operator_type: str = "laplacian",
            potential_type: Optional[str] = None,
            potential_strength: Optional[float] = None,
            n_neighbors: int = 30,
            decimation_config=None,
            verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute eigendecomposition for specified operator (Laplacian or Schrödinger).

        Args:
            file_path: Path to the mesh file
            vertices: Processed vertices
            faces: Processed faces
            num_eigenfunctions: Number of eigenfunctions to compute
            operator_type: "laplacian" or "schrodinger"
            potential_type: Type of potential (for Schrödinger)
            potential_strength: β value (for Schrödinger)
            n_neighbors: Number of neighbors for k-NN graph and curvature estimation
            decimation_config: Decimation configuration
            verbose: Print progress

        Returns:
            Tuple of (eigenvectors, eigenvalues, vertex_areas, potential)
            potential is None for Laplacian
        """
        try:
            if verbose:
                print(f"Computing {operator_type} eigendecomposition for {file_path.name}...")

            if operator_type == "laplacian":
                # Standard Laplacian
                eigenvectors, eigenvalues, vertex_areas = compute_robust_normalized_laplacian_eigenvectors(
                    vertices, num_eigenfunctions, n_neighbors=n_neighbors
                )
                potential = None

            elif operator_type == "schrodinger":
                # Schrödinger operator
                if potential_type is None:
                    raise ValueError("potential_type required for Schrödinger operator")
                if potential_strength is None:
                    potential_strength = 5.0  # Default

                eigenvectors, eigenvalues, vertex_areas, potential = compute_robust_schrodinger_eigenvectors(
                    vertices,
                    num_eigenfunctions,
                    n_neighbors=n_neighbors,
                    potential_type=potential_type,
                    potential_strength=potential_strength
                )
            else:
                raise ValueError(f"Unknown operator type: {operator_type}")

            # Save to cache
            if self.enabled:
                cache_key = self.generate_operator_cache_key(
                    file_path, vertices, faces,
                    operator_type, potential_type, potential_strength,
                    n_neighbors, decimation_config
                )

                self.save_schrodinger_to_cache(
                    cache_key,
                    eigenvectors,
                    eigenvalues,
                    vertex_areas,
                    potential if potential is not None else torch.tensor([]),
                    operator_type,
                    potential_type,
                    potential_strength
                )

                if verbose:
                    print(f"Saved {operator_type} eigendecomposition to cache")

            return eigenvectors, eigenvalues, vertex_areas, potential

        except Exception as e:
            if verbose:
                print(f"ERROR computing eigendecomposition: {e}")
            raise e

    def load_operator_eigendecomposition(
            self,
            file_path: Path,
            vertices: np.ndarray,
            faces: Optional[np.ndarray],
            num_eigenfunctions: int,
            operator_type: str = "laplacian",
            potential_type: Optional[str] = None,
            potential_strength: Optional[float] = None,
            n_neighbors: int = 30,
            decimation_config=None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Load eigendecomposition from cache for specified operator.

        Returns:
            Tuple of (eigenvectors, eigenvalues, vertex_areas, potential)
            All None if not cached or insufficient eigenfunctions
        """
        if not self.enabled:
            return None, None, None, None

        cache_key = self.generate_operator_cache_key(
            file_path, vertices, faces,
            operator_type, potential_type, potential_strength,
            n_neighbors, decimation_config
        )

        result = self.load_schrodinger_from_cache(cache_key)

        if result is not None:
            eigenvectors, eigenvalues, vertex_areas, potential, metadata = result

            # Check sufficient eigenfunctions
            if eigenvectors.shape[1] >= num_eigenfunctions:
                return (
                    eigenvectors[:, :num_eigenfunctions],
                    eigenvalues[:num_eigenfunctions],
                    vertex_areas,
                    potential
                )

        return None, None, None, None

    def get_cache_info(self, cache_key: str) -> Optional[dict]:
        """
        Get information about a cached item without loading the full data.

        Args:
            cache_key: Unique cache key

        Returns:
            Dictionary with cache info or None if not cached
        """
        if not self.enabled:
            return None

        cache_path = self.get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                if isinstance(cached_data, dict) and 'num_eigenfunctions' in cached_data:
                    return {
                        'num_eigenfunctions': cached_data['num_eigenfunctions'],
                        'timestamp': cached_data.get('timestamp', 0),
                        'file_size': cache_path.stat().st_size,
                        'cache_path': cache_path
                    }
            except:
                pass

        return None

    def list_cache_contents(self) -> list:
        """
        List all items in the cache with their metadata.

        Returns:
            List of dictionaries containing cache item information
        """
        if not self.enabled:
            return []

        cache_items = []

        for cache_file in self.cache_dir.glob("eigendecomposition_*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                if isinstance(cached_data, dict):
                    cache_items.append({
                        'cache_key': cache_file.stem.replace('eigendecomposition_', ''),
                        'num_eigenfunctions': cached_data.get('num_eigenfunctions', 'unknown'),
                        'timestamp': cached_data.get('timestamp', 0),
                        'file_size': cache_file.stat().st_size,
                        'cache_path': cache_file
                    })
            except:
                # Skip corrupted cache files
                continue

        return sorted(cache_items, key=lambda x: x['timestamp'], reverse=True)

    def clear_cache(self, confirm: bool = False) -> int:
        """
        Clear all cache files.

        Args:
            confirm: If True, actually delete files. If False, just return count.

        Returns:
            Number of files that would be/were deleted
        """
        if not self.enabled:
            return 0

        cache_files = list(self.cache_dir.glob("eigendecomposition_*.pkl"))

        if confirm:
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                except:
                    pass

        return len(cache_files)


def create_eigen_cache_manager(cache_dir: Optional[str] = None,
                               enabled: bool = True) -> EigenCacheManager:
    """
    Create an EigenCacheManager instance.

    Args:
        cache_dir: Directory to store cache files
        enabled: Whether caching is enabled

    Returns:
        EigenCacheManager instance
    """
    return EigenCacheManager(cache_dir=cache_dir, enabled=enabled)


# =============================================================================
# File Scanning Utilities
# =============================================================================

def scan_files(root_dirs: List[Path],
               file_size: Optional[Tuple[Optional[float], Optional[float]]],
               max_items: Optional[Union[int, float]],
               file_extensions: List[str],
               ) -> List[Path]:
    """
    Scan directories for mesh files with filtering options.

    This function provides shared file scanning logic that can be used by both
    PointCloudDataset and standalone cache computation tools.

    Args:
        root_dirs: List of directories to scan for mesh files
        min_file_size_mb: Minimum file size in MB to consider
        max_file_size_mb: Maximum file size in MB to consider
        max_items: Maximum number of items to include (int for absolute, float for fraction)
        replications: Number of replications per file
        file_extensions: List of file extensions to search for (default: common mesh formats)

    Returns:
        List of mesh file paths
    """
    all_paths: List[Path] = []

    # Scan all root directories
    for root_dir in root_dirs:
        root_path = Path(root_dir)
        if not root_path.exists():
            print(f"Warning: Directory {root_path} does not exist, skipping...")
            continue

        for ext in file_extensions:
            all_paths.extend(root_path.rglob(ext))

    # Filter files based on size range (converting MB to bytes)
    if file_size is not None:
        min_file_size_mb, max_file_size_mb = file_size
        if min_file_size_mb is not None:
            min_bytes = min_file_size_mb * (1024 * 1024)
        else:
            min_bytes = 0

        if max_file_size_mb is not None:
            max_bytes = max_file_size_mb * (1024 * 1024)
        else:
            max_bytes = np.inf

        filtered_paths = []
        for path in all_paths:
            try:
                file_size = path.stat().st_size
                if min_bytes <= file_size <= max_bytes:
                    filtered_paths.append(path)
            except OSError:
                # Skip files that can't be accessed
                continue
    else:
        filtered_paths = all_paths

    # Apply max_items limit
    if max_items is not None:
        if isinstance(max_items, int):
            filtered_paths = filtered_paths[:max_items]
        elif isinstance(max_items, float):
            if not 0 < max_items <= 1:
                raise ValueError("When max_items is a float, it must be between 0 and 1")
            num_items = int(len(filtered_paths) * max_items)
            filtered_paths = filtered_paths[:num_items]
        else:
            raise TypeError("max_items must be int, float, or None")

    return filtered_paths


# Add these functions to utils.py

def load_geometry(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load geometry from file.

    Args:
        file_path: Path to the mesh file

    Returns:
        Tuple of (vertices, faces)
    """
    import pymeshlab

    ms: pymeshlab.MeshSet = pymeshlab.MeshSet()
    ms.load_new_mesh(str(file_path))

    vertices = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix()

    return vertices, faces


def apply_decimation(vertices: np.ndarray, faces: np.ndarray, decimation_config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply mesh decimation using quadric edge collapse.

    Args:
        vertices: Original mesh vertices (N, 3)
        faces: Original mesh faces (M, 3)
        decimation_config: DecimationConfig object

    Returns:
        Tuple of (decimated_vertices, decimated_faces)
    """
    import pymeshlab
    import omegaconf

    # Create a MeshSet for decimation
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, faces))

    if decimation_config.target_face_num_range is not None:
        if isinstance(decimation_config.target_face_num_range, omegaconf.ListConfig):
            min_target_face_num, max_target_face_num = decimation_config.target_face_num_range
            # Use randint for integer face counts instead of uniform
            target_face_num = np.random.randint(min_target_face_num, max_target_face_num + 1)
        else:
            target_face_num = decimation_config.target_face_num_range

        target_perc = target_face_num / faces.shape[0]
    else:
        # Draw a random target percentage from the configured range
        if isinstance(decimation_config.target_perc_range, omegaconf.ListConfig):
            min_perc, max_perc = decimation_config.target_perc_range
            target_perc = np.random.uniform(min_perc, max_perc)
        else:
            target_perc = decimation_config.target_perc_range

    # Select a random quality threshold using config range
    if isinstance(decimation_config.quality_thr_range, omegaconf.ListConfig):
        min_quality, max_quality = decimation_config.quality_thr_range
        quality_thr = np.random.uniform(min_quality, max_quality)
    else:
        quality_thr = decimation_config.quality_thr_range

    # Apply decimation only on selected faces (those not marked for preservation)
    ms.meshing_decimation_quadric_edge_collapse(
        targetperc=target_perc,
        qualitythr=quality_thr,
        preserveboundary=decimation_config.preserve_boundary,
        preservenormal=decimation_config.preserve_normal,
        preservetopology=decimation_config.preserve_topology,
        optimalplacement=decimation_config.optimal_placement,
        planarquadric=decimation_config.planar_quadric,
        selected=False
    )

    # Get decimated mesh data
    decimated_mesh = ms.current_mesh()
    decimated_vertices = decimated_mesh.vertex_matrix()
    decimated_faces = decimated_mesh.face_matrix()

    return decimated_vertices, decimated_faces


def prepare_geometry(file_path: Path,
                     decimation_config: Optional = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare geometry data up to the point of eigendecomposition computation.

    Args:
        file_path: Path to the mesh file
        decimation_config: Optional decimation configuration

    Returns:
        Tuple of (vertices, faces)
    """
    # Step 1: Load and normalize geometry
    vertices, faces = load_geometry(file_path=file_path)
    # vertices = normalize_mesh_to_unit_area(vertices=vertices, faces=faces)
    vertices = normalize_to_unit_sphere(points=vertices)

    # Step 2: Apply decimation only if configured
    if decimation_config is not None:
        vertices, faces = apply_decimation(vertices=vertices, faces=faces, decimation_config=decimation_config)

    return vertices, faces


def estimate_normals(points: np.ndarray,
                     k_neighbors: int = 10,
                     k_orient: int = 10,
                     lambda_param: float = 0.0,
                     cos_alpha_tol: float = 1.0) -> np.ndarray:
    """
    Estimate normals for a point cloud and orient them consistently using tangent planes.

    Args:
        points: torch.Tensor of shape (K, 3) containing 3D points
        k_neighbors: Number of nearest neighbors to use for normal estimation
        k_orient: Number of neighbors for normal orientation consistency
        lambda_param: Weight parameter for orientation propagation
        cos_alpha_tol: Cosine tolerance for orientation propagation

    Returns:
        torch.Tensor of shape (K, 3) containing the estimated normals
    """
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
    )

    try:
        # Try orienting normals consistently
        pcd.orient_normals_consistent_tangent_plane(
            k_orient,
            lambda_param,
            cos_alpha_tol
        )
    except RuntimeError as e:
        print(f"Warning: Failed to orient normals consistently.")

    # Optional: ensure normals are normalized
    pcd.normalize_normals()

    return np.array(pcd.normals)


def compute_shot_features(points: np.ndarray, normals: np.ndarray, radius: float) -> np.ndarray:
    return hd.compute_shot(points, normals, points, normals, radius)


def compute_risp_features(points: np.ndarray, normals: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Compute RISP features for each point in the point cloud without loops.

    :param points: numpy array of shape (N, 3) containing point coordinates
    :param normals: numpy array of shape (N, 3) containing normal vectors for each point
    :param k: number of nearest neighbors to consider
    :return: numpy array of shape (N, 14, k) containing RISP features for each point
    """
    N = points.shape[0]
    tree: cKDTree = cKDTree(points)
    _, indices = tree.query(points, k=k + 1)  # +1 because the first neighbor is the point itself

    # Remove the first column (self-distances and self-indices)
    indices = indices[:, 1:]

    # Get neighbors
    neighbors = points[indices]  # Shape: (N, k, 3)
    neighbor_normals = normals[indices]

    # Compute relative positions
    rel_pos: np.ndarray = neighbors - points[:, np.newaxis, :]  # Shape: (N, k, 3)

    # Project neighbors onto tangent plane
    # First compute dot product of rel_pos with normals
    dots = np.sum(rel_pos * normals[:, np.newaxis, :], axis=2)  # Shape: (N, k)

    # Then subtract the normal component
    proj_neighbors: np.ndarray = rel_pos - dots[..., np.newaxis] * normals[:, np.newaxis, :]  # Shape: (N, k, 3)

    # for i in range(100):
    #     for j in range(k):
    #         x = np.cross(proj_neighbors[i, 0], proj_neighbors[i, j+1])
    #         bla1 = x / np.linalg.norm(x)
    #         bla2 = normals[i]
    #         pass

    # Compute angles in tangent plane
    # Step 1: Get vector perpendicular to normal in tangent plane as reference direction
    ref_dir_x = np.array([1.0, 0.0, 0.0])  # Can be any vector not parallel to normal
    ref_dir_x = ref_dir_x - np.sum(ref_dir_x * normals, axis=1, keepdims=True) * normals
    ref_dir_x = ref_dir_x / (np.linalg.norm(ref_dir_x, axis=1, keepdims=True) + 1e-16)

    ref_dir_y = np.cross(ref_dir_x, normals)
    ref_dir_y = ref_dir_y / (np.linalg.norm(ref_dir_y, axis=1, keepdims=True) + 1e-16)

    # Step 2: Compute angles relative to this reference direction
    x_coord = np.sum(proj_neighbors * ref_dir_x[:, np.newaxis, :], axis=2)
    y_coord = np.sum(proj_neighbors * ref_dir_y[:, np.newaxis, :], axis=2)
    angles = np.arctan2(y_coord, x_coord)

    # Sort neighbors by angle
    sort_idx = np.argsort(angles, axis=1)

    row_idx = np.arange(N)[:, np.newaxis]
    sorted_neighbors = neighbors[row_idx, sort_idx]
    sorted_neighbor_normals = neighbor_normals[row_idx, sort_idx]
    sorted_rel_pos = rel_pos[row_idx, sort_idx]

    # Compute edge vectors
    e_i = sorted_rel_pos
    e_i_minus_1 = np.roll(sorted_rel_pos, 1, axis=1)
    e_i_plus_1 = np.roll(sorted_rel_pos, -1, axis=1)
    n_i = sorted_neighbors
    n_i_minus_1 = np.roll(sorted_neighbors, 1, axis=1)
    n_i_plus_1 = np.roll(sorted_neighbors, -1, axis=1)

    # Compute RISP features
    L_0 = np.linalg.norm(e_i, axis=2)
    phi_1 = compute_angle_between_vectors(e_i_minus_1, e_i)
    phi_2 = compute_angle_between_vectors(e_i_plus_1, e_i)
    phi_3 = compute_angle_between_vectors(e_i_minus_1, n_i - n_i_minus_1)
    phi_4 = compute_angle_between_vectors(e_i_plus_1, n_i_plus_1 - n_i)
    phi_5 = compute_angle_between_vectors(np.cross(e_i_plus_1, e_i), np.cross(e_i_minus_1, e_i))

    alpha_1 = compute_angle_between_vectors(np.broadcast_to(normals[:, np.newaxis, :], e_i.shape), e_i)
    alpha_2 = compute_angle_between_vectors(np.broadcast_to(normals[:, np.newaxis, :], e_i.shape), e_i_minus_1)

    nn_i = sorted_neighbor_normals
    nn_i_minus_1 = np.roll(sorted_neighbor_normals, 1, axis=1)
    nn_i_plus_1 = np.roll(sorted_neighbor_normals, -1, axis=1)

    # w_i = np.cross(e_i, e_i_minus_1)
    # w_i = w_i / (np.linalg.norm(w_i, axis=2, keepdims=True) + 1e-10)

    beta_1 = compute_angle_between_vectors(nn_i, e_i)
    beta_2 = compute_angle_between_vectors(nn_i, n_i - n_i_minus_1)

    # n_i_minus_1 = np.cross(e_i_minus_1, e_i_minus_1 - np.roll(sorted_neighbors, 1, axis=1) + points[:, np.newaxis, :])
    # n_i_minus_1 = n_i_minus_1 / (np.linalg.norm(n_i_minus_1, axis=2, keepdims=True) + 1e-10)

    theta_1 = compute_angle_between_vectors(nn_i_minus_1, e_i_minus_1)
    theta_2 = compute_angle_between_vectors(nn_i_minus_1, n_i - n_i_minus_1)

    gamma_1 = compute_angle_between_vectors(nn_i_plus_1, n_i_plus_1 - n_i)
    gamma_2 = compute_angle_between_vectors(nn_i_plus_1, e_i_plus_1)

    risp_features = np.stack([
        L_0, phi_1, phi_2, phi_3, phi_4, phi_5,
        alpha_1, alpha_2, beta_1, beta_2,
        theta_1, theta_2, gamma_1, gamma_2], axis=-1)

    # risp_features_max = np.max(risp_features, axis=1)
    # risp_features_min = np.min(risp_features, axis=1)
    # risp_features_mean = np.mean(risp_features, axis=1)
    # risp_features_concat = np.concat([risp_features_max, risp_features_min, risp_features_mean], axis=1)
    # return risp_features_max
    return risp_features


def compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute the angle between two sets of vectors."""
    v1_n = v1 / (np.linalg.norm(v1, axis=2, keepdims=True) + 1e-10)
    v2_n = v2 / (np.linalg.norm(v2, axis=2, keepdims=True) + 1e-10)
    return np.arccos(np.clip(np.sum(v1_n * v2_n, axis=2), -1.0, 1.0))


def import_object(full_type_name: str) -> Type:
    module_name, class_name = full_type_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_input_args(forward_method: Callable) -> List[str]:
    signature = inspect.signature(forward_method)
    return [
        param.name for param in signature.parameters.values()
        if param.default == param.empty and param.name != 'self'
    ]


def centroid_to_origin(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0, keepdims=True)
    centered_points = points - centroid
    return centered_points


def normalize_to_unit_cube(points: np.ndarray) -> np.ndarray:
    points = centroid_to_origin(points=points)
    p_max = points.max(axis=0)
    p_min = points.min(axis=0)
    center = (p_max + p_min) / 2
    scale = (p_max - p_min).max()
    return (points - center) / scale


def normalize_to_unit_sphere(points: np.ndarray) -> np.ndarray:
    """
    Rescales a point cloud to fit within a unit sphere centered at the origin.

    Args:
        points (np.ndarray): Point cloud array of shape (K, 3) where K is the number of points

    Returns:
        np.ndarray: Normalized point cloud of shape (K, 3) fitting within a unit sphere

    Raises:
        ValueError: If input array doesn't have shape (K, 3)
    """
    points = centroid_to_origin(points=points)

    # Find the maximum distance from the origin to any point
    distances = np.linalg.norm(points, axis=1)
    max_distance = np.max(distances)

    # Scale the points to fit within a unit sphere
    normalized_points = points / max_distance

    return normalized_points


def normalize_mesh_to_unit_area(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = centroid_to_origin(points=vertices)

    # Calculate the current surface area
    current_area = igl.doublearea(vertices, faces).sum() / 2.0

    # Calculate scaling factor
    scale_factor = 1.0 / np.sqrt(current_area)

    # Scale the vertices
    normalized_vertices = vertices * scale_factor

    # Verify the new area
    # new_area = igl.doublearea(normalized_vertices, faces).sum() / 2.0
    # print(f"Original area: {current_area}")
    # print(f"Normalized area: {new_area}")

    return normalized_vertices


def random_rotation_matrix() -> torch.Tensor:
    """
    Generate a random 3D rotation matrix.

    Returns:
    torch.Tensor: A 3x3 orthonormal rotation matrix.
    """
    # Generate a random 3x3 matrix
    random_matrix: torch.Tensor = torch.randn(3, 3)

    # Perform QR decomposition
    q, r = torch.linalg.qr(random_matrix)

    # Ensure proper rotation matrix (determinant = 1)
    d: torch.Tensor = torch.diag(torch.sign(torch.diag(r)))
    rotation_matrix: torch.Tensor = torch.mm(q, d)

    # Ensure right-handed coordinate system
    if torch.det(rotation_matrix) < 0:
        rotation_matrix[:, 0] *= -1

    return rotation_matrix


def compute_canonical_pose_pca(points: torch.Tensor) -> torch.Tensor:
    """
    Compute the canonical pose of a 3D point cloud.

    This function centers the point cloud at the origin and aligns its principal axes
    with the coordinate axes.

    Args:
    points (torch.Tensor): Tensor of shape (N, 3) containing N 3D points.

    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        - points_canonical: Points in canonical pose, shape (N, 3)
        - R: Rotation matrix, shape (3, 3)
        - t: Translation vector, shape (3,)
    """
    # Center the points
    center: torch.Tensor = torch.mean(points, dim=0)
    centered_points: torch.Tensor = points - center

    # Compute covariance matrix
    cov: torch.Tensor = torch.mm(centered_points.t(), centered_points) / (points.shape[0] - 1)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    R: torch.Tensor = eigenvectors[:, sorted_indices]

    # Ensure right-handed coordinate system
    if torch.det(R) < 0:
        R[:, 2] *= -1

    # Transform points to canonical pose
    points_canonical: torch.Tensor = torch.mm(centered_points, R)

    return points_canonical


def faces_to_edges(faces: torch.Tensor) -> torch.Tensor:
    """
    Convert triangle faces to edge indices

    Args:
        faces: torch.LongTensor of shape [N, 3] containing triangular faces

    Returns:
        edge_index: torch.LongTensor of shape [2, E] containing unique edges
    """
    # Get all edges from faces (including duplicates)
    # For each triangle, get its 3 edges
    edges: torch.Tensor = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)

    # Sort edges to ensure (v1, v2) and (v2, v1) are treated as the same edge
    edges = torch.sort(edges, dim=1)[0]

    # Remove duplicate edges
    edges = torch.unique(edges, dim=0)

    # Convert to PyG edge_index format (2, E)
    edge_index: torch.Tensor = edges.t().contiguous()

    return edge_index


def farthest_point_sampling(vertices: np.ndarray, num_samples: int, random_start: bool = True) -> np.ndarray:
    """Perform farthest point sampling using Open3D.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions
        num_samples (int): Number of points to sample
        random_start (bool): Whether to use random initialization for FPS

    Returns:
        np.ndarray: Indices of sampled vertices
    """
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # Set start_index based on random_start parameter
    start_index = np.random.randint(len(vertices)) if random_start else 0

    # Perform FPS using Open3D with specified start index
    num_samples = num_samples if num_samples < vertices.shape[0] else vertices.shape[0]
    sampled_pcd = pcd.farthest_point_down_sample(num_samples, start_index=start_index)
    sampled_points = np.asarray(sampled_pcd.points)

    # Find the indices of these points in the original vertices array
    tree = cKDTree(vertices)
    _, indices = tree.query(sampled_points)

    return indices.astype(np.int32)


def split_results_by_nodes(results: torch.Tensor, batch: Batch) -> List[torch.Tensor]:
    return [results[batch.batch == i] for i in range(batch.num_graphs)]


def split_results_by_graphs(results: torch.Tensor, batch: Batch) -> List[torch.Tensor]:
    return [results[i] for i in range(batch.num_graphs)]


def scale_by_half(scalar_functions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w_half = torch.diag(torch.sqrt(weights))
    weighted_scalar_functions = w_half @ scalar_functions
    return weighted_scalar_functions


def scale_by_half_inv(scalar_functions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w_inv_half = torch.diag(1 / torch.sqrt(weights))
    weighted_scalar_functions = w_inv_half @ scalar_functions
    return weighted_scalar_functions


def _setup_projection_batch(
        eigenvectors_basis: torch.Tensor,
        weights: torch.Tensor,
        scalar_functions: torch.Tensor,
        max_eigenvectors: int
) -> tuple:
    """
    Common setup for both normalized and unnormalized LÃƒâ€šÃ‚Â²-projections.

    Returns:
        tuple: (mass_matrix, M_f, M_A, At_M_f, mask_lower, A_expanded, M_A_expanded)
    """
    num_vertices, num_eigenvectors = eigenvectors_basis.shape
    device = eigenvectors_basis.device

    # Construct mass matrix M
    mass_matrix = torch.diag(weights)  # [num_vertices, num_vertices]

    # Pre-compute common matrix products
    M_f = torch.matmul(mass_matrix, scalar_functions)  # [num_vertices, num_dims]
    M_A = torch.matmul(mass_matrix, eigenvectors_basis)  # [num_vertices, num_eigenvectors]
    At_M_f = torch.matmul(eigenvectors_basis.T, M_f)  # [num_eigenvectors, num_dims]

    # Create triangular mask for selecting first k eigenvectors at each level
    mask_lower = torch.tril(torch.ones(max_eigenvectors, num_eigenvectors, device=device))

    # Expand matrices for batch operations
    A_expanded = eigenvectors_basis.unsqueeze(0).expand(max_eigenvectors, -1, -1)
    M_A_expanded = M_A.unsqueeze(0).expand(max_eigenvectors, -1, -1)

    return mass_matrix, M_f, M_A, At_M_f, mask_lower, A_expanded, M_A_expanded


def _apply_reconstruction(
        A_expanded: torch.Tensor,
        coefficients: torch.Tensor,
        mask_lower: torch.Tensor
) -> torch.Tensor:
    """
    Common reconstruction step: f_proj = A @ c with proper masking.

    Args:
        A_expanded: Expanded eigenvectors [max_eigenvectors, num_vertices, num_eigenvectors]
        coefficients: Projection coefficients [max_eigenvectors, num_eigenvectors, num_dims]
        mask_lower: Triangular mask [max_eigenvectors, num_eigenvectors]

    Returns:
        torch.Tensor: Reconstructed functions [max_eigenvectors, num_vertices, num_dims]
    """
    # Apply mask to eigenvectors and coefficients
    A_masked = A_expanded * mask_lower.unsqueeze(1)
    coefficients_masked = coefficients * mask_lower.unsqueeze(-1)

    # Compute reconstructions: f_proj = A @ c
    reconstructed_functions = torch.bmm(A_masked, coefficients_masked)

    return reconstructed_functions


def project_functions_unnormalized(
        eigenvectors_basis: torch.Tensor,
        weights: torch.Tensor,
        scalar_functions: torch.Tensor,
        max_eigenvectors: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized LÂ²-projection for unnormalized Laplacian eigenvectors.

    Simple case: coefficients = (váµ¢áµ€Mf) / (váµ¢áµ€Mváµ¢) since eigenvectors are M-orthogonal.

    Returns:
        Tuple of (reconstructed_functions, scalar_functions) for consistent interface.
    """
    # Common setup
    _, M_f, M_A, At_M_f, mask_lower, A_expanded, M_A_expanded = _setup_projection_batch(
        eigenvectors_basis, weights, scalar_functions, max_eigenvectors
    )

    # Compute denominators: váµ¢áµ€Mváµ¢ for all eigenvectors
    denominators = torch.sum(eigenvectors_basis * M_A, dim=0)  # [num_eigenvectors]

    # Compute all projection coefficients: numerator / denominator
    coefficients = At_M_f / denominators.unsqueeze(1)  # [num_eigenvectors, num_dims]

    # Expand coefficients for all levels
    coefficients_expanded = coefficients.unsqueeze(0).expand(max_eigenvectors, -1, -1)

    # Apply reconstruction
    reconstructed = _apply_reconstruction(A_expanded, coefficients_expanded, mask_lower)

    return reconstructed, scalar_functions


def project_functions_normalized(
        eigenvectors_basis: torch.Tensor,
        weights: torch.Tensor,
        scalar_functions: torch.Tensor,
        max_eigenvectors: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized LÂ²-projection for normalized Laplacian eigenvectors.

    Complex case: must solve Gc = b where G = Aáµ€MA for each level k.

    Returns:
        Tuple of (reconstructed_functions, scalar_functions) for consistent interface.
    """
    num_eigenvectors = eigenvectors_basis.shape[1]
    device = eigenvectors_basis.device

    # Common setup
    _, M_f, M_A, At_M_f, mask_lower, A_expanded, M_A_expanded = _setup_projection_batch(
        eigenvectors_basis, weights, scalar_functions, max_eigenvectors
    )

    # Apply mask to get A_k and M_A_k for each level k
    A_masked = A_expanded * mask_lower.unsqueeze(1)
    M_A_masked = M_A_expanded * mask_lower.unsqueeze(1)

    # Compute all Gram matrices G_k = A_k^T @ M @ A_k simultaneously
    G_batch = torch.bmm(A_masked.transpose(1, 2), M_A_masked)  # [max_eigenvectors, num_eigenvectors, num_eigenvectors]

    # Add regularization for numerical stability
    reg_term = 1e-8 * torch.eye(num_eigenvectors, device=device).unsqueeze(0).expand(max_eigenvectors, -1, -1)
    G_batch_reg = G_batch + reg_term

    # Create batch of RHS vectors b_k = A_k^T @ M @ f
    At_M_f_expanded = At_M_f.unsqueeze(0).expand(max_eigenvectors, -1, -1)
    b_batch = At_M_f_expanded * mask_lower.unsqueeze(-1)

    # Solve all linear systems G_k @ c_k = b_k simultaneously
    coefficients_batch = torch.linalg.solve(G_batch_reg, b_batch)

    # Apply reconstruction
    reconstructed = _apply_reconstruction(A_expanded, coefficients_batch, mask_lower)

    return reconstructed, scalar_functions


def project_functions_whitened(
        eigenvectors_basis: torch.Tensor,
        weights: torch.Tensor,
        scalar_functions: torch.Tensor,
        max_eigenvectors: int,
        whiten: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure Euclidean projection, optionally in whitened domain.

    When whiten=True (default): Applies the Best Bases theorem for L_sym.
    Whitens signals first: g = M^(1/2) f, then projects.

    When whiten=False: Pure Euclidean projection on raw signals f.
    No scaling applied, just fÌ‚ = Q Q^T f.

    Args:
        eigenvectors_basis: Eigenvector basis Q [n_vertices, n_eigenvectors]
                           Assumed to be Euclidean orthonormal (Q^T Q = I)
        weights: Vertex area weights M [n_vertices] (only used if whiten=True)
        scalar_functions: Raw target functions f [n_vertices, n_dims]
        max_eigenvectors: Maximum number of eigenvectors to use
        whiten: If True, whiten signals with M^(1/2) before projection.
                If False, use raw signals directly.

    Returns:
        Tuple of (reconstructed, signals) where:
            reconstructed: Reconstructed signals [max_eigenvectors, n_vertices, n_dims]
            signals: Original signals (whitened if whiten=True) [n_vertices, n_dims]
    """
    num_vertices, num_eigenvectors = eigenvectors_basis.shape
    device = eigenvectors_basis.device

    # Step 1: Optionally whiten the signals: g = M^(1/2) f
    if whiten:
        sqrt_weights = torch.sqrt(weights)  # [n_vertices]
        signals = scalar_functions * sqrt_weights.unsqueeze(-1)  # [n_vertices, n_dims]
    else:
        signals = scalar_functions  # Use raw signals directly

    # Step 2: Compute projection coefficients for all eigenvectors
    # coefficients = Q^T signals  (simple Euclidean projection)
    coefficients = torch.matmul(eigenvectors_basis.T, signals)  # [n_eigenvectors, n_dims]

    # Step 3: Create triangular mask for selecting first k eigenvectors at each level
    mask_lower = torch.tril(torch.ones(max_eigenvectors, num_eigenvectors, device=device))

    # Step 4: Expand for batch operations
    Q_expanded = eigenvectors_basis.unsqueeze(0).expand(max_eigenvectors, -1, -1)  # [max_k, n_vertices, n_eig]
    coefficients_expanded = coefficients.unsqueeze(0).expand(max_eigenvectors, -1, -1)  # [max_k, n_eig, n_dims]

    # Step 5: Apply mask and reconstruct: signals_k = Q_k Q_k^T signals = Q_k c_k
    Q_masked = Q_expanded * mask_lower.unsqueeze(1)  # [max_k, n_vertices, n_eig]
    coefficients_masked = coefficients_expanded * mask_lower.unsqueeze(-1)  # [max_k, n_eig, n_dims]

    # Reconstruct: signals_hat = Q @ c
    reconstructed = torch.bmm(Q_masked, coefficients_masked)  # [max_k, n_vertices, n_dims]

    return reconstructed, signals


def reconstruction_error(
        eigenvectors: torch.Tensor,
        scalar_functions: torch.Tensor,
        weights: torch.Tensor,
        max_eigenvectors: int,
        projection_method: ProjectionMethod,
        use_weighted_norm: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute reconstruction error and eigenvalues from scalar function projection.

    Args:
        eigenvectors: Eigenvector basis [n_vertices, n_eigenvectors]
        scalar_functions: Target functions to reconstruct [n_vertices, n_dims]
        weights: Vertex area weights [n_vertices]
        max_eigenvectors: Maximum number of eigenvectors to use for reconstruction
        projection_method: Which projection method to use (UNNORMALIZED, NORMALIZED, WHITENED, or EUCLIDEAN)
        use_weighted_norm: Whether to use M-weighted norm for the loss computation.
                          Note: For WHITENED and EUCLIDEAN, this is ignored since the Euclidean norm
                          is the theoretically correct one.

    Returns:
        reconstruction_loss: Mean reconstruction error
        eigenvalues_unweighted: Eigenvalues from unweighted norm errors
        eigenvalues_weighted: Eigenvalues from weighted norm errors
    """
    # Call the appropriate projection method
    if projection_method == ProjectionMethod.WHITENED:
        # Pure Euclidean projection in whitened domain (Best Bases theorem)
        reconstructed, original = project_functions_whitened(
            eigenvectors_basis=eigenvectors,
            weights=weights,
            scalar_functions=scalar_functions,
            max_eigenvectors=max_eigenvectors,
            whiten=True
        )

    elif projection_method == ProjectionMethod.EUCLIDEAN:
        # Pure Euclidean projection without whitening (raw signals)
        reconstructed, original = project_functions_whitened(
            eigenvectors_basis=eigenvectors,
            weights=weights,
            scalar_functions=scalar_functions,
            max_eigenvectors=max_eigenvectors,
            whiten=False
        )

    elif projection_method == ProjectionMethod.NORMALIZED:
        # M-weighted projection for normalized Laplacian eigenvectors
        reconstructed, original = project_functions_normalized(
            eigenvectors_basis=eigenvectors,
            scalar_functions=scalar_functions,
            weights=weights,
            max_eigenvectors=max_eigenvectors
        )

    elif projection_method == ProjectionMethod.UNNORMALIZED:
        # M-orthogonal eigenvectors
        scaled_eigenvectors = scale_by_half_inv(scalar_functions=eigenvectors, weights=weights)
        reconstructed, original = project_functions_unnormalized(
            eigenvectors_basis=scaled_eigenvectors,
            scalar_functions=scalar_functions,
            weights=weights,
            max_eigenvectors=max_eigenvectors
        )
    else:
        raise ValueError(f"Unknown projection method: {projection_method}")

    # Compute reconstruction error
    error = original - reconstructed

    if projection_method in (ProjectionMethod.WHITENED, ProjectionMethod.EUCLIDEAN):
        # For WHITENED/EUCLIDEAN: use pure Euclidean norm
        # use_weighted_norm is ignored since Euclidean norm is the correct one
        norm_squared = torch.sum(error ** 2, dim=1)  # [max_k, n_dims]

        mean_errors = torch.mean(norm_squared, dim=1)  # [max_k]
        eigenvalues = 1.0 / mean_errors

        reconstruction_loss = torch.mean(norm_squared)

        # For WHITENED/EUCLIDEAN, both eigenvalue estimates are the same
        return reconstruction_loss, eigenvalues, eigenvalues

    else:
        # For NORMALIZED and UNNORMALIZED: compute both norms
        unweighted_error = error
        weighted_error = error * torch.sqrt(weights).unsqueeze(0).unsqueeze(-1)

        norm_unweighted = torch.sum(unweighted_error ** 2, dim=1)
        norm_weighted = torch.sum(weighted_error ** 2, dim=1)

        # Compute eigenvalues from both norms
        mean_errors_unweighted = torch.mean(norm_unweighted, dim=1)
        eigenvalues_unweighted = 1.0 / mean_errors_unweighted

        mean_errors_weighted = torch.mean(norm_weighted, dim=1)
        eigenvalues_weighted = 1.0 / mean_errors_weighted

        # Use selected norm for the loss
        norm_for_loss = norm_weighted if use_weighted_norm else norm_unweighted
        reconstruction_loss = torch.mean(norm_for_loss)

        return reconstruction_loss, eigenvalues_unweighted, eigenvalues_weighted


def is_close_to_identity(matrix, rtol=1e-3, atol=1e-3):
    """
    Check if a matrix is close to the identity matrix.

    Args:
        matrix (torch.Tensor): The matrix to check
        rtol (float): Relative tolerance
        atol (float): Absolute tolerance

    Returns:
        bool: True if the matrix is close to identity, False otherwise
    """
    # Get the shape to create appropriate identity matrix
    batch_dims = matrix.shape[:-2]
    n = matrix.shape[-1]

    # Create identity matrix with the same shape and device as input
    identity = torch.eye(n, device=matrix.device)
    if batch_dims:
        identity = identity.expand(*batch_dims, n, n)

    # Check if matrices are close
    return torch.allclose(matrix, identity, rtol=rtol, atol=atol)


def knn_graph(
        x: Union[torch.Tensor, np.ndarray],
        k: int = 6,
        loop: bool = False,
        batch: Optional[Union[torch.Tensor, np.ndarray]] = None,
        flow: Literal["source_to_target", "target_to_source"] = "source_to_target"
) -> torch.Tensor:
    """
    Compute edge_index for PyTorch Geometric using scikit-learn's NearestNeighbors

    Args:
        x: Node features tensor or numpy array with shape [N, dim]
        k: Number of nearest neighbors
        loop: Whether to include self-loops
        batch: Optional batch vector, which assigns each node to a specific example
        flow: Edge direction, either "source_to_target" (default) or "target_to_source"
                  - "source_to_target": Edges point from center node to neighbors
                  - "target_to_source": Edges point from neighbors to center node

    Returns:
        edge_index: COO edge index tensor with shape [2, E]
    """
    # Convert tensor to numpy if needed
    if isinstance(x, torch.Tensor):
        x_np: np.ndarray = x.detach().cpu().numpy()
    else:
        x_np: np.ndarray = x

    # Handle batched data if needed
    if batch is not None:
        if isinstance(batch, torch.Tensor):
            batch_np: np.ndarray = batch.detach().cpu().numpy()
        else:
            batch_np: np.ndarray = batch

        edge_index_list: list = []
        max_node_idx: int = 0

        # Process each batch separately
        for b in range(batch_np.max() + 1):
            batch_mask: np.ndarray = (batch_np == b)
            x_batch: np.ndarray = x_np[batch_mask]

            # Skip batches with less than k+1 nodes
            if len(x_batch) <= k:
                continue

            # Compute KNN for this batch
            neighbor_model = NearestNeighbors(n_neighbors=k + (0 if loop else 1))
            neighbor_model.fit(x_batch)
            distances: np.ndarray
            indices: np.ndarray
            distances, indices = neighbor_model.kneighbors(x_batch)

            # Generate source and target indices
            rows: np.ndarray = np.repeat(np.arange(len(x_batch)), k)
            if not loop:
                indices = indices[:, 1:]  # Remove self-loops
            cols: np.ndarray = indices.flatten()

            # Adjust indices for global node indexing
            rows += max_node_idx
            cols += max_node_idx

            # Add to edge list
            batch_edge_index = np.vstack((rows, cols))
            edge_index_list.append(batch_edge_index)

            # Update max node index for next batch
            max_node_idx += len(x_batch)

        edge_index = np.hstack(edge_index_list)

    else:
        # No batch processing needed
        # Compute KNN
        neighbor_model = NearestNeighbors(n_neighbors=k + (0 if loop else 1))
        neighbor_model.fit(x_np)
        distances: np.ndarray
        indices: np.ndarray
        distances, indices = neighbor_model.kneighbors(x_np)

        # Generate source and target indices
        rows: np.ndarray = np.repeat(np.arange(len(x_np)), k)
        if not loop:
            indices = indices[:, 1:]  # Remove self-loops
        cols: np.ndarray = indices.flatten()

        # Create edge_index based on the specified direction
        if flow == "source_to_target":
            edge_index: np.ndarray = np.vstack((cols, rows))  # From neighbors to center nodes
        elif flow == "target_to_source":
            edge_index: np.ndarray = np.vstack((rows, cols))  # From center nodes to their neighbors
        else:
            raise ValueError("Direction must be 'source_to_target' or 'target_to_source'")

    # Convert to torch tensor
    return torch.tensor(edge_index, dtype=torch.long)


def rebuild_batch_from_list(batch: Batch, property_name: str, property_tensor_list: List[torch.Tensor]) -> Batch:
    data_list = batch.to_data_list()
    for data, tensor in zip(data_list, property_tensor_list):
        data[property_name] = tensor
    return Batch.from_data_list(data_list)


def rebuild_batch_from_tensor(batch: Batch, property_name: str, property_tensor: torch.Tensor) -> Batch:
    tensor_list = property_tensor.split(split_size=batch.batch.bincount().tolist(), dim=0)
    return rebuild_batch_from_list(batch=batch, property_name=property_name, property_tensor_list=tensor_list)


def rebuild_batch_from_dictionary_of_lists(batch: Batch, property_dict: Dict[str, List[torch.Tensor]]) -> Batch:
    for property_name, property_tensor_list in property_dict.items():
        batch = rebuild_batch_from_list(batch=batch, property_name=property_name, property_tensor_list=property_tensor_list)
    return batch


def align_eigenvector_signs(eigenvectors: Union[torch.Tensor, np.ndarray],
                            reference_eigenvectors: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Align eigenvector signs to match reference eigenvectors.

    Args:
        eigenvectors: Eigenvectors to align [N, K]
        reference_eigenvectors: Reference eigenvectors [N, K]

    Returns:
        Sign-aligned eigenvectors with same type as input
    """
    # Remember original type
    return_torch = isinstance(eigenvectors, torch.Tensor)

    # Convert to numpy for computation
    if isinstance(eigenvectors, torch.Tensor):
        eigenvectors_np = eigenvectors.detach().cpu().numpy()
    else:
        eigenvectors_np = eigenvectors.copy()

    if isinstance(reference_eigenvectors, torch.Tensor):
        reference_np = reference_eigenvectors.detach().cpu().numpy()
    else:
        reference_np = reference_eigenvectors.copy()

    # Determine the number of eigenvectors to align
    min_eigenvectors = min(eigenvectors_np.shape[1], reference_np.shape[1])

    # Normalize eigenvectors to unit length for accurate cosine computation
    norms_eig = np.linalg.norm(eigenvectors_np[:, :min_eigenvectors], axis=0) + 1e-8
    norms_ref = np.linalg.norm(reference_np[:, :min_eigenvectors], axis=0) + 1e-8

    eigenvectors_norm = eigenvectors_np[:, :min_eigenvectors] / norms_eig[None, :]
    reference_norm = reference_np[:, :min_eigenvectors] / norms_ref[None, :]

    # Compute cosine similarities (without absolute value)
    cosine_similarities = np.sum(eigenvectors_norm * reference_norm, axis=0)

    # Create aligned eigenvectors
    aligned_eigenvectors = eigenvectors_np.copy()

    # Flip signs where cosine similarity is negative
    for i in range(min_eigenvectors):
        if cosine_similarities[i] < 0:
            aligned_eigenvectors[:, i] *= -1

    # Convert back to original type
    if return_torch:
        return torch.from_numpy(aligned_eigenvectors).to(dtype=eigenvectors.dtype, device=eigenvectors.device)
    else:
        return aligned_eigenvectors


def compute_eigenvector_cosine_similarities(eigenvectors_1: torch.Tensor,
                                            eigenvectors_2: torch.Tensor) -> torch.Tensor:
    """
    Compute absolute cosine similarities between corresponding eigenvectors from two sets.
    Note: If eigenvectors are sign-aligned, cosine similarities should be positive.

    Args:
        eigenvectors_1: First set of eigenvectors [N, K] (PyTorch tensor on CPU)
        eigenvectors_2: Second set of eigenvectors [N, K] (PyTorch tensor on CPU)

    Returns:
        Tensor of absolute cosine similarities [min(K1, K2)]
    """
    # # Assert inputs are PyTorch tensors
    # assert isinstance(eigenvectors_1, torch.Tensor), f"eigenvectors_1 must be a torch.Tensor, got {type(eigenvectors_1)}"
    # assert isinstance(eigenvectors_2, torch.Tensor), f"eigenvectors_2 must be a torch.Tensor, got {type(eigenvectors_2)}"
    #
    # # Ensure tensors are on CPU
    # eigenvectors_1 = eigenvectors_1.detach().cpu()
    # eigenvectors_2 = eigenvectors_2.detach().cpu()
    #
    # # Determine the number of eigenvectors to compare
    # min_eigenvectors = min(eigenvectors_1.shape[1], eigenvectors_2.shape[1])
    #
    # # Slice to match dimensions
    # eig1 = eigenvectors_1[:, :min_eigenvectors]
    # eig2 = eigenvectors_2[:, :min_eigenvectors]
    #
    # # Normalize all eigenvectors to unit length
    # norms_1 = torch.norm(eig1, dim=0) + 1e-8
    # norms_2 = torch.norm(eig2, dim=0) + 1e-8
    #
    # eig1_norm = eig1 / norms_1.unsqueeze(0)
    # eig2_norm = eig2 / norms_2.unsqueeze(0)
    #
    # # Compute cosine similarities (element-wise dot products)
    # cosine_sims = torch.sum(eig1_norm * eig2_norm, dim=0)

    cos_sim = F.cosine_similarity(eigenvectors_1, eigenvectors_2, dim=0)

    # Take absolute value to handle sign ambiguity
    return torch.abs(cos_sim)


# =============================================================================
# Shared Core Functions
# =============================================================================

def _compute_pyfm_base(vertices: np.ndarray, faces: np.ndarray, num_eigenfunctions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, sparse.spmatrix]:
    """
    Core PyFM computation logic shared between normalized and unnormalized versions.

    Args:
        vertices: Vertex positions [num_vertices, 3]
        faces: Face connectivity [num_faces, 3]
        num_eigenfunctions: Number of eigenfunctions to compute

    Returns:
        Tuple of (eigenvectors, eigenvalues, vertex_areas, laplacian_matrix)
    """
    # Create pyFM mesh from vertices and faces
    mesh = TriMesh(vertices, faces)

    # Process the mesh and compute the Laplacian spectrum
    # Set intrinsic=False for using extrinsic Laplacian
    mesh.process(k=num_eigenfunctions, intrinsic=False, verbose=False)

    # Retrieve eigenvalues and eigenfunctions
    eigenvalues = mesh.eigenvalues
    eigenvectors = mesh.eigenvectors
    vertex_areas = mesh.vertex_areas

    # Calculate the Laplace-Beltrami operator
    n = len(vertex_areas)
    M_inv = sparse.diags(1.0 / vertex_areas)
    laplacian = M_inv @ mesh.W

    return eigenvectors, eigenvalues, vertex_areas, laplacian


def _compute_robust_base(vertices: np.ndarray, num_eigenfunctions: int, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core robust Laplacian computation logic shared between normalized and unnormalized versions.

    Args:
        vertices: Vertex positions [num_vertices, 3]
        num_eigenfunctions: Number of eigenfunctions to compute
        n_neighbors: Number of neighbors for k-NN graph construction

    Returns:
        Tuple of (eigenvectors, eigenvalues, vertex_areas)
    """
    # Compute robust Laplacian for point clouds
    L_robust, M_robust = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=n_neighbors)

    # Compute eigenvectors using scipy
    evals_robust, evecs_robust = sla.eigsh(L_robust, num_eigenfunctions, M_robust, sigma=1e-8)

    # Sort eigenvalues and eigenvectors in ascending order
    sort_indices = np.argsort(evals_robust)
    evals_robust = evals_robust[sort_indices]
    evecs_robust = evecs_robust[:, sort_indices]

    # Extract vertex areas from mass matrix diagonal
    vertex_areas = M_robust.diagonal()

    return evecs_robust, evals_robust, vertex_areas


# =============================================================================
# Potential Functions for Schrödinger Operator
# =============================================================================

def compute_potential(
        vertices: np.ndarray,
        potential_type: str,
        n_neighbors: int = 30,
        normalize: bool = True
) -> np.ndarray:
    """
    Compute potential function V(x) for Schrödinger operator.

    The potential determines where eigenfunctions will concentrate:
    - Eigenfunctions concentrate where V is LOW
    - Eigenfunctions are suppressed where V is HIGH

    Args:
        vertices: Vertex positions [num_vertices, 3]
        potential_type: Type of potential:
            - "curvature": V = local curvature (eigenfuncs avoid sharp regions)
            - "inverse_curvature": V = -curvature (eigenfuncs at sharp features)
            - "center_distance": V = distance from centroid (eigenfuncs at center)
            - "height": V = z-coordinate (eigenfuncs at bottom)
            - "inverse_height": V = -z (eigenfuncs at top)
            - "random": V = random values (for testing)
        n_neighbors: Number of neighbors for curvature estimation
        normalize: Whether to normalize V to [0, 1]

    Returns:
        V: [num_vertices] potential values
    """
    num_vertices = len(vertices)

    if potential_type == "random":
        V = np.random.rand(num_vertices)

    elif potential_type == "curvature":
        V = _estimate_curvature_numpy(vertices, n_neighbors)

    elif potential_type == "inverse_curvature":
        curv = _estimate_curvature_numpy(vertices, n_neighbors)
        V = curv.max() - curv

    elif potential_type == "center_distance":
        centroid = vertices.mean(axis=0)
        V = np.linalg.norm(vertices - centroid, axis=1)

    elif potential_type == "height":
        V = vertices[:, 2] - vertices[:, 2].min()

    elif potential_type == "inverse_height":
        V = vertices[:, 2].max() - vertices[:, 2]

    else:
        raise ValueError(f"Unknown potential type: {potential_type}")

    # Normalize to [0, 1]
    if normalize and V.max() > V.min():
        V = (V - V.min()) / (V.max() - V.min())

    return V.astype(np.float64)


def _estimate_curvature_numpy(
        vertices: np.ndarray,
        n_neighbors: int = 30
) -> np.ndarray:
    """
    Estimate local curvature from point cloud using PCA of local neighborhood.

    Curvature is estimated as the ratio of smallest to largest eigenvalue
    of the local covariance matrix. High ratio = high curvature.

    Args:
        vertices: [num_vertices, 3] point positions
        n_neighbors: Number of neighbors for local PCA

    Returns:
        curvature: [num_vertices] curvature estimate (higher = more curved)
    """
    num_vertices = len(vertices)

    # Build kNN
    tree = cKDTree(vertices)
    _, indices = tree.query(vertices, k=n_neighbors + 1)  # +1 for self

    curvature = np.zeros(num_vertices)

    for i in range(num_vertices):
        # Get neighbors (exclude self)
        neighbor_indices = indices[i, 1:]
        neighbors = vertices[neighbor_indices]

        # Center at vertex i
        centered = neighbors - vertices[i]

        # Covariance matrix
        cov = centered.T @ centered / len(neighbors)

        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)

        # Curvature ~ ratio of smallest to largest eigenvalue
        ratio = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        curvature[i] = ratio

    return curvature


def _compute_schrodinger_base(
        vertices: np.ndarray,
        num_eigenfunctions: int,
        n_neighbors: int,
        potential_type: str,
        potential_strength: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core Schrödinger operator computation: H = L + β·diag(V)

    Solves generalized eigenvalue problem: H·φ = λ·M·φ

    Args:
        vertices: Vertex positions [num_vertices, 3]
        num_eigenfunctions: Number of eigenfunctions to compute
        n_neighbors: Number of neighbors for k-NN graph construction
        potential_type: Type of potential function
        potential_strength: β coefficient for potential term

    Returns:
        Tuple of (eigenvectors, eigenvalues, vertex_areas, potential)
    """
    # Compute robust Laplacian
    L_robust, M_robust = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=n_neighbors)

    # Compute potential
    V = compute_potential(vertices, potential_type, n_neighbors=n_neighbors, normalize=True)

    # Form Schrödinger operator: H = L + β·diag(V)
    H = L_robust + sparse.diags(potential_strength * V)

    # Solve generalized eigenvalue problem: H·φ = λ·M·φ
    evals, evecs = sla.eigsh(H, num_eigenfunctions, M_robust, sigma=1e-8)

    # Sort eigenvalues and eigenvectors in ascending order
    sort_indices = np.argsort(evals)
    evals = evals[sort_indices]
    evecs = evecs[:, sort_indices]

    # Extract vertex areas from mass matrix diagonal
    vertex_areas = M_robust.diagonal()

    return evecs, evals, vertex_areas, V


def compute_robust_schrodinger_eigenvectors(
        vertices: np.ndarray,
        num_eigenfunctions: int = 100,
        n_neighbors: int = 30,
        potential_type: str = "curvature",
        potential_strength: float = 5.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Schrödinger operator eigenvectors: H = -Δ + V(x)

    The eigenfunctions will concentrate where potential V is LOW.

    Args:
        vertices: Vertex positions [num_vertices, 3]
        num_eigenfunctions: Number of eigenfunctions to compute
        n_neighbors: Number of neighbors for k-NN graph construction
        potential_type: Type of potential function
        potential_strength: β coefficient for potential term

    Returns:
        Tuple of (normalized_eigenvectors, eigenvalues, vertex_areas, potential) as torch tensors
    """
    # Compute base eigendecomposition
    eigenvectors, eigenvalues, vertex_areas, potential = _compute_schrodinger_base(
        vertices, num_eigenfunctions, n_neighbors, potential_type, potential_strength
    )

    # Convert to torch tensors
    eigenvectors_torch = torch.from_numpy(eigenvectors).float()
    eigenvalues_torch = torch.from_numpy(eigenvalues).float()
    vertex_areas_torch = torch.from_numpy(vertex_areas).float()
    potential_torch = torch.from_numpy(potential).float()

    # Apply normalization
    normalized_eigenvectors = _normalize_eigenvectors(eigenvectors_torch, vertex_areas_torch)

    return normalized_eigenvectors, eigenvalues_torch, vertex_areas_torch, potential_torch


def _normalize_eigenvectors(eigenvectors: torch.Tensor, vertex_areas: torch.Tensor) -> torch.Tensor:
    """
    Apply normalization to eigenvectors using the existing scale_by_half function.

    Args:
        eigenvectors: Eigenvectors to normalize [num_vertices, num_eigenfunctions]
        vertex_areas: Vertex area weights [num_vertices]

    Returns:
        Normalized eigenvectors
    """
    # Apply the same normalization used in your existing pipeline
    normalized_eigenvectors = scale_by_half(scalar_functions=eigenvectors, weights=vertex_areas)
    return normalized_eigenvectors


# =============================================================================
# PyFM-based Functions
# =============================================================================

def compute_pyfm_unnormalized_laplacian_eigenfunctions(vertices: np.ndarray,
                                                       faces: np.ndarray,
                                                       num_eigenfunctions: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute unnormalized Laplacian eigenfunctions using PyFM.

    Args:
        vertices: Vertex positions [num_vertices, 3]
        faces: Face connectivity [num_faces, 3]
        num_eigenfunctions: Number of eigenfunctions to compute

    Returns:
        Tuple of (eigenvectors, eigenvalues, vertex_areas) as torch tensors
    """
    if faces is None:
        raise ValueError("PyFM methods require face connectivity. Use robust methods for point clouds.")

    # Compute base eigendecomposition
    eigenvectors, eigenvalues, vertex_areas, _ = _compute_pyfm_base(vertices, faces, num_eigenfunctions)

    # Convert to torch tensors
    eigenvectors_torch = torch.from_numpy(eigenvectors).float()
    eigenvalues_torch = torch.from_numpy(eigenvalues).float()
    vertex_areas_torch = torch.from_numpy(vertex_areas).float()

    return eigenvectors_torch, eigenvalues_torch, vertex_areas_torch


def compute_pyfm_normalized_laplacian_eigenfunctions(vertices: np.ndarray,
                                                     faces: np.ndarray,
                                                     num_eigenfunctions: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute normalized Laplacian eigenfunctions using PyFM.

    Args:
        vertices: Vertex positions [num_vertices, 3]
        faces: Face connectivity [num_faces, 3]
        num_eigenfunctions: Number of eigenfunctions to compute

    Returns:
        Tuple of (normalized_eigenvectors, eigenvalues, vertex_areas) as torch tensors
    """
    if faces is None:
        raise ValueError("PyFM methods require face connectivity. Use robust methods for point clouds.")

    # Compute base eigendecomposition
    eigenvectors, eigenvalues, vertex_areas, _ = _compute_pyfm_base(vertices, faces, num_eigenfunctions)

    # Convert to torch tensors
    eigenvectors_torch = torch.from_numpy(eigenvectors).float()
    eigenvalues_torch = torch.from_numpy(eigenvalues).float()
    vertex_areas_torch = torch.from_numpy(vertex_areas).float()

    # Apply normalization
    normalized_eigenvectors = _normalize_eigenvectors(eigenvectors_torch, vertex_areas_torch)

    return normalized_eigenvectors, eigenvalues_torch, vertex_areas_torch


# =============================================================================
# Robust Laplacian Functions
# =============================================================================

def compute_robust_unnormalized_laplacian_eigenvectors(vertices: np.ndarray,
                                                       num_eigenfunctions: int = 100,
                                                       n_neighbors: int = 30) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute unnormalized Laplacian eigenvectors using robust Laplacian method.

    Args:
        vertices: Vertex positions [num_vertices, 3]
        num_eigenfunctions: Number of eigenfunctions to compute
        n_neighbors: Number of neighbors for k-NN graph construction

    Returns:
        Tuple of (eigenvectors, eigenvalues, vertex_areas) as torch tensors
    """
    # Compute base eigendecomposition
    eigenvectors, eigenvalues, vertex_areas = _compute_robust_base(vertices, num_eigenfunctions, n_neighbors)

    # Convert to torch tensors
    eigenvectors_torch = torch.from_numpy(eigenvectors).float()
    eigenvalues_torch = torch.from_numpy(eigenvalues).float()
    vertex_areas_torch = torch.from_numpy(vertex_areas).float()

    return eigenvectors_torch, eigenvalues_torch, vertex_areas_torch


def compute_robust_normalized_laplacian_eigenvectors(vertices: np.ndarray,
                                                     num_eigenfunctions: int = 100,
                                                     n_neighbors: int = 30) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute normalized Laplacian eigenvectors using robust Laplacian method.

    Args:
        vertices: Vertex positions [num_vertices, 3]
        num_eigenfunctions: Number of eigenfunctions to compute
        n_neighbors: Number of neighbors for k-NN graph construction

    Returns:
        Tuple of (normalized_eigenvectors, eigenvalues, vertex_areas) as torch tensors
    """
    # Compute base eigendecomposition
    eigenvectors, eigenvalues, vertex_areas = _compute_robust_base(vertices, num_eigenfunctions, n_neighbors)

    # Convert to torch tensors
    eigenvectors_torch = torch.from_numpy(eigenvectors).float()
    eigenvalues_torch = torch.from_numpy(eigenvalues).float()
    vertex_areas_torch = torch.from_numpy(vertex_areas).float()

    # Apply normalization
    normalized_eigenvectors = _normalize_eigenvectors(eigenvectors_torch, vertex_areas_torch)

    return normalized_eigenvectors, eigenvalues_torch, vertex_areas_torch


def orthogonalize_columns(X: torch.Tensor) -> torch.Tensor:
    """Make columns orthogonal but keep their magnitudes."""
    Q = X.clone()
    n_cols = Q.shape[1]

    for i in range(1, n_cols):  # Start from second column
        for j in range(i):
            # Remove projection onto previous columns
            proj = (Q[:, i] @ Q[:, j]) / (Q[:, j] @ Q[:, j] + 1e-8)
            Q[:, i] = Q[:, i] - proj * Q[:, j]

    return Q


def compute_distances(pos: torch.Tensor, edge_index: torch.Tensor, use_cosine: bool) -> torch.Tensor:
    """
    Compute distances for edges based on metric.

    Args:
        pos: Node features [n_nodes, feature_dim]
        edge_index: Edge connectivity [2, n_edges]
        use_cosine: Whether to use cosine distance (True) or Euclidean (False)

    Returns:
        Distances for each edge [n_edges]
    """
    i, j = edge_index[0], edge_index[1]

    if use_cosine:
        # Cosine distance = 1 - cosine_similarity
        cos_sim = F.cosine_similarity(pos[i], pos[j])
        distances = 1 - cos_sim
    else:
        # Euclidean distance
        distances = torch.norm(pos[i] - pos[j], dim=1)

    return distances


def load_cached_file_list(
        root_dirs: Union[List[Union[str, Path]], str, Path],
        cache_basename: str = 'file_scan_cache',
        prefer_pickle: bool = True,
        validate_exists: bool = False
) -> List[Path]:
    """
    Load cached file lists from multiple root directories.

    Args:
        root_dirs: Single directory or list of directories containing cache files
        cache_basename: Base name of cache files (default: 'file_scan_cache')
        prefer_pickle: If True, try pickle first (faster), else try JSON first
        validate_exists: If True, filter out files that no longer exist

    Returns:
        List of Path objects for all cached files across all directories

    Raises:
        FileNotFoundError: If no cache file is found in a directory

    Examples:
        # Load from single directory
        files = load_cached_file_list('/path/to/data')

        # Load from multiple directories
        files = load_cached_file_list(['/path/to/data1', '/path/to/data2'])

        # Prefer JSON format
        files = load_cached_file_list('/path/to/data', prefer_pickle=False)

        # Validate that files still exist
        files = load_cached_file_list('/path/to/data', validate_exists=True)
    """
    # Normalize input to list of Paths
    if isinstance(root_dirs, (str, Path)):
        root_dirs = [Path(root_dirs)]
    else:
        root_dirs = [Path(d) for d in root_dirs]

    all_file_paths = []

    for root_dir in root_dirs:
        if not root_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {root_dir}")

        # Determine which cache file to try first
        pickle_cache = root_dir / f"{cache_basename}.pkl"
        json_cache = root_dir / f"{cache_basename}.json"

        cache_files = [pickle_cache, json_cache] if prefer_pickle else [json_cache, pickle_cache]

        loaded = False
        for cache_file in cache_files:
            if cache_file.exists():
                try:
                    if cache_file.suffix == '.pkl':
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                    else:  # JSON
                        with open(cache_file, 'r') as f:
                            data = json.load(f)

                    # Extract file paths (handle both string and Path objects)
                    cached_files = data.get('files', [])
                    file_paths = [Path(f) if isinstance(f, str) else f for f in cached_files]

                    # Optionally validate that files still exist
                    if validate_exists:
                        file_paths = [f for f in file_paths if f.exists()]

                    all_file_paths.extend(file_paths)
                    loaded = True
                    print(f"Loaded {len(file_paths)} files from cache: {cache_file}")
                    break

                except Exception as e:
                    print(f"Warning: Failed to load cache {cache_file}: {e}")
                    continue

        if not loaded:
            raise FileNotFoundError(
                f"No valid cache file found in {root_dir}. "
                f"Expected: {cache_basename}.pkl or {cache_basename}.json"
            )

    return all_file_paths


# ============================================================================
# Manifold Learning and Clustering Utilities
# ============================================================================

def compute_pca_embedding(
        features: np.ndarray,
        n_components: int,
        random_state: int = 42
) -> np.ndarray:
    """
    Compute PCA embedding.

    Args:
        features: Input features [n_samples, feature_dim]
        n_components: Target dimensionality
        random_state: Random seed for reproducibility

    Returns:
        PCA embedding [n_samples, n_components]
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(features)

    return embedding


def compute_umap_embedding(
        features: np.ndarray,
        n_components: int,
        k_neighbors: int,
        use_cosine: bool = False,
        min_dist: float = 0.1,
        random_state: Optional[int] = None
) -> np.ndarray:
    """
    Compute UMAP embedding.

    Args:
        features: Input features [n_samples, feature_dim]
        n_components: Target dimensionality
        k_neighbors: Number of neighbors for UMAP
        use_cosine: Whether to use cosine distance instead of euclidean
        min_dist: UMAP min_dist parameter
        random_state: Random seed (None enables parallelism)

    Returns:
        UMAP embedding [n_samples, n_components]
    """
    from umap import UMAP

    metric = 'cosine' if use_cosine else 'euclidean'

    umap = UMAP(
        n_components=n_components,
        n_neighbors=k_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    embedding = umap.fit_transform(features)

    return embedding


def compute_tsne_embedding(
        features: np.ndarray,
        n_components: int,
        perplexity: int,
        random_state: int = 42,
        max_iter: int = 1000
) -> np.ndarray:
    """
    Compute t-SNE embedding.

    Args:
        features: Input features [n_samples, feature_dim]
        n_components: Target dimensionality
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
        max_iter: Maximum number of iterations

    Returns:
        t-SNE embedding [n_samples, n_components]
    """
    from sklearn.manifold import TSNE

    # t-SNE's barnes_hut algorithm only supports n_components <= 3
    # Use exact method for higher dimensions
    method = 'barnes_hut' if n_components <= 3 else 'exact'

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=max_iter,
        method=method
    )
    embedding = tsne.fit_transform(features)

    return embedding


def compute_isomap_embedding(
        features: np.ndarray,
        n_components: int,
        k_neighbors: int,
        use_cosine: bool = False
) -> np.ndarray:
    """
    Compute Isomap embedding.

    Args:
        features: Input features [n_samples, feature_dim]
        n_components: Target dimensionality
        k_neighbors: Number of neighbors for Isomap
        use_cosine: Whether to use cosine distance instead of euclidean

    Returns:
        Isomap embedding [n_samples, n_components]
    """
    from sklearn.manifold import Isomap

    metric = 'cosine' if use_cosine else 'euclidean'

    isomap = Isomap(
        n_components=n_components,
        n_neighbors=k_neighbors,
        metric=metric
    )
    embedding = isomap.fit_transform(features)

    return embedding


def compute_manifold_embeddings(
        features: np.ndarray,
        n_components: int,
        k_neighbors: int,
        methods: List[str],
        use_cosine: bool = False,
        min_dist: float = 0.1,
        random_state: Optional[int] = 42
) -> Dict[str, Optional[np.ndarray]]:
    """
    Apply multiple manifold learning methods to features.

    This function calls individual compute_*_embedding functions and handles
    exceptions, returning None for methods that fail.

    Args:
        features: Input features [n_samples, feature_dim]
        n_components: Target dimensionality
        k_neighbors: Number of neighbors (for methods that use it)
        methods: List of method names from {'pca', 'umap', 'tsne', 'isomap'}
        use_cosine: Whether to use cosine distance
        min_dist: UMAP min_dist parameter
        random_state: Random seed for reproducibility (None for UMAP enables parallelism)

    Returns:
        Dictionary mapping method names to embeddings [n_samples, n_components]
        or None if the method failed
    """
    embeddings = {}

    if 'pca' in methods:
        try:
            embeddings['pca'] = compute_pca_embedding(
                features=features,
                n_components=n_components,
                random_state=random_state if random_state is not None else 42
            )
        except Exception as e:
            print(f"PCA failed: {e}")
            embeddings['pca'] = None

    if 'umap' in methods:
        try:
            embeddings['umap'] = compute_umap_embedding(
                features=features,
                n_components=n_components,
                k_neighbors=k_neighbors,
                use_cosine=use_cosine,
                min_dist=min_dist,
                random_state=random_state
            )
        except Exception as e:
            print(f"UMAP failed: {e}")
            embeddings['umap'] = None

    if 'tsne' in methods:
        try:
            # Compute perplexity based on k_neighbors
            perplexity = min(max(5, k_neighbors), 50)
            embeddings['tsne'] = compute_tsne_embedding(
                features=features,
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state if random_state is not None else 42
            )
        except Exception as e:
            print(f"t-SNE failed: {e}")
            embeddings['tsne'] = None

    if 'isomap' in methods:
        try:
            embeddings['isomap'] = compute_isomap_embedding(
                features=features,
                n_components=n_components,
                k_neighbors=k_neighbors,
                use_cosine=use_cosine
            )
        except Exception as e:
            print(f"Isomap failed: {e}")
            embeddings['isomap'] = None

    return embeddings


def compute_kmeans_clustering(
        embeddings: np.ndarray,
        n_clusters: int,
        random_state: int = 42,
        n_init: int = 10
) -> np.ndarray:
    """
    Run k-means clustering and return predicted labels.

    Args:
        embeddings: Point embeddings [n_points, embedding_dim]
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        n_init: Number of k-means initializations

    Returns:
        Predicted cluster labels [n_points]
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init
    )
    predicted_labels = kmeans.fit_predict(embeddings)

    return predicted_labels


def compute_clustering_metrics(
        predicted_labels: np.ndarray,
        true_labels: np.ndarray
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Compute clustering quality metrics.

    Args:
        predicted_labels: Predicted cluster assignments [n_points]
        true_labels: Ground truth class labels [n_points]

    Returns:
        Tuple of (NMI, ARI, Completeness, AMI, Homogeneity, V-Measure, Fowlkes-Mallows)
    """
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
        completeness_score,
        adjusted_mutual_info_score,
        homogeneity_score,
        v_measure_score,
        fowlkes_mallows_score
    )

    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)
    ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    fmi = fowlkes_mallows_score(true_labels, predicted_labels)

    return nmi, ari, completeness, ami, homogeneity, v_measure, fmi


def compute_graph_laplacian_eigenvectors(
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        n_eigenvectors: int,
        use_cosine: bool
) -> torch.Tensor:
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
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    n_nodes = pos.shape[0]

    # Compute distances based on k-NN graph metric
    distances = compute_distances(pos=pos, edge_index=edge_index, use_cosine=use_cosine)

    # Gaussian weighting: exp(-dÃƒâ€šÃ‚Â²/ÃƒÂÃ†â€™Ãƒâ€šÃ‚Â²)
    sigma = torch.median(distances)
    weights = torch.exp(-distances ** 2 / (sigma ** 2 + 1e-10))

    # Convert to scipy sparse (COO format)
    edge_index_np = edge_index.cpu().numpy()
    weights_np = weights.cpu().numpy()

    W = sp.csr_matrix(
        (weights_np, (edge_index_np[0], edge_index_np[1])),
        shape=(n_nodes, n_nodes)
    )

    # Make symmetric (for undirected graph)
    W = (W + W.T) / 2

    # Compute degree matrix
    degrees = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees + 1e-10))

    # Normalized Laplacian: L = I - D^(-1/2) W D^(-1/2)
    L = sp.eye(n_nodes) - D_inv_sqrt @ W @ D_inv_sqrt

    # Compute eigenvectors (smallest eigenvalues)
    eigenvalues, eigenvectors = eigsh(L, k=n_eigenvectors, which='SM')

    return torch.from_numpy(eigenvectors).float().to(pos.device)


def generate_random_orthogonal_basis(
        n_points: int,
        n_eigenvectors: int,
        device: torch.device,
        random_state: Optional[int] = None
) -> torch.Tensor:
    """
    Generate random orthogonal basis using QR decomposition.

    Args:
        n_points: Number of points
        n_eigenvectors: Number of vectors in basis
        device: Device to create tensor on
        random_state: Random seed for reproducibility (None for non-deterministic)

    Returns:
        Random orthogonal matrix [n_points, n_eigenvectors]
    """
    if random_state is not None:
        # Save current state
        torch_state = torch.get_rng_state()
        # Set seed
        torch.manual_seed(random_state)

    random_matrix = torch.randn(n_points, n_eigenvectors, device=device)
    Q, R = torch.linalg.qr(random_matrix)

    if random_state is not None:
        # Restore state
        torch.set_rng_state(torch_state)

    return Q
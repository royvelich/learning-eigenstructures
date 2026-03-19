#!/usr/bin/env python3
"""
Standalone application for computing eigendecomposition cache.
Supports both Laplacian and Schrödinger operators.
Uses multiprocessing.Pool for better Linux compatibility.

Usage:
    # Laplacian (default, no operator config)
    python compute_eigen_cache.py --config-name=cache_config

    # Schrödinger - use a config file with operator defined:
    python compute_eigen_cache.py --config-name=cache_config_schrodinger

    # Config file example (cache_config_schrodinger.yaml):
    #   operator:
    #     _target_: neural_laplacian.configs.OperatorConfig
    #     type: schrodinger
    #     potential_type: curvature
    #     potential_strength: 5.0
"""

# standard library
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import multiprocessing as mp
from functools import partial
import random

# hydra
import hydra
from omegaconf import DictConfig, OmegaConf

# numpy
import numpy as np

# pymeshlab
import pymeshlab

# neural-laplacian
from neural_laplacian import utils
from neural_laplacian.configs import OperatorConfig

# progress tracking
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def compute_single_file_legacy(file_path: Path, config_dict: dict) -> Dict[str, Any]:
    """
    Process a single mesh file using the ORIGINAL Laplacian-only logic.
    This is the backward-compatible path when no operator config is specified.
    """
    try:
        print(f"[Worker] Starting: {file_path.name}")

        from omegaconf import DictConfig
        config = DictConfig(config_dict)

        print(f"[Worker] Preparing geometry: {file_path.name}")
        vertices, faces = utils.prepare_geometry(
            file_path=file_path,
            decimation_config=getattr(config, 'decimation_config', None)
        )
        print(f"[Worker] Prepared geometry {file_path.name}: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

        print(f"[Worker] Creating cache manager for: {file_path.name}")
        cache_manager = utils.create_eigen_cache_manager(
            cache_dir=config.cache_dir,
            enabled=True
        )

        print(f"[Worker] Checking cache for: {file_path.name}")
        gt_eigenvectors, gt_eigenvalues, gt_vertex_areas = cache_manager.load_eigendecomposition(
            file_path=file_path,
            vertices=vertices,
            faces=faces,
            num_eigenfunctions=config.num_eigenfunctions,
            decimation_config=getattr(config, 'decimation_config', None)
        )

        if gt_eigenvectors is not None:
            print(f"[Worker] Found in cache: {file_path.name} ({gt_eigenvectors.shape[1]} eigenfunctions)")
            return {'status': 'cached', 'file_path': str(file_path)}

        print(f"[Worker] Computing eigendecomposition for: {file_path.name}")
        start_time = time.time()
        gt_eigenvectors, gt_eigenvalues, gt_vertex_areas = cache_manager.compute_eigendecomposition(
            file_path=file_path,
            vertices=vertices,
            faces=faces,
            num_eigenfunctions=config.num_eigenfunctions,
            decimation_config=getattr(config, 'decimation_config', None),
            verbose=False
        )
        computation_time = time.time() - start_time

        if gt_eigenvectors is not None:
            print(f"[Worker] Completed: {file_path.name} ({computation_time:.1f}s, {gt_eigenvectors.shape[1]} eigenfunctions)")
            return {
                'status': 'computed',
                'file_path': str(file_path),
                'computation_time': computation_time,
                'num_eigenfunctions': gt_eigenvectors.shape[1]
            }
        else:
            print(f"[Worker] FAILED: {file_path.name} - Computation failed")
            return {'status': 'error', 'file_path': str(file_path), 'error': 'Computation failed'}

    except Exception as e:
        print(f"[Worker] ERROR: {file_path.name} - {str(e)}")
        return {'status': 'error', 'file_path': str(file_path), 'error': str(e)}


def compute_single_file_with_operator(file_path: Path, config_dict: dict, operator_dict: dict) -> Dict[str, Any]:
    """
    Process a single mesh file with explicit operator configuration.
    Supports both Laplacian and Schrödinger operators.

    Args:
        file_path: Path to mesh file
        config_dict: Hydra config as dict
        operator_dict: OperatorConfig fields as dict (for multiprocessing pickling)
    """
    try:
        print(f"[Worker] Starting: {file_path.name}")

        from omegaconf import DictConfig
        config = DictConfig(config_dict)

        # Recreate OperatorConfig from dict (multiprocessing requirement)
        operator_config = OperatorConfig(
            type=operator_dict['type'],
            potential_type=operator_dict.get('potential_type'),
            potential_strength=operator_dict.get('potential_strength'),
            n_neighbors=operator_dict.get('n_neighbors', 30)
        )

        operator_type = operator_config.type
        potential_type = operator_config.potential_type
        potential_strength = operator_config.potential_strength
        n_neighbors = operator_config.n_neighbors

        print(f"[Worker] Preparing geometry: {file_path.name}")
        vertices, faces = utils.prepare_geometry(
            file_path=file_path,
            decimation_config=getattr(config, 'decimation_config', None)
        )
        print(f"[Worker] Prepared geometry {file_path.name}: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

        print(f"[Worker] Creating cache manager for: {file_path.name}")
        cache_manager = utils.create_eigen_cache_manager(
            cache_dir=config.cache_dir,
            enabled=True
        )

        print(f"[Worker] Checking cache for: {file_path.name} (operator={operator_type}, n_neighbors={n_neighbors})")
        gt_eigenvectors, gt_eigenvalues, gt_vertex_areas, gt_potential = cache_manager.load_operator_eigendecomposition(
            file_path=file_path,
            vertices=vertices,
            faces=faces,
            num_eigenfunctions=config.num_eigenfunctions,
            operator_type=operator_type,
            potential_type=potential_type,
            potential_strength=potential_strength,
            n_neighbors=n_neighbors,
            decimation_config=getattr(config, 'decimation_config', None)
        )

        if gt_eigenvectors is not None:
            print(f"[Worker] Found in cache: {file_path.name} ({gt_eigenvectors.shape[1]} eigenfunctions)")
            return {'status': 'cached', 'file_path': str(file_path)}

        print(f"[Worker] Computing {operator_type} eigendecomposition for: {file_path.name}")
        start_time = time.time()
        gt_eigenvectors, gt_eigenvalues, gt_vertex_areas, gt_potential = cache_manager.compute_operator_eigendecomposition(
            file_path=file_path,
            vertices=vertices,
            faces=faces,
            num_eigenfunctions=config.num_eigenfunctions,
            operator_type=operator_type,
            potential_type=potential_type,
            potential_strength=potential_strength,
            n_neighbors=n_neighbors,
            decimation_config=getattr(config, 'decimation_config', None),
            verbose=False
        )
        computation_time = time.time() - start_time

        if gt_eigenvectors is not None:
            potential_info = ""
            if operator_type == "schrodinger":
                potential_info = f", potential={potential_type}, β={potential_strength}, k={n_neighbors}"
            print(f"[Worker] Completed: {file_path.name} ({computation_time:.1f}s, {gt_eigenvectors.shape[1]} eigenfunctions{potential_info})")
            return {
                'status': 'computed',
                'file_path': str(file_path),
                'computation_time': computation_time,
                'num_eigenfunctions': gt_eigenvectors.shape[1],
                'operator_type': operator_type
            }
        else:
            print(f"[Worker] FAILED: {file_path.name} - Computation failed")
            return {'status': 'error', 'file_path': str(file_path), 'error': 'Computation failed'}

    except Exception as e:
        print(f"[Worker] ERROR: {file_path.name} - {str(e)}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'file_path': str(file_path), 'error': str(e)}


def compute_single_file(file_path: Path, config_dict: dict, operator_dict: Optional[dict] = None) -> Dict[str, Any]:
    """
    Process a single mesh file. Routes to legacy or operator-aware method.

    Args:
        file_path: Path to mesh file
        config_dict: Hydra config as dict
        operator_dict: OperatorConfig as dict, or None for legacy Laplacian-only flow
    """
    if operator_dict is None:
        # Backward compatible: use original Laplacian-only logic
        return compute_single_file_legacy(file_path, config_dict)
    else:
        # New: use operator-aware logic
        return compute_single_file_with_operator(file_path, config_dict, operator_dict)


@hydra.main(version_base="1.2", config_path="config/caching", config_name=None)
def main(config: DictConfig) -> None:
    """Main function."""

    # Try to instantiate operator config using hydra if present
    operator_config: Optional[OperatorConfig] = None
    operator_dict: Optional[dict] = None

    if 'operator' in config and config.operator is not None:
        # Use hydra.utils.instantiate to create OperatorConfig
        operator_config = hydra.utils.instantiate(config.operator)
        # Convert to dict for multiprocessing (dataclass is not always picklable)
        operator_dict = {
            'type': operator_config.type,
            'potential_type': operator_config.potential_type,
            'potential_strength': operator_config.potential_strength,
            'n_neighbors': operator_config.n_neighbors
        }

    print("Computing eigendecomposition cache...")
    print(f"Cache directory: {config.cache_dir}")
    print(f"Number of eigenfunctions: {config.num_eigenfunctions}")
    print(f"Workers: {config.num_workers}")

    if operator_config is None:
        print(f"Operator: laplacian (default)")
    else:
        print(f"Operator: {operator_config}")

    # Find mesh files
    print("Scanning for mesh files...")
    min_file_size_mb = getattr(config, 'min_file_size_mb', 0.0)
    max_file_size_mb = getattr(config, 'max_file_size_mb', float('inf'))
    mesh_files = utils.scan_files(
        root_dirs=[Path(folder) for folder in config.mesh_folders],
        file_size=tuple([min_file_size_mb, max_file_size_mb]),
        max_items=getattr(config, 'max_files', None),
        file_extensions=['*.obj', '*.ply', '*.off', '*.stl']
    )

    if not mesh_files:
        print("No mesh files found!")
        return

    print(f"Found {len(mesh_files)} mesh files to process")
    print(f"First 5 files: {[f.name for f in mesh_files[:5]]}")

    # Initialize stats
    stats = {'total': 0, 'computed': 0, 'cached': 0, 'error': 0}
    start_time = time.time()

    # Convert config to dict for multiprocessing
    config_dict = OmegaConf.to_container(config, resolve=True)
    print(f"Config dict keys: {list(config_dict.keys())}")

    # Process files
    if config.num_workers > 1:
        print(f"Setting up multiprocessing with {config.num_workers} workers...")

        # Set start method for Linux compatibility
        print("Setting multiprocessing start method to 'spawn'...")
        try:
            mp.set_start_method('spawn', force=True)
            print("Successfully set start method to 'spawn'")
        except RuntimeError as e:
            print(f"Start method already set: {e}")

        try:
            print("Creating multiprocessing pool...")
            with mp.Pool(processes=config.num_workers) as pool:
                print(f"Pool created with {config.num_workers} processes")

                # Create partial function with config and operator_dict
                print("Creating worker function...")
                worker_func = partial(compute_single_file, config_dict=config_dict, operator_dict=operator_dict)

                print("Starting parallel processing...")
                if HAS_TQDM:
                    # Process with progress bar and timeout handling
                    print("Using tqdm progress bar with timeout handling")
                    results = []

                    # Use imap_unordered for better error handling
                    result_iter = pool.imap_unordered(worker_func, mesh_files, chunksize=20)

                    with tqdm(total=len(mesh_files), desc="Processing") as pbar:
                        for i, result in enumerate(result_iter):
                            results.append(result)
                            pbar.update(1)

                            # Print progress every 50 items
                            if (i + 1) % 50 == 0:
                                print(f"\nProcessed {i + 1}/{len(mesh_files)} files")
                                print(f"Last result: {result['status']} - {Path(result['file_path']).name}")

                            # Check for hanging workers every 100 items
                            if (i + 1) % 100 == 0:
                                print(f"Checkpoint at {i + 1} files - checking pool health...")
                else:
                    # Process without progress bar
                    print("Processing without progress bar")
                    results = []
                    result_iter = pool.imap_unordered(worker_func, mesh_files, chunksize=1)

                    for i, result in enumerate(result_iter):
                        results.append(result)
                        if (i + 1) % 50 == 0:
                            print(f"Processed {i + 1}/{len(mesh_files)} files")

                print("Parallel processing completed")

        except Exception as e:
            print(f"Multiprocessing failed with error: {e}")
            print("Error type:", type(e).__name__)
            import traceback
            print("Full traceback:")
            traceback.print_exc()

            print("Falling back to sequential processing...")
            results = []
            files = tqdm(mesh_files, desc="Processing (sequential)") if HAS_TQDM else mesh_files
            for i, file_path in enumerate(files):
                print(f"Sequential processing {i + 1}/{len(mesh_files)}: {file_path.name}")
                result = compute_single_file(file_path, config_dict, operator_dict)
                results.append(result)
                print(f"Sequential result: {result['status']}")
    else:
        # Sequential processing
        print("Processing sequentially (num_workers = 1)...")
        results = []
        files = tqdm(mesh_files, desc="Processing") if HAS_TQDM else mesh_files
        for i, file_path in enumerate(files):
            print(f"Sequential processing {i + 1}/{len(mesh_files)}: {file_path.name}")
            result = compute_single_file(file_path, config_dict, operator_dict)
            results.append(result)
            print(f"Sequential result: {result['status']}")

    print("Processing completed, counting results...")
    # Count results
    for result in results:
        stats['total'] += 1
        stats[result['status']] += 1

    print("Results counted, generating summary...")
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nSummary:")
    print(f"Total processed: {stats['total']}")
    print(f"Computed: {stats['computed']}")
    print(f"Already cached: {stats['cached']}")
    print(f"Errors: {stats['error']}")
    print(f"Time: {elapsed_time:.1f}s")

    if stats['computed'] > 0:
        print(f"Avg time per computation: {elapsed_time / stats['computed']:.1f}s")


if __name__ == "__main__":
    main()
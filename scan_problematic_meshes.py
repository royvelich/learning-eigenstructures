#!/usr/bin/env python3
"""
Standalone application for scanning mesh files and identifying problematic ones.
Outputs a text file with full paths of files that couldn't be opened.

Usage:
    python scan_problematic_meshes.py --config-name=cache_config
    or
    python scan_problematic_meshes.py --config-name=cache_config --output=problematic_files.txt
"""

# standard library
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import multiprocessing as mp
from functools import partial
import argparse
from datetime import datetime

# hydra
import hydra
from omegaconf import DictConfig

# numpy
import numpy as np

# pymeshlab
import pymeshlab

# neural-laplacian
from neural_laplacian import utils

# progress tracking
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def test_single_file(file_path: Path) -> Dict[str, Any]:
    """Test if a single mesh file can be opened and processed."""
    result = {
        'file_path': str(file_path),
        'status': 'unknown',
        'error': None,
        'vertices': 0,
        'faces': 0
    }

    try:
        # Try to load with pymeshlab directly first
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(file_path))

        # Get basic mesh info
        vertices = ms.current_mesh().vertex_matrix()
        faces = ms.current_mesh().face_matrix()

        result['vertices'] = vertices.shape[0]
        result['faces'] = faces.shape[0]

        # Now try the full geometry preparation pipeline
        vertices_processed, faces_processed = utils.prepare_geometry(
            file_path=file_path,
            decimation_config=None
        )

        # Check for degenerate cases
        if vertices_processed.shape[0] < 4:
            result['status'] = 'error'
            result['error'] = f"Too few vertices after processing: {vertices_processed.shape[0]}"
        elif faces_processed.shape[0] < 1:
            result['status'] = 'error'
            result['error'] = f"No faces after processing"
        else:
            result['status'] = 'success'

    except FileNotFoundError as e:
        result['status'] = 'error'
        result['error'] = f"File not found: {str(e)}"
    except PermissionError as e:
        result['status'] = 'error'
        result['error'] = f"Permission denied: {str(e)}"
    except pymeshlab.PyMeshLabException as e:
        result['status'] = 'error'
        result['error'] = f"PyMeshLab error: {str(e)}"
    except Exception as e:
        result['status'] = 'error'
        result['error'] = f"Unexpected error: {type(e).__name__}: {str(e)}"

    return result


def scan_files_sequential(mesh_files: List[Path]) -> List[Dict[str, Any]]:
    """Process files sequentially."""
    results = []

    files = tqdm(mesh_files, desc="Scanning") if HAS_TQDM else mesh_files

    for i, file_path in enumerate(files):
        if not HAS_TQDM and (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(mesh_files)} files")

        result = test_single_file(file_path)
        results.append(result)

        if result['status'] == 'error' and not HAS_TQDM:
            print(f"ERROR: {file_path.name} - {result['error']}")

    return results


def scan_files_parallel(mesh_files: List[Path], num_workers: int) -> List[Dict[str, Any]]:
    """Process files in parallel."""
    results = []

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    try:
        with mp.Pool(processes=num_workers) as pool:
            if HAS_TQDM:
                result_iter = pool.imap_unordered(test_single_file, mesh_files, chunksize=20)

                with tqdm(total=len(mesh_files), desc="Scanning") as pbar:
                    for result in result_iter:
                        results.append(result)
                        pbar.update(1)

                        if result['status'] == 'error':
                            pbar.set_postfix_str(f"Last error: {Path(result['file_path']).name}")
            else:
                result_iter = pool.imap_unordered(test_single_file, mesh_files, chunksize=20)

                for i, result in enumerate(result_iter):
                    results.append(result)

                    if (i + 1) % 100 == 0:
                        print(f"Processed {i + 1}/{len(mesh_files)} files")

                    if result['status'] == 'error':
                        print(f"ERROR: {Path(result['file_path']).name} - {result['error']}")

    except Exception as e:
        print(f"Parallel processing failed: {e}")
        print("Falling back to sequential processing...")
        return scan_files_sequential(mesh_files)

    return results


def write_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Write problematic files to output file."""
    problematic_files = [r for r in results if r['status'] == 'error']

    with open(output_file, 'w') as f:
        # Write header
        f.write(f"# Problematic Mesh Files Report\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total files scanned: {len(results)}\n")
        f.write(f"# Problematic files: {len(problematic_files)}\n")
        f.write("#" * 80 + "\n\n")

        # Write problematic files with error details
        for result in problematic_files:
            f.write(f"FILE: {result['file_path']}\n")
            f.write(f"ERROR: {result['error']}\n")
            f.write("-" * 40 + "\n")

    # Also create a simple list version
    simple_output = output_file.replace('.txt', '_list.txt')
    with open(simple_output, 'w') as f:
        for result in problematic_files:
            f.write(f"{result['file_path']}\n")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary statistics."""
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    errors = sum(1 for r in results if r['status'] == 'error')

    print("\n" + "=" * 60)
    print("SCAN SUMMARY")
    print("=" * 60)
    print(f"Total files scanned:     {total}")
    print(f"Successfully opened:     {successful} ({100 * successful / total:.1f}%)")
    print(f"Failed to open:          {errors} ({100 * errors / total:.1f}%)")

    if errors > 0:
        print("\nError breakdown:")
        error_types = {}
        for r in results:
            if r['status'] == 'error' and r['error']:
                # Categorize error types
                if 'PyMeshLab error' in r['error']:
                    error_type = 'PyMeshLab error'
                elif 'Too few vertices' in r['error']:
                    error_type = 'Degenerate mesh (too few vertices)'
                elif 'No faces' in r['error']:
                    error_type = 'Degenerate mesh (no faces)'
                elif 'File not found' in r['error']:
                    error_type = 'File not found'
                elif 'Permission denied' in r['error']:
                    error_type = 'Permission denied'
                else:
                    error_type = 'Other'

                error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error_type}: {count}")

    print("=" * 60)


@hydra.main(version_base="1.2", config_path="config/caching", config_name=None)
def main(config: DictConfig) -> None:
    """Main function."""
    # Parse additional command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='problematic_mesh_files.txt',
                        help='Output file for problematic files list')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override number of workers from config')
    args, _ = parser.parse_known_args()

    output_file = args.output
    num_workers = args.num_workers if args.num_workers is not None else config.get('num_workers', 8)

    print("Scanning mesh files for problems...")
    print(f"Output file: {output_file}")
    print(f"Workers: {num_workers}")

    # Find mesh files
    print("\nScanning directories:")
    for folder in config.mesh_folders:
        print(f"  - {folder}")

    mesh_files = utils.scan_files(
        root_dirs=[Path(folder) for folder in config.mesh_folders],
        min_file_size_mb=config.get('min_file_size_mb', 0.0),
        max_file_size_mb=config.get('max_file_size_mb', float('inf')),
        max_items=config.get('max_files', None),
        file_extensions=['*.obj', '*.ply', '*.off', '*.stl']
    )

    if not mesh_files:
        print("No mesh files found!")
        return

    print(f"\nFound {len(mesh_files)} mesh files to scan")

    # Process files
    start_time = time.time()

    if num_workers > 1:
        print(f"Using parallel processing with {num_workers} workers...")
        results = scan_files_parallel(mesh_files, num_workers)
    else:
        print("Using sequential processing...")
        results = scan_files_sequential(mesh_files)

    elapsed_time = time.time() - start_time

    # Write results to file
    write_results(results, output_file)

    # Print summary
    print_summary(results)

    print(f"\nScan completed in {elapsed_time:.1f} seconds")
    print(f"Results written to: {output_file}")
    print(f"Simple list written to: {output_file.replace('.txt', '_list.txt')}")

    # Print first few problematic files as examples
    problematic = [r for r in results if r['status'] == 'error']
    if problematic and len(problematic) <= 10:
        print("\nProblematic files:")
        for r in problematic[:10]:
            print(f"  - {Path(r['file_path']).name}: {r['error']}")
    elif problematic:
        print(f"\nShowing first 10 of {len(problematic)} problematic files:")
        for r in problematic[:10]:
            print(f"  - {Path(r['file_path']).name}: {r['error']}")


if __name__ == "__main__":
    main()
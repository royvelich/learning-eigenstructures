#!/usr/bin/env python3
"""
Standalone application for scanning directories and caching the file lists.

Scans each root directory separately and saves the list of found files
as a cache file in that directory for later reuse.

Usage:
    python cache_file_scan.py --root-dirs /path/to/data1 /path/to/data2 --extensions *.obj *.ply
    python cache_file_scan.py --root-dirs /path/to/data --min-size 0.1 --max-size 10.0
    python cache_file_scan.py --root-dirs /path/to/data --max-items 1000
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

# Import the scan_files function from neural_laplacian utils
from neural_laplacian import utils


def save_file_list_json(file_paths: List[Path], output_path: Path, metadata: dict) -> None:
    """Save file list as JSON (human-readable)."""
    data = {
        'metadata': metadata,
        'files': [str(p) for p in file_paths]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved JSON cache: {output_path}")


def save_file_list_pickle(file_paths: List[Path], output_path: Path, metadata: dict) -> None:
    """Save file list as pickle (faster to load)."""
    data = {
        'metadata': metadata,
        'files': file_paths
    }
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved pickle cache: {output_path}")


def scan_and_cache_directory(
        root_dir: Path,
        file_extensions: List[str],
        file_size: Optional[Tuple[Optional[float], Optional[float]]],
        max_items: Optional[int],
        cache_basename: str,
        save_json: bool,
        save_pickle: bool
) -> List[Path]:
    """
    Scan a single directory and save the results to cache files.

    Args:
        root_dir: Directory to scan
        file_extensions: List of file extensions to search for
        file_size: Tuple of (min_mb, max_mb) for file size filtering
        max_items: Maximum number of items to include
        cache_basename: Base name for cache files
        save_json: Whether to save JSON cache
        save_pickle: Whether to save pickle cache

    Returns:
        List of found file paths
    """
    print(f"\nScanning directory: {root_dir}")

    if not root_dir.exists():
        print(f"  Warning: Directory does not exist, skipping...")
        return []

    # Scan files using utils.scan_files
    file_paths = utils.scan_files(
        root_dirs=[root_dir],
        file_size=file_size,
        max_items=max_items,
        file_extensions=file_extensions
    )

    if not file_paths:
        print(f"  No files found matching criteria")
        return []

    print(f"  Found {len(file_paths)} files")

    # Prepare metadata
    metadata = {
        'root_dir': str(root_dir),
        'scan_date': datetime.now().isoformat(),
        'num_files': len(file_paths),
        'file_extensions': file_extensions,
        'min_file_size_mb': file_size[0] if file_size else None,
        'max_file_size_mb': file_size[1] if file_size else None,
        'max_items': max_items
    }

    # Save cache files in the root directory
    if save_json:
        json_cache_path = root_dir / f"{cache_basename}.json"
        save_file_list_json(file_paths, json_cache_path, metadata)

    if save_pickle:
        pickle_cache_path = root_dir / f"{cache_basename}.pkl"
        save_file_list_pickle(file_paths, pickle_cache_path, metadata)

    return file_paths


def main():
    parser = argparse.ArgumentParser(
        description='Scan directories for files and cache the results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan multiple directories for mesh files
  python cache_file_scan.py --root-dirs /data/meshes1 /data/meshes2 --extensions *.obj *.ply

  # Scan with file size filtering
  python cache_file_scan.py --root-dirs /data/meshes --min-size 0.5 --max-size 10.0

  # Limit number of files
  python cache_file_scan.py --root-dirs /data/meshes --max-items 5000

  # Custom cache filename
  python cache_file_scan.py --root-dirs /data/meshes --cache-name my_scan_cache
        """
    )

    parser.add_argument(
        '--root-dirs',
        nargs='+',
        required=True,
        help='Root directories to scan (one cache file per directory)'
    )

    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['*.obj', '*.ply', '*.off', '*.stl'],
        help='File extensions to search for (default: *.obj *.ply *.off *.stl)'
    )

    parser.add_argument(
        '--min-size',
        type=float,
        default=None,
        help='Minimum file size in MB (optional)'
    )

    parser.add_argument(
        '--max-size',
        type=float,
        default=None,
        help='Maximum file size in MB (optional)'
    )

    parser.add_argument(
        '--max-items',
        type=int,
        default=None,
        help='Maximum number of files to include per directory (optional)'
    )

    parser.add_argument(
        '--cache-name',
        type=str,
        default='file_scan_cache',
        help='Base name for cache files (default: file_scan_cache)'
    )

    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Save only JSON cache (default: save both JSON and pickle)'
    )

    parser.add_argument(
        '--pickle-only',
        action='store_true',
        help='Save only pickle cache (default: save both JSON and pickle)'
    )

    args = parser.parse_args()

    # Determine which formats to save
    if args.json_only:
        save_json, save_pickle = True, False
    elif args.pickle_only:
        save_json, save_pickle = False, True
    else:
        save_json, save_pickle = True, True

    # Prepare file size tuple
    file_size = None
    if args.min_size is not None or args.max_size is not None:
        file_size = (args.min_size, args.max_size)

    print("=" * 70)
    print("FILE SCAN CACHE GENERATOR")
    print("=" * 70)
    print(f"Root directories: {len(args.root_dirs)}")
    print(f"File extensions: {args.extensions}")
    print(f"File size filter: {file_size}")
    print(f"Max items per directory: {args.max_items}")
    print(f"Cache basename: {args.cache_name}")
    print(f"Save JSON: {save_json}, Save Pickle: {save_pickle}")
    print("=" * 70)

    # Scan each directory separately and save cache
    total_files = 0
    for root_dir_str in args.root_dirs:
        root_dir = Path(root_dir_str)
        file_paths = scan_and_cache_directory(
            root_dir=root_dir,
            file_extensions=args.extensions,
            file_size=file_size,
            max_items=args.max_items,
            cache_basename=args.cache_name,
            save_json=save_json,
            save_pickle=save_pickle
        )
        total_files += len(file_paths)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Directories scanned: {len(args.root_dirs)}")
    print(f"Total files found: {total_files}")
    print("Cache files saved in each root directory")
    print("=" * 70)


if __name__ == "__main__":
    main()
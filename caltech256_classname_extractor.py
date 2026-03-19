#!/usr/bin/env python3
"""
Extract names from subdirectories with format ###.name
and save them to a text file.
"""

import os
import sys
from pathlib import Path


def extract_names(directory_path, output_file='names.txt'):
    """
    Extract names after the dot from subdirectories with format ###.name

    Args:
        directory_path: Path to the directory containing subdirectories
        output_file: Name of the output text file (default: names.txt)
    """
    # Convert to Path object
    dir_path = Path(directory_path)

    # Check if directory exists
    if not dir_path.exists():
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    if not dir_path.is_dir():
        print(f"Error: '{directory_path}' is not a directory.")
        return

    # Get all subdirectories
    subdirs = [d for d in dir_path.iterdir() if d.is_dir()]

    # Extract names after the dot
    names = []
    for subdir in subdirs:
        dir_name = subdir.name
        if '.' in dir_name:
            # Split by dot and take everything after the first dot
            name_after_dot = dir_name.split('.', 1)[1]
            names.append(name_after_dot)

    # Sort names alphabetically (optional, remove if you want original order)
    # names.sort()

    # Write to output file
    output_path = dir_path / output_file
    with open(output_path, 'w') as f:
        for name in names:
            f.write(f"{name}\n")

    print(f"Extracted {len(names)} names from subdirectories.")
    print(f"Output saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_subdir_names.py <directory_path> [output_file]")
        print("Example: python extract_subdir_names.py /path/to/data names.txt")
        sys.exit(1)

    directory_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'names.txt'

    extract_names(directory_path, output_file)


if __name__ == "__main__":
    main()
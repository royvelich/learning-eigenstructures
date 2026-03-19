#!/usr/bin/env python3
"""
Generic image dataset mapper - creates mappings for any dataset organized in category folders.
Works with any dataset structure where images are organized in category subdirectories.
"""

import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, Optional


def create_image_to_label_mapping(dataset_root: str, output_path: Optional[str] = None) -> Dict:
    """
    Create a mapping from image filenames to class IDs and class names.

    Works with any dataset organized in category folders:
    dataset_root/
        category1/
            image1.jpg
            image2.jpg
            ...
        category2/
            image1.jpg
            ...
        ...

    Args:
        dataset_root: Path to the folder containing category subdirectories
        output_path: Path to save the pickle file (default: image_mapping.pkl in current dir)

    Returns:
        Dictionary mapping image filenames to their labels
    """
    if output_path is None:
        output_path = 'image_mapping.pkl'

    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise ValueError(f"Directory not found: {dataset_root}")

    print(f"Processing dataset from: {dataset_root}\n")

    # Get all category folders and sort them alphabetically for consistent IDs
    category_folders = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

    if not category_folders:
        raise ValueError(f"No subdirectories found in {dataset_root}")

    print(f"Found {len(category_folders)} categories")

    # Dictionary to store all mappings
    image_to_label_map = {}

    # Image extensions to look for
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
                        '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.TIFF', '.WEBP'}

    # Process each category folder
    for category_idx, category_folder in enumerate(category_folders):
        category_name = category_folder.name

        # Get all image files in this category
        image_files = [f for f in category_folder.iterdir()
                       if f.is_file() and f.suffix in image_extensions]

        if not image_files:
            print(f"Warning: No images found in {category_name}")
            continue

        # Process each image file
        for image_file in image_files:
            # Store mapping with just the filename (not full path)
            # Use alphabetical index as class_id for consistency
            image_to_label_map[image_file.name] = {
                'class_id': category_idx,
                'class_name': category_name
            }

        # Progress update every 50 categories
        if (category_idx + 1) % 50 == 0:
            print(f"  Processed {category_idx + 1}/{len(category_folders)} categories...")

    print()

    if not image_to_label_map:
        raise ValueError("No images found in any category!")

    # Save to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(image_to_label_map, f)

    print(f"Successfully processed {len(image_to_label_map)} images")
    print(f"Saved mapping dictionary to: {output_path}")
    print()

    # Show some statistics
    class_counts = {}
    for info in image_to_label_map.values():
        class_name = info['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print("Class distribution:")
    sorted_classes = sorted(class_counts.items())

    # Show first 10 categories
    for class_name, count in sorted_classes[:10]:
        print(f"  {class_name}: {count} images")

    if len(sorted_classes) > 10:
        print(f"  ... ({len(sorted_classes) - 10} more categories)")
    print()

    # Show example mapping
    print("Dictionary structure example:")
    example_key = list(image_to_label_map.keys())[0]
    print(f"  '{example_key}': {image_to_label_map[example_key]}")
    print()

    # Summary
    print("Summary:")
    print(f"  Total categories: {len(class_counts)}")
    print(f"  Total images: {len(image_to_label_map)}")
    print(f"  Average images per category: {len(image_to_label_map) / len(class_counts):.1f}")

    return image_to_label_map


def create_category_list(dataset_root: str, output_path: Optional[str] = None) -> None:
    """
    Create a text file with all category names (one per line), sorted alphabetically.

    Args:
        dataset_root: Path to the folder containing category subdirectories
        output_path: Path to save the text file (default: categories.txt)
    """
    if output_path is None:
        output_path = 'categories.txt'

    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise ValueError(f"Directory not found: {dataset_root}")

    # Get all category folders and sort them alphabetically
    category_folders = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])

    if not category_folders:
        raise ValueError(f"No subdirectories found in {dataset_root}")

    # Save to text file
    with open(output_path, 'w') as f:
        for category_name in category_folders:
            f.write(f"{category_name}\n")

    print(f"Saved {len(category_folders)} category names to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create image filename to label mapping for any dataset organized in category folders',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'dataset_root',
        type=str,
        help='Path to folder containing category subdirectories'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for pickle file (default: image_mapping.pkl)'
    )
    parser.add_argument(
        '--categories-txt',
        action='store_true',
        help='Also create a text file with category names (categories.txt)'
    )

    args = parser.parse_args()

    # Create the mapping
    create_image_to_label_mapping(args.dataset_root, args.output)

    # Optionally create category list text file
    if args.categories_txt:
        print()
        create_category_list(args.dataset_root)


if __name__ == '__main__':
    main()
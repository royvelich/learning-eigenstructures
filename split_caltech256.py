#!/usr/bin/env python3
"""
Split Caltech-256 dataset into train/val/test sets with uniform random sampling.
Creates a new directory structure with symbolic links or copies.
Test split is now optional - if not specified, only train/val split is created.
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def validate_split_ratios(train: float, val: float, test: Optional[float] = None) -> None:
    """Validate that split ratios sum to 1.0"""
    if test is None:
        # Two-way split: train and val only
        total = train + val
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if train <= 0 or val <= 0:
            raise ValueError("Train and val ratios must be positive")
    else:
        # Three-way split: train, val, and test
        total = train + val + test
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if train <= 0 or val < 0 or test < 0:
            raise ValueError("Split ratios must be non-negative, and train must be positive")


def get_category_images(category_path: Path) -> List[Path]:
    """Get all image files from a category folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    return [f for f in category_path.iterdir()
            if f.is_file() and f.suffix in image_extensions]


def split_images(images: List[Path], train_ratio: float, val_ratio: float,
                 test_ratio: Optional[float], seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split images into train/val/test sets with uniform random sampling.

    Args:
        images: List of image paths
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set (None for two-way split)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_images, val_images, test_images)
        test_images will be empty list if test_ratio is None
    """
    # Shuffle images with seed for reproducibility
    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)

    if test_ratio is None:
        # Two-way split: train and val only
        train_images = shuffled[:n_train]
        val_images = shuffled[n_train:]
        test_images = []
    else:
        # Three-way split: train, val, and test
        n_val = int(n_total * val_ratio)
        train_images = shuffled[:n_train]
        val_images = shuffled[n_train:n_train + n_val]
        test_images = shuffled[n_train + n_val:]

    return train_images, val_images, test_images


def create_split_structure(source_dir: Path, output_dir: Path,
                           train_ratio: float, val_ratio: float, test_ratio: Optional[float],
                           seed: int, use_symlinks: bool = False) -> Dict[str, int]:
    """
    Create train/val/test split directory structure.

    Args:
        source_dir: Path to 256_ObjectCategories folder
        output_dir: Path to output directory
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set (None for two-way split)
        seed: Random seed
        use_symlinks: If True, create symbolic links; if False, copy files

    Returns:
        Dictionary with split statistics
    """
    # Create output directories
    splits = ['train', 'val']
    if test_ratio is not None:
        splits.append('test')

    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # Get all category folders
    category_folders = sorted([d for d in source_dir.iterdir() if d.is_dir()])

    stats = {
        'total_categories': len(category_folders),
        'train_images': 0,
        'val_images': 0,
        'test_images': 0
    }

    print(f"Processing {len(category_folders)} categories...")
    if test_ratio is None:
        print(f"Split ratios - Train: {train_ratio:.2f}, Val: {val_ratio:.2f}")
    else:
        print(f"Split ratios - Train: {train_ratio:.2f}, Val: {val_ratio:.2f}, Test: {test_ratio:.2f}")
    print(f"Random seed: {seed}")
    print(f"Mode: {'Symbolic links' if use_symlinks else 'Copy files'}\n")

    # Process each category
    for idx, category_folder in enumerate(category_folders):
        category_name = category_folder.name

        # Get all images in this category
        images = get_category_images(category_folder)

        if not images:
            print(f"Warning: No images found in {category_name}")
            continue

        # Split images for this category
        train_imgs, val_imgs, test_imgs = split_images(
            images, train_ratio, val_ratio, test_ratio, seed + idx
        )

        # Create category folders in each split
        for split in splits:
            (output_dir / split / category_name).mkdir(exist_ok=True)

        # Copy or link images to appropriate splits
        for img in train_imgs:
            dst = output_dir / 'train' / category_name / img.name
            if use_symlinks:
                dst.symlink_to(img.resolve())
            else:
                shutil.copy2(img, dst)

        for img in val_imgs:
            dst = output_dir / 'val' / category_name / img.name
            if use_symlinks:
                dst.symlink_to(img.resolve())
            else:
                shutil.copy2(img, dst)

        if test_ratio is not None:
            for img in test_imgs:
                dst = output_dir / 'test' / category_name / img.name
                if use_symlinks:
                    dst.symlink_to(img.resolve())
                else:
                    shutil.copy2(img, dst)

        # Update stats
        stats['train_images'] += len(train_imgs)
        stats['val_images'] += len(val_imgs)
        stats['test_images'] += len(test_imgs)

        # Progress update
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(category_folders)} categories...")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Split Caltech-256 dataset into train/val(/test) sets with uniform random sampling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'source_dir',
        type=str,
        help='Path to 256_ObjectCategories folder'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to output directory for split dataset'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Fraction of data for training set'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.3,
        help='Fraction of data for validation set'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=None,
        help='Fraction of data for test set (optional, if not set only train/val split is created)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--symlinks',
        action='store_true',
        help='Create symbolic links instead of copying files (saves space)'
    )

    args = parser.parse_args()

    # Validate inputs
    validate_split_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if output_dir.exists():
        response = input(f"Output directory {output_dir} already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(output_dir)

    # Create split
    print(f"\nCreating split dataset...")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}\n")

    stats = create_split_structure(
        source_dir=source_dir,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        use_symlinks=args.symlinks
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("Split Summary:")
    print(f"{'=' * 60}")
    print(f"Total categories: {stats['total_categories']}")
    print(f"Train images:     {stats['train_images']:,} ({args.train_ratio:.1%})")
    print(f"Val images:       {stats['val_images']:,} ({args.val_ratio:.1%})")
    if args.test_ratio is not None:
        print(f"Test images:      {stats['test_images']:,} ({args.test_ratio:.1%})")
    print(f"Total images:     {sum([stats['train_images'], stats['val_images'], stats['test_images']]):,}")
    print(f"{'=' * 60}")

    print(f"\n✓ Dataset split created successfully!")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── 001.ak47/")
    print(f"  │   ├── 002.american-flag/")
    print(f"  │   └── ...")
    print(f"  ├── val/")
    print(f"  │   └── ...")
    if args.test_ratio is not None:
        print(f"  └── test/")
        print(f"      └── ...")


if __name__ == '__main__':
    main()
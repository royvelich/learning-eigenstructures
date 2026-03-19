#!/usr/bin/env python3
"""
Simple STL-10 Dataset Downloader - Downloads and extracts images
"""

import argparse
import os
import shutil
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def clean_corrupted_download(path):
    """Remove corrupted download files"""
    stl10_dir = os.path.join(path, 'stl10_binary')
    if os.path.exists(stl10_dir):
        print("Cleaning up corrupted/incomplete download...")
        shutil.rmtree(stl10_dir)

    # Also remove the tar.gz file if it exists
    tar_file = os.path.join(path, 'stl10_binary.tar.gz')
    if os.path.exists(tar_file):
        os.remove(tar_file)
        print("Removed incomplete tar.gz file")


def main():
    parser = argparse.ArgumentParser(description='Download STL-10 dataset and extract images')
    parser.add_argument('--path', type=str, required=True,
                        help='Directory where to save the dataset images')
    parser.add_argument('--clean', action='store_true',
                        help='Clean corrupted downloads before starting')
    args = parser.parse_args()

    # Clean up if requested or if previous download failed
    if args.clean:
        clean_corrupted_download(args.path)

    print(f"Downloading STL-10 dataset to: {args.path}")
    print("This will download ~2.6GB of data...")
    print("If download fails, run with --clean flag to retry\n")

    transform = transforms.ToTensor()

    try:
        # Download and save train images
        print("Downloading and extracting train images...")
        train_data = datasets.STL10(root=args.path, split='train', download=True, transform=transform)
        train_dir = os.path.join(args.path, 'train')
        os.makedirs(train_dir, exist_ok=True)

        for i, (img, label) in enumerate(train_data):
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(train_dir, f'{i:05d}_{label}.png'))
        print(f"✓ Saved {len(train_data)} train images")

        # Download and save test images
        print("\nDownloading and extracting test images...")
        test_data = datasets.STL10(root=args.path, split='test', download=True, transform=transform)
        test_dir = os.path.join(args.path, 'test')
        os.makedirs(test_dir, exist_ok=True)

        for i, (img, label) in enumerate(test_data):
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(test_dir, f'{i:05d}_{label}.png'))
        print(f"✓ Saved {len(test_data)} test images")

        # Download and save unlabeled images
        print("\nDownloading and extracting unlabeled images...")
        unlabeled_data = datasets.STL10(root=args.path, split='unlabeled', download=True, transform=transform)
        unlabeled_dir = os.path.join(args.path, 'unlabeled')
        os.makedirs(unlabeled_dir, exist_ok=True)

        for i, (img, _) in enumerate(unlabeled_data):
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(unlabeled_dir, f'{i:05d}.png'))
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/100000 unlabeled images...")
        print(f"✓ Saved {len(unlabeled_data)} unlabeled images")

        print(f"\n✓ Done! All images saved to: {args.path}/")
        print(f"  - {args.path}/train/ (5,000 images)")
        print(f"  - {args.path}/test/ (8,000 images)")
        print(f"  - {args.path}/unlabeled/ (100,000 images)")

    except RuntimeError as e:
        if "File not found or corrupted" in str(e):
            print("\n❌ Download failed or file corrupted!")
            print("This can happen due to network interruption.")
            print("\nTo fix this, run:")
            print(f"  python {__file__} --path {args.path} --clean")
            print("\nThis will clean up the corrupted download and retry.")
        else:
            raise


if __name__ == '__main__':
    main()
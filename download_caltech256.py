#!/usr/bin/env python3
"""
Simple Caltech-256 Dataset Downloader - Downloads and extracts images
"""

import argparse
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Official Caltech-256 download URL
CALTECH256_URL = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar"


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download a file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def clean_corrupted_download(path):
    """Remove corrupted download files"""
    caltech256_dir = os.path.join(path, '256_ObjectCategories')
    if os.path.exists(caltech256_dir):
        print("Cleaning up corrupted/incomplete download...")
        shutil.rmtree(caltech256_dir)

    # Also remove the tar file if it exists
    tar_file = os.path.join(path, '256_ObjectCategories.tar')
    if os.path.exists(tar_file):
        os.remove(tar_file)
        print("Removed incomplete tar file")


def main():
    parser = argparse.ArgumentParser(description='Download Caltech-256 dataset and extract images')
    parser.add_argument('--path', type=str, required=True,
                        help='Directory where to save the dataset images')
    parser.add_argument('--clean', action='store_true',
                        help='Clean corrupted downloads before starting')
    parser.add_argument('--save-flat', action='store_true',
                        help='Save all images in flat directory structure instead of category folders')
    args = parser.parse_args()

    # Clean up if requested or if previous download failed
    if args.clean:
        clean_corrupted_download(args.path)

    # Create output directory
    os.makedirs(args.path, exist_ok=True)

    print(f"Downloading Caltech-256 dataset to: {args.path}")
    print("This will download ~1.2GB of data...")
    print("If download fails, run with --clean flag to retry\n")

    tar_path = os.path.join(args.path, '256_ObjectCategories.tar')
    extracted_dir = os.path.join(args.path, '256_ObjectCategories')

    try:
        # Check if already downloaded and extracted
        if os.path.exists(extracted_dir) and len(os.listdir(extracted_dir)) > 250:
            print("✓ Dataset already downloaded and extracted")
        else:
            # Download the tar file if not already present
            if not os.path.exists(tar_path):
                print(f"Downloading from: {CALTECH256_URL}")
                download_file(CALTECH256_URL, tar_path)
                print("✓ Download complete")
            else:
                print("✓ Tar file already exists, skipping download")

            # Extract the tar file
            print("\nExtracting images...")
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=args.path)
            print("✓ Extraction complete")

        # Count images and categories
        categories = sorted([d for d in os.listdir(extracted_dir)
                             if os.path.isdir(os.path.join(extracted_dir, d))])

        total_images = 0
        for category in categories:
            category_path = os.path.join(extracted_dir, category)
            num_images = len([f for f in os.listdir(category_path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total_images += num_images

        print(f"\nDataset Statistics:")
        print(f"  Total images: {total_images}")
        print(f"  Total categories: {len(categories)}")
        print(f"  Categories include: {', '.join(categories[:5])}...")

        if args.save_flat:
            # Save in flat structure with category labels in filenames
            output_dir = os.path.join(args.path, 'images_flat')
            os.makedirs(output_dir, exist_ok=True)

            print("\nCreating flat directory structure...")
            image_idx = 0

            for category_idx, category in enumerate(categories):
                category_path = os.path.join(extracted_dir, category)
                image_files = sorted([f for f in os.listdir(category_path)
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

                for img_file in image_files:
                    src_path = os.path.join(category_path, img_file)
                    ext = os.path.splitext(img_file)[1]
                    # Create filename: INDEX_CATEGORYID_CATEGORYNAME.ext
                    dst_filename = f'{image_idx:05d}_{category_idx:03d}_{category}{ext}'
                    dst_path = os.path.join(output_dir, dst_filename)
                    shutil.copy2(src_path, dst_path)
                    image_idx += 1

                if (category_idx + 1) % 50 == 0:
                    print(f"  Processed {category_idx + 1}/{len(categories)} categories...")

            print(f"\n✓ Saved {total_images} images to: {output_dir}/")
            print(f"  Filename format: INDEX_CATEGORYID_CATEGORYNAME.ext")
        else:
            print(f"\n✓ Images organized by category in: {extracted_dir}/")
            print(f"  Structure: Each subfolder contains images for one category")

        # Optionally remove tar file to save space
        if os.path.exists(tar_path):
            print(f"\nNote: You can delete {tar_path} to save space (1.2GB)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf download was interrupted, run with --clean flag:")
        print(f"  python {__file__} --path {args.path} --clean")
        raise


if __name__ == '__main__':
    main()
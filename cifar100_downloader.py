"""
CIFAR-100 Dataset Downloader
Downloads CIFAR-100 dataset and extracts images with their class labels.
"""

import os
import pickle
import urllib.request
import tarfile
import numpy as np
from PIL import Image
from pathlib import Path
import json
import argparse


class CIFAR100Downloader:
    """Download and extract CIFAR-100 dataset."""

    def __init__(self, root_dir="./cifar100_data"):
        self.root_dir = Path(root_dir)
        self.url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        self.dataset_dir = self.root_dir / "cifar-100-python"
        self.images_dir = self.root_dir / "images"

    def download(self):
        """Download the CIFAR-100 dataset."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        tar_path = self.root_dir / "cifar-100-python.tar.gz"

        if tar_path.exists():
            print(f"Dataset already downloaded at {tar_path}")
            return tar_path

        print(f"Downloading CIFAR-100 from {self.url}...")
        urllib.request.urlretrieve(self.url, tar_path)
        print(f"Downloaded to {tar_path}")
        return tar_path

    def extract(self, tar_path):
        """Extract the downloaded tar file."""
        if self.dataset_dir.exists():
            print(f"Dataset already extracted at {self.dataset_dir}")
            return

        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(self.root_dir)
        print(f"Extracted to {self.dataset_dir}")

    def load_batch(self, batch_file):
        """Load a CIFAR-100 batch file."""
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        return batch

    def get_label_names(self):
        """Load fine and coarse label names."""
        meta_file = self.dataset_dir / "meta"
        meta = self.load_batch(meta_file)

        fine_label_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]
        coarse_label_names = [name.decode('utf-8') for name in meta[b'coarse_label_names']]

        return fine_label_names, coarse_label_names

    def save_images(self, split='train'):
        """
        Save images from the dataset to disk.

        Args:
            split: 'train' or 'test'
        """
        # Load label names
        fine_label_names, coarse_label_names = self.get_label_names()

        # Load the appropriate batch
        if split == 'train':
            batch_file = self.dataset_dir / "train"
        else:
            batch_file = self.dataset_dir / "test"

        batch = self.load_batch(batch_file)

        # Extract data
        images = batch[b'data']
        fine_labels = batch[b'fine_labels']
        coarse_labels = batch[b'coarse_labels']
        filenames = [name.decode('utf-8') for name in batch[b'filenames']]

        # Reshape images: CIFAR-100 stores images as (N, 3072) where 3072 = 32*32*3
        # Need to reshape to (N, 3, 32, 32) then transpose to (N, 32, 32, 3)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        # Create output directory
        split_dir = self.images_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = []

        print(f"Saving {len(images)} {split} images...")
        for idx, (img, fine_label, coarse_label, filename) in enumerate(
                zip(images, fine_labels, coarse_labels, filenames)):

            # Create subdirectory for each fine class
            class_name = fine_label_names[fine_label]
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)

            # Save image
            img_pil = Image.fromarray(img.astype('uint8'))
            img_path = class_dir / filename
            img_pil.save(img_path)

            # Store metadata
            metadata.append({
                'filename': filename,
                'path': str(img_path.relative_to(self.images_dir)),
                'fine_label': int(fine_label),
                'fine_label_name': class_name,
                'coarse_label': int(coarse_label),
                'coarse_label_name': coarse_label_names[coarse_label],
                'index': idx
            })

            if (idx + 1) % 1000 == 0:
                print(f"  Saved {idx + 1}/{len(images)} images...")

        # Save metadata as JSON
        metadata_file = split_dir / f"{split}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(images)} {split} images to {split_dir}")
        print(f"Metadata saved to {metadata_file}")

        return metadata

    def get_dataset_info(self):
        """Print dataset information."""
        fine_label_names, coarse_label_names = self.get_label_names()

        print("\n" + "=" * 60)
        print("CIFAR-100 Dataset Information")
        print("=" * 60)
        print(f"Number of fine classes: {len(fine_label_names)}")
        print(f"Number of coarse classes: {len(coarse_label_names)}")
        print(f"\nCoarse classes: {', '.join(coarse_label_names)}")
        print(f"\nFirst 10 fine classes: {', '.join(fine_label_names[:10])}")
        print("=" * 60 + "\n")

    def run(self):
        """Run the complete download and extraction pipeline."""
        # Download
        tar_path = self.download()

        # Extract
        self.extract(tar_path)

        # Print dataset info
        self.get_dataset_info()

        # Save training images
        print("\nProcessing training set...")
        train_metadata = self.save_images('train')

        # Save test images
        print("\nProcessing test set...")
        test_metadata = self.save_images('test')

        print("\n" + "=" * 60)
        print("Download and extraction complete!")
        print(f"Images saved to: {self.images_dir}")
        print(f"Training images: {len(train_metadata)}")
        print(f"Test images: {len(test_metadata)}")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download and extract CIFAR-100 dataset with images and labels.'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory where the dataset will be downloaded and extracted'
    )

    args = parser.parse_args()

    downloader = CIFAR100Downloader(root_dir=args.output_dir)
    downloader.run()


if __name__ == "__main__":
    main()
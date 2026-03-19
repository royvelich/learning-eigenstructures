import numpy as np
import os
import argparse


def extract_labels(data_folder, output_folder=None):
    """
    Extract labels from STL-10 binary label files.

    Args:
        data_folder: Path to folder containing test_y.bin and train_y.bin
        output_folder: Path to save output text files (default: same as data_folder)
    """
    if output_folder is None:
        output_folder = data_folder

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # File names to process
    label_files = ['test_y.bin', 'train_y.bin']

    for label_file in label_files:
        input_path = os.path.join(data_folder, label_file)

        # Check if file exists
        if not os.path.exists(input_path):
            print(f"Warning: {label_file} not found in {data_folder}")
            continue

        # Read labels (stored as uint8)
        labels = np.fromfile(input_path, dtype=np.uint8)

        # Create output filename
        output_filename = label_file.replace('.bin', '_labels.txt')
        output_path = os.path.join(output_folder, output_filename)

        # Save labels to text file (one label per line)
        np.savetxt(output_path, labels, fmt='%d')

        print(f"Extracted {len(labels)} labels from {label_file}")
        print(f"Saved to: {output_path}")
        print(f"Label range: {labels.min()} to {labels.max()}")
        print(f"Unique labels: {sorted(np.unique(labels))}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Extract class labels from STL-10 binary files to text files'
    )
    parser.add_argument(
        'data_folder',
        type=str,
        help='Path to folder containing test_y.bin and train_y.bin files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output folder for text files (default: same as data_folder)'
    )

    args = parser.parse_args()

    extract_labels(args.data_folder, args.output)


if __name__ == '__main__':
    main()
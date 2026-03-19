import os
import pickle
import argparse
from pathlib import Path

# Official mapping from torchvision.datasets.imagenette
# Source: https://docs.pytorch.org/vision/0.18/_modules/torchvision/datasets/imagenette.html
WNID_TO_CLASS = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute"
}


def create_image_to_label_mapping(dataset_root, output_path=None):
    """
    Create a mapping from image filenames to class IDs and class names for Imagenette2.

    The Imagenette2 dataset is organized in folders like:
    imagenette2/
        train/
            n01440764/
                ILSVRC2012_val_00000293.JPEG
                ...
            n02102040/
                ...
        val/
            n01440764/
            n02102040/
            ...

    Args:
        dataset_root: Path to the train or val folder containing WordNet ID subdirectories
        output_path: Path to save the pickle file (default: imagenette_mapping.pkl in current dir)
    """
    if output_path is None:
        output_path = 'imagenette_mapping.pkl'

    dataset_root = Path(dataset_root)

    print(f"Processing Imagenette2 dataset from: {dataset_root}\n")

    # Dictionary to store all mappings
    image_to_label_map = {}

    # Image extensions to look for
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}

    # Get all WordNet ID folders (should be 10 folders)
    wnid_folders = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

    if len(wnid_folders) != 10:
        print(f"Warning: Expected 10 class folders, found {len(wnid_folders)}")

    # Process each WordNet ID folder
    for wnid_idx, wnid_folder in enumerate(wnid_folders):
        wnid = wnid_folder.name

        # Get the class name from our mapping
        if wnid in WNID_TO_CLASS:
            class_name = WNID_TO_CLASS[wnid]
        else:
            print(f"Warning: Unknown WordNet ID {wnid}, using as class name")
            class_name = wnid

        # Assign a class ID based on sorted order of WordNet IDs
        # This ensures consistency across train and val splits
        class_id = sorted(WNID_TO_CLASS.keys()).index(wnid) if wnid in WNID_TO_CLASS else wnid_idx

        # Get all image files in this class folder
        image_files = [f for f in wnid_folder.iterdir()
                       if f.is_file() and f.suffix in image_extensions]

        if not image_files:
            print(f"Warning: No images found in {wnid_folder}")
            continue

        # Process each image file
        for image_file in image_files:
            # Store mapping with just the filename (not full path)
            image_to_label_map[image_file.name] = {
                'class_id': class_id,
                'class_name': class_name
            }

    print(f"Processed {len(wnid_folders)} classes")
    print()

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
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} images")
    print()

    # Show example mapping
    print("Dictionary structure example:")
    example_key = list(image_to_label_map.keys())[0]
    print(f"  '{example_key}': {image_to_label_map[example_key]}")
    print()

    return image_to_label_map


def create_category_list(output_path=None):
    """
    Create a text file with all category names (one per line), sorted by WordNet ID.
    Similar to class_names.txt for STL-10.

    Args:
        output_path: Path to save the text file (default: imagenette_categories.txt)
    """
    if output_path is None:
        output_path = 'imagenette_categories.txt'

    # Get category names sorted by WordNet ID
    category_names = [WNID_TO_CLASS[wnid] for wnid in sorted(WNID_TO_CLASS.keys())]

    # Save to text file
    with open(output_path, 'w') as f:
        for category_name in category_names:
            f.write(f"{category_name}\n")

    print(f"Saved {len(category_names)} category names to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create image filename to label mapping for Imagenette2 dataset'
    )
    parser.add_argument(
        'dataset_root',
        type=str,
        help='Path to train or val folder containing WordNet ID subdirectories (e.g., /path/to/imagenette2/train)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for pickle file (default: imagenette_mapping.pkl)'
    )
    parser.add_argument(
        '--categories-txt',
        action='store_true',
        help='Also create a text file with category names (imagenette_categories.txt)'
    )

    args = parser.parse_args()

    # Create the mapping
    create_image_to_label_mapping(args.dataset_root, args.output)

    # Optionally create category list text file
    if args.categories_txt:
        print()
        create_category_list()


if __name__ == '__main__':
    main()
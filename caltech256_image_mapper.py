import os
import pickle
import argparse
from pathlib import Path


def create_image_to_label_mapping(dataset_root, output_path=None):
    """
    Create a mapping from image filenames to class IDs and class names for Caltech-256.

    The Caltech-256 dataset is organized in folders like:
    256_ObjectCategories/ (or train/ or val/)
        001.ak47/
            001_0001.jpg
            001_0002.jpg
            ...
        002.american-flag/
            002_0001.jpg
            ...
        ...

    Args:
        dataset_root: Path to the folder containing category subdirectories (e.g., 256_ObjectCategories, train, or val)
        output_path: Path to save the pickle file (default: caltech256_mapping.pkl in current dir)
    """
    if output_path is None:
        output_path = 'caltech256_mapping.pkl'

    categories_dir = dataset_root

    if not os.path.exists(categories_dir):
        raise ValueError(f"Directory not found: {categories_dir}")

    print(f"Processing dataset from: {categories_dir}\n")

    # Get all category folders and sort them
    category_folders = sorted([d for d in os.listdir(categories_dir)
                               if os.path.isdir(os.path.join(categories_dir, d))])

    print(f"Found {len(category_folders)} categories")

    # Dictionary to store all mappings
    image_to_label_map = {}

    # Image extensions to look for
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    # Process each category folder
    for category_idx, category_folder in enumerate(category_folders):
        category_path = os.path.join(categories_dir, category_folder)

        # Extract category ID and name from folder name
        # Format is typically: "XXX.category-name" or "XXX.category_name"
        # e.g., "001.ak47", "002.american-flag", "257.clutter"
        parts = category_folder.split('.', 1)
        if len(parts) == 2:
            category_id_str, category_name = parts
            try:
                category_id = int(category_id_str)
            except ValueError:
                print(f"Warning: Could not parse category ID from {category_folder}, using index {category_idx}")
                category_id = category_idx
        else:
            # Fallback: use the folder name as category name and index as ID
            category_name = category_folder
            category_id = category_idx

        # Get all image files in this category
        image_files = [f for f in os.listdir(category_path)
                       if os.path.splitext(f)[1] in image_extensions]

        if not image_files:
            print(f"Warning: No images found in {category_folder}")
            continue

        # Process each image file
        for image_file in image_files:
            # Store mapping with just the filename (not full path)
            image_to_label_map[image_file] = {
                'class_id': category_id,
                'class_name': category_name,
                'category_folder': category_folder
            }

        if (category_idx + 1) % 50 == 0:
            print(f"  Processed {category_idx + 1}/{len(category_folders)} categories...")

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

    print("Class distribution (first 10 categories):")
    for i, (class_name, count) in enumerate(sorted(class_counts.items())[:10]):
        print(f"  {class_name}: {count} images")
    print(f"  ... ({len(class_counts) - 10} more categories)")
    print()

    # Show example mapping
    print("Dictionary structure example:")
    example_key = list(image_to_label_map.keys())[0]
    print(f"  '{example_key}': {image_to_label_map[example_key]}")
    print()

    # Show category name examples
    print("Sample category names:")
    unique_categories = sorted(set(info['class_name'] for info in image_to_label_map.values()))
    for cat in unique_categories[:10]:
        print(f"  - {cat}")
    print(f"  ... ({len(unique_categories) - 10} more categories)")

    return image_to_label_map


def create_category_list(dataset_root, output_path=None):
    """
    Create a text file with all category names (one per line), sorted by category ID.
    Similar to class_names.txt for STL-10.

    Args:
        dataset_root: Path to the folder containing category subdirectories
        output_path: Path to save the text file (default: caltech256_categories.txt)
    """
    if output_path is None:
        output_path = 'caltech256_categories.txt'

    categories_dir = dataset_root

    if not os.path.exists(categories_dir):
        raise ValueError(f"Directory not found: {categories_dir}")

    # Get all category folders and sort them
    category_folders = sorted([d for d in os.listdir(categories_dir)
                               if os.path.isdir(os.path.join(categories_dir, d))])

    # Extract category names (removing the numeric prefix)
    category_names = []
    for category_folder in category_folders:
        parts = category_folder.split('.', 1)
        if len(parts) == 2:
            category_name = parts[1]
        else:
            category_name = category_folder
        category_names.append(category_name)

    # Save to text file
    with open(output_path, 'w') as f:
        for category_name in category_names:
            f.write(f"{category_name}\n")

    print(f"Saved {len(category_names)} category names to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create image filename to label mapping for Caltech-256 dataset'
    )
    parser.add_argument(
        'dataset_root',
        type=str,
        help='Path to folder containing category subdirectories (e.g., 256_ObjectCategories, train, or val)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for pickle file (default: caltech256_mapping.pkl)'
    )
    parser.add_argument(
        '--categories-txt',
        action='store_true',
        help='Also create a text file with category names (caltech256_categories.txt)'
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
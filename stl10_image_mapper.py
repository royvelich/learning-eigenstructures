import os
import pickle
import argparse
from pathlib import Path


def load_class_names(class_names_file):
    """Load class names from text file (one class per line)."""
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def extract_class_id_from_filename(filename):
    """
    Extract class ID from filename with format: ####_CLASSID.png
    Returns the class ID as an integer.
    """
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]

    # Find the underscore and take everything after it
    underscore_index = name_without_ext.rfind('_')
    if underscore_index == -1:
        raise ValueError(f"No underscore found in filename: {filename}")

    class_id_str = name_without_ext[underscore_index + 1:]
    class_id = int(class_id_str)

    return class_id


def create_image_to_label_mapping(image_folders, class_names_file, output_path=None):
    """
    Create a mapping from image filenames to class IDs and class names.

    Args:
        image_folders: List of paths to folders containing image files
        class_names_file: Path to text file with class names (one per line)
        output_path: Path to save the pickle file (default: image_to_label_map.pkl in current dir)
    """
    if output_path is None:
        output_path = 'image_to_label_map.pkl'

    # Load class names
    class_names = load_class_names(class_names_file)
    print(f"Loaded {len(class_names)} class names: {class_names}")
    print()

    # Dictionary to store all mappings
    image_to_label_map = {}

    # Get all image files (png, jpg, jpeg)
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}

    # Process each folder
    for image_folder in image_folders:
        if not os.path.exists(image_folder):
            print(f"Warning: Folder not found: {image_folder}")
            continue

        image_files = [f for f in os.listdir(image_folder)
                       if os.path.splitext(f)[1] in image_extensions]

        print(f"Processing {image_folder}: found {len(image_files)} image files")

        # Process each image file
        for image_file in sorted(image_files):
            try:
                # Extract class ID from filename
                class_id = extract_class_id_from_filename(image_file)

                # Get class name (assuming class_id is 0-indexed or adjust as needed)
                # If your class IDs are 1-indexed, use: class_names[class_id - 1]
                class_name = class_names[class_id]

                # Store mapping with just the filename
                image_to_label_map[image_file] = {
                    'class_id': class_id,
                    'class_name': class_name
                }

            except (ValueError, IndexError) as e:
                print(f"  Warning: Could not process {image_file}: {e}")
                continue

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

    return image_to_label_map


def main():
    parser = argparse.ArgumentParser(
        description='Create image filename to label mapping from files with format ####_CLASSID.png'
    )
    parser.add_argument(
        'image_folders',
        type=str,
        nargs='+',
        help='Paths to folders containing image files (can specify multiple)'
    )
    parser.add_argument(
        'class_names_file',
        type=str,
        help='Path to text file with class names (one per line)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for pickle file (default: image_to_label_map.pkl)'
    )

    args = parser.parse_args()

    create_image_to_label_mapping(args.image_folders, args.class_names_file, args.output)


if __name__ == '__main__':
    main()
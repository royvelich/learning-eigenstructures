import os
import pickle
import argparse
from scipy.io import loadmat


def create_image_to_label_mapping(images_folder, imagelabels_mat_path, output_path=None):
    """
    Create a mapping from image filenames to class IDs for Oxford 102 Flowers.

    Args:
        images_folder: Path to folder containing all flower images
        imagelabels_mat_path: Path to imagelabels.mat file containing the labels
        output_path: Path to save the pickle file (default: image_to_label_map.pkl)
    """
    if output_path is None:
        output_path = 'image_to_label_map.pkl'

    # Load the imagelabels.mat file
    print(f"Loading labels from {imagelabels_mat_path}...")
    labels_data = loadmat(imagelabels_mat_path)

    # Extract labels (1-indexed, values 1-102)
    labels = labels_data['labels'].flatten()

    print(f"Loaded {len(labels)} labels")
    print(f"Label range: {labels.min()} to {labels.max()}")
    print()

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_images = sorted([f for f in os.listdir(images_folder)
                         if os.path.splitext(f)[1] in image_extensions])

    print(f"Found {len(all_images)} images in {images_folder}")
    print()

    # Create the mapping dictionary
    image_to_label_map = {}

    for img_file in all_images:
        # Extract the index from filename
        name_without_ext = os.path.splitext(img_file)[0]

        try:
            # Handle various naming conventions
            if 'image_' in name_without_ext:
                img_idx = int(name_without_ext.split('_')[-1])
            else:
                # Try to extract any number sequence
                import re
                numbers = re.findall(r'\d+', name_without_ext)
                if numbers:
                    img_idx = int(numbers[-1])
                else:
                    print(f"Warning: Could not parse index from filename: {img_file}")
                    continue

            # Get the label for this image (mat file uses 1-indexing)
            if 1 <= img_idx <= len(labels):
                class_id = int(labels[img_idx - 1])  # Convert to 0-indexed for array access

                # Store mapping
                image_to_label_map[img_file] = {
                    'class_id': class_id
                }
            else:
                print(f"Warning: Image index {img_idx} out of range for labels array")

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not process {img_file}: {e}")
            continue

    # Save to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(image_to_label_map, f)

    print(f"Successfully processed {len(image_to_label_map)} images")
    print(f"Saved mapping dictionary to: {output_path}")
    print()

    # Show some statistics
    class_counts = {}
    for info in image_to_label_map.values():
        class_id = info['class_id']
        class_counts[class_id] = class_counts.get(class_id, 0) + 1

    print(f"Number of unique classes: {len(class_counts)}")
    print(f"Class ID range: {min(class_counts.keys())} to {max(class_counts.keys())}")
    print(f"Images per class - Min: {min(class_counts.values())}, Max: {max(class_counts.values())}")
    print()

    # Show example mapping
    print("Dictionary structure example:")
    example_key = list(image_to_label_map.keys())[0]
    print(f"  '{example_key}': {image_to_label_map[example_key]}")

    return image_to_label_map


def main():
    parser = argparse.ArgumentParser(
        description='Create image filename to class ID mapping for Oxford 102 Flowers'
    )
    parser.add_argument(
        'images_folder',
        type=str,
        help='Path to folder containing all flower images'
    )
    parser.add_argument(
        'imagelabels_mat',
        type=str,
        help='Path to imagelabels.mat file containing the labels'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for pickle file (default: image_to_label_map.pkl)'
    )

    args = parser.parse_args()

    create_image_to_label_mapping(args.images_folder, args.imagelabels_mat, args.output)


if __name__ == '__main__':
    main()
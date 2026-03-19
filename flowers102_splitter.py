import os
import shutil
import argparse
from scipy.io import loadmat


def split_oxford_flowers(images_folder, setid_mat_path, output_folder):
    """
    Split Oxford 102 Flowers images into train, val, and test folders.

    Args:
        images_folder: Path to folder containing all flower images
        setid_mat_path: Path to setid.mat file containing the splits
        output_folder: Path to output folder where train/val/test folders will be created
    """
    # Load the setid.mat file
    print(f"Loading splits from {setid_mat_path}...")
    setid_data = loadmat(setid_mat_path)

    # Extract the splits (these are 1-indexed image IDs)
    train_ids = setid_data['trnid'].flatten()
    val_ids = setid_data['valid'].flatten()
    test_ids = setid_data['tstid'].flatten()

    print(f"Train set: {len(train_ids)} images")
    print(f"Validation set: {len(val_ids)} images")
    print(f"Test set: {len(test_ids)} images")
    print()

    # Create output directories
    train_dir = os.path.join(output_folder, 'train')
    val_dir = os.path.join(output_folder, 'val')
    test_dir = os.path.join(output_folder, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"Created output directories in {output_folder}")
    print()

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_images = sorted([f for f in os.listdir(images_folder)
                         if os.path.splitext(f)[1] in image_extensions])

    print(f"Found {len(all_images)} images in {images_folder}")
    print()

    # Create mapping from image index to filename
    # Oxford flowers images are typically named image_00001.jpg, image_00002.jpg, etc.
    # The index in the mat file corresponds to this numbering
    image_index_to_filename = {}
    for img_file in all_images:
        # Extract the index from filename (assuming format like image_00001.jpg)
        name_without_ext = os.path.splitext(img_file)[0]
        # Try to extract number from the filename
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
                    continue
            image_index_to_filename[img_idx] = img_file
        except (ValueError, IndexError):
            print(f"Warning: Could not parse index from filename: {img_file}")
            continue

    # Copy images to respective folders
    def copy_images(image_ids, dest_dir, split_name):
        copied = 0
        for img_id in image_ids:
            if img_id in image_index_to_filename:
                src_path = os.path.join(images_folder, image_index_to_filename[img_id])
                dst_path = os.path.join(dest_dir, image_index_to_filename[img_id])
                shutil.copy2(src_path, dst_path)
                copied += 1
            else:
                print(f"Warning: Image ID {img_id} not found in image folder")
        print(f"Copied {copied} images to {split_name}")

    print("Copying images to splits...")
    copy_images(train_ids, train_dir, 'train')
    copy_images(val_ids, val_dir, 'val')
    copy_images(test_ids, test_dir, 'test')
    print()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Split Oxford 102 Flowers dataset into train/val/test folders'
    )
    parser.add_argument(
        'images_folder',
        type=str,
        help='Path to folder containing all flower images'
    )
    parser.add_argument(
        'setid_mat',
        type=str,
        help='Path to setid.mat file containing the splits'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Path to output folder where train/val/test folders will be created'
    )

    args = parser.parse_args()

    split_oxford_flowers(args.images_folder, args.setid_mat, args.output_folder)


if __name__ == '__main__':
    main()
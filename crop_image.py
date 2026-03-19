#!/usr/bin/env python3
"""
Script to crop an image into multiple parts along the height dimension.
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def is_transparent_line(img_array, y, alpha_threshold=10, min_transparent_ratio=0.95):
    """
    Check if a horizontal line at height y is almost completely transparent.

    Args:
        img_array: numpy array of the image with shape (height, width, channels)
        y: y-coordinate (row) to check
        alpha_threshold: pixels with alpha <= this value are considered transparent (0-255)
        min_transparent_ratio: minimum ratio of transparent pixels (0.0-1.0)

    Returns:
        True if the line is mostly transparent, False otherwise
    """
    # Check if image has alpha channel
    if img_array.shape[2] < 4:
        return False

    width = img_array.shape[1]
    alpha_channel = img_array[y, :, 3]

    # Count pixels below the alpha threshold
    num_transparent = (alpha_channel <= alpha_threshold).sum()
    transparent_ratio = num_transparent / width

    return transparent_ratio >= min_transparent_ratio


def find_nearest_transparent_line(img_array, target_y, search_range=50, alpha_threshold=10, min_transparent_ratio=0.95):
    """
    Find the nearest transparent line to the target y-coordinate, searching only downward.

    Args:
        img_array: numpy array of the image
        target_y: target y-coordinate
        search_range: maximum distance to search downward
        alpha_threshold: pixels with alpha <= this value are considered transparent
        min_transparent_ratio: minimum ratio of transparent pixels for a line

    Returns:
        y-coordinate of the nearest transparent line, or target_y if none found
    """
    height = img_array.shape[0]

    # Search only downward from target
    for distance in range(search_range + 1):
        y = target_y + distance
        if y < height and is_transparent_line(img_array, y, alpha_threshold, min_transparent_ratio):
            return y

    # If no transparent line found, return the target
    return target_y


def crop_image_by_height(image_path, num_parts, output_folder, alpha_threshold=10, min_transparent_ratio=0.95, search_range=100):
    """
    Crop an image into multiple parts along the height dimension.
    Adjusts crop boundaries to align with transparent pixel lines.

    Args:
        image_path: Path to the input image
        num_parts: Number of parts to divide the image into
        output_folder: Path to the output folder where cropped images will be saved
        alpha_threshold: pixels with alpha <= this value are considered transparent (0-255)
        min_transparent_ratio: minimum ratio of transparent pixels for a line (0.0-1.0)
        search_range: maximum pixels to search for transparent line
    """
    # Load the image
    img = Image.open(image_path)

    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    width, height = img.size

    # Convert to numpy array for easier pixel access
    img_array = np.array(img)

    # Calculate the approximate height of each part
    part_height = height // num_parts

    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the base name of the input image (without extension)
    base_name = Path(image_path).stem
    extension = Path(image_path).suffix

    print(f"Image size: {width}x{height}")
    print(f"Cropping into {num_parts} parts (approximate height: {part_height} each)")
    print(f"Transparency settings: alpha_threshold={alpha_threshold}, min_ratio={min_transparent_ratio}")
    print(f"Output folder: {output_folder}")
    print()

    # Find optimal crop boundaries
    crop_boundaries = [0]  # Start with top of image

    for i in range(1, num_parts):
        target_y = i * part_height
        # Find the nearest transparent line
        adjusted_y = find_nearest_transparent_line(img_array, target_y, search_range, alpha_threshold, min_transparent_ratio)
        crop_boundaries.append(adjusted_y)

        if adjusted_y != target_y:
            print(f"Adjusted boundary {i}: {target_y} -> {adjusted_y} (offset: {adjusted_y - target_y})")

    crop_boundaries.append(height)  # End with bottom of image
    print()

    # Crop and save each part
    for i in range(num_parts):
        top = crop_boundaries[i]
        bottom = crop_boundaries[i + 1]

        crop_box = (0, top, width, bottom)

        # Crop the image
        cropped_img = img.crop(crop_box)

        # Save the cropped image
        output_filename = f"{base_name}_part_{i + 1:02d}{extension}"
        output_file_path = output_path / output_filename
        cropped_img.save(output_file_path)

        print(f"Saved part {i + 1}: {output_filename} (size: {cropped_img.size[0]}x{cropped_img.size[1]}, rows {top}-{bottom})")

    print(f"\nSuccessfully cropped image into {num_parts} parts!")


def main():
    parser = argparse.ArgumentParser(
        description="Crop an image into multiple parts along the height dimension."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image"
    )
    parser.add_argument(
        "num_parts",
        type=int,
        help="Number of parts to divide the image into"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the output folder"
    )
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=10,
        help="Pixels with alpha <= this value are considered transparent (0-255, default: 10)"
    )
    parser.add_argument(
        "--min-transparent-ratio",
        type=float,
        default=0.95,
        help="Minimum ratio of transparent pixels for a line to be considered transparent (0.0-1.0, default: 0.95)"
    )
    parser.add_argument(
        "--search-range",
        type=int,
        default=100,
        help="Maximum pixels to search up/down for transparent line (default: 100)"
    )

    args = parser.parse_args()

    # Validate inputs
    if args.num_parts <= 0:
        print("Error: num_parts must be a positive integer")
        return

    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        return

    if not 0 <= args.alpha_threshold <= 255:
        print("Error: alpha_threshold must be between 0 and 255")
        return

    if not 0.0 <= args.min_transparent_ratio <= 1.0:
        print("Error: min_transparent_ratio must be between 0.0 and 1.0")
        return

    # Run the cropping function
    crop_image_by_height(
        args.image_path,
        args.num_parts,
        args.output_folder,
        args.alpha_threshold,
        args.min_transparent_ratio,
        args.search_range
    )


if __name__ == "__main__":
    main()
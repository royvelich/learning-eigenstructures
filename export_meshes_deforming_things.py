import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import trimesh
from joblib import Parallel, delayed


def anime_read(filename):
    """
    filename: .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: [nv, 3], vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: [nt, 3], triangle face data of the 1st frame
        offset_data: [nf-1,nv,3], 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    f.close()
    return nf, nv, nt, vert_data, face_data, offset_data


def write_obj(vertices, faces, filename):
    """
    Write mesh to OBJ file using trimesh.

    Args:
        vertices: [nv, 3] array of vertex positions
        faces: [nt, 3] array of face indices (0-indexed)
        filename: output .obj file path
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(filename)


def extract_anime_to_obj(anime_path, output_dir):
    """
    Extract all frames from an .anime file to OBJ files.

    Args:
        anime_path: path to .anime file
        output_dir: directory to save OBJ files

    Returns:
        tuple: (anime_path, nf, success, error_msg)
    """
    try:
        # Read anime file
        nf, nv, nt, vert_data, face_data, offset_data = anime_read(anime_path)

        # Get the base name from the anime file (e.g., "clarie_run" from "clarie_run.anime")
        base_name = Path(anime_path).stem

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save first frame
        first_frame_path = os.path.join(output_dir, f"{base_name}_frame_0000.obj")
        write_obj(vert_data, face_data, first_frame_path)

        # Save subsequent frames by applying offsets
        for frame_idx in range(nf - 1):
            frame_vertices = vert_data + offset_data[frame_idx]
            frame_path = os.path.join(output_dir, f"{base_name}_frame_{frame_idx + 1:04d}.obj")
            write_obj(frame_vertices, face_data, frame_path)

        return (anime_path, nf, True, None)
    except Exception as e:
        return (anime_path, 0, False, str(e))


def process_dataset(root_dir, output_dir, n_jobs=-1):
    """
    Process entire DeformingThings4D dataset in parallel.

    Args:
        root_dir: root directory of DeformingThings4D dataset
        output_dir: output directory for OBJ files
        n_jobs: number of parallel jobs (-1 uses all available cores)
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)

    # Find all .anime files
    anime_files = list(root_path.rglob("*.anime"))

    if not anime_files:
        print(f"No .anime files found in {root_dir}")
        return

    print(f"Found {len(anime_files)} .anime files")
    print(f"Processing with {n_jobs if n_jobs > 0 else 'all available'} CPU cores...")

    # Prepare arguments for parallel processing
    tasks = []
    for anime_file in anime_files:
        # Get relative path to maintain directory structure
        rel_path = anime_file.relative_to(root_path)

        # Create output directory maintaining the same structure
        seq_output_dir = output_path / rel_path.parent

        tasks.append((str(anime_file), str(seq_output_dir)))

    # Process in parallel with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_anime_to_obj)(anime_path, out_dir)
        for anime_path, out_dir in tqdm(tasks, desc="Processing animations")
    )

    # Collect statistics
    total_frames = 0
    successful = 0
    failed = 0

    for anime_path, nf, success, error_msg in results:
        if success:
            total_frames += nf
            successful += 1
            print(f"  ✓ Processed {Path(anime_path).stem}: {nf} frames")
        else:
            failed += 1
            print(f"  ✗ Error processing {Path(anime_path).stem}: {error_msg}")

    print(f"\n{'=' * 60}")
    print(f"Extraction complete!")
    print(f"Total animations: {len(anime_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total frames extracted: {total_frames}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract DeformingThings4D anime files to OBJ format"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory of DeformingThings4D dataset"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for OBJ files"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 uses all available cores, default: -1)"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process dataset
    process_dataset(args.root_dir, args.output_dir, args.n_jobs)
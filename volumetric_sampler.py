#!/usr/bin/env python3
"""
Volumetric Point Cloud Sampler (SDF-Based)
Converts surface meshes to volumetric point clouds using Signed Distance Fields.
Supports multiple backends including GPU-accelerated options.

Performance comparison:
- point-cloud-utils: Fast CPU-based SDF (uses efficient C++)
- mesh_to_sdf: General purpose, works with non-watertight meshes
- Kaolin: GPU-accelerated (requires PyTorch + CUDA)

For best performance with watertight meshes: point-cloud-utils
For GPU acceleration with PyTorch models: Kaolin
For robustness with problematic meshes: mesh_to_sdf
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import warnings
import time

import numpy as np
import trimesh

# Check which backends are available
HAS_PCU = False
HAS_MESH_TO_SDF = False
HAS_KAOLIN = False

try:
    import point_cloud_utils as pcu

    HAS_PCU = True
except ImportError:
    pass

try:
    import mesh_to_sdf

    HAS_MESH_TO_SDF = True
except ImportError:
    pass

try:
    import kaolin
    import torch

    HAS_KAOLIN = True
except ImportError:
    pass


class SDFVolumetricSampler:
    """Samples points from inside 3D mesh volumes using Signed Distance Fields."""

    def __init__(self, n_points: int = 100000, include_surface: bool = False,
                 backend: str = 'auto', use_gpu: bool = True):
        """
        Initialize the SDF-based volumetric sampler.

        Args:
            n_points: Number of points to sample inside the volume
            include_surface: Whether to include all surface vertices
            backend: SDF computation backend ('auto', 'pcu', 'mesh_to_sdf', 'kaolin')
            use_gpu: Whether to use GPU acceleration if available (for Kaolin)
        """
        self.n_points = n_points
        self.include_surface = include_surface
        self.use_gpu = use_gpu and torch.cuda.is_available() if HAS_KAOLIN else False

        # Select backend
        if backend == 'auto':
            if HAS_PCU:
                self.backend = 'pcu'
            elif HAS_MESH_TO_SDF:
                self.backend = 'mesh_to_sdf'
            elif HAS_KAOLIN:
                self.backend = 'kaolin'
            else:
                raise ImportError(
                    "No SDF backend available. Install one of:\n"
                    "  pip install point-cloud-utils\n"
                    "  pip install mesh-to-sdf\n"
                    "  pip install kaolin (requires PyTorch + CUDA)"
                )
        else:
            self.backend = backend
            if backend == 'pcu' and not HAS_PCU:
                raise ImportError("point-cloud-utils not installed. Run: pip install point-cloud-utils")
            elif backend == 'mesh_to_sdf' and not HAS_MESH_TO_SDF:
                raise ImportError("mesh-to-sdf not installed. Run: pip install mesh-to-sdf")
            elif backend == 'kaolin' and not HAS_KAOLIN:
                raise ImportError("Kaolin not installed. Requires PyTorch + CUDA")

    def check_watertight(self, mesh: trimesh.Trimesh) -> bool:
        """Check if mesh is watertight (closed)."""
        return mesh.is_watertight

    def get_surface_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Get all surface vertices from mesh."""
        return np.array(mesh.vertices)

    def compute_sdf_pcu(self, points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute SDF using point-cloud-utils (Fast C++ implementation).

        Args:
            points: Query points (N, 3)
            mesh: Input mesh

        Returns:
            SDF values (N,) - negative inside, positive outside
        """
        v = np.array(mesh.vertices, dtype=np.float32)
        f = np.array(mesh.faces, dtype=np.int32)

        # Compute signed distances
        sdfs, _, _ = pcu.signed_distance_to_mesh(points, v, f)
        return sdfs

    def compute_sdf_mesh_to_sdf(self, points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute SDF using mesh_to_sdf (Robust, works with non-watertight).

        Args:
            points: Query points (N, 3)
            mesh: Input mesh

        Returns:
            SDF values (N,) - positive inside, negative outside (inverted!)
        """
        # Note: mesh_to_sdf returns positive inside, negative outside (opposite convention)
        sdfs = mesh_to_sdf.mesh_to_sdf(mesh, points, sign_method='normal')
        return -sdfs  # Invert to match standard convention (negative inside)

    def compute_sdf_kaolin(self, points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute SDF using Kaolin (GPU-accelerated with PyTorch).

        Args:
            points: Query points (N, 3)
            mesh: Input mesh

        Returns:
            SDF values (N,) - negative inside, positive outside
        """
        import torch
        import kaolin.ops.mesh
        import kaolin.metrics.trianglemesh

        # Convert to torch tensors
        device = 'cuda' if self.use_gpu else 'cpu'
        vertices = torch.from_numpy(np.array(mesh.vertices)).float().to(device)
        faces = torch.from_numpy(np.array(mesh.faces)).long().to(device)
        query_points = torch.from_numpy(points).float().to(device)

        # Add batch dimension
        vertices_batch = vertices.unsqueeze(0)  # (1, V, 3)
        query_points_batch = query_points.unsqueeze(0)  # (1, N, 3)

        # Index vertices by faces to get face vertices
        face_vertices = kaolin.ops.mesh.index_vertices_by_faces(
            vertices_batch, faces
        )  # (1, F, 3, 3)

        # Compute unsigned distance from each query point to the mesh surface
        distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            query_points_batch, face_vertices
        )  # (1, N)

        # Determine sign using ray casting (inside/outside test)
        # check_sign returns True for points inside the mesh
        is_inside = kaolin.ops.mesh.check_sign(
            vertices_batch, faces, query_points_batch
        )  # (1, N) boolean

        # Convert to signed distance
        # Inside points: negative distance
        # Outside points: positive distance
        signed_distance = torch.where(
            is_inside,
            -distance,  # Negative for inside
            distance  # Positive for outside
        ).squeeze(0)  # Remove batch dimension

        return signed_distance.cpu().numpy()

    def sample_volume_sdf(self, mesh: trimesh.Trimesh, n_samples: int) -> np.ndarray:
        """
        Sample points inside mesh volume using SDF.

        Strategy:
        1. Generate candidate points in bounding box
        2. Compute SDF for all candidates
        3. Keep only points with negative SDF (inside)
        4. Subsample to target count

        Args:
            mesh: Input mesh (should be watertight for best results)
            n_samples: Number of points to sample

        Returns:
            Array of points inside the mesh (N x 3)
        """
        start_time = time.time()

        if not self.check_watertight(mesh) and self.backend == 'pcu':
            warnings.warn(
                "Mesh is not watertight. point-cloud-utils may give incorrect results. "
                "Consider using backend='mesh_to_sdf' for non-watertight meshes."
            )

        # Get bounding box with some padding
        bbox_min = mesh.bounds[0] - 0.05 * (mesh.bounds[1] - mesh.bounds[0])
        bbox_max = mesh.bounds[1] + 0.05 * (mesh.bounds[1] - mesh.bounds[0])
        bbox_size = bbox_max - bbox_min

        print(f"  Using SDF backend: {self.backend}")
        if self.backend == 'kaolin' and self.use_gpu:
            print(f"  GPU acceleration: Enabled")

        # Generate candidate points (oversample to account for rejection)
        # Estimate fill ratio to determine oversample factor
        try:
            mesh_volume = abs(mesh.volume)
            bbox_volume = np.prod(bbox_size)
            fill_ratio = mesh_volume / bbox_volume if bbox_volume > 0 else 0.5
        except:
            fill_ratio = 0.5

        oversample_factor = max(2.0, 1.0 / (fill_ratio + 0.1))
        n_candidates = int(n_samples * oversample_factor * 1.5)

        print(f"  Generating {n_candidates:,} candidate points...")
        candidates = np.random.random((n_candidates, 3))
        candidates = bbox_min + candidates * bbox_size

        # Compute SDF in batches
        print(f"  Computing SDF for {n_candidates:,} points...")
        batch_size = 100000 if self.backend != 'kaolin' else 500000  # Larger batches for GPU

        all_sdfs = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            if self.backend == 'pcu':
                sdfs = self.compute_sdf_pcu(batch, mesh)
            elif self.backend == 'mesh_to_sdf':
                sdfs = self.compute_sdf_mesh_to_sdf(batch, mesh)
            elif self.backend == 'kaolin':
                sdfs = self.compute_sdf_kaolin(batch, mesh)

            all_sdfs.append(sdfs)

        all_sdfs = np.concatenate(all_sdfs)

        # Keep only interior points (negative SDF)
        inside_mask = all_sdfs < 0
        interior_points = candidates[inside_mask]

        elapsed = time.time() - start_time
        print(f"  Found {len(interior_points):,} interior points in {elapsed:.2f}s")

        if len(interior_points) == 0:
            warnings.warn("No interior points found. Mesh may be problematic.")
            return np.array([])

        # Subsample to target count
        if len(interior_points) >= n_samples:
            indices = np.random.choice(len(interior_points), n_samples, replace=False)
            final_points = interior_points[indices]
            print(f"  Subsampled to {n_samples:,} points")
        else:
            final_points = interior_points
            shortfall = n_samples - len(interior_points)
            print(f"  Using {len(interior_points):,} points (short by {shortfall:,})")

        return final_points

    def process_mesh(self, mesh_path: Path) -> Optional[np.ndarray]:
        """Process a single mesh file."""
        try:
            print(f"\n📦 Processing: {mesh_path.name}")

            # Load mesh
            print(f"  Loading mesh...")
            mesh = trimesh.load(mesh_path, force='mesh')

            # Convert to Trimesh if it's a Scene
            if isinstance(mesh, trimesh.Scene):
                print(f"  Converting Scene to single mesh...")
                mesh = mesh.dump(concatenate=True)

            if not isinstance(mesh, trimesh.Trimesh):
                print(f"  ❌ Could not load as Trimesh object")
                return None

            print(f"  Mesh info: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
            print(f"  Bounding box size: {mesh.bounds[1] - mesh.bounds[0]}")

            # Check if watertight
            is_watertight = self.check_watertight(mesh)
            print(f"  Watertight: {'✓ Yes' if is_watertight else '✗ No'}")

            if not is_watertight and self.backend == 'pcu':
                print(f"  ⚠️  Mesh is not watertight - attempting to fix...")
                trimesh.repair.fill_holes(mesh)

                is_watertight = self.check_watertight(mesh)
                if is_watertight:
                    print(f"  ✓ Successfully made mesh watertight")
                else:
                    print(f"  ⚠️  Still not watertight. Results may be inaccurate.")

            # Sample volume points using SDF
            volume_points = self.sample_volume_sdf(mesh, self.n_points)

            if volume_points.size == 0:
                print(f"  ❌ No points sampled from volume")
                if self.include_surface:
                    surface_pts = self.get_surface_points(mesh)
                    if surface_pts.size > 0:
                        print(f"  ✓ Using {len(surface_pts):,} surface points only")
                        return surface_pts
                return None
            else:
                if self.include_surface:
                    surface_pts = self.get_surface_points(mesh)
                    if surface_pts.size > 0:
                        all_points = np.vstack([volume_points, surface_pts])
                        print(f"  ✓ Combined {len(volume_points):,} volume + {len(surface_pts):,} surface points")
                    else:
                        all_points = volume_points
                else:
                    all_points = volume_points
                    print(f"  ✓ Sampled {len(volume_points):,} volume points")

            return all_points

        except Exception as e:
            print(f"  ❌ Error processing {mesh_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def save_point_cloud(self, points: np.ndarray, output_path: Path) -> bool:
        """Save point cloud to PLY file."""
        try:
            point_cloud = trimesh.points.PointCloud(points)
            point_cloud.export(output_path)
            return True
        except Exception as e:
            print(f"  ❌ Error saving point cloud: {str(e)}")
            return False


def get_mesh_files(input_dir: Path, extensions: List[str] = None) -> List[Path]:
    """Get all mesh files from directory."""
    if extensions is None:
        extensions = ['.obj', '.ply', '.off', '.stl', '.mesh', '.3ds', '.dae']

    mesh_files = []
    for ext in extensions:
        mesh_files.extend(input_dir.glob(f'*{ext}'))
        mesh_files.extend(input_dir.glob(f'*{ext.upper()}'))

    return sorted(set(mesh_files))


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Convert surface meshes to volumetric point clouds using Signed Distance Fields'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory containing mesh files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for point clouds (default: input_dir/volumetric_clouds_sdf)'
    )
    parser.add_argument(
        '--n_points',
        type=int,
        default=100000,
        help='Number of points to sample per mesh (default: 100000)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['auto', 'pcu', 'mesh_to_sdf', 'kaolin'],
        default='auto',
        help='SDF computation backend (default: auto)'
    )
    parser.add_argument(
        '--no_gpu',
        action='store_true',
        help='Disable GPU acceleration (Kaolin only)'
    )
    parser.add_argument(
        '--include_surface',
        action='store_true',
        help='Include all surface vertices in addition to volume points'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=None,
        help='Mesh file extensions to process (default: common mesh formats)'
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'volumetric_clouds_sdf'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get mesh files
    mesh_files = get_mesh_files(input_dir, args.extensions)
    if not mesh_files:
        print(f"❌ No mesh files found in {input_dir}")
        sys.exit(1)

    # Print available backends
    print(f"🔧 Available SDF backends:")
    if HAS_PCU:
        print(f"  ✓ point-cloud-utils (fast C++)")
    if HAS_MESH_TO_SDF:
        print(f"  ✓ mesh-to-sdf (robust)")
    if HAS_KAOLIN:
        gpu_status = "with CUDA" if torch.cuda.is_available() else "CPU only"
        print(f"  ✓ Kaolin ({gpu_status})")

    if not (HAS_PCU or HAS_MESH_TO_SDF or HAS_KAOLIN):
        print(f"  ❌ No backends available! Install at least one:")
        print(f"     pip install point-cloud-utils")
        print(f"     pip install mesh-to-sdf")
        sys.exit(1)

    print(f"\n🔍 Found {len(mesh_files)} mesh files")
    print(f"📂 Output directory: {output_dir}")
    print(f"🎯 Points per mesh: {args.n_points:,} volume points")

    if args.include_surface:
        print(f"🎯 Surface points: All vertices will be included")
    print("-" * 50)

    # Initialize sampler
    sampler = SDFVolumetricSampler(
        n_points=args.n_points,
        include_surface=args.include_surface,
        backend=args.backend,
        use_gpu=not args.no_gpu
    )

    # Process each mesh
    successful = 0
    failed = 0

    for mesh_path in mesh_files:
        output_filename = f"{mesh_path.stem}_volumetric_sdf.ply"
        output_path = output_dir / output_filename

        if output_path.exists():
            print(f"\n⏭ Skipping {mesh_path.name} (already processed)")
            successful += 1
            continue

        points = sampler.process_mesh(mesh_path)

        if points is not None and len(points) > 0:
            if sampler.save_point_cloud(points, output_path):
                print(f"  💾 Saved to: {output_path.name}")
                successful += 1
            else:
                failed += 1
        else:
            failed += 1

    # Print summary
    print("\n" + "=" * 50)
    print(f"✅ Successfully processed: {successful}/{len(mesh_files)} meshes")
    if failed > 0:
        print(f"⚠️  Failed: {failed} meshes")
    print(f"📂 Output directory: {output_dir}")


if __name__ == "__main__":
    main()
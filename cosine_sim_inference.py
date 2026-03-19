"""
Inference Script for Neural Laplacian Models

This script loads a trained model checkpoint and runs feed-forward inference
on all shapes in the validation dataset(s).

Usage:
    python inference.py ++checkpoint_path=/path/to/checkpoint.ckpt
"""

# standard library
import os
from pathlib import Path
from typing import List, Dict
import csv
from datetime import datetime

# hydra
import hydra
from omegaconf import DictConfig

# torch
import torch

# pytorch lightning
import pytorch_lightning as pl

# numpy
import numpy as np

# neural-laplacian
from neural_laplacian.modules.laplacian_modules import LaplacianPredictorModule3D


def compute_average_cosine_similarities_by_k(cosine_similarities_list: List[torch.Tensor],
                                              k_ranges: List[int]) -> Dict[str, float]:
    """
    Compute average cosine similarities for different k ranges across all shapes.

    Args:
        cosine_similarities_list: List of cosine similarity tensors, one per shape
        k_ranges: List of k values to compute averages up to (e.g., [5, 10, 15, ..., 50])

    Returns:
        Dictionary mapping k range to average cosine similarity
    """
    results = {}

    for k in k_ranges:
        # Collect all cosine similarities up to and including index k-1 (since k eigenvectors means indices 0 to k-1)
        # For k<=5, we take eigenvectors 0,1,2,3,4 (5 eigenvectors total)
        all_sims_up_to_k = []

        for cosine_sims in cosine_similarities_list:
            if cosine_sims is not None:
                # Take first k eigenvectors (indices 0 to k-1, which gives us k eigenvectors)
                sims_k = cosine_sims[:k]
                all_sims_up_to_k.append(sims_k)

        if len(all_sims_up_to_k) > 0:
            # Concatenate all similarities and compute mean
            all_sims_tensor = torch.cat(all_sims_up_to_k)
            avg_sim = torch.mean(all_sims_tensor).item()
            results[f"k<={k}"] = avg_sim
        else:
            results[f"k<={k}"] = float('nan')

    return results


@hydra.main(version_base="1.2", config_path="config/training")
def main(cfg: DictConfig):
    """Main inference function."""

    # Set seed for reproducibility
    pl.seed_everything(cfg.globals.seed)

    # Check if checkpoint path is provided
    if not hasattr(cfg, 'checkpoint_path') or cfg.checkpoint_path is None:
        raise ValueError("Please provide a checkpoint path using ++checkpoint_path=<path>")

    # Load the model from checkpoint
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    # Note: We pass compute_metrics_on_predict=True to override the checkpoint's saved value
    # This is necessary for checkpoints created before this parameter was added
    model = LaplacianPredictorModule3D.load_from_checkpoint(
        cfg.checkpoint_path,
        compute_metrics_on_predict=True
    )
    model.eval()

    # The model is already configured to compute metrics, but we can still use enable/disable
    # model.enable_metrics_on_predict()  # Not needed since we set it in load_from_checkpoint

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Create data module
    data_module = hydra.utils.instantiate(cfg.data_module.module)

    # Get validation dataloader(s)
    val_dataloaders = data_module.val_dataloader()

    # Handle single dataloader or list of dataloaders
    if not isinstance(val_dataloaders, list):
        val_dataloaders = [val_dataloaders]

    # Define k ranges to compute statistics for
    k_ranges = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]



    # Store results for all datasets
    all_results = []

    # Run inference on each validation dataloader
    for dataloader_idx, val_dataloader in enumerate(val_dataloaders):
        # Get dataset name
        dataset_name = val_dataloader.dataset.name if hasattr(val_dataloader.dataset, 'name') else f"dataset_{dataloader_idx}"

        print(f"\n[Dataset {dataloader_idx + 1}/{len(val_dataloaders)}]: {dataset_name}")
        print(f"  Total shapes: {len(val_dataloader.dataset)}")

        # Collect cosine similarities for all shapes in this dataset
        all_cosine_similarities = []

        # Iterate through all batches in the validation dataloader
        for batch_idx, batch in enumerate(val_dataloader):
            # Move batch to device
            batch = batch.to(device)

            # Run forward pass
            with torch.no_grad():
                laplacian_prediction, processed_batch = model.predict_step(batch, batch_idx=batch_idx)

            # Collect cosine similarities from this batch
            if laplacian_prediction.cosine_similarities_list is not None:
                for cosine_sims in laplacian_prediction.cosine_similarities_list:
                    if cosine_sims is not None:
                        all_cosine_similarities.append(cosine_sims.cpu())

        # Compute average cosine similarities for different k ranges
        if len(all_cosine_similarities) > 0:
            avg_similarities = compute_average_cosine_similarities_by_k(all_cosine_similarities, k_ranges)

            print(f"  Average Cosine Similarities:")
            for k_range, avg_sim in avg_similarities.items():
                print(f"    {k_range}: {avg_sim:.6f}")

            # Store results for CSV
            result_row = {
                'dataset_name': dataset_name,
                'num_shapes': len(val_dataloader.dataset),
                'num_shapes_with_gt': len(all_cosine_similarities)
            }
            result_row.update(avg_similarities)
            all_results.append(result_row)
        else:
            print(f"  Warning: No ground truth eigenvectors available for this dataset")

            # Store empty results
            result_row = {
                'dataset_name': dataset_name,
                'num_shapes': len(val_dataloader.dataset),
                'num_shapes_with_gt': 0
            }
            for k in k_ranges:
                result_row[f"k<={k}"] = float('nan')
            all_results.append(result_row)

    print("\n" + "=" * 80)
    print("Inference complete!")

    # Save results to CSV
    if len(all_results) > 0:
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = Path(cfg.checkpoint_path).stem
        output_filename = f"inference_results_{checkpoint_name}_{timestamp}.csv"

        # Write to CSV
        with open(output_filename, 'w', newline='') as csvfile:
            fieldnames = ['dataset_name', 'num_shapes', 'num_shapes_with_gt'] + [f"k<={k}" for k in k_ranges]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in all_results:
                writer.writerow(result)

        print(f"\nResults saved to: {output_filename}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
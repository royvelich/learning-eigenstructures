import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from neural_laplacian.datasets import ValidationDataset


def reconstruct_normalized_laplacian(eigenvectors: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct normalized Laplacian from spectral decomposition.

    Args:
        eigenvectors: [N, K] matrix of eigenvectors
        eigenvalues: [K] vector of eigenvalues

    Returns:
        L_norm: [N, N] reconstructed normalized Laplacian
    """
    Lambda = torch.diag(eigenvalues)
    L_norm = eigenvectors @ Lambda @ eigenvectors.T
    return L_norm


def reconstruct_standard_laplacian(eigenvectors: torch.Tensor, eigenvalues: torch.Tensor, vertex_areas: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct standard Laplacian from normalized Laplacian eigenvectors/eigenvalues.

    The relationship is: L_norm = M^(-1/2) L M^(-1/2)
    So: L = M^(1/2) L_norm M^(1/2)

    Args:
        eigenvectors: [N, K] matrix of normalized Laplacian eigenvectors
        eigenvalues: [K] vector of eigenvalues
        vertex_areas: [N] vector of vertex areas (mass matrix diagonal)

    Returns:
        L_standard: [N, N] reconstructed standard Laplacian
    """
    # Reconstruct normalized Laplacian first
    L_norm = reconstruct_normalized_laplacian(eigenvectors, eigenvalues)

    # Create mass matrix square root
    M_sqrt = torch.diag(torch.sqrt(vertex_areas))

    # Transform back to standard Laplacian: L = M^(1/2) L_norm M^(1/2)
    L_standard = M_sqrt @ L_norm @ M_sqrt

    return L_standard


def plot_laplacian_heatmaps(L_norm: torch.Tensor, L_standard: torch.Tensor, item_idx: int, save_path: Path = None):
    """
    Plot heatmaps of normalized and standard Laplacians side by side.

    Args:
        L_norm: Normalized Laplacian matrix [N, N]
        L_standard: Standard Laplacian matrix [N, N]
        item_idx: Index of the current item
        save_path: Optional path to save the plot
    """
    # Convert to numpy for plotting
    L_norm_np = L_norm.detach().cpu().numpy()
    L_standard_np = L_standard.detach().cpu().numpy()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot normalized Laplacian
    sns.heatmap(L_norm_np,
                ax=ax1,
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={'label': 'Value'})
    ax1.set_title(f'Normalized Laplacian (Item {item_idx})\nShape: {L_norm_np.shape}')
    ax1.set_xlabel('Vertex Index')
    ax1.set_ylabel('Vertex Index')

    # Plot standard Laplacian
    sns.heatmap(L_standard_np,
                ax=ax2,
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={'label': 'Value'})
    ax2.set_title(f'Standard Laplacian (Item {item_idx})\nShape: {L_standard_np.shape}')
    ax2.set_xlabel('Vertex Index')
    ax2.set_ylabel('Vertex Index')

    # Add statistics as text
    norm_stats = f'Min: {L_norm_np.min():.4f}, Max: {L_norm_np.max():.4f}\nMean: {L_norm_np.mean():.4f}, Std: {L_norm_np.std():.4f}'
    std_stats = f'Min: {L_standard_np.min():.4f}, Max: {L_standard_np.max():.4f}\nMean: {L_standard_np.mean():.4f}, Std: {L_standard_np.std():.4f}'

    ax1.text(0.02, 0.98, norm_stats, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.02, 0.98, std_stats, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f'laplacian_heatmaps_item_{item_idx:03d}.png', dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {save_path / f'laplacian_heatmaps_item_{item_idx:03d}.png'}")

    plt.show()


def analyze_laplacian_properties(L_norm: torch.Tensor, L_standard: torch.Tensor, item_idx: int):
    """
    Analyze and print properties of the Laplacian matrices.

    Args:
        L_norm: Normalized Laplacian matrix [N, N]
        L_standard: Standard Laplacian matrix [N, N]
        item_idx: Index of the current item
    """
    print(f"\n=== Laplacian Analysis for Item {item_idx} ===")

    # Convert to numpy
    L_norm_np = L_norm.detach().cpu().numpy()
    L_standard_np = L_standard.detach().cpu().numpy()

    print(f"Matrix size: {L_norm_np.shape[0]} x {L_norm_np.shape[1]}")

    # Check symmetry
    norm_symmetric = np.allclose(L_norm_np, L_norm_np.T, atol=1e-6)
    std_symmetric = np.allclose(L_standard_np, L_standard_np.T, atol=1e-6)
    print(f"Normalized Laplacian symmetric: {norm_symmetric}")
    print(f"Standard Laplacian symmetric: {std_symmetric}")

    # Check sparsity
    norm_nnz = np.count_nonzero(np.abs(L_norm_np) > 1e-6)
    std_nnz = np.count_nonzero(np.abs(L_standard_np) > 1e-6)
    total_entries = L_norm_np.size

    print(f"Normalized Laplacian sparsity: {norm_nnz}/{total_entries} ({100 * norm_nnz / total_entries:.2f}% non-zero)")
    print(f"Standard Laplacian sparsity: {std_nnz}/{total_entries} ({100 * std_nnz / total_entries:.2f}% non-zero)")

    # Check row sums (should be close to zero for Laplacian)
    norm_row_sums = np.sum(L_norm_np, axis=1)
    std_row_sums = np.sum(L_standard_np, axis=1)

    print(f"Normalized Laplacian row sum stats - Mean: {norm_row_sums.mean():.6f}, Max abs: {np.abs(norm_row_sums).max():.6f}")
    print(f"Standard Laplacian row sum stats - Mean: {std_row_sums.mean():.6f}, Max abs: {np.abs(std_row_sums).max():.6f}")

    # Eigenvalue analysis
    try:
        norm_eigenvals = np.linalg.eigvals(L_norm_np)
        std_eigenvals = np.linalg.eigvals(L_standard_np)

        norm_eigenvals = np.sort(norm_eigenvals)
        std_eigenvals = np.sort(std_eigenvals)

        print(f"Normalized Laplacian eigenvalue range: [{norm_eigenvals[0]:.6f}, {norm_eigenvals[-1]:.6f}]")
        print(f"Standard Laplacian eigenvalue range: [{std_eigenvals[0]:.6f}, {std_eigenvals[-1]:.6f}]")

    except Exception as e:
        print(f"Could not compute eigenvalues: {e}")


@hydra.main(version_base="1.2", config_path="config/training")
def main(cfg: DictConfig):
    """Main function for Laplacian heatmap visualization."""

    # Seed for reproducibility
    pl.seed_everything(cfg.globals.seed)

    # Use the data module instantiated by Hydra
    data_module = hydra.utils.instantiate(cfg.data_module)
    data_loader = data_module.train_dataloader()

    # Create output directory for saving plots
    output_dir = Path("laplacian_heatmaps")

    # Visualization loop
    for batch_idx, batch in enumerate(data_loader):
        # Visualize each item in the batch
        for idx in range(len(batch)):
            data = batch[idx]
            item_idx = batch_idx * data_loader.batch_size + idx

            print(f"\n{'=' * 50}")
            print(f"Processing batch {batch_idx + 1}, item {idx + 1} (global item {item_idx + 1})")
            print(f"{'=' * 50}")

            # Check if we have predicted components
            if not (hasattr(data, 'pred_eigenvectors') and
                    hasattr(data, 'pred_eigenvalues') and
                    hasattr(data, 'pred_vertex_areas')):
                print("Warning: This data doesn't have predicted components. Skipping...")
                continue

            # Extract predicted components
            pred_eigenvectors = data.pred_eigenvectors
            pred_eigenvalues = data.pred_eigenvalues
            pred_vertex_areas = data.pred_vertex_areas

            print(f"Data shape - Vertices: {data.pos.shape[0]}")
            print(f"Eigenvectors: {pred_eigenvectors.shape}")
            print(f"Eigenvalues: {pred_eigenvalues.shape}")
            print(f"Vertex areas: {pred_vertex_areas.shape}")

            # Reconstruct both Laplacians
            print("Reconstructing Laplacians...")
            L_norm = reconstruct_normalized_laplacian(pred_eigenvectors, pred_eigenvalues)
            L_standard = reconstruct_standard_laplacian(pred_eigenvectors, pred_eigenvalues, pred_vertex_areas)

            # Analyze properties
            analyze_laplacian_properties(L_norm, L_standard, item_idx)

            # Plot heatmaps
            print("Plotting heatmaps...")
            plot_laplacian_heatmaps(L_norm, L_standard, item_idx, save_path=output_dir)

            # User interaction
            user_input = input(f"\nItem {item_idx + 1} complete. Press Enter to continue, 's' to save and continue, 'q' to quit: ")
            if user_input.lower() == 'q':
                print(f"\nVisualization complete! Plots saved to: {output_dir}")
                return
            elif user_input.lower() == 's':
                plot_laplacian_heatmaps(L_norm, L_standard, item_idx, save_path=output_dir)


if __name__ == "__main__":
    main()
import torch
import wandb

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

def generate_random_rbf(n_centers=5, sigma=1.0, weight_scale=1.0, device='cuda'):
    kernel_types = ['gaussian', 'inverse_quadratic', 'multiquadric']
    kernel = np.random.choice(kernel_types)

    centers = torch.rand(n_centers, device=device)  # (n_centers,)
    weights = torch.randn(n_centers, device=device) * weight_scale  # (n_centers,)

    def kernel_function(distances_sq):
        if kernel == 'gaussian':
            return torch.exp(-distances_sq / (2 * sigma ** 2))
        elif kernel == 'inverse_quadratic':
            return 1 / (1 + distances_sq / sigma ** 2)
        elif kernel == 'multiquadric':
            return torch.sqrt(1 + distances_sq / sigma ** 2)
        else:
            raise ValueError(f"Unknown kernel type: {kernel}")

    def rbf(x):
        # x shape: (batch_size, num_points)
        batch_size, num_points = x.shape

        # Compute distances_sq: (batch_size, num_points, n_centers)
        distances_sq = (x.unsqueeze(-1) - centers.view(1, 1, -1)) ** 2

        # Compute kernel values: (batch_size, num_points, n_centers)
        basis_vals = kernel_function(distances_sq)

        # Multiply by weights and sum over centers, output shape: (batch_size, num_points)
        result = basis_vals @ weights

        return result  # shape: (batch_size, num_points)

    return rbf

def save_combined_plots(k, epoch, non_uniform_x, f, w_non, uniform_x, f_uniform, w_uniform):
    sns.set_style('whitegrid')

    num_eigen = k
    num_rows = num_eigen + 1
    num_cols = 2

    x_min_non, x_max_non = np.min(non_uniform_x), np.max(non_uniform_x)
    x_min_uniform, x_max_uniform = np.min(uniform_x), np.max(uniform_x)

    height_ratios = [1] * num_eigen + [3]

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols,
                            figsize=(18, (num_rows + 2) * 2),
                            gridspec_kw={'height_ratios': height_ratios})

    for i in range(num_eigen):
        ax_non = axes[i, 0]
        ax_non.plot(non_uniform_x, f[i], color=f'C{i}', label=f'Eigenvector {i+1}')
        ax_non.scatter(non_uniform_x, f[i], color='blue', s=10)
        ax_non.set_xlim(x_min_non, x_max_non)
        ax_non.set_ylim(-0.5, 0.5)
        ax_non.set_ylabel('Value', fontsize=10)
        ax_non.legend(fontsize=8)

        ax_uniform = axes[i, 1]
        ax_uniform.plot(uniform_x, f_uniform[i], color=f'C{i}', label=f'Eigenvector {i+1}')
        ax_uniform.scatter(uniform_x, f_uniform[i], color='blue', s=10)
        ax_uniform.set_xlim(x_min_uniform, x_max_uniform)
        ax_uniform.set_ylim(-0.5, 0.5)
        ax_uniform.set_ylabel('Value', fontsize=10)
        ax_uniform.legend(fontsize=8)

    # Use weights explicitly for dot products
    # weighted_f_non = f * w_non[np.newaxis, :]
    weighted_f_non = f
    dot_non = weighted_f_non @ f.T
    sns.heatmap(dot_non, annot=True, fmt=".2f", cmap='viridis',
                ax=axes[num_eigen, 0], square=True)
    axes[num_eigen, 0].set_title('Weighted Non-Uniform Dot-Product Matrix', fontsize=10)
    axes[num_eigen, 0].set_xlabel('Index', fontsize=10)
    axes[num_eigen, 0].set_ylabel('Index', fontsize=10)

    # weighted_f_uniform = f_uniform * w_uniform[np.newaxis, :]
    weighted_f_uniform = f_uniform
    dot_uniform = weighted_f_uniform @ f_uniform.T
    sns.heatmap(dot_uniform, annot=True, fmt=".2f", cmap='viridis',
                ax=axes[num_eigen, 1], square=True)
    axes[num_eigen, 1].set_title('Weighted Uniform Dot-Product Matrix', fontsize=10)
    axes[num_eigen, 1].set_xlabel('Index', fontsize=10)
    axes[num_eigen, 1].set_ylabel('Index', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    fig.text(0.25, 0.96, "Non-Uniform Sampling (Weighted)", ha='center', fontsize=16, weight='bold')
    fig.text(0.75, 0.96, "Uniform Sampling (Weighted)", ha='center', fontsize=16, weight='bold')

    # Log directly to WandB without saving/loading
    wandb.log({
        f"Weighted Combined Plots at epoch {epoch}": [wandb.Image(fig)]
    })

    plt.close(fig)
import torch
import wandb
import os

import numpy as np
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from PIL import Image
from src.losses.losses import OrthogonalLoss, ReconstructionLoss


class SineActivation(nn.Module):
    def __init__(self):
        super(SineActivation, self).__init__()
    
    def forward(self, x):
        return torch.sin(x)

class ManifoldLaplacianNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        if self.hparams.activation == "sine":
            self.activation = SineActivation()
        elif self.hparams.activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        first_layer = [
            nn.Linear(self.hparams.signal_length, self.hparams.hidden_dim),
            self.activation
        ]

        hidden_layer = [
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            self.activation
        ]
        
        last_layer = [
            nn.Linear(self.hparams.hidden_dim, self.hparams.signal_length*self.hparams.k),
        ]

        layers = first_layer + self.hparams.hidden_layers * hidden_layer + last_layer

        self.model = nn.Sequential(*layers)

        self.orthogonality_criterion = OrthogonalLoss(k=self.hparams.k)
        self.reconstruction_criterion = ReconstructionLoss(k=self.hparams.k)
        self.anchor_points_criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        non_uniform_x, non_uniform_y = batch
        f = self(non_uniform_x)
        orthogonality_loss = self.orthogonality_criterion(f)
        reconstruction_loss = self.reconstruction_criterion(f, non_uniform_y)
        # anchor_points_loss = self.anchor_points_criterion(anchor_signal_1, anchor_signal2)

        # loss = orthogonality_loss + reconstruction_loss + anchor_points_loss
        # loss = orthogonality_loss + reconstruction_loss
        loss = reconstruction_loss + orthogonality_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Expects batch to be a tuple: (non_uniform_x, non_uniform_y)
        where each is a torch.Tensor.
        """
        # Unpack the batch.
        non_uniform_x, _ = batch

        # Compute the eigenvectors using the network for non-uniform input.
        # (Make sure to compute f before converting non_uniform_x to numpy.)
        f = self(non_uniform_x)
        f = f.detach().cpu().numpy().reshape(self.hparams.k, self.hparams.signal_length)

        # Convert non_uniform_x (and non_uniform_y if needed) to NumPy arrays.
        non_uniform_x = non_uniform_x.detach().cpu().numpy().squeeze()
        # non_uniform_y is not used in the plot here, but you could use it if needed.

        # Create uniform input: equally spaced values between 0 and 1.
        uniform_x = np.linspace(0, 1, self.hparams.signal_length)
        uniform_x_tensor = torch.tensor(uniform_x, dtype=torch.float32).to(self.device)
        f_uniform = self(uniform_x_tensor)

        f_uniform = f_uniform.detach().cpu().numpy().reshape(self.hparams.k, self.hparams.signal_length)

        # Create and log the combined plot.
        self.save_combined_plots(non_uniform_x, f, uniform_x, f_uniform)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        # If total_epochs = 100, then steps happen at epoch 30 and 70.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50, 90],  # decay at epochs 30 and 70
            gamma=0.1
        )

        return [optimizer], [scheduler]
        # return optimizer
    

    def save_combined_plots(self, non_uniform_x, f, uniform_x, f_uniform):
        """
        Creates one combined figure with two columns:
        Left Column (Non-Uniform):
            - The top self.hparams.k subplots: each eigenvector (with ylims [-0.15, 0.15])
            plotted versus non_uniform_x (line + scatter markers).
            - The bottom subplot: a heatmap of the dot-product matrix for non-uniform eigenvectors.
        Right Column (Uniform):
            - The top self.hparams.k subplots: each eigenvector (with ylims [-0.15, 0.15])
            plotted versus uniform_x (line + scatter markers).
            - The bottom subplot: a heatmap of the dot-product matrix for uniform eigenvectors.
        Additionally, titles are added above each column.
        """
        sns.set_style('whitegrid')

        num_eigen = self.hparams.k           # Number of eigenvectors (and eigenvector subplots per block)
        num_rows = num_eigen + 1             # k rows for eigenvector plots + 1 row for the heatmap
        num_cols = 2                       # Left column: non-uniform, Right column: uniform

        # Determine x-axis limits for each case.
        x_min_non, x_max_non = np.min(non_uniform_x), np.max(non_uniform_x)
        x_min_uniform, x_max_uniform = np.min(uniform_x), np.max(uniform_x)

        # Set height ratios: give each eigenvector row a ratio of 1 and the heatmap a larger ratio (e.g., 3).
        height_ratios = [1] * num_eigen + [3]

        # Adjust overall figure size.
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols,
                                figsize=(18, (num_rows + 2) * 2),
                                gridspec_kw={'height_ratios': height_ratios})

        # --- Plot Eigenvectors ---
        for i in range(num_eigen):
            # Left Column: Non-uniform eigenvectors.
            ax_non = axes[i, 0]
            ax_non.plot(non_uniform_x, f[i], color=f'C{i}', label=f'Eigenvector {i+1}')
            ax_non.scatter(non_uniform_x, f[i], color='blue', s=10)
            ax_non.set_xlim(x_min_non, x_max_non)
            ax_non.set_ylim(-0.15, 0.15)
            ax_non.set_ylabel('Value', fontsize=10)
            ax_non.legend(fontsize=8)

            # Right Column: Uniform eigenvectors.
            ax_uniform = axes[i, 1]
            ax_uniform.plot(uniform_x, f_uniform[i], color=f'C{i}', label=f'Eigenvector {i+1}')
            ax_uniform.scatter(uniform_x, f_uniform[i], color='blue', s=10)
            ax_uniform.set_xlim(x_min_uniform, x_max_uniform)
            ax_uniform.set_ylim(-0.15, 0.15)
            ax_uniform.set_ylabel('Value', fontsize=10)
            ax_uniform.legend(fontsize=8)

        # --- Plot Dot-Product Heatmaps ---
        # Left Column: Non-uniform dot-product matrix.
        ax_heat_non = axes[num_eigen, 0]
        dot_non = np.dot(f, f.T)
        sns.heatmap(dot_non, annot=True, fmt=".2f", cmap='viridis',
                    ax=ax_heat_non, square=True)
        ax_heat_non.set_title('Non-Uniform Dot-Product Matrix', fontsize=10)
        ax_heat_non.set_xlabel('Index', fontsize=10)
        ax_heat_non.set_ylabel('Index', fontsize=10)
        ax_heat_non.set_aspect('equal', adjustable='box')

        # Right Column: Uniform dot-product matrix.
        ax_heat_uniform = axes[num_eigen, 1]
        dot_uniform = np.dot(f_uniform, f_uniform.T)
        sns.heatmap(dot_uniform, annot=True, fmt=".2f", cmap='viridis',
                    ax=ax_heat_uniform, square=True)
        ax_heat_uniform.set_title('Uniform Dot-Product Matrix', fontsize=10)
        ax_heat_uniform.set_xlabel('Index', fontsize=10)
        ax_heat_uniform.set_ylabel('Index', fontsize=10)
        ax_heat_uniform.set_aspect('equal', adjustable='box')

        # Adjust layout to make space for column titles.
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        # --- Add Column Titles ---
        # Use fig.text to add titles above each column.
        fig.text(0.25, 0.96, "Non-Uniform Sampling", ha='center', va='center', fontsize=16, weight='bold')
        fig.text(0.75, 0.96, "Uniform Sampling", ha='center', va='center', fontsize=16, weight='bold')

        # Save the figure to a temporary file.
        combined_filename = 'combined_plot.png'
        plt.savefig(combined_filename, dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Log the image to Weights & Biases and remove the temporary file.
        image = Image.open(combined_filename)
        wandb.log({f"Combined Non-Uniform & Uniform Plots at epoch {self.current_epoch}": [wandb.Image(image)]})
        os.remove(combined_filename)
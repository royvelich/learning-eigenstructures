import torch
import wandb
import io

import numpy as np
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from PIL import Image
from src.losses.losses import *

class SineActivation(nn.Module):
    def __init__(self):
        super(SineActivation, self).__init__()
    
    def forward(self, x):
        return torch.sin(x)

class AnchorPointsLaplacian(pl.LightningModule):
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
            nn.Linear(self.hparams.hidden_dim, self.hparams.signal_length * self.hparams.k),
        ]

        layers = first_layer + self.hparams.hidden_layers * hidden_layer + last_layer

        self.model = nn.Sequential(*layers)

        # self.orthonormality_criterion = WeightedOrthonormalLoss()
        # self.reconstruction_criterion = WeightedReconstructionLoss()

        self.orthonormality_criterion = OrthogonalLoss(self.hparams.k)
        self.reconstruction_criterion = ReconstructionLoss(self.hparams.k)

        # Register manifold_x as a buffer so that it automatically moves with the model.
        manifold = np.linspace(-np.pi, np.pi, self.hparams.signal_length * 20)
        self.register_buffer('manifold_x', torch.tensor(manifold, dtype=torch.float32))
    
    def forward(self, x):
        return self.model(x)

    def on_train_batch_start(self, batch, batch_idx):
        manifold_y = batch
        # Create indices on the same device as manifold_x
        unsorted_indices = torch.randperm(self.manifold_x.numel(), device=self.manifold_x.device)[:self.hparams.signal_length]
        sorted_indices, _ = torch.sort(unsorted_indices)

        # FIXME: WORKAROUND - JUST TO TEST
        sorted_indices = torch.linspace(0, self.manifold_x.numel() - 1, self.hparams.signal_length).long().to(self.manifold_x.device)
        # print("SHIT")
        self.non_uniform_x = self.manifold_x[sorted_indices]
        # Ensure non_uniform_y is on the same device
        self.non_uniform_y = manifold_y[:, sorted_indices].to(self.manifold_x.device)

    def training_step(self, batch, batch_idx):
        f = self(self.non_uniform_x)
        # f = f.reshape(self.hparams.k, self.hparams.signal_length)

        # ortho_loss = self.orthonormality_criterion(f, self.non_uniform_x)
        # recon_loss = self.reconstruction_criterion(f, self.non_uniform_x, self.non_uniform_y)

        #FIXME: TEST SOMETHING
        f = f.unsqueeze(0).repeat(self.hparams.batch_size, 1)

        ortho_loss = self.orthonormality_criterion(f)
        
        recon_loss = self.reconstruction_criterion(f, self.non_uniform_y)
        
        loss = ortho_loss + recon_loss
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ortho_loss", ortho_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_validation_batch_start(self, batch, batch_idx):
        manifold_y = batch

        unsorted_indices = torch.randperm(self.manifold_x.numel(), device=self.manifold_x.device)[:self.hparams.signal_length]
        sorted_indices, _ = torch.sort(unsorted_indices)

        self.non_uniform_x = self.manifold_x[sorted_indices]
        self.non_uniform_y = manifold_y[:, sorted_indices].to(self.manifold_x.device)

        linear_indices = torch.linspace(0, self.manifold_x.numel() - 1, self.hparams.signal_length).long().to(self.manifold_x.device)
        self.uniform_x = self.manifold_x[linear_indices]
        self.uniform_y = manifold_y[:, linear_indices].to(self.manifold_x.device)

    def validation_step(self, batch, batch_idx):
        """
        Expects batch to be a tuple: (non_uniform_x, non_uniform_y)
        where each is a torch.Tensor.
        """
        non_uniform_f = self(self.non_uniform_x)
        non_uniform_f = non_uniform_f.detach().cpu().numpy().reshape(self.hparams.k, self.hparams.signal_length)
        non_uniform_x = self.non_uniform_x.detach().cpu().numpy().squeeze()
    
        f_uniform = self(self.uniform_x)
        f_uniform = f_uniform.detach().cpu().numpy().reshape(self.hparams.k, self.hparams.signal_length)
        uniform_x = self.uniform_x.detach().cpu().numpy().squeeze()

        self.save_combined_plots(non_uniform_x, non_uniform_f, uniform_x, f_uniform)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50, 90],
            gamma=0.1
        )

        return [optimizer], [scheduler]
    
    def save_combined_plots(self, non_uniform_x, f, uniform_x, f_uniform):
        """
        Creates one combined figure with two columns:
        Left Column (Non-Uniform):
            - The top self.hparams.k subplots: each eigenvector (with ylims [-0.15, 0.15])
            plotted versus non_uniform_x (line + scatter markers).
            - The bottom subplot: a heatmap of the weighted dot-product matrix for non-uniform eigenvectors.
            The weighting uses trapezoidal rule dt, as in the orthonormal loss.
        Right Column (Uniform):
            - The top self.hparams.k subplots: each eigenvector (with ylims [-0.15, 0.15])
            plotted versus uniform_x (line + scatter markers).
            - The bottom subplot: a heatmap of the weighted dot-product matrix for uniform eigenvectors.
            The weighting uses trapezoidal rule dt.
        Additionally, titles are added above each column.
        The figure is then logged to Weights & Biases without saving to disk.
        """
        sns.set_style('whitegrid')

        num_eigen = self.hparams.k
        num_rows = num_eigen + 1
        num_cols = 2

        # Determine x-axis limits
        x_min_non, x_max_non = np.min(non_uniform_x), np.max(non_uniform_x)
        x_min_uniform, x_max_uniform = np.min(uniform_x), np.max(uniform_x)

        height_ratios = [1] * num_eigen + [3]

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols,
                                figsize=(18, (num_rows + 2) * 2),
                                gridspec_kw={'height_ratios': height_ratios})

        # Plot eigenvectors for non-uniform and uniform sampling
        for i in range(num_eigen):
            # Non-uniform subplot
            ax_non = axes[i, 0]
            ax_non.plot(non_uniform_x, f[i], color=f'C{i}', label=f'Eigenvector {i+1}')
            ax_non.scatter(non_uniform_x, f[i], color='blue', s=10)
            ax_non.set_xlim(x_min_non, x_max_non)
            ax_non.set_ylim(-0.15, 0.15)
            ax_non.set_ylabel('Value', fontsize=10)
            ax_non.legend(fontsize=8)

            # Uniform subplot
            ax_uniform = axes[i, 1]
            ax_uniform.plot(uniform_x, f_uniform[i], color=f'C{i}', label=f'Eigenvector {i+1}')
            ax_uniform.scatter(uniform_x, f_uniform[i], color='blue', s=10)
            ax_uniform.set_xlim(x_min_uniform, x_max_uniform)
            ax_uniform.set_ylim(-0.15, 0.15)
            ax_uniform.set_ylabel('Value', fontsize=10)
            ax_uniform.legend(fontsize=8)

        # --- Compute weighted dot-product (Gram) matrices using trapezoidal rule dt ---

        # For non-uniform sampling:
        L_non = len(non_uniform_x)
        dx_non = np.diff(non_uniform_x)
        w_non = np.empty(L_non, dtype=np.float32)
        w_non[0] = dx_non[0] / 2
        w_non[-1] = dx_non[-1] / 2
        if L_non > 2:
            w_non[1:-1] = (dx_non[:-1] + dx_non[1:]) / 2
        else:
            if L_non == 2:
                w_non[1] = dx_non[0] / 2

        # Multiply each eigenvector by sqrt(dt) so that the weighted inner product becomes:
        #   G_non = (f * sqrt(w_non)) @ (f * sqrt(w_non)).T
        weighted_f = f * np.sqrt(w_non)
        G_non = np.dot(weighted_f, weighted_f.T)

        # For uniform sampling:
        L_uniform = len(uniform_x)
        dx_uniform = np.diff(uniform_x)
        w_uniform = np.empty(L_uniform, dtype=np.float32)
        w_uniform[0] = dx_uniform[0] / 2
        w_uniform[-1] = dx_uniform[-1] / 2
        if L_uniform > 2:
            w_uniform[1:-1] = (dx_uniform[:-1] + dx_uniform[1:]) / 2
        else:
            if L_uniform == 2:
                w_uniform[1] = dx_uniform[0] / 2

        weighted_f_uniform = f_uniform * np.sqrt(w_uniform)
        G_uniform = np.dot(weighted_f_uniform, weighted_f_uniform.T)

        # --- Plot the weighted dot-product matrices as heatmaps ---

        ax_heat_non = axes[num_eigen, 0]
        sns.heatmap(G_non, annot=True, fmt=".2f", cmap='viridis',
                    ax=ax_heat_non, square=True)
        ax_heat_non.set_title('Non-Uniform Weighted Dot-Product Matrix', fontsize=10)
        ax_heat_non.set_xlabel('Index', fontsize=10)
        ax_heat_non.set_ylabel('Index', fontsize=10)
        ax_heat_non.set_aspect('equal', adjustable='box')

        ax_heat_uniform = axes[num_eigen, 1]
        sns.heatmap(G_uniform, annot=True, fmt=".2f", cmap='viridis',
                    ax=ax_heat_uniform, square=True)
        ax_heat_uniform.set_title('Uniform Weighted Dot-Product Matrix', fontsize=10)
        ax_heat_uniform.set_xlabel('Index', fontsize=10)
        ax_heat_uniform.set_ylabel('Index', fontsize=10)
        ax_heat_uniform.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        fig.text(0.25, 0.96, "Non-Uniform Sampling", ha='center', va='center', fontsize=16, weight='bold')
        fig.text(0.75, 0.96, "Uniform Sampling", ha='center', va='center', fontsize=16, weight='bold')

        # Save the figure to an in-memory buffer and log to wandb without writing to disk
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({f"Combined Non-Uniform & Uniform Plots at epoch {self.current_epoch}": [wandb.Image(image)]})

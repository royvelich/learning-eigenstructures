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

def init_identity(linear_layer):
    """
    Initializes the weight of a Linear layer to an identity matrix if possible.
    If the layer is not square, fills the overlapping diagonal elements with 1.
    """
    out_features, in_features = linear_layer.weight.shape
    # Zero out everything
    linear_layer.weight.data.zero_()
    # Fill the diagonal with 1 up to the min of out_features, in_features
    for i in range(min(out_features, in_features)):
        linear_layer.weight.data[i, i] = 1.0

class SineActivation(nn.Module):
    def __init__(self):
        super(SineActivation, self).__init__()
    
    def forward(self, x):
        return torch.sin(x)

class LaplacianNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        first_layer = [
            nn.Linear(self.hparams.signal_length, self.hparams.hidden_dim),
            nn.ReLU()
        ]

        hidden_layer = [
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.ReLU()
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
        f = self(batch)
        orthogonality_loss = self.orthogonality_criterion(f)
        reconstruction_loss = self.reconstruction_criterion(f, batch)

        loss = orthogonality_loss + reconstruction_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        f = self(batch)
        f = f.detach().cpu().numpy()
        f = f.reshape(self.hparams.k, self.hparams.signal_length)

        sns.set_style('whitegrid')
        palette = sns.color_palette('husl', self.hparams.k)

        fig, axes = plt.subplots(self.hparams.k, 1, figsize=(12, self.hparams.k * 2), sharex=True)
        for i in range(self.hparams.k):
            sns.lineplot(x=np.arange(self.hparams.signal_length), y=f[i], ax=axes[i], 
                        color=palette[i], label=f'Eigen Vector {i+1}')
            axes[i].set_ylabel('Value', fontsize=12)
            axes[i].legend(fontsize=10)
            axes[i].tick_params(axis='both', labelsize=10)
            axes[i].set_ylim(-0.18, 0.18)  # Customize the range as needed

        # Customize x-axis and overall layout
        axes[-1].set_xlabel('Signal Length', fontsize=12)
        axes[-1].set_xlim(0, self.hparams.signal_length)

        plt.tight_layout()

        # Save the figure to a PNG file. Adjust 'dpi' and filename as needed.
        plt.savefig('my_plot.png', dpi=200, bbox_inches='tight')

        # Close the figure to free memory/resources
        plt.close()

        image = Image.open('my_plot.png')
        wandb.log({f"eigen_vectors at epoch {self.current_epoch}": [wandb.Image(image)]})
        os.remove('my_plot.png')


        dot_product_matrix = np.dot(f, f.T)
        plt.figure(figsize=(10, 8))
        sns.heatmap(dot_product_matrix, annot=True, fmt=".2f", cmap='viridis')
        plt.title('$B^{T} B$')
        plt.xlabel('Index')
        plt.ylabel('Index')
        
        plt.tight_layout()
        plt.savefig('basis.png', dpi=200, bbox_inches='tight')
        plt.close()

        image = Image.open('basis.png')
        wandb.log({f"basis at epoch {self.current_epoch}": [wandb.Image(image)]})
        os.remove('basis.png')
        

        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        # If total_epochs = 100, then steps happen at epoch 30 and 70.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[100, 200],  # decay at epochs 30 and 70
            gamma=0.1
        )

        return [optimizer], [scheduler]
        # return optimizer

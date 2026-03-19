import torch
import wandb
import os

import numpy as np
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from models.area_weights_net import AreaWeightsNet
from models.eigenvalues_net import EigenValuesNet
from models.res_block_1d import ResBlock1D
from losses.orthonormality import OrthonormalityLoss
from losses.reconstruction import ReconstructionLoss
from losses.anchor_consistency import AnchorConsistencyLoss
from utils import generate_random_rbf, save_combined_plots
class DecompositionLaplacianNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.area_weights_net = AreaWeightsNet(self.hparams.signal_length, self.hparams.hidden_dim)
        self.eigenvalues_net = EigenValuesNet(self.hparams.signal_length, self.hparams.hidden_dim, self.hparams.k)

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

        # res_blocks = [
        #     ResBlock1D(self.hparams.k)
        #     for _ in range(20)]

        # layers = res_blocks + [
        #     nn.Linear(self.hparams.signal_length, self.hparams.signal_length)
        # ]

        self.model = nn.Sequential(*layers)

        self.orthonormality_criterion = OrthonormalityLoss()
        self.reconstruction_criterion = ReconstructionLoss()
        self.anchor_criterion = AnchorConsistencyLoss()

    def forward(self, x):
        x = self.model(x)
        # x = x.squeeze(1)
        return x
    
    def on_train_batch_start(self, batch, batch_idx):
        self.rbf_fn = generate_random_rbf(n_centers=10, sigma=0.1)

    def training_step(self, batch, batch_idx):
        x, mask = batch
        B, signal_length = x.shape

        # Validate signal_length
        if signal_length != self.hparams.signal_length:
            print(f"Warning: x has shape {x.shape}, expected {(B, self.hparams.signal_length)}. "
                  f"Interpolating x to match signal_length={self.hparams.signal_length}.")
            x_old = x
            x = torch.linspace(
                x_old[:, 0].unsqueeze(1), 
                x_old[:, -1].unsqueeze(1), 
                self.hparams.signal_length, 
                device=x.device, 
                dtype=x.dtype
            ).expand(B, self.hparams.signal_length)  # Shape: (B, signal_length)

        # Generate random RBF parameters for each batch element
        centers, weights = generate_random_rbf_batch(
            batch_size=B,
            n_centers=10,  # Match original n_centers
            sigma=0.1,     # Match original sigma
            device=x.device,
            x_range=(x[:, 0].min().item(), x[:, -1].max().item())
        )

        # Apply batched RBF functions
        f = apply_rbf_batch(x, centers, weights, sigma=0.1)  # Shape: (B, signal_length)
        area_weights = self.area_weights_net(x)  # Shape: (B, signal_length)
        # print(f"Shape of f: {f.shape}")
        f_hat = self(x).view(-1, self.hparams.k, self.hparams.signal_length)
        
        loss = 0.0
        ## Orthonormality criterion
        # loss = self.orthonormality_criterion(f_hat, x, area_weights)
        ## Reconstruction criterion
        loss += self.reconstruction_criterion(f_hat, f, x, area_weights)
        ## Anchor points criterion
        # loss += self.anchor_criterion(f, f_hat, eigenvalues, mask)

        # weight_sums = w.sum(dim=1)  # shape (B,)
        
        # loss += 0.01 * ((weight_sums - 1.0) ** 2).mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, mask = batch
        print(f"Shape of x: {x.shape}")
        # Non-uniform sampling
        w = self.area_weights_net(x).detach().cpu().numpy().squeeze()
        print(f"Shape of w: {w.shape}")
        f_hat = self(x).view(-1, self.hparams.k, self.hparams.signal_length).detach().cpu().numpy().squeeze()
        print(f"Shape of f_hat: {f_hat.shape}")
        x = x.detach().cpu().numpy().squeeze()

        # Uniform sampling over [0,1]
        x_uniform = torch.linspace(0, 1, steps=self.hparams.signal_length).unsqueeze(0).to(self.device)
        w_uniform = self.area_weights_net(x_uniform).detach().cpu().numpy().squeeze()
        f_hat_uniform = self(x_uniform).view(-1, self.hparams.k, self.hparams.signal_length).detach().cpu().numpy().squeeze()
        x_uniform = x_uniform.detach().cpu().numpy().squeeze()

        print(f_hat_uniform.shape)

        # Pass both sets of data to plotting
        save_combined_plots(self.hparams.k, self.current_epoch, x, f_hat, w, x_uniform, f_hat_uniform, w_uniform)
        

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


def generate_random_rbf_batch(batch_size, n_centers, sigma, device, x_range=(0, 1)):
    """
    Generate batched RBF functions with random centers for each batch element.
    
    Args:
        batch_size (int): Number of batch elements.
        n_centers (int): Number of RBF centers per function.
        sigma (float): Width of Gaussian kernels.
        device (torch.device): Device for tensors.
        x_range (tuple): Range for random centers (min, max).
    
    Returns:
        centers: Shape (B, n_centers), random centers for each batch element.
        weights: Shape (B, n_centers), random weights for each center.
    """
    centers = torch.rand(batch_size, n_centers, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    weights = torch.randn(batch_size, n_centers, device=device)
    return centers, weights

def apply_rbf_batch(x, centers, weights, sigma):
    """
    Apply batched RBF functions to input x.
    
    Args:
        x (torch.Tensor): Shape (B, signal_length), input points.
        centers (torch.Tensor): Shape (B, n_centers), RBF centers.
        weights (torch.Tensor): Shape (B, n_centers), RBF weights.
        sigma (float): Width of Gaussian kernels.
    
    Returns:
        torch.Tensor: Shape (B, signal_length), RBF outputs.
    """
    B, signal_length = x.shape
    B, n_centers = centers.shape
    
    # Reshape for broadcasting: (B, signal_length, n_centers)
    x_expanded = x.unsqueeze(-1)
    centers_expanded = centers.unsqueeze(1)
    
    # Compute Gaussian kernels
    dists = (x_expanded - centers_expanded) ** 2 / (2 * sigma ** 2)
    kernels = torch.exp(-dists)  # Shape: (B, signal_length, n_centers)
    
    # Apply weights and sum over centers
    output = (kernels * weights.unsqueeze(1)).sum(dim=-1)  # Shape: (B, signal_length)
    return output

# def apply_rbf_batch(
#     x: torch.Tensor,
#     centers: torch.Tensor,
#     weights: torch.Tensor,
#     kernel: str = "gaussian",
#     epsilon: float = 1.0,
#     poly_degree: int = 3
# ) -> torch.Tensor:
#     """
#     Apply batched RBF functions to input x, using a variety of kernel types.

#     Args:
#         x (torch.Tensor): Shape (B, signal_length), input points.
#         centers (torch.Tensor): Shape (B, n_centers), RBF centers.
#         weights (torch.Tensor): Shape (B, n_centers), RBF weights.
#         kernel (str): One of
#             - "gaussian"            exp( - (epsilon * r)**2 )
#             - "multiquadric"        sqrt(1 + (epsilon * r)**2)
#             - "inverse_multiquadric" 1 / sqrt(1 + (epsilon * r)**2)
#             - "inverse_quadratic"    1 / (1 + (epsilon * r)**2)
#             - "polynomial"          r**poly_degree
#             - "thin_plate"          r**2 * log(r)
#         epsilon (float): Shape parameter (ε) for kernels that need it.
#         poly_degree (int): Degree k for the “polynomial” (polyharmonic) spline.

#     Returns:
#         torch.Tensor: Shape (B, signal_length), the RBF-weighted sum.
#     """
#     # B, signal_length = x.shape
#     # B, n_centers     = centers.shape

#     # compute pairwise |x-c| distances, shape (B, signal_length, n_centers)
#     r = torch.abs(x.unsqueeze(-1) - centers.unsqueeze(1))

#     if kernel == "gaussian":
#         # φ(r) = exp(-(ε r)^2)
#         phi = torch.exp(- (epsilon * r) ** 2)

#     elif kernel == "multiquadric":
#         # φ(r) = sqrt(1 + (ε r)^2)
#         phi = torch.sqrt(1 + (epsilon * r) ** 2)

#     elif kernel == "inverse_multiquadric":
#         # φ(r) = 1 / sqrt(1 + (ε r)^2)
#         phi = 1.0 / torch.sqrt(1 + (epsilon * r) ** 2)

#     elif kernel == "inverse_quadratic":
#         # φ(r) = 1 / (1 + (ε r)^2)
#         phi = 1.0 / (1 + (epsilon * r) ** 2)

#     elif kernel == "polynomial":
#         # φ(r) = r^k
#         phi = r.pow(poly_degree)

#     elif kernel == "thin_plate":
#         # φ(r) = r^2 * log(r), with φ(0)=0
#         # avoid log(0) by masking
#         # r_safe = r.clone(); r_safe[r_safe==0] = 1
#         r_safe = torch.where(r == 0, torch.ones_like(r), r)
#         phi = r_safe ** 2 * torch.log(r_safe)
#         phi = torch.where(r == 0, torch.zeros_like(r), phi)

#     else:
#         raise ValueError(f"Unknown kernel type '{kernel}'")

#     # apply weights: (B, 1, n_centers) * (B, signal_length, n_centers) -> sum over centers
#     out = (phi * weights.unsqueeze(1)).sum(dim=-1)
#     return out
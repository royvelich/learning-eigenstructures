import torch

import torch.nn as nn
import torch.nn.functional as F

class OrthonormalityLoss(nn.Module):
    def __init__(self):
        super(OrthonormalityLoss, self).__init__()

    def forward(self, f_hat, x, area_weights):
        B, K, L = f_hat.shape

        # Compute pairwise inner products using the trapezoid rule:
        # For each batch, compute the outer product of f_hat along dimension L,
        # then integrate the product over L.
        # inner_products = torch.trapz(f_hat.unsqueeze(2) * f_hat.unsqueeze(1), x.unsqueeze(1).expand(-1, 10, -1), dim=-1)  # Shape (B, K, K)
        W = torch.diag_embed(area_weights)

        weighted_f_hat = torch.einsum("bll, bkl->bkl", W, f_hat)
        inner_products = torch.einsum("bkl, bml->bkm", weighted_f_hat, f_hat)

        # Create identity matrix target (B, K, K)
        identity = torch.eye(K, device=f_hat.device).unsqueeze(0).expand(B, -1, -1)

        # Compute mean squared error between the integrated inner products and the identity matrix
        loss = F.mse_loss(inner_products, identity)

        return loss
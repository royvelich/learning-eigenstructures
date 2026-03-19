import torch

import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, basis, signal, x, area_weights):
        B, K, D = basis.shape

        # # Validate input shapes
        # if signal.shape != (B, D):
        #     raise ValueError(f"Expected signal shape {(B, D)}, got {signal.shape}")
        # if x.shape[0] != B:
        #     raise ValueError(f"Expected x batch size {B}, got {x.shape[0]}")
        
        # # If x has incorrect dimension, interpolate to match D
        # if x.shape[1] != D:
        #     print(f"Warning: x has shape {x.shape}, expected {(B, D)}. Interpolating x to match D={D}.")
        #     x_old = x
        #     # Create new x with D points, linearly spaced within the same range
        #     x = torch.linspace(
        #         x_old[:, 0].unsqueeze(1), 
        #         x_old[:, -1].unsqueeze(1), 
        #         D, 
        #         device=x.device, 
        #         dtype=x.dtype
        #     ).expand(B, D)  # Shape: (B, D)

        # # Compute trapezoidal rule weights
        # h = x[:, 1:] - x[:, :-1]  # Shape: (B, D-1)
        # weights = torch.zeros(B, D, dtype=x.dtype, device=x.device)
        # weights[:, 0] = h[:, 0] / 2
        # if D > 2:
        #     weights[:, 1:-1] = (h[:, :-1] + h[:, 1:]) / 2
        # weights[:, -1] = h[:, -1] / 2

        # # Construct diagonal matrices W, shape: (B, D, D)
        # W = torch.zeros(B, D, D, dtype=x.dtype, device=x.device)
        # indices = torch.arange(D, device=x.device)
        # W[:, indices, indices] = weights

        # # Compute W^{1/2} and W^{-1/2} for weighted QR decomposition
        # sqrt_weights = torch.sqrt(weights)  # Shape: (B, D)
        # inv_sqrt_weights = 1.0 / torch.sqrt(weights + 1e-10)  # Shape: (B, D), added epsilon for stability
        # W_sqrt = torch.zeros_like(W)
        # W_inv_sqrt = torch.zeros_like(W)
        # W_sqrt[:, indices, indices] = sqrt_weights
        # W_inv_sqrt[:, indices, indices] = inv_sqrt_weights

        # # Transpose basis to (B, D, K) for QR decomposition
        # basis_transposed = basis.transpose(1, 2)  # Shape: (B, D, K)

        # # Transform basis: W^{1/2} @ basis_transposed, shape: (B, D, K)
        # transformed_basis = torch.einsum("bdd, bdk->bdk", W_sqrt, basis_transposed)

        # # Perform batched QR decomposition on transformed_basis
        # Q, _ = torch.linalg.qr(transformed_basis)  # Q: (B, D, K), R: (B, K, K)

        # Transform Q back: Q @ W^{-1/2}, shape: (B, D, K)
        # orthogonal_basis = torch.bmm(W_inv_sqrt, Q).transpose(1, 2)  # Shape: (B, D, K)
        # Q = Q.transpose(1, 2)



        # Randomly sample sub_k (number of basis vectors to use) per batch element
        sub_ks = torch.randint(1, K + 1, (B,), device=basis.device)  # (B,)
        # print(f"Shape of sub_ks: {sub_ks.size()}")

        # Create mask to select first sub_ks[i] vectors per batch element
        range_ = torch.arange(K, device=basis.device).unsqueeze(0)  # (1, k)
        # print(f"Shape of range_: {range_.size()}")
        mask = (range_ < sub_ks.unsqueeze(1)).unsqueeze(-1).float()      # (B, k, 1)
        # print(f"Shape of mask: {mask.size()}")

        # Apply mask to basis vectors (zero-out unused vectors)
        truncated_basis = basis * mask  # (B, k, D)

        # Compute weighted coefficients using weights w
        # print(f"Shape of signal: {signal.shape}")
        # print(f"Shape of mul: {mul.shape}")
        # print(f"Shape of x: {x.shape}")
        # Put area weights on a diagonal of a matrix
        W = torch.diag_embed(area_weights)  # Shape: (B, D, D)

        # Multiply W by each vector in the basis
        weighted_basis = torch.einsum("bdd,bkd->bkd", W, truncated_basis)  # Shape: (B, K, D)
        # Compute coefficients using weighted least squares
        coefficients = torch.einsum("bkd,bd->bk", weighted_basis, signal)  # Shape: (B, K)
        # print(coefficients.size())
        # coefficients = torch.einsum("bkd,bd->bk", basis, signal)

        # Reconstruct signal from truncated_basis and weighted coefficients
        reconstructed_signal = torch.einsum("bk,bkd->bd", coefficients, weighted_basis)  # (B, D)

        # Compute weighted MSE loss
        # mse_per_example = ((signal - reconstructed_signal) ** 2 * w).mean(dim=1)  # (B,)
        weighted_error = (torch.einsum("bdd,bd->bd", W, signal - reconstructed_signal))**2

        # Return average loss over batch
        loss = weighted_error.mean()

        return loss

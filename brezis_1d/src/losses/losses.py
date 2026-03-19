import torch

import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn

class WeightedOrthonormalLoss(nn.Module):
    """
    Computes the orthonormality loss for a set of functions f sampled at points x.
    
    Given:
      - f: Tensor of shape (k, L) where each row represents a sampled function.
      - x: 1D Tensor of length L containing the (sorted) sampling points.
    
    The loss approximates the deviation of the weighted inner product
      G_{ij} = ∑_m w_m f_i(x_m) f_j(x_m)
    from the identity matrix, using the trapezoidal rule weights w.
    """
    def __init__(self):
        super(WeightedOrthonormalLoss, self).__init__()

    def forward(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the loss computation.
        
        Parameters:
          f (torch.Tensor): Tensor of shape (k, L).
          x (torch.Tensor): 1D Tensor of length L (must be sorted).
        
        Returns:
          torch.Tensor: Scalar loss value.
        """
        # Number of sample points.
        L = x.shape[0]
        
        # Compute differences between consecutive x-values.
        dx = x[1:] - x[:-1]
        
        # Compute trapezoidal rule weights.
        w = torch.empty(L, device=x.device, dtype=x.dtype)
        w[0] = dx[0] / 2
        w[-1] = dx[-1] / 2
        if L > 2:
            w[1:-1] = (dx[:-1] + dx[1:]) / 2
        else:
            # When L == 2, there's only one interval.
            w[1] = dx[0] / 2
        
        # Multiply f by sqrt(w) to incorporate the weights into the dot product.
        f_weighted = f * torch.sqrt(w)
        # Compute the Gram matrix G.
        G = f_weighted @ f_weighted.T
        
        # Identity matrix for the target orthonormal condition.
        I = torch.eye(f.shape[0], device=f.device, dtype=f.dtype)
        
        # Compute the mean squared error between G and I.
        loss = torch.mean((G - I) ** 2)
        return loss

class OrthogonalLoss(nn.Module):
    def __init__(self, k):
        super(OrthogonalLoss, self).__init__()

        self.k = k

    def forward(self, basis):

        basis = basis.view(basis.size(0), self.k, -1)

        # Compute the Gram matrix
        gram_matrix = torch.matmul(basis, basis.transpose(1, 2))

        # Compute the loss as the mean squared error between the Gram matrix and the identity matrix
        loss = torch.mean((gram_matrix - torch.eye(self.k, device=basis.device)) ** 2)

        return loss

class WeightedReconstructionLoss(nn.Module):
    """
    Computes the reconstruction loss for a batch of signals y (sampled on x)
    using a randomly chosen subset of k functions (eigenfunctions) f sampled on x.

    For each signal y (shape: (signal_length,)), the reconstruction is computed by
    projecting y onto a random subset of the k functions f (each of shape (signal_length,))
    using the weighted inner product (with weights computed via the trapezoidal rule) and then
    re-synthesizing y as a linear combination of the chosen functions.

    Mathematically, for each eigenfunction f_i and signal y:
        c_i = ∫ f_i(x) y(x) dx ≈ sum_m f_i(x_m) y(x_m) * dt_m,
    and the reconstructed signal is:
        y_reconstructed = ∑_{i in subset} c_i f_i(x).

    The loss is the mean squared error between y and y_reconstructed.
    """
    def __init__(self):
        super(WeightedReconstructionLoss, self).__init__()

    def forward(self, f: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            f (torch.Tensor): Tensor of shape (k, L) where each row is an eigenfunction
                              sampled at the points in x.
            x (torch.Tensor): 1D tensor of shape (L,) representing the (sorted) sampling points.
            y (torch.Tensor): Tensor of shape (batch_size, L) containing the batch of signals.

        Returns:
            torch.Tensor: Scalar reconstruction loss.
        """
        # Compute trapezoidal rule weights for x.
        L = x.shape[0]
        dx = x[1:] - x[:-1]
        w = torch.empty(L, device=x.device, dtype=x.dtype)
        w[0] = dx[0] / 2.0
        w[-1] = dx[-1] / 2.0
        if L > 2:
            w[1:-1] = (dx[:-1] + dx[1:]) / 2.0
        else:
            w[1] = dx[0] / 2.0

        batch_size = y.shape[0]
        k_total = f.shape[0]

        # For each sample in the batch, we will randomly choose a subset of eigenfunctions.
        # To do so without an explicit loop we build a mask of shape (batch_size, k_total)
        # that indicates which eigenfunctions to use for each sample.
        # 1. For each sample, generate random numbers for each eigenfunction.
        U = torch.rand(batch_size, k_total, device=f.device)
        # 2. Compute the ranks along the eigenfunction dimension.
        #    (i.e. each row gets values 0 ... k_total-1 in random order)
        ranks = U.argsort(dim=1).argsort(dim=1)  # shape: (batch_size, k_total)
        # 3. For each sample, randomly choose a subset size between 1 and k_total.
        sub_k_tensor = torch.randint(1, k_total + 1, (batch_size,), device=f.device)
        # 4. Create a mask: for sample n, mask[n, i] = 1 if ranks[n, i] < sub_k_tensor[n], else 0.
        mask = (ranks < sub_k_tensor.unsqueeze(1)).float()  # shape: (batch_size, k_total)

        # Compute projection coefficients for each sample for all eigenfunctions.
        # For each sample n and eigenfunction i:
        #   c_all[n, i] = sum_m f[i, m] * w[m] * y[n, m]
        # We compute this via an einsum: "kl,nl->nk", where f*w has shape (k_total, L)
        c_all = torch.einsum("kl,nl->nk", f * w, y)  # shape: (batch_size, k_total)

        # Zero out the coefficients for eigenfunctions not selected.
        c_masked = mask * c_all  # shape: (batch_size, k_total)

        # Reconstruct each sample as a linear combination of the eigenfunctions:
        #   y_reconstructed[n] = sum_i c_masked[n,i] * f[i, :]
        y_reconstructed = torch.einsum("nk,kl->nl", c_masked, f)  # shape: (batch_size, L)

        # Compute the mean squared error between y and y_reconstructed.
        loss = torch.mean((y - y_reconstructed) ** 2)
        return loss

class ReconstructionLoss(nn.Module):
    def __init__(self, k):
        super(ReconstructionLoss, self).__init__()

        self.k = k

    def forward(self, basis, signal):
        # Reshape basis to (B, k, D) if needed
        B = basis.size(0)
        D = signal.size(1)
        basis = basis.view(B, self.k, -1)  # (B, k, D)

        # 1) Sample a random sub_k (in [1..k]) for each element in the batch
        sub_ks = torch.randint(
            low=1, high=self.k + 1, size=(B,), device=basis.device
        )  # shape: (B,)

        # 2) Create a mask that retains only the first sub_ks[i] vectors for the i-th element
        #    We'll do this by comparing range [0..k-1] with sub_ks.
        #    mask shape => (B, k, 1), True/False for which basis vectors to keep
        range_ = torch.arange(self.k, device=basis.device).unsqueeze(0)  # (1, k)
        mask = range_ < sub_ks.unsqueeze(1)  # (B, k)
        mask = mask.unsqueeze(-1)            # (B, k, 1)
        mask = mask.float()

        # 3) Mask out the unused basis vectors
        #    The multiplication effectively sets unused parts to zero.
        truncated_basis = basis * mask  # (B, k, D)

        # 4) Compute coefficients (shape (B, k)) and then reconstruct (shape (B, D))
        #    coefficients[b, k] = sum over D of truncated_basis[b, k, D] * signal[b, D]
        coefficients = torch.einsum("bkd,bd->bk", truncated_basis, signal)

        #    reconstructed[b, D] = sum over k of coefficients[b, k] * truncated_basis[b, k, D]
        reconstructed_signal = torch.einsum("bk,bkd->bd", coefficients, truncated_basis)

        # 5) Compute MSE per example (mean over D), then scale by sub_ks, then average over batch
        loss = (signal - reconstructed_signal).pow(2).mean()   # shape: (B,)
        # loss_per_example = mse * sub_ks.float()                     # shape: (B,)
        # loss = loss_per_example.mean()                              # final scalar

        return loss

        # #FIXME: The loss is not entirely correct. Need to choose random sub_k for each element in the batch

        # basis = basis.view(basis.size(0), self.k, -1)

        # # Pick a single sub_k randomly from 1..k
        # sub_k = torch.randint(1, self.k+1, (1,)).item()

        # # Use only the first sub_k components of the basis
        # truncated_basis = basis[:, :sub_k, :]

        # # Compute coefficients
        # coefficients = torch.einsum("bnd, bd->bn", truncated_basis, signal)

        # # Reconstruct the signal
        # reconstructed_signal = torch.einsum("bn, bnd->bd", coefficients, truncated_basis)

        # # Compute MSE loss for the chosen sub_k
        # loss = sub_k * torch.mean((signal - reconstructed_signal) ** 2)

        # return loss


    # def forward(self, basis, signal):
    #     total_loss = 0.0

    #     # Reshape basis
    #     basis = basis.view(basis.size(0), self.k, -1)

    #     for sub_k in range(1, self.k + 1):
    #         # Use only the first sub_k components of the basis
    #         truncated_basis = basis[:, :sub_k, :]

    #         # Compute coefficients
    #         coefficients = torch.einsum("bnd, bd->bn", truncated_basis, signal)

    #         # Reconstruct the signal using truncated basis
    #         reconstructed_signal = torch.einsum("bn, bnd->bd", coefficients, truncated_basis)

    #         # Compute the loss for this sub_k
    #         loss = torch.mean((signal - reconstructed_signal) ** 2)

    #         # Accumulate the loss
    #         total_loss += loss

    #     return total_loss
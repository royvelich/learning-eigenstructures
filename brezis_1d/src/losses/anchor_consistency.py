import torch
import torch.nn as nn
import torch.nn.functional as F

# class AnchorConsistencyLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, f, f_hat, eigenvalues, mask):
#         B, K, L = f_hat.shape

#         # Compute all transformations at once: (B, L, L)
#         basis_T = f_hat.transpose(1, 2)                         # (B, L, K)
#         eig_diag = torch.diag_embed(eigenvalues)                # (B, K, K)
#         transforms = basis_T @ eig_diag @ basis_T.transpose(1, 2)  # (B, L, L)

#         # Apply transformations to f: (B, L)
#         f_transformed = torch.bmm(transforms, f.unsqueeze(-1)).squeeze(-1)

#         loss = 0.0
#         count = 0

#         # Pre-mask transformed functions
#         masked_f = [f_transformed[i][mask[i].bool()] for i in range(B)]

#         for a in range(B):
#             masked_f_a = masked_f[a]  # already masked, shape: (num_anchors_a,)
#             for b in range(a + 1, B):
#                 masked_f_b = masked_f[b]  # shape: (num_anchors_b,)

#                 # Direct MSE (assumes dimensions match)
#                 mse = F.mse_loss(masked_f_a, masked_f_b)

#                 loss += mse
#                 count += 1

#         loss /= count
#         return loss

class AnchorConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f, f_hat, eigenvalues, mask):
        B, K, L = f_hat.shape

        # Compute transformations once for entire batch
        basis_T = f_hat.transpose(1, 2)  # (B, L, K)
        eig_diag = torch.diag_embed(eigenvalues)  # (B, K, K)
        transforms = basis_T @ eig_diag @ basis_T.transpose(1, 2)  # (B, L, L)

        # Transform f: (B, L)
        f_transformed = torch.bmm(transforms, f.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # Apply masks (assuming equal masked lengths per batch element)
        masked_f = [f_transformed[i][mask[i].bool()] for i in range(B)]  # list of (N,) tensors

        # Stack into tensor: shape (B, N), assumes all masked_f[i] have same N
        masked_f_tensor = torch.stack(masked_f)  # (B, N)

        # Compute pairwise differences efficiently:
        # (B, 1, N) - (1, B, N) → (B, B, N)
        diffs = masked_f_tensor.unsqueeze(1) - masked_f_tensor.unsqueeze(0)

        # Take upper-triangular indices (a < b) without diagonal to avoid duplicate pairs
        triu_indices = torch.triu_indices(B, B, offset=1)

        # Compute MSE over selected pairs
        squared_diffs = diffs[triu_indices[0], triu_indices[1], :] ** 2  # (num_pairs, N)
        mse_per_pair = squared_diffs.mean(dim=1)  # (num_pairs,)

        # Final loss is average over all pairs
        loss = mse_per_pair.mean()

        return loss


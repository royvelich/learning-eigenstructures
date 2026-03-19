import torch
import unittest
import numpy as np
from src.losses.losses import WeightedOrthonormalLoss  # Adjust the import path as needed

class TestWeightedOrthonormalLoss(unittest.TestCase):
    def test_single_function_uniform(self):
        # k = 1, L = 10; x is equally spaced in [0, 1]
        L = 10
        x = torch.linspace(0, 1, L)
        # For equally spaced points on [0,1], the trapezoidal weights sum to 1.
        # A constant function with value 1 normalized by sqrt(sum(w)) is orthonormal.
        # Compute weights manually:
        dx = x[1:] - x[:-1]
        w = torch.empty(L, dtype=torch.float32)
        w[0] = dx[0] / 2.0
        w[-1] = dx[-1] / 2.0
        if L > 2:
            w[1:-1] = (dx[:-1] + dx[1:]) / 2.0
        else:
            w[1] = dx[0] / 2.0
        f0 = torch.ones(L) / torch.sqrt(torch.sum(w))
        f = f0.unsqueeze(0)  # shape: (1, L)
        
        loss_fn = WeightedOrthonormalLoss()
        loss = loss_fn(f, x)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_two_functions_uniform(self):
        # k = 2, L = 3; use a simple uniformly spaced x.
        x = torch.tensor([0.0, 0.5, 1.0])
        # Trapezoidal weights: [0.25, 0.5, 0.25]
        # Construct two functions:
        # f0 = [2, 0, 0] so weighted norm = 2^2*0.25 = 1.
        # f1 = [0, sqrt(2), 0] so weighted norm = (sqrt(2))^2*0.5 = 1.
        f0 = torch.tensor([2.0, 0.0, 0.0])
        f1 = torch.tensor([0.0, np.sqrt(2), 0.0])
        f = torch.stack([f0, f1], dim=0)
        
        loss_fn = WeightedOrthonormalLoss()
        loss = loss_fn(f, x)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_two_functions_nonuniform(self):
        # k = 2, L = 5; x is non-uniformly spaced.
        # Define x (sorted) non-uniformly:
        x = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0], dtype=torch.float32)
        L = x.shape[0]
        # Compute trapezoidal weights (as in the loss function)
        dx = x[1:] - x[:-1]
        w = torch.empty(L, dtype=torch.float32)
        w[0] = dx[0] / 2.0
        w[-1] = dx[-1] / 2.0
        if L > 2:
            w[1:-1] = (dx[:-1] + dx[1:]) / 2.0
        else:
            w[1] = dx[0] / 2.0

        # f0: constant function, normalized: f0 = ones(L)/sqrt(sum(w))
        f0 = torch.ones(L) / torch.sqrt(torch.sum(w))
        # f1: a linear function, shifted by the weighted mean of x, then normalized.
        weighted_mean = torch.sum(x * w) / torch.sum(w)
        # Compute weighted variance:
        weighted_var = torch.sum(w * (x - weighted_mean)**2)
        f1 = (x - weighted_mean) / torch.sqrt(weighted_var)
        # f0 and f1 are orthonormal with respect to the weighted inner product.
        f = torch.stack([f0, f1], dim=0)

        loss_fn = WeightedOrthonormalLoss()
        loss = loss_fn(f, x)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_non_orthonormal(self):
        # k = 2, L = 5; use random f (the loss should be nonnegative).
        torch.manual_seed(42)
        L = 5
        x = torch.linspace(0, 1, L)
        f = torch.randn((2, L))
        loss_fn = WeightedOrthonormalLoss()
        loss = loss_fn(f, x)
        self.assertGreaterEqual(loss.item(), 0.0)

if __name__ == '__main__':
    unittest.main()

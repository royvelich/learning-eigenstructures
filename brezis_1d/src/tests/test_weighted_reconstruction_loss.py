import torch
import unittest
import numpy as np
from src.losses.losses import WeightedReconstructionLoss  # Adjust this import as needed

class TestWeightedReconstructionLoss(unittest.TestCase):
    def setUp(self):
        # Fix the random seed for reproducibility
        torch.manual_seed(0)

    def compute_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute trapezoidal rule weights given the sampling points x.
        """
        L = x.shape[0]
        dx = x[1:] - x[:-1]
        w = torch.empty(L, dtype=x.dtype, device=x.device)
        w[0] = dx[0] / 2.0
        w[-1] = dx[-1] / 2.0
        if L > 2:
            w[1:-1] = (dx[:-1] + dx[1:]) / 2.0
        else:
            w[1] = dx[0] / 2.0
        return w

    def test_zero_signal(self):
        # If y is zero, the reconstruction loss should be 0.
        L = 10
        k = 3
        x = torch.linspace(0, 1, L)
        f = torch.randn((k, L))  # arbitrary eigenfunctions
        y = torch.zeros((5, L))  # batch of 5 zero signals
        loss_fn = WeightedReconstructionLoss()
        loss = loss_fn(f, x, y)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_exact_reconstruction_k1_uniform(self):
        # When k = 1 and y exactly equals the normalized eigenfunction f,
        # the reconstruction should be perfect (loss = 0).
        L = 10
        x = torch.linspace(0, 1, L)  # uniform sampling
        w = self.compute_weights(x)
        # Create a constant function f; normalize it with respect to the weighted norm.
        norm = torch.sqrt(torch.sum(w))
        f_single = torch.ones(L) / norm  # shape: (L,)
        f = f_single.unsqueeze(0)  # shape: (1, L)
        # Let y equal f for each signal in the batch.
        y = f.repeat(5, 1)  # batch size = 5
        loss_fn = WeightedReconstructionLoss()
        loss = loss_fn(f, x, y)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_exact_reconstruction_nonuniform(self):
        # Test exact reconstruction using non-uniformly spaced x.
        # When k = 1 and y equals f (normalized w.r.t. non-uniform x),
        # loss should be zero.
        L = 10
        # Create non-uniform, sorted sampling points.
        x = torch.sort(torch.rand(L))[0]  # sorted random numbers in [0,1]
        w = self.compute_weights(x)
        norm = torch.sqrt(torch.sum(w))
        f_single = torch.ones(L) / norm
        f = f_single.unsqueeze(0)  # shape: (1, L)
        y = f.repeat(3, 1)  # batch size = 3
        loss_fn = WeightedReconstructionLoss()
        loss = loss_fn(f, x, y)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_nonnegative_loss_random(self):
        # For random eigenfunctions and random signals,
        # the loss should be nonnegative.
        L = 10
        k = 4
        batch_size = 7
        x = torch.linspace(0, 1, L)
        f = torch.randn((k, L))
        y = torch.randn((batch_size, L))
        loss_fn = WeightedReconstructionLoss()
        loss = loss_fn(f, x, y)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_loss_is_scalar(self):
        # Check that the returned loss is a scalar, even for a batch of signals.
        L = 8
        k = 3
        batch_size = 4
        x = torch.linspace(0, 1, L)
        f = torch.randn((k, L))
        y = torch.randn((batch_size, L))
        loss_fn = WeightedReconstructionLoss()
        loss = loss_fn(f, x, y)
        self.assertEqual(loss.dim(), 0)

if __name__ == '__main__':
    unittest.main()

import torch

import numpy as np

from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d

class AnchorPointsDataset(Dataset):
    def __init__(self, length=100, signals_per_epoch=1000, anchor_ratio=0.5):
        self.length = length
        self.signals_per_epoch = signals_per_epoch
        self.anchor_ratio = anchor_ratio

        self.manifold_x = np.linspace(0, 1, self.length * 20)

        self.anchor_points_x = np.linspace(self.manifold_x[0], self.manifold_x[-1], int(self.length * anchor_ratio))
        self.anchor_points_x.sort()

        self.anchor_indices = np.searchsorted(self.manifold_x, self.anchor_points_x)

        self.non_uniform_points_x = np.setdiff1d(self.manifold_x, self.anchor_points_x)

        self.rng = np.random.default_rng(seed=1994)

    def __len__(self):
        return self.signals_per_epoch

    def __getitem__(self, idx):

        noise = np.random.rand(len(self.manifold_x))
        smoothing_factor = self.rng.uniform(low=50, high=150)
        manifold_y = gaussian_filter1d(noise, smoothing_factor)

        manifold_y_at_anchor_points = manifold_y[self.anchor_indices]

        sampling_x_1 = self._get_non_uniform_sampling()
        sampling_x_2 = self._get_non_uniform_sampling()

        samplings_x = np.stack((sampling_x_1, sampling_x_2), axis=0).reshape(2, -1)
        

        return torch.tensor(self.anchor_points_x, dtype=torch.float32), torch.tensor(samplings_x, dtype=torch.float32), torch.tensor(manifold_y_at_anchor_points, dtype=torch.float32)

        
    
    def _get_non_uniform_sampling(self):
        
        non_uniform_sampling = np.random.choice(self.non_uniform_points_x, int((1 - self.anchor_ratio) * self.length), replace=False)
        non_uniform_sampling = np.concatenate((non_uniform_sampling, self.anchor_points_x))
        non_uniform_sampling.sort()

        return non_uniform_sampling
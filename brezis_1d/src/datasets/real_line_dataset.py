import torch

import numpy as np

from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d

class RealLineDataset(Dataset):
    def __init__(self, length=100, signals_per_epoch=1000):
        self.length = length
        self.signals_per_epoch = signals_per_epoch

        self.manifold_x = np.linspace(0, 1, self.length * 20)

        self.rng = np.random.default_rng(seed=1994)
    
    def __len__(self):
        return self.signals_per_epoch
    
    def __getitem__(self, idx):
        
        noise = np.random.rand(len(self.manifold_x))
        smoothing_factor = self.rng.uniform(low=50, high=150)
        manifold_y = gaussian_filter1d(noise, smoothing_factor)

        non_uniform_x = np.random.choice(self.manifold_x, self.length, replace=False)
        non_uniform_x.sort()

        non_uniform_y = manifold_y[np.searchsorted(self.manifold_x, non_uniform_x)]

        return torch.tensor(non_uniform_x, dtype=torch.float32), torch.tensor(non_uniform_y, dtype=torch.float32)
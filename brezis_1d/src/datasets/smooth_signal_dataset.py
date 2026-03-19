import torch

import numpy as np

from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d

class SmoothSignalDataset(Dataset):
    """
    Dataset that generates a densely sampled smooth signal
    """
    def __init__(self, length=100, signals_per_epoch=1000, anchor_ratio=0.5):
        self.length = length
        self.signals_per_epoch = signals_per_epoch
        self.anchor_ratio = anchor_ratio

        self.manifold_x = np.linspace(-np.pi, np.pi, self.length * 20)

        self.rng = np.random.default_rng(seed=1994)
    
    def __len__(self) -> int:
        return self.signals_per_epoch
    
    def __getitem__(self, idx) -> torch.Tensor:
        noise = np.random.rand(len(self.manifold_x))
        smoothing_factor = self.rng.uniform(low=200, high=500)
        manifold_y = gaussian_filter1d(noise, smoothing_factor)
        
        return torch.tensor(manifold_y, dtype=torch.float32)
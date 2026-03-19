import torch

import numpy as np

from torch.utils.data import Dataset

class ZeroOne(Dataset):
    def __init__(self, length=1000, signals_per_epoch=1000, fixed_doman=None):
        self.length = length
        self.signals_per_epoch = signals_per_epoch

        self.anchor_points_x = np.linspace(0, 1, int(self.length * 0.2))
        self.x = fixed_doman
        self.x.sort()
        print(f"Generated x: {self.x}")
    
    def __len__(self):
        return self.signals_per_epoch
    
    def __getitem__(self, idx):
        
        # x = np.concatenate([self.anchor_points_x, x])
        
        # x = np.linspace(0, 1, self.length)
        # x = np.geomspace(0.00000001, 1, self.length)
        # print(x)
        # self.x = np.random.uniform(0, 1, size=self.length)
        # self.x.sort()
        mask = np.isin(self.x, self.anchor_points_x)

        return torch.tensor(self.x, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
    

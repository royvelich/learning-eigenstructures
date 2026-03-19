import numpy as np

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from datasets.zero_one import ZeroOne

class ZeroOneDatamodule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
    
    def setup(self, stage=None):
        # domain = np.random.beta(8, 2, size=self.hparams.signal_length)
        domain = np.random.uniform(0, 1, size=self.hparams.signal_length)
        self.train_dataset = ZeroOne(self.hparams.signal_length, self.hparams.signals_per_epoch, domain)
        self.val_dataset = ZeroOne(self.hparams.signal_length, 1, domain)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, prefetch_factor=2)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.hparams.num_workers)
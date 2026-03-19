import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.datasets.anchor_points_dataset import AnchorPointsDataset

class AnchorPointsDatamodule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
    
    def setup(self, stage=None):
        self.train_dataset = AnchorPointsDataset(self.hparams.signal_length, self.hparams.signals_per_epoch)
        self.val_dataset = AnchorPointsDataset(self.hparams.signal_length, 1)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, prefetch_factor=2, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.hparams.num_workers)
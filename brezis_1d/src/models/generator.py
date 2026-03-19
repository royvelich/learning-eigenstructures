import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.losses.losses import DirichletLoss


class Generator(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.lr = lr

        self.criterion = DirichletLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        print(batch[0].size())
        f = self(batch[0])
        loss = self.criterion(f)
        print(loss)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

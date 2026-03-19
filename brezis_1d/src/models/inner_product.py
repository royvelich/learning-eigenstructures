import torch.nn as nn
import pytorch_lightning as pl

class InnerProduct(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.maxpool = nn.MaxPool1d(kernel_size=3)

        self.output_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):

        x = self.mlp(x)
        x = x.permute(0, 2, 1)
        x = self.maxpool(x)
        x = x.squeeze(2)
        x = self.output_block(x)

        return x
        


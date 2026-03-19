import torch

import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from src.losses.losses import OrthogonalLoss, ReconstructionLoss

# Modulation size is 256

class SineLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SirenModel(pl.LightningModule):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., lr=1e-3, epochs=2000, k=5, weight_decay=1e-4):
        super().__init__()

        self.lr = lr
        
        self.k = k
        self.weight_decay = weight_decay

        self.orthogonality_criterion = OrthogonalLoss(k=k)
        self.reconstruction_criterion = ReconstructionLoss(k=k)
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)


    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        f = self(batch)
        orthogonality_loss = self.orthogonality_criterion(f)
        reconstruction_loss = self.reconstruction_criterion(f, batch)

        loss = orthogonality_loss + reconstruction_loss
        # loss = reconstruction_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # If total_epochs = 100, then steps happen at epoch 30 and 70.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[80, 90],  # decay at epochs 30 and 70
            gamma=0.1
        )

        return [optimizer], [scheduler]
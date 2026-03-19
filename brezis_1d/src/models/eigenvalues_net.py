import torch.nn as nn

class EigenValuesNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EigenValuesNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        return self.model(x)
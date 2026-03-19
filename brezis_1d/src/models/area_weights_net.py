import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class AreaWeightsNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.conv = nn.Conv1d(1, 1, 3, stride=1, padding=1, bias=False, padding_mode='circular')
        # self.conv = ResBlock1D(1, kernel_size=5, stride=1)

        self.model = nn.Sequential(
            # nn.ReLU(),
            # nn.Linear(input_dim, input_dim),
            # nn.ReLU(),
            # nn.Linear(input_dim, input_dim),
            # nn.Softplus()
            # nn.Linear(input_dim, input_dim, bias=False),
            # nn.Softplus(),
            # TridiagonalLinear(input_dim),
            nn.Softplus(),
        )

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        x = self.conv(x)
        # print(x.shape)
        x = x.squeeze(1)
        # weights = self.model(x)
        weights = self.model(x)
        # print(x.shape)
        # weights = self.model(x)
        # weights = weights / weights.sum(dim=1, keepdim=True)
        return weights

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, dilation=1):
        super(ResBlock1D, self).__init__()

        padding = ((kernel_size - 1) // 2) * dilation

        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            # nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            # nn.BatchNorm1d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        identity = x
        # print(x.shape)
        out = self.conv_block(x)
        # print(out.shape)
        out += identity
        out = self.relu(out)
        # print(f"Shape of out: {out.shape}")

        return out

class TridiagonalLinear(nn.Module):
    def __init__(self, features: int):
        """
        A linear layer (features → features) with no bias and
        a weight matrix that’s always tridiagonal.
        """
        super().__init__()
        self.features = features

        # the raw weight parameters
        self.weight = nn.Parameter(torch.Tensor(features, features))
        # a fixed mask that has 1s on the main, sub- and super-diagonals
        self.register_buffer('mask', self._make_mask(features))

        self.reset_parameters()

    @staticmethod
    def _make_mask(n: int) -> torch.Tensor:
        m = torch.zeros(n, n)
        idx = torch.arange(n)
        m[idx, idx] = 1.0               # main diagonal
        if n > 1:
            m[idx[:-1], idx[1:]] = 1.0  # super-diagonal
            m[idx[1:], idx[:-1]] = 1.0  # sub-diagonal
        return m

    def reset_parameters(self):
        # standard Kaiming initialization, then zero out off-diagonals
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mask ensures only the tri-diagonals are ever “seen” by the forward
        Wtri = self.weight * self.mask
        return F.linear(x, Wtri, bias=None)
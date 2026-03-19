import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, dilation=1):
        super(ResBlock1D, self).__init__()

        padding = ((kernel_size - 1) // 2) * dilation

        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
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
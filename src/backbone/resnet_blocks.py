import torch.nn as nn
from layers.activation import get_activation
from layers.normalization import get_normalization
from layers.conv_layer import Conv2dRSC

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = Conv2dRSC(in_channels, out_channels, 3, stride, 1)
        self.bn1 = get_normalization(out_channels)
        self.relu = get_activation("relu")
        self.conv2 = Conv2dRSC(out_channels, out_channels, 3, 1, 1)
        self.bn2 = get_normalization(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv2dRSC(in_channels, out_channels, 1, stride),
                get_normalization(out_channels)
            )

    def forward(self, x, mask=None):
        identity = x
        out = self.conv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out, mask)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x, mask)

        out += identity
        out = self.relu(out)
        return out

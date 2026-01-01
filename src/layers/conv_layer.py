import torch
import torch.nn as nn

class Conv2dRSC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x, mask=None):
        x = self.conv(x)
        if mask is not None:
            x = x * mask  # becarefull the shape
        return x

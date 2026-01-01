import torch.nn as nn

def get_normalization(channels, norm_type="batch"):
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Normalization {norm_type} not supported")

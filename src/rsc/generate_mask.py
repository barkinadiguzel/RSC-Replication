import torch

def generate_mask(features, grad, p=0.3, rsc_type="spatial"):
    B, C, H, W = features.size()

    if rsc_type == "spatial":
        g_avg = grad.mean(dim=1)
        threshold = torch.quantile(g_avg.view(B, -1), 1 - p, dim=1)
        mask = (g_avg < threshold.view(B, 1, 1)).float()
        mask = mask.unsqueeze(1).expand(-1, C, -1, -1)  

    elif rsc_type == "channel":
        g_avg = grad.mean(dim=(2, 3))
        threshold = torch.quantile(g_avg, 1 - p, dim=1)
        mask = (g_avg < threshold.unsqueeze(1)).float()
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  

    else:
        raise ValueError("rsc_type must be 'spatial' or 'channel'")

    return mask

import torch
from .mask_generator import generate_mask

class RSCStrategy:
    def __init__(self, p=0.3, rsc_type="spatial", batch_pct=1.0):
        self.p = p
        self.rsc_type = rsc_type
        self.batch_pct = batch_pct

    def select_samples(self, batch_size):
        num_samples = int(batch_size * self.batch_pct)
        idx = torch.randperm(batch_size)[:num_samples]
        return idx

    def apply(self, features, grad):
        mask = generate_mask(features, grad, p=self.p, rsc_type=self.rsc_type)
        return features * mask

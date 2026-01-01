import torch
from .mask_generator import generate_mask

def apply_rsc(features, model_top, labels, loss_fn, p=0.3, rsc_type="spatial"):
    features.requires_grad_(True)
    logits = model_top(features)
    loss = loss_fn(logits, labels)
    grad = torch.autograd.grad(loss, features, retain_graph=True)[0]
    mask = generate_mask(features, grad, p=p, rsc_type=rsc_type)
  
    features_rsc = features * mask
    
    return features_rsc

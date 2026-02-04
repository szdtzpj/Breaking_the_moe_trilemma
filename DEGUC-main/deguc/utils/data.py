import torch

def synthetic_batch(batch_size, dim, device):
    return torch.randn(batch_size, dim, device=device)
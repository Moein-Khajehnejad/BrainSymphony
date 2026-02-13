import torch
import torch.nn as nn

class TokenMasker1D(nn.Module):
    """Masks time points for the Temporal Transformer"""
    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        # x: (B, R, T)
        B, R, T = x.shape
        mask = torch.ones((B, T), device=x.device)
        num_mask = int(self.mask_ratio * T)
        
        for b in range(B):
            idx = torch.randperm(T)[:num_mask]
            mask[b, idx] = 0
        return mask # (B, T)

class TokenMasker2D(nn.Module):
    """Masks ROIs for the Spatial Transformer"""
    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        # x: (B, R, T)
        B, R, T = x.shape
        mask = torch.ones((B, R), device=x.device)
        num_mask = int(self.mask_ratio * R)
        
        for b in range(B):
            idx = torch.randperm(R)[:num_mask]
            mask[b, idx] = 0
        return mask # (B, R)

class TimePointMasker(nn.Module):
    """Masks individual (ROI, Time) entries independently"""
    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        B, R, T = x.shape
        mask = torch.ones((B, R, T), device=x.device)
        num_mask = int(self.mask_ratio * T)
        
        for b in range(B):
            for r in range(R):
                idx = torch.randperm(T)[:num_mask]
                mask[b, r, idx] = 0
        return mask

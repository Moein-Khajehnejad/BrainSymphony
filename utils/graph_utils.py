import torch
import torch.nn.functional as F

def mask_random_edges(adj, mask_ratio=0.2):
    """
    Randomly masks edges in adjacency matrix.
    adj: (B, N, N)
    Returns: mask (B, N, N) where 1=Keep, 0=Mask
    """
    if adj.dim() == 3:
        B, N, _ = adj.shape
        mask = torch.ones_like(adj)
        num_mask = int(mask_ratio * N * N)
        for b in range(B):
            indices = torch.randperm(N*N)[:num_mask]
            m_flat = torch.ones(N*N, device=adj.device)
            m_flat[indices] = 0
            mask[b] = m_flat.view(N, N)
        return mask
    return torch.ones_like(adj) # Fallback

def weighted_mse_loss(pred_adj, true_adj, mask):
    """
    Calculates MSE only on masked edges.
    """
    return F.mse_loss(pred_adj * (1-mask), true_adj * (1-mask))

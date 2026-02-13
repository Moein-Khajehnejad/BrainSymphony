import torch
import torch.nn as nn

class AdaptiveGatingFusion(nn.Module):
    """
    Fuses Functional (fMRI) and Structural (SC) embeddings using a learnable gate.
    
    Formula:
    E_fused = Gate * E_func + (1 - Gate) * E_struct
    Gate = Sigmoid(Linear([E_func || E_struct]))
    """
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embed_func, embed_struct):
        """
        embed_func: (B, R, D) - From fMRI Perceiver
        embed_struct: (B, R, D) - From Structural Graph Transformer
        """
        # Concatenate features along the channel dimension
        combined = torch.cat([embed_func, embed_struct], dim=-1) # (B, R, 2D)
        
        # Compute gate
        gate = self.sigmoid(self.gate_proj(combined)) # (B, R, D)
        
        # Weighted sum
        fused = gate * embed_func + (1 - gate) * embed_struct
        
        return fused, gate

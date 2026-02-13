import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# Positional Encodings
# ==========================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, dim)

    def forward(self, x):
        """
        x: (batch, time_steps, dim)
        """
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :], self.pe[:, :seq_len, :]

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim))
        self.ln = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        """
        B, N, _ = x.shape
        pe = self.pe[:, :N, :].expand(B, -1, -1)
        return x + pe, pe

class BrainGradientPositionalEncoding(nn.Module):
    def __init__(self, dim, num_gradients=10, delta=0.5):
        super().__init__()
        self.num_gradients = num_gradients
        self.delta = delta
        self.proj = nn.Linear(self.num_gradients, dim)
        self.ln = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, gm, x):
        """
        gm: Gradient maps tensor (batch, ROIs, num_gradients) or (ROIs, num_gradients)
        """
        G_proj = self.proj(gm)  
        return self.ln(x + self.alpha * G_proj), self.alpha * G_proj

# ==========================================
# Transformer Blocks
# ==========================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape 
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, H, T, D_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = (attn_weights @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        out = self.fc_out(out)
        return out, attn_weights

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normed_x = self.norm1(x)
        attn_out, weights = self.attn(normed_x)
        x = x + self.dropout(attn_out)
        
        normed_mlp = self.norm2(x)
        mlp_out = self.mlp(normed_mlp)
        x = x + self.dropout(mlp_out)
        return x, weights

# ==========================================
# Encoders
# ==========================================

class TemporalTransformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, mlp_dim, time_steps, max_len):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_dim) for _ in range(num_layers)])
        self.pos_enc = SinusoidalPositionalEncoding(dim, max_len)
        # Projects ROI dimension (R) to latent dim (D)
        # Note: Input is (B, R, T) -> Transposed to (B, T, R) -> Projected to (B, T, D)
        self.embedding_layer = nn.LazyLinear(dim) 
    
    def forward(self, x):
        """
        x: (batch, ROIs, time_steps)
        Returns: (batch, time_steps, dim)
        """
        x = x.transpose(1, 2)  # (B, T, R)
        x = self.embedding_layer(x)  # (B, T, D)
        x, raw_pe = self.pos_enc(x)

        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_all.append(attn_weights)
        return x, attn_weights_all, raw_pe

class SpatialTransformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, mlp_dim, time_steps, max_len, 
                 num_gradients=10, delta=0.5, pos_enc_type="learnable"):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_dim) for _ in range(num_layers)])
        self.embedding_layer = nn.Linear(time_steps, dim)
        self.pos_enc_type = pos_enc_type

        if pos_enc_type == "learnable":
            self.pos_enc = LearnablePositionalEncoding(dim, max_len)
        elif pos_enc_type == "brain_gradient":
            self.pos_enc = BrainGradientPositionalEncoding(dim, num_gradients, delta)
        else:
            raise ValueError(f"Unknown pos_enc_type: {pos_enc_type}")

    def forward(self, x, gm=None):
        """
        x: (batch, ROIs, time_steps)
        gm: Gradient Maps (optional, required if pos_enc_type='brain_gradient')
        Returns: (batch, ROIs, dim)
        """
        x = self.embedding_layer(x) # (B, R, D)
        
        if self.pos_enc_type == "brain_gradient":
            x, raw_pe = self.pos_enc(gm, x)
        else:
            x, raw_pe = self.pos_enc(x)

        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_all.append(attn_weights)
        return x, attn_weights_all, raw_pe

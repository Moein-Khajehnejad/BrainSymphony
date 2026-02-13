import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatial_temporal import TransformerBlock

class PerceiverIOModelROIWise(nn.Module):
    def __init__(self,
                 input_dim,          # Total dim after concat (e.g. 3*128)
                 latent_dim=128,
                 num_rois=450,
                 num_heads=8,
                 time_steps=200,
                 dropout=0.1,
                 num_transformer_blocks=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_rois = num_rois
        self.time_steps = time_steps

        # One learnable latent vector per ROI
        self.latents = nn.Parameter(torch.randn(num_rois, latent_dim))

        # Input projection (Projecting large fused input to latent dim)
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.input_ln = nn.LayerNorm(latent_dim)

        # Cross-Attention: Latents query the fused input
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim,
                                                 num_heads=num_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(latent_dim)

        # Latent Transformer (Processing interactions between ROIs)
        mlp_dim = latent_dim * 4
        self.latent_transformers = nn.ModuleList([
            TransformerBlock(dim=latent_dim, num_heads=num_heads, 
                             mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        # Decoder Head: Map latent (B, R, D) back to Time Series (B, R, T)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.GELU(),
            nn.Linear(latent_dim*2, time_steps)
        )

        # Feature alignment projections
        # We assume input_dim is roughly 3 equal parts, but flexible
        self.roi_proj = nn.Linear(128, input_dim // 3) 
        self.time_proj = nn.Linear(128, input_dim // 3)
        self.signal_proj = nn.Linear(64, input_dim // 3)

    @staticmethod
    def _apply_linear_in_chunks(x2d, layer, chunk=200_000):
        outs = []
        with torch.cuda.amp.autocast(enabled=False):
            x = x2d.float()
            w = layer.weight.float()
            b = layer.bias.float() if layer.bias is not None else None
            for i in range(0, x.shape[0], chunk):
                outs.append(F.linear(x[i:i+chunk], w, b))
        return torch.cat(outs, dim=0).to(x2d.dtype)
    
    def forward(self, roi_embeds, time_embeds, context_features, roi_pos_pe):
        """
        roi_embeds:       (B, R, D_roi)
        time_embeds:      (B, T, D_time)
        context_features: (B, R, T, D_signal)
        roi_pos_pe:       (B, R, latent_dim) - Spatial PE to add to latents
        """
        B, R, D_roi = roi_embeds.shape
        _, T, D_time = time_embeds.shape

        # 1. Expand and Concat (Fusion)
        roi_exp  = roi_embeds.unsqueeze(2).expand(-1, -1, T, -1)   # (B, R, T, D)
        time_exp = time_embeds.unsqueeze(1).expand(-1, R, -1, -1)  # (B, R, T, D)
        
        roi_proj    = self.roi_proj(roi_exp)
        time_proj   = self.time_proj(time_exp)
        signal_proj = self.signal_proj(context_features)

        fused = torch.cat([roi_proj, time_proj, signal_proj], dim=-1) # (B, R, T, input_dim)
        
        # 2. Project Input to Latent Space (Chunked for memory efficiency)
        fused_2d = fused.view(B * R * T, -1)
        proj_2d = self._apply_linear_in_chunks(fused_2d, self.input_proj)
        fused_proj = proj_2d.view(B, R * T, self.latent_dim)
        fused_proj = self.input_ln(fused_proj)

        # 3. Cross Attention
        # Add spatial positional encoding to the latents
        latents_query = self.latents.unsqueeze(0).expand(B, -1, -1) + roi_pos_pe
        
        latents, attn_weights = self.cross_attn(query=latents_query, 
                                                key=fused_proj, 
                                                value=fused_proj)
        latents = self.cross_attn_norm(latents)

        # 4. Latent Processing
        latent_attentions = []
        for block in self.latent_transformers:
            latents, w = block(latents)
            latent_attentions.append(w)

        # 5. Decode
        recon = self.head(latents) # (B, R, T)
        
        return recon, latents, attn_weights

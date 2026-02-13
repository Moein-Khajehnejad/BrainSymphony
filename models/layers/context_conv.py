import torch
import torch.nn as nn

class MaskedConv1DTemporalModel(nn.Module):
    def __init__(self, input_channels=1, signal_dim=64, decoder_hidden_dims=[32, 8]):
        super().__init__()
        self.signal_dim = signal_dim

        # Encoder: Conv1D over time for each ROI
        self.conv1d = nn.Conv1d(
            in_channels=input_channels,
            out_channels=signal_dim,
            kernel_size=5,
            padding=2  # preserves time length
        )

        # Learnable token to replace masked time points
        self.mask_token = nn.Parameter(torch.randn(1))

        # Decoder for self-supervised pretraining (MLP)
        decoder_layers = []
        prev_dim = signal_dim
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.GELU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, 1))
        self.decoder = nn.Sequential(*decoder_layers)

    def _mlp_decode_in_chunks(self, feats_2d: torch.Tensor, chunk: int = 200_000) -> torch.Tensor:
        outs = []
        # Force float32 for stability
        with torch.cuda.amp.autocast(enabled=False):
            f32 = feats_2d.float()
            for i in range(0, f32.shape[0], chunk):
                outs.append(self.decoder(f32[i:i+chunk]))
        return torch.cat(outs, dim=0)

    def forward(self, x, mask=None):
        """
        x: (B, R, T)
        mask: (B, R, T) boolean or 0/1, where 0/False indicates masked
        """
        B, R, T = x.shape
        device, dtype = x.device, x.dtype
    
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool, device=device)
        else:
            if mask.dtype != torch.bool:
                mask = mask > 0
            mask = mask.to(device)
    
        token = self.mask_token.to(device=device, dtype=dtype)
        x_masked = torch.where(mask, x, token.expand_as(x))
    
        # Flatten ROIs into batch for Conv1D: (B*R, 1, T)
        x_flat = x_masked.contiguous().view(B * R, 1, T)
        features = self.conv1d(x_flat)                   # (B*R, D, T)
        features = features.transpose(1, 2).contiguous() # (B*R, T, D)
    
        # Decode for reconstruction loss
        BR, TT, D = features.shape
        feats_2d = features.reshape(BR * TT, D)
        decoded  = self._mlp_decode_in_chunks(feats_2d)
        preds    = decoded.view(B, R, T)
    
        # Reshape features back to (B, R, T, D)
        features = features.view(B, R, T, D)
        
        return preds, features

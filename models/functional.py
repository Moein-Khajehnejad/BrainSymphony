
import torch
import torch.nn as nn

from .layers import (
    SpatialTransformer, 
    TemporalTransformer, 
    MaskedConv1DTemporalModel, 
    PerceiverIOModelROIWise
)

class BrainSymphonyFMRI(nn.Module):
    def __init__(self, config):
        """
        config: Dictionary containing hyperparameters
        """
        super().__init__()
        
        self.spatial = SpatialTransformer(
            num_layers=config['num_layers'],
            dim=config['dim'],
            num_heads=config['num_heads'],
            mlp_dim=config['mlp_dim'],
            time_steps=config['time_steps'],
            max_len=config['max_len_spatial'],
            num_gradients=config.get('num_gradients', 10),
            delta=config.get('delta', 0.5),
            pos_enc_type=config.get('pos_enc_type', 'learnable')
        )

        self.temporal = TemporalTransformer(
            num_layers=config['num_layers'],
            dim=config['dim'],
            num_heads=config['num_heads'],
            mlp_dim=config['mlp_dim'],
            time_steps=config['time_steps'],
            max_len=config['max_len_temp']
        )

        self.context = MaskedConv1DTemporalModel(
            input_channels=1, 
            signal_dim=64, 
            decoder_hidden_dims=[32, 8]
        )

        # Input dim for Perceiver is 3 parts: ROI_dim (128) + Time_dim (128) + Context_dim (128 approx)
        # We assume the projectors in Perceiver align these.
        # Based on notebook: input_dim=64*3 (if signal_proj=64) or similar. 
        # You set input_dim=192 in notebook (64*3).
        
        self.perceiver = PerceiverIOModelROIWise(
            input_dim=config['dim'] * 3, # e.g. 128*3 = 384
            latent_dim=config['dim'],
            num_rois=450,
            num_heads=2,
            time_steps=config['time_steps'],
            dropout=config.get('dropout', 0.1)
        )

    def load_pretrained_components(self, spatial_path, temporal_path, context_path):
        """Helper to load the self-supervised weights"""
        # Note: You might need to adjust keys if saved with 'module.' prefix or wrapper classes
        try:
            self.spatial.load_state_dict(torch.load(spatial_path), strict=False)
            print("Loaded Spatial Transformer")
        except Exception as e:
            print(f"Error loading spatial: {e}")

        try:
            self.temporal.load_state_dict(torch.load(temporal_path), strict=False)
            print("Loaded Temporal Transformer")
        except Exception as e:
            print(f"Error loading temporal: {e}")

        try:
            self.context.load_state_dict(torch.load(context_path), strict=False)
            print("Loaded Context Model")
        except Exception as e:
            print(f"Error loading context: {e}")

    def forward(self, fmri, gm=None):
        """
        fmri: (B, R, T)
        gm: Gradient Maps (optional)
        """
        # 1. Component Encoding (Frozen or Finetuned)
        # We assume no masking during fusion/inference phase
        
        # Spatial
        roi_embeds, _, roi_pos_pe = self.spatial(fmri, gm) # (B, R, D)
        
        # Temporal
        time_embeds, _, _ = self.temporal(fmri) # (B, T, D)
        
        # Context
        _, context_features = self.context(fmri, mask=None) # (B, R, T, D_signal)

        # 2. Perceiver Fusion
        recon, latents, attn = self.perceiver(roi_embeds, time_embeds, context_features, roi_pos_pe)
        
        return recon, latents, attn

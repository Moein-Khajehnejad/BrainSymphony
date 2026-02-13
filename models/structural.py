import torch.nn as nn
from .layers.signed_graph import SignedGraphTransformer

class BrainSymphonyStructural(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SignedGraphTransformer(
            num_nodes=config['rois'],
            in_dim=config['rois'], # Features are usually One-Hot or connectivity profiles
            hidden_dim=config['dim'],
            num_layers=config['num_layers_struct'],
            num_heads=config['num_heads_struct'],
            dropout=config.get('dropout', 0.1),
            attention_type=config.get('attention_type', 'first_hop'),
            max_hops=config.get('max_hops', 2),
            diffusion_theta=config.get('diffusion_theta', 0.1),
            pos_encoding_type=config.get('pos_enc_type_struct', 'learnable')
        )

    def forward(self, x, adj_matrix, mask_edges=None):
        """
        x: Node features (B, R, InDim)
        adj_matrix: (B, R, R)
        mask_edges: (B, R, R) (Optional, for pretraining)
        """
        h, pred_weights, attn = self.encoder(x, adj_matrix, mask_edges)
        return h, pred_weights, attn

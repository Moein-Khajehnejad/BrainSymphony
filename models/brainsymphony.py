import torch
import torch.nn as nn
from .functional import BrainSymphonyFMRI
from .structural import BrainSymphonyStructural
from .fusion import AdaptiveGatingFusion

class BrainSymphony(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mode = config.get('mode', 'multimodal') # 'multimodal', 'functional', 'structural'
        
        # Initialize modules based on mode
        if self.mode in ['functional', 'multimodal']:
            self.functional_module = BrainSymphonyFMRI(config)
            
        if self.mode in ['structural', 'multimodal']:
            self.structural_module = BrainSymphonyStructural(config)
            
        if self.mode == 'multimodal':
            self.fusion_module = AdaptiveGatingFusion(config['dim'])

    def forward(self, batch):
        """
        batch: Dict containing 'fmri', 'gm', 'sc_adj', 'sc_feats', 'sc_mask'
        """
        outputs = {}
        
        # 1. Functional Branch
        func_embeds = None
        if self.mode in ['functional', 'multimodal'] and 'fmri' in batch:
            # recon, latents, attn
            recon, func_embeds, func_attn = self.functional_module(
                batch['fmri'], 
                gm=batch.get('gm', None)
            )
            outputs['func_recon'] = recon
            outputs['func_embeds'] = func_embeds
            outputs['func_attn'] = func_attn

        # 2. Structural Branch
        struct_embeds = None
        if self.mode in ['structural', 'multimodal'] and 'sc_adj' in batch:
            # h, pred_weights, attn
            struct_embeds, pred_weights, struct_attn = self.structural_module(
                batch['sc_feats'], 
                batch['sc_adj'], 
                mask_edges=batch.get('sc_mask', None)
            )
            outputs['struct_embeds'] = struct_embeds
            outputs['struct_pred_weights'] = pred_weights
            outputs['struct_attn'] = struct_attn

        # 3. Fusion
        if self.mode == 'multimodal' and func_embeds is not None and struct_embeds is not None:
            fused_embeds, gates = self.fusion_module(func_embeds, struct_embeds)
            outputs['fused_embeds'] = fused_embeds
            outputs['fusion_gates'] = gates
        
        return outputs

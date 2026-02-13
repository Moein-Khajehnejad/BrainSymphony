import torch
from torch.utils.data import Dataset
import numpy as np

class BrainSymphonyDataset(Dataset):
    """
    Unified Dataset for BrainSymphony.
    Handles fMRI (BOLD), Structural Connectivity (SC), and Gradient Maps (GM).
    """
    def __init__(self, fmri_path=None, sc_path=None, gm_path=None, 
                 transform=True, mode='multimodal'):
        """
        Args:
            fmri_path (str): Path to .pt file containing fMRI tensors (N, ROIs, Time).
            sc_path (str): Path to .pt file containing Adjacency matrices (N, ROIs, ROIs).
            gm_path (str): Path to .pt file containing Gradient Maps (N, ROIs, G).
            transform (bool): Apply robust scaling to fMRI.
            mode (str): 'multimodal', 'functional', or 'structural'.
        """
        self.mode = mode
        self.transform = transform
        
        # Load Data
        self.fmri_data = None
        self.sc_data = None
        self.gm_data = None
        
        if mode in ['functional', 'multimodal'] and fmri_path:
            self.fmri_data = torch.load(fmri_path)
            # Ensure shape is (N, R, T)
            if self.transform:
                self.fmri_data = self._robust_scale(self.fmri_data)
                
        if mode in ['structural', 'multimodal'] and sc_path:
            self.sc_data = torch.load(sc_path) # (N, R, R)
            
        if gm_path:
            self.gm_data = torch.load(gm_path) # (N, R, G)

        # Validation
        if self.fmri_data is not None:
            self.length = len(self.fmri_data)
        elif self.sc_data is not None:
            self.length = len(self.sc_data)
        else:
            raise ValueError("No data loaded! Check paths and mode.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {}
        
        # Functional Data
        if self.fmri_data is not None:
            sample['fmri'] = self.fmri_data[idx].float()
            
        # Structural Data
        if self.sc_data is not None:
            adj = self.sc_data[idx].float()
            sample['sc_adj'] = adj
            # For Graph Transformer, node features are often the adjacency itself or identity
            sample['sc_feats'] = adj # Using connectivity profile as features
            
        # Gradient Maps (for Spatial PE)
        if self.gm_data is not None:
            sample['gm'] = self.gm_data[idx].float()
            
        return sample

    def _robust_scale(self, data):
        """
        Robust scaling: (x - median) / IQR
        """
        print("Applying Robust Scaling...")
        N, R, T = data.shape
        reshaped = data.permute(1, 0, 2).reshape(R, -1) # (R, N*T)
        
        median = reshaped.median(dim=1, keepdim=True).values
        q1 = torch.quantile(reshaped, 0.25, dim=1, keepdim=True)
        q3 = torch.quantile(reshaped, 0.75, dim=1, keepdim=True)
        iqr = q3 - q1
        
        scaled = (reshaped - median) / (iqr + 1e-8)
        return scaled.reshape(R, N, T).permute(1, 0, 2)

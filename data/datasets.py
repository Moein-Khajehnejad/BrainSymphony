import torch
from torch.utils.data import Dataset
import numpy as np
import os

class BrainSymphonyDataset(Dataset):
    """
    Unified Dataset for BrainSymphony.
    Handles fMRI (BOLD), Structural Connectivity (SC), and Gradient Maps (GM).
    Robustly loads tensors whether they are saved directly or inside a dictionary.
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
        
        # --- Functional Data Loading ---
        if mode in ['functional', 'multimodal'] and fmri_path:
            if not os.path.exists(fmri_path):
                raise FileNotFoundError(f"fMRI file not found: {fmri_path}")
            
            # Helper to find tensor inside dict if necessary
            self.fmri_data = self._safe_load_tensor(fmri_path, ['fmri_data', 'data', 'fmri'])
            
            if self.transform:
                self.fmri_data = self._robust_scale(self.fmri_data)
                
        # --- Structural Data Loading ---
        if mode in ['structural', 'multimodal'] and sc_path:
            if not os.path.exists(sc_path):
                raise FileNotFoundError(f"SC file not found: {sc_path}")
                
            self.sc_data = self._safe_load_tensor(sc_path, ['fc_data', 'adj', 'sc_data', 'sc']) # (N, R, R)
            
        # --- Gradient Maps Loading ---
        if gm_path:
            if not os.path.exists(gm_path):
                 print(f"Warning: Gradient Map file not found at {gm_path}. Spatial PE might fail if required.")
            else:
                self.gm_data = self._safe_load_tensor(gm_path, ['G_tensor', 'gradients', 'gm']) # (N, R, G)

        # Validation
        if self.fmri_data is not None:
            self.length = len(self.fmri_data)
        elif self.sc_data is not None:
            self.length = len(self.sc_data)
        else:
            raise ValueError("No valid data loaded! Check paths and mode.")

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

    def _safe_load_tensor(self, path, key_search_list):
        """Helper to load tensor whether it's raw or inside a dict"""
        print(f"Loading {path}...")
        loaded = torch.load(path, map_location='cpu') # Load to CPU first to avoid OOM
        
        if isinstance(loaded, torch.Tensor):
            return loaded
            
        if isinstance(loaded, dict):
            # Try to find the right key
            for key in key_search_list:
                if key in loaded:
                    print(f"  -> Found key '{key}' in dictionary.")
                    return loaded[key]
            
            # If keys don't match, print available keys for debugging
            raise KeyError(f"Could not find valid key {key_search_list} in {path}. Available keys: {list(loaded.keys())}")
            
        raise ValueError(f"File {path} format not recognized (must be Tensor or Dict).")

    def _robust_scale(self, data):
        """
        Robust scaling: (x - median) / IQR
        """
        print("  -> Applying Robust Scaling...")
        N, R, T = data.shape
        reshaped = data.permute(1, 0, 2).reshape(R, -1) # (R, N*T)
        
        median = reshaped.median(dim=1, keepdim=True).values
        q1 = torch.quantile(reshaped, 0.25, dim=1, keepdim=True)
        q3 = torch.quantile(reshaped, 0.75, dim=1, keepdim=True)
        iqr = q3 - q1
        
        scaled = (reshaped - median) / (iqr + 1e-8)
        return scaled.reshape(R, N, T).permute(1, 0, 2)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# Imports from our modular package
from models import BrainSymphonyFMRI, BrainSymphonyStructural, BrainSymphony
from data.dataset import BrainSymphonyDataset
from utils.masking import TokenMasker1D, TokenMasker2D, TimePointMasker
from utils.graph_utils import mask_random_edges

def parse_args():
    parser = argparse.ArgumentParser(description="Train BrainSymphony Foundation Model")
    
    # Data Paths
    parser.add_argument('--fmri_path', type=str, default=None, help='Path to fMRI .pt file')
    parser.add_argument('--sc_path', type=str, default=None, help='Path to SC .pt file')
    parser.add_argument('--gm_path', type=str, default=None, help='Path to Gradient Maps .pt file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    
    # Training Mode
    parser.add_argument('--mode', type=str, choices=['functional', 'structural', 'multimodal'], 
                        required=True, help='Which component to train')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Model Hyperparameters
    parser.add_argument('--rois', type=int, default=450)
    parser.add_argument('--time_steps', type=int, default=200)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--mlp_dim', type=int, default=512)
    
    # Structural Specific
    parser.add_argument('--num_layers_struct', type=int, default=6)
    parser.add_argument('--attention_type', type=str, default='first_hop')
    
    return parser.parse_args()

def train_functional(args, device):
    """Pretrains the fMRI branch (Spatial+Temporal+Context+Perceiver)"""
    print("--- Starting Functional Pretraining ---")
    
    # Dataset
    dataset = BrainSymphonyDataset(args.fmri_path, gm_path=args.gm_path, mode='functional')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model
    config = vars(args)
    config['max_len_spatial'] = args.rois + 50
    config['max_len_temp'] = 5000
    model = BrainSymphonyFMRI(config).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        with tqdm(loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                
                fmri = batch['fmri'].to(device)
                gm = batch.get('gm', None)
                if gm is not None: gm = gm.to(device)
                
                # Forward pass (Auto-masking handled inside model or we can explicitly mask here)
                # Note: For strict pretraining, we usually mask inside the loop. 
                # Here we assume the model handles reconstruction internally or we mask input.
                
                # Simple reconstruction task
                recon, _, _ = model(fmri, gm)
                
                loss = loss_fn(recon, fmri)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{args.save_dir}/functional_epoch_{epoch+1}.pth")
            print(f"Saved checkpoint to {args.save_dir}")

def train_structural(args, device):
    """Pretrains the Structural Graph Transformer"""
    print("--- Starting Structural Pretraining ---")
    
    dataset = BrainSymphonyDataset(sc_path=args.sc_path, mode='structural')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    config = vars(args)
    config['num_heads_struct'] = args.num_heads
    model = BrainSymphonyStructural(config).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        with tqdm(loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                
                adj = batch['sc_adj'].to(device)
                feats = batch['sc_feats'].to(device)
                
                # Create mask
                mask = mask_random_edges(adj, mask_ratio=0.2)
                
                # Forward
                _, pred_weights, _ = model(feats, adj, mask_edges=mask)
                
                # Calculate loss only on masked edges
                # Targets are the true weights of masked edges
                # (Logic handled inside model helper or here)
                # Using the helper from model class for simplicity if it returns dense preds:
                # But our SignedGraphTransformer returns flattened preds/targets
                
                loss = loss_fn(pred_weights[0], pred_weights[1]) # pred vs target
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{args.save_dir}/structural_epoch_{epoch+1}.pth")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    if args.mode == 'functional':
        train_functional(args, device)
    elif args.mode == 'structural':
        train_structural(args, device)
    elif args.mode == 'multimodal':
        print("Multimodal training not fully implemented in this script version.")
        print("Use the 'BrainSymphony' class to load pretrained weights and fuse.")

if __name__ == "__main__":
    main()

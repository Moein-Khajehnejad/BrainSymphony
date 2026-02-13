import torch
import os
from models import BrainSymphony

def load_brainsymphony(checkpoint_root='./checkpoints', device='cpu'):
    """
    Loads the BrainSymphony model from versioned subfolders.
    
    Expected Structure:
        checkpoint_root/
        ├── functional_v1/      # spatial.pth, temporal.pth, context.pth, fusion.pth
        └── structural_v1/      # structural.pth
    """
    
    # 1. Define Model Configuration
    config = {
        'mode': 'multimodal', 
        'rois': 450,
        'time_steps': 200,
        'dim': 128,
        'num_layers': 6,
        'num_heads': 4,
        'mlp_dim': 512,
        'max_len_spatial': 500,
        'max_len_temp': 5000,
        'pos_enc_type': 'learnable',
        
        # Structural specific
        'num_layers_struct': 6,
        'num_heads_struct': 4,
        'attention_type': 'first_hop',
        'pos_enc_type_struct': 'learnable'
    }

    print(f"Initializing BrainSymphony architecture on {device}...")
    model = BrainSymphony(config).to(device)
    
    # 2. Define Subfolder Paths (Hardcoded standard names)
    func_dir = os.path.join(checkpoint_root, 'functional_v1')
    struct_dir = os.path.join(checkpoint_root, 'structural_v1')

    # Define file paths
    paths = {
        'spatial': os.path.join(func_dir, 'spatial.pth'),
        'temporal': os.path.join(func_dir, 'temporal.pth'),
        'context': os.path.join(func_dir, 'context.pth'),
        'fusion': os.path.join(func_dir, 'fusion.pth'),
        
        'structural': os.path.join(struct_dir, 'structural.pth')
    }

    # Track loading status
    loaded_functional = False
    loaded_structural = False

    # 3. Load Functional Branch
    if os.path.exists(paths['spatial']):
        print(f">> Found Functional checkpoints in {func_dir}...")
        try:
            model.functional_module.spatial.load_state_dict(torch.load(paths['spatial'], map_location=device))
            model.functional_module.temporal.load_state_dict(torch.load(paths['temporal'], map_location=device))
            
            if os.path.exists(paths['context']):
                model.functional_module.context.load_state_dict(torch.load(paths['context'], map_location=device))
            if os.path.exists(paths['fusion']):
                model.functional_module.perceiver.load_state_dict(torch.load(paths['fusion'], map_location=device))
            
            loaded_functional = True
            print("   ✅ Functional branch loaded successfully.")
        except Exception as e:
            print(f"   ⚠️ Error loading Functional branch: {e}")
    else:
        print(f"   ℹ️ Functional folder not found or empty: {func_dir}")

    # 4. Load Structural Branch
    if os.path.exists(paths['structural']):
        print(f">> Found Structural checkpoint in {struct_dir}...")
        try:
            model.structural_module.encoder.load_state_dict(torch.load(paths['structural'], map_location=device))
            loaded_structural = True
            print("   ✅ Structural branch loaded successfully.")
        except Exception as e:
            print(f"   ⚠️ Error loading Structural branch: {e}")
    else:
        print(f"   ℹ️ Structural folder/file not found: {paths['structural']}")

    # 5. Final Check
    if not loaded_functional and not loaded_structural:
        print(f"\n⚠️ WARNING: No checkpoints found in {checkpoint_root}. Model initialized with random weights.")
    elif loaded_functional and loaded_structural:
        print("\n✅ Model is fully loaded in MULTIMODAL mode.")
    else:
        print("\n⚠️ Partial load complete (One branch only).")

    model.eval()
    return model

if __name__ == "__main__":
    model = load_brainsymphony()

import torch
import os
import sys
from models import BrainSymphony

def load_brainsymphony(checkpoint_dir='./checkpoints', device='cpu'):
    """
    Loads the BrainSymphony model flexibly. 
    It checks for available checkpoints and loads what is found.
    
    Args:
        checkpoint_dir (str): Path to the directory containing .pth files.
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        model (nn.Module): The initialized model with loaded weights.
    """
    
    # 1. Define Model Configuration
    # (Ensure these match your training settings)
    config = {
        'mode': 'multimodal', # Initialize full architecture to allow flexible loading
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
    
    # 2. Define Expected Paths
    # Note: Ensure your structural file is renamed to 'structural.pth'
    paths = {
        'spatial': os.path.join(checkpoint_dir, 'spatial.pth'),
        'temporal': os.path.join(checkpoint_dir, 'temporal.pth'),
        'context': os.path.join(checkpoint_dir, 'context.pth'),
        'fusion': os.path.join(checkpoint_dir, 'fusion.pth'),
        'structural': os.path.join(checkpoint_dir, 'structural.pth')
    }

    # Track what we successfully loaded
    loaded_functional = False
    loaded_structural = False

    # 3. Load Functional Branch
    if os.path.exists(paths['spatial']) and os.path.exists(paths['temporal']):
        print(">> Found Functional checkpoints. Loading...")
        try:
            model.functional_module.spatial.load_state_dict(torch.load(paths['spatial'], map_location=device))
            model.functional_module.temporal.load_state_dict(torch.load(paths['temporal'], map_location=device))
            # Context and Fusion might be optional or part of the set, check them too
            if os.path.exists(paths['context']):
                model.functional_module.context.load_state_dict(torch.load(paths['context'], map_location=device))
            if os.path.exists(paths['fusion']):
                model.functional_module.perceiver.load_state_dict(torch.load(paths['fusion'], map_location=device))
            
            loaded_functional = True
            print("   ✅ Functional branch loaded successfully.")
        except Exception as e:
            print(f"   ⚠️ Error loading Functional branch: {e}")
    else:
        print("   ℹ️ Functional checkpoints (spatial.pth/temporal.pth) NOT found. Skipping.")

    # 4. Load Structural Branch
    if os.path.exists(paths['structural']):
        print(">> Found Structural checkpoint. Loading...")
        try:
            # We load into the encoder part of the structural module
            model.structural_module.encoder.load_state_dict(torch.load(paths['structural'], map_location=device))
            loaded_structural = True
            print("   ✅ Structural branch loaded successfully.")
        except Exception as e:
            print(f"   ⚠️ Error loading Structural branch: {e}")
    else:
        print("   ℹ️ Structural checkpoint (structural.pth) NOT found. Skipping.")

    # 5. Final Safety Check
    if not loaded_functional and not loaded_structural:
        raise RuntimeError(
            "\n❌ CRITICAL ERROR: No checkpoints were loaded!\n"
            f"   Expected files in {checkpoint_dir}: spatial.pth, temporal.pth, OR structural.pth.\n"
            "   Please check your folder path and filenames."
        )

    # 6. Mode Adjustment (Optional)
    # If we only loaded one branch, we can inform the user or adjust the model mode conceptually
    if loaded_functional and not loaded_structural:
        print("\n⚠️ WARNING: Model is running in FUNCTIONAL-ONLY mode (Structural weights are random).")
    elif loaded_structural and not loaded_functional:
        print("\n⚠️ WARNING: Model is running in STRUCTURAL-ONLY mode (Functional weights are random).")
    else:
        print("\n✅ Model is fully loaded in MULTIMODAL mode.")

    model.eval()
    return model

if __name__ == "__main__":
    # Example usage
    try:
        model = load_brainsymphony(checkpoint_dir='./checkpoints', device='cpu')
        
        # Quick test inference (optional)
        # dummy_batch = {'fmri': torch.randn(1, 450, 200)}
        # out = model(dummy_batch)
        # print("Inference test passed.")
        
    except RuntimeError as e:
        print(e)

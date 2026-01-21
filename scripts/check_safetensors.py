import os
import re
from safetensors import safe_open

def check_safetensor_shapes(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Cannot find file {file_path}")
        return
    
    print(f"Reading file: {file_path}")
    
    unique_structures = {}
    all_block_indices = set()
    action_block_indices = set()

    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                block_match = re.search(r'blocks\.(\d+)\.', key)
                if block_match:
                    index = int(block_match.group(1))
                    all_block_indices.add(index)
                    if "action_model" in key:
                        action_block_indices.add(index)
                # Store structural info
                shape = f.get_slice(key).get_shape()
                generic_key = re.sub(r'\.\d+\.', '.x.', key)
                if generic_key not in unique_structures:
                    unique_structures[generic_key] = shape

        print(f"\n{'Generic Parameter Name':<60} | {'Shape (Example)':<20}")
        print("-" * 85)
        for g_key in sorted(unique_structures.keys()):
            print(f"{g_key:<60} | {str(unique_structures[g_key]):<20}")
        print("-" * 85)

        print(f"Total Blocks found:  {len(all_block_indices)} (Indices: {min(all_block_indices) if all_block_indices else 0}-{max(all_block_indices) if all_block_indices else 0})")
        print(f"Action Blocks found: {len(action_block_indices)} (Indices: {sorted(list(action_block_indices))})")
        print(f"Unique structural components: {len(unique_structures)}")

    except Exception as e:
        print(f"Failed to read file: {e}")

if __name__ == "__main__":
    # path_to_file = "Matrix-Game-2.0-Student-Diffusers/transformer/diffusion_pytorch_model.safetensors" 
    # path_to_file = "Matrix-Game-2.0-Student-Diffusers/transformer_processed/diffusion_pytorch_model.safetensors" 
    # path_to_file = "Matrix-Game-2.0-TempleRun-Diffusers/transformer/diffusion_pytorch_model.safetensors"
    # path_to_file = "Matrix-Game-2.0-GTA-Diffusers/transformer/diffusion_pytorch_model.safetensors"
    path_to_file = "checkpoints/matrixgame_ode_init/checkpoint-100/transformer/diffusion_pytorch_model.safetensors"
    check_safetensor_shapes(path_to_file)
import torch
import os
import json
from pathlib import Path
from safetensors.torch import load_file, save_file

N = 15
NEW_KEYBOARD_DIM_IN = 3
input_dir = "Matrix-Game-2.0-Student-Diffusers/transformer"
output_dir = "Matrix-Game-2.0-Student-Diffusers/transformer_processed"

def process_transformer_folder(src, dst):
    src_path = Path(src)
    dst_path = Path(dst)

    config_file = src_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"cannot find config.json in {src}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(config)
    blocks = config["action_config"]["blocks"]
    print(blocks)
    blocks = blocks[:N]
    config["action_config"]["blocks"] = blocks
    config["action_config"]["enable_mouse"] = False
    config["action_config"]["keyboard_dim_in"] = NEW_KEYBOARD_DIM_IN
    print(blocks)

    model_file = src_path / "diffusion_pytorch_model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"cannot find diffusion_pytorch_model.safetensors in {src}")

    tensors = load_file(str(model_file))
    new_tensors = {}
    mouse_key_prefixes = (
        "action_model.mouse_mlp",
        "action_model.t_qkv",
        "action_model.img_attn_q_norm",
        "action_model.img_attn_k_norm",
        "action_model.proj_mouse",
    )
    for key, val in tensors.items():
        if "action_model" in key:
            parts = key.split('.')
            try:
                block_idx = int(parts[1])
                if block_idx >= N:
                    continue
            except (ValueError, IndexError):
                pass
            if any(prefix in key for prefix in mouse_key_prefixes):
                continue

            if "action_model.keyboard_embed" in key and key.endswith("weight"):
                in_dim = val.shape[-1]
                if in_dim < NEW_KEYBOARD_DIM_IN:
                    pad = NEW_KEYBOARD_DIM_IN - in_dim
                    val = torch.nn.functional.pad(val, (0, pad))
                elif in_dim > NEW_KEYBOARD_DIM_IN:
                    val = val[..., :NEW_KEYBOARD_DIM_IN].contiguous() # make it contiguous
        new_tensors[key] = val

    dst_path.mkdir(parents=True, exist_ok=True)    
    save_file(new_tensors, str(dst_path / "diffusion_pytorch_model.safetensors"))
    with open(dst_path / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"weights number changed: {len(tensors)} -> {len(new_tensors)}")

if __name__ == "__main__":
    process_transformer_folder(input_dir, output_dir)

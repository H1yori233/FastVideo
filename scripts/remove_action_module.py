import torch
import os
import json
from pathlib import Path
from safetensors.torch import load_file, save_file

from fastvideo.models.dits.matrixgame.action_module import ActionModule

N = 15
NEW_KEYBOARD_DIM_IN = 3
RNG_SEED = 42
input_dir = "Matrix-Game-2.0-Foundation-Diffusers/transformer"
output_dir = "Matrix-Game-2.0-Student-Diffusers/transformer"

def process_transformer_folder(src, dst):
    src_path = Path(src)
    dst_path = Path(dst)

    config_file = src_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"cannot find config.json in {src}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(config)
    action_config = config["action_config"]
    blocks = action_config["blocks"]
    print(blocks)
    blocks = blocks[:N]
    action_config["blocks"] = blocks
    action_config["enable_mouse"] = False
    action_config["keyboard_dim_in"] = NEW_KEYBOARD_DIM_IN
    print(blocks)

    model_file = src_path / "diffusion_pytorch_model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"cannot find diffusion_pytorch_model.safetensors in {src}")

    tensors = load_file(str(model_file))
    new_tensors = {}

    for key, val in tensors.items():
        if ".action_model." in key:
            continue
        new_tensors[key] = val

    torch.manual_seed(RNG_SEED)
    action_module = ActionModule(
        mouse_dim_in=action_config.get("mouse_dim_in", 2),
        keyboard_dim_in=action_config.get("keyboard_dim_in", 6),
        hidden_size=action_config.get("hidden_size", 128),
        img_hidden_size=action_config.get("img_hidden_size", 1536),
        keyboard_hidden_dim=action_config.get("keyboard_hidden_dim", 1024),
        mouse_hidden_dim=action_config.get("mouse_hidden_dim", 1024),
        vae_time_compression_ratio=action_config.get("vae_time_compression_ratio", 4),
        windows_size=action_config.get("windows_size", 3),
        heads_num=action_config.get("heads_num", 16),
        patch_size=action_config.get("patch_size", [1, 2, 2]),
        qk_norm=action_config.get("qk_norm", True),
        qkv_bias=action_config.get("qkv_bias", False),
        rope_dim_list=action_config.get("rope_dim_list", [8, 28, 28]),
        rope_theta=action_config.get("rope_theta", 256),
        mouse_qk_dim_list=action_config.get("mouse_qk_dim_list", [8, 28, 28]),
        enable_mouse=action_config.get("enable_mouse", True),
        enable_keyboard=action_config.get("enable_keyboard", True),
        local_attn_size=action_config.get("local_attn_size", 6),
        blocks=action_config.get("blocks", []),
    )
    action_state = action_module.state_dict()
    for block_idx in blocks:
        for name, val in action_state.items():
            new_tensors[f"blocks.{block_idx}.action_model.{name}"] = val.detach().clone()

    dst_path.mkdir(parents=True, exist_ok=True)    
    save_file(new_tensors, str(dst_path / "diffusion_pytorch_model.safetensors"))
    with open(dst_path / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"weights number changed: {len(tensors)} -> {len(new_tensors)}")

if __name__ == "__main__":
    process_transformer_folder(input_dir, output_dir)

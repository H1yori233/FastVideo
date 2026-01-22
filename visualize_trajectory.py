import argparse
import os
import torch
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from PIL import Image

# Add potential paths to sys.path if needed, assuming running from FastVideo root or locally
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
fastvideo_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
if fastvideo_root not in sys.path:
    sys.path.append(fastvideo_root)

from fastvideo import PipelineConfig
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.models.vaes.wanvae import AutoencoderKLWan
from fastvideo.utils import save_decoded_latents_as_video, maybe_download_model

def main():
    parser = argparse.ArgumentParser(description="Visualize Trajectory from Parquet file")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to the input parquet file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save output videos")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Precision for decoding")
    
    args = parser.parse_args()
    
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]
    print(f"Using device: {args.device}, dtype: {args.dtype}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
    print(f"Loading model from {args.model_path}...")
    model_path = maybe_download_model(args.model_path)
    
    # Load configuration
    pipeline_config = PipelineConfig.from_pretrained(model_path)
    vae_config = WanVAEConfig(load_encoder=False, load_decoder=True)
    vae = AutoencoderKLWan(vae_config)
    
    vae.to(args.device, dtype=torch_dtype)
    vae.eval()

    # Move config tensors to device if necessary and cast
    if hasattr(vae.config, "scaling_factor") and isinstance(vae.config.scaling_factor, torch.Tensor):
        vae.config.scaling_factor = vae.config.scaling_factor.to(args.device, dtype=torch_dtype)
    if hasattr(vae.config, "shift_factor") and isinstance(vae.config.shift_factor, torch.Tensor):
        vae.config.shift_factor = vae.config.shift_factor.to(args.device, dtype=torch_dtype)

    # Read Parquet
    print(f"Reading parquet file: {args.parquet_path}")
    table = pq.read_table(args.parquet_path)
    
    # Iterate over rows
    num_visualized = 0
    
    pbar = tqdm(total=min(args.num_samples, table.num_rows))
    
    for i in range(table.num_rows):
        if num_visualized >= args.num_samples:
            break
            
        row = table.slice(i, length=1)
        record = row.to_pydict()
        
        video_id = record["id"][0]
        
        # Parse Latents
        shape = record["trajectory_latents_shape"][0]
        dtype = record["trajectory_latents_dtype"][0]
        dtype = np.dtype(dtype)
        
        latents_bytes = record["trajectory_latents_bytes"][0]
        # Copy to avoid read-only warning
        latents_np = np.copy(np.frombuffer(latents_bytes, dtype=dtype).reshape(shape))
        
        latents_tensor = torch.from_numpy(latents_np).to(args.device, dtype=torch_dtype)
        if latents_tensor.ndim == 6 and latents_tensor.shape[0] == 1:
            latents_tensor = latents_tensor.squeeze(0)
        
        print(f"Decoding video {video_id} with shape {latents_tensor.shape}...")
        
        # create subfolder
        vid_output_dir = os.path.join(args.output_dir, str(video_id))
        os.makedirs(vid_output_dir, exist_ok=True)
        
        # Decode only the last step
        steps = latents_tensor.shape[0]
        indices_to_decode = [steps - 1]
        
        for step in tqdm(indices_to_decode, desc=f"Decoding {video_id}", leave=False):
            latent_step = latents_tensor[step].unsqueeze(0) # [1, C, T, H, W]
            
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    # Apply scaling/shifting if needed before decode?
                    
                    if hasattr(vae.config, "scaling_factor"):
                         latent_step = latent_step / vae.config.scaling_factor
                    elif hasattr(vae, "scaling_factor"):
                         latent_step = latent_step / vae.scaling_factor
                    
                    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
                         latent_step = latent_step + vae.config.shift_factor
                    elif hasattr(vae, "shift_factor") and vae.shift_factor is not None:
                         latent_step = latent_step + vae.shift_factor
                         
                    decoded_video = vae.decode(latent_step)
                
                # Save video
                save_path = os.path.join(vid_output_dir, f"step_{step:03d}.mp4")
                save_decoded_latents_as_video(decoded_video.float(), save_path, fps=25) 
                
        print(f"Saved {steps} steps to {vid_output_dir}")
        num_visualized += 1
        pbar.update(1)
        
    pbar.close()

if __name__ == "__main__":
    main()

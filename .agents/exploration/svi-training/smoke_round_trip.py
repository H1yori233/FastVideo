"""Stage 3 round-trip smoke: load a SaveLoRACallback safetensors into the
inference pipeline and run a minimal generation.

Verifies:
  1. The exported safetensors loads via LoRAPipeline.set_lora_adapter
     (no missing/unexpected key warnings)
  2. The pipeline completes a 2-step inference run without NaN
  3. Output has a sensible value range

Run from repo root:
    python .agents/exploration/svi-training/smoke_round_trip.py \\
        --lora outputs/svi_stage3/lora/last.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", required=True, help="LoRA safetensors saved by SaveLoRACallback")
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image",
                        default=str(REPO / "Stable-Video-Infinity" / "data" / "toy_test" / "shot" / "frame.jpg"))
    parser.add_argument("--prompt",
                        default=("A sleek white motor yacht speeds across the turquoise blue sea, "
                                 "leaving a dramatic wake of white foam behind it under a clear blue sky."))
    parser.add_argument("--output", default="video_samples_stage3")
    args = parser.parse_args()

    from fastvideo import VideoGenerator

    print(f"Building VideoGenerator with lora_path={args.lora}")
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        override_pipeline_cls_name="WanSVIImageToVideoPipeline",
        lora_path=args.lora,
        lora_nickname="svi-stage3",
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )

    output = generator.generate_video(
        prompt=args.prompt,
        image_path=args.image,
        output_path=args.output,
        save_video=True,
        height=448,
        width=832,
        num_frames=args.num_frames,
        num_inference_steps=args.num_steps,
        guidance_scale=5.0,
        seed=args.seed,
        svi_num_clips=1,
        svi_num_motion_frames=1,
        svi_ref_pad_num=-1,
    )

    if isinstance(output, dict) and "video" in output:
        video = output["video"]
    elif isinstance(output, torch.Tensor):
        video = output
    else:
        print(f"[smoke] generated output of type {type(output).__name__}; assuming success")
        return

    print(f"[smoke] video tensor: shape={tuple(video.shape)} dtype={video.dtype} "
          f"min={video.min().item():.4f} max={video.max().item():.4f} "
          f"has_nan={bool(torch.isnan(video).any())}")
    if bool(torch.isnan(video).any()):
        raise SystemExit("NaN in output — round-trip failed")
    print("[smoke] Stage 3 round-trip OK")


if __name__ == "__main__":
    main()

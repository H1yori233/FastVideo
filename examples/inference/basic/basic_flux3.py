# SPDX-License-Identifier: Apache-2.0
"""Generate or edit an image from a short prompt with FLUX.3 Image Klein."""
import argparse
import json

from fastvideo import VideoGenerator
from fastvideo.api import EngineConfig, GeneratorConfig, OffloadConfig, PipelineSelection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="black-forest-labs/flux-3-image-klein",
                        help="Official Hugging Face model ID or local official snapshot")
    parser.add_argument("--prompt", default="A red bicycle beside a white wall, morning sunlight.")
    parser.add_argument("--reference", action="append", default=[], help="Reference image path; repeat for multiple images")
    parser.add_argument("--boxes", type=json.loads, default=None, help="JSON list of normalized layout boxes")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--auto-size", action="store_true", help="Match the first reference instead of width/height")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--no-pup", action="store_true", help="Use an already expanded caption")
    parser.add_argument("--pup-max-new-tokens", type=int, default=3584)
    parser.add_argument("--output-path", default="outputs/flux3/image.png")
    args = parser.parse_args()
    generator = VideoGenerator.from_config(GeneratorConfig(
        model_path=args.model_path,
        engine=EngineConfig(num_gpus=1, use_fsdp_inference=False, offload=OffloadConfig(
            dit=False, dit_layerwise=False, text_encoder=False, vae=False,
            pin_cpu_memory=False, lazy_module_load=True)),
        pipeline=PipelineSelection(workload_type="i2i" if args.reference else "t2i"),
    ))
    try:
        generator.generate({
            "prompt": args.prompt,
            "inputs": {"references": args.reference},
            "sampling": {"width": args.width, "height": args.height,
                         "num_inference_steps": args.steps, "seed": args.seed},
            "extensions": {
                "flux3_boxes": args.boxes,
                "flux3_auto_size": args.auto_size,
                "flux3_upsample_prompt": not args.no_pup,
                "flux3_pup_max_new_tokens": args.pup_max_new_tokens,
            },
            "output": {"output_path": args.output_path, "save_video": True, "return_frames": False},
        })
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()

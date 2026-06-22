from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_lingbotworld_fast"


def main():
    # LingBot-World-Fast is the block-causal, DMD-distilled streaming variant.
    # It generates latent frames chunk-by-chunk with a rolling KV cache, so the
    # whole clip is produced in a handful of denoising steps per chunk.
    generator = VideoGenerator.from_pretrained(
        "robbyant/lingbot-world-fast-diffusers",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    prompt = (
        "The video presents a soaring journey through a fantasy jungle. Floating "
        "islands drift past distant waterfalls as an ancient gothic castle "
        "approaches against a bright sky.")
    image_path = "https://raw.githubusercontent.com/Robbyant/lingbot-world/main/examples/00/image.jpg"

    generator.generate_video(
        prompt,
        image_path=image_path,
        output_path=OUTPUT_PATH,
        save_video=True,
        num_frames=81,
        height=480,
        width=832,
    )


if __name__ == "__main__":
    main()

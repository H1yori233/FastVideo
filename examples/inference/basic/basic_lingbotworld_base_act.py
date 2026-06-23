from fastvideo import VideoGenerator
from fastvideo.models.dits.lingbotworld.act_utils import prepare_action_embedding

OUTPUT_PATH = "video_samples_lingbotworld_act"


def main():
    generator = VideoGenerator.from_pretrained(
        "FastVideo/LingBot-World-Base-Act-Diffusers",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    num_frames = 81
    prompt = "The video presents a soaring journey through a fantasy jungle. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."
    image_path = "https://raw.githubusercontent.com/Robbyant/lingbot-world/main/examples/00/image.jpg"

    action_string = "w-20,wl-12,none-8,wd-20,none-8,s-16"
    c2ws_plucker_emb, num_frames = prepare_action_embedding(
        action_string=action_string,
        num_frames=num_frames,
        height=480,
        width=832,
        spatial_scale=8,
    )

    generator.generate_video(
        prompt,
        image_path=image_path,
        output_path=OUTPUT_PATH,
        save_video=True,
        num_frames=num_frames,
        height=480,
        width=832,
        c2ws_plucker_emb=c2ws_plucker_emb,
    )


if __name__ == "__main__":
    main()

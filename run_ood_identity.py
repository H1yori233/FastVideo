import cloudpickle
import torch

from fastvideo import VideoGenerator
from fastvideo.models.dits.matrixgame2.utils import create_action_presets

# ----------------------------------------------------------------------------
MODEL_PATH = "FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers"
IMAGE = ("https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/"
         "Matrix-Game-2/demo_images/universal/0000.png")
HEIGHT, WIDTH = 352, 640        # token grid = (HEIGHT//16, WIDTH//16)
NUM_FRAMES = 201                # must be 4k+1
KEYBOARD_DIM = 4                # Base model: W/S/A/D
NUM_INFERENCE_STEPS = 50
OUTPUT_PATH = "fg_anchored_output"

# FG_BOX = (r0, r1, c0, c1) of the SUBJECT in the token grid.
FG_BOX = (8, 18, 14, 26)
FG_BOOST = 4.0
FG_QSUPP = 6.0
FG_SIGMA = (5.0, 5.0)
SINK_SIZE = 1
LOCAL_ATTN_SIZE = 6
# ----------------------------------------------------------------------------


def _enable_injection(worker):
    """Configure the foreground-anchored injector inside the model worker."""
    from fastvideo.models.dits.matrixgame2.attn_injection import INJECTOR
    from fastvideo.models.dits.matrixgame2.causal_model import (
        CausalMatrixGame2SelfAttention)
    transformer = worker.pipeline.get_module("transformer")
    for module in transformer.modules():
        if isinstance(module, CausalMatrixGame2SelfAttention):
            module.sink_size = SINK_SIZE
            module.local_attn_size = LOCAL_ATTN_SIZE
    INJECTOR.prepare(transformer)
    INJECTOR.grid_rows, INJECTOR.grid_cols = HEIGHT // 16, WIDTH // 16
    INJECTOR.fg_box = FG_BOX
    INJECTOR.fg_sigma = FG_SIGMA
    INJECTOR.fg_boost = FG_BOOST
    INJECTOR.fg_qsupp = FG_QSUPP
    INJECTOR.on()
    return "injection enabled"


def main():
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=1,
        use_fsdp_inference=False,    # set True if the GPU is out of memory
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    # The model lives in a worker process, so enable injection there via RPC.
    generator.executor.collective_rpc(cloudpickle.dumps(_enable_injection))

    actions = create_action_presets(NUM_FRAMES, keyboard_dim=KEYBOARD_DIM)
    grid_sizes = torch.tensor([(NUM_FRAMES + 3) // 4, HEIGHT // 8, WIDTH // 8])

    generator.generate_video(
        prompt="",
        image_path=IMAGE,
        mouse_cond=actions["mouse"].unsqueeze(0),
        keyboard_cond=actions["keyboard"].unsqueeze(0),
        grid_sizes=grid_sizes,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=42,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()

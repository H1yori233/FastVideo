"""Driver: run upstream ``train_svi.py`` for ONE step with the
``UpstreamStep1DumpCallback`` injected, producing the parity fixture under
``outputs/parity_fixture/step01/``.

Monkey-patches ``pl.Trainer.__init__`` so the dump callback rides along
with whatever Trainer upstream constructs. ER is forced off
(``--noise_prob 0 --y_prob 0 --latent_prob 0 --clean_prob 1
--clean_buffer_update_prob 0``) so the captured step is pure flow-match.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
UPSTREAM = REPO / "Stable-Video-Infinity"
EXPL = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(UPSTREAM / "data/toy_train/svi-film-shot"))
    parser.add_argument("--output", default=str(REPO / "outputs/parity_fixture/step01"))
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args_cli = parser.parse_args()

    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(EXPL))

    import lightning as pl
    from upstream_step1_dump_callback import UpstreamStep1DumpCallback

    weights_dir = UPSTREAM / "weights/Wan2.1-I2V-14B-480P"
    dit_shards = ",".join(
        sorted(str(p) for p in weights_dir.glob("diffusion_pytorch_model-0000?-of-00007.safetensors")))
    if not dit_shards:
        raise FileNotFoundError(f"No Wan I2V shards under {weights_dir}")

    # Resolve to an absolute path BEFORE we chdir into the upstream repo,
    # otherwise a relative ``--output`` ends up nested under Stable-Video-Infinity/.
    output_root = os.path.abspath(args_cli.output)
    os.makedirs(output_root, exist_ok=True)

    sys.argv = [
        "train_svi.py",
        "--learning_rate", str(args_cli.lr),
        "--lora_rank", str(args_cli.rank),
        "--lora_alpha", str(args_cli.alpha),
        "--dataset_path", args_cli.dataset,
        "--dit_path", dit_shards,
        "--vae_path", str(weights_dir / "Wan2.1_VAE.pth"),
        "--text_encoder_path", str(weights_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
        "--image_encoder_path", str(weights_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        "--max_epochs", "1",
        "--train_architecture", "lora",
        "--use_gradient_checkpointing",
        "--training_strategy", "auto",
        "--output_path", args_cli.output,
        "--y_error_num", "1",
        "--num_motion_frames", "1",
        "--num_grids", "50",
        "--ref_pad_num", "-1",
        "--noise_prob", "0",
        "--y_prob", "0",
        "--latent_prob", "0",
        "--clean_prob", "1",
        "--clean_buffer_update_prob", "0",
        "--buffer_warmup_iter", "0",
        "--buffer_replacement_strategy", "random",
        "--dataloader_num_workers", "0",
        "--exp_prefix", "step1",
    ]

    os.chdir(str(UPSTREAM))
    pl.seed_everything(int(args_cli.seed), workers=True)

    callback = UpstreamStep1DumpCallback(root=output_root)
    original_init = pl.Trainer.__init__

    def patched_init(self, *args, **kwargs):
        callbacks = list(kwargs.get("callbacks") or [])
        callbacks.append(callback)
        kwargs["callbacks"] = callbacks
        kwargs["max_steps"] = 1
        kwargs["max_epochs"] = -1
        kwargs["devices"] = 1
        print(f"[dump_step1_upstream] Trainer kwargs: max_steps={kwargs['max_steps']}, "
              f"devices={kwargs['devices']}",
              flush=True)
        original_init(self, *args, **kwargs)

    pl.Trainer.__init__ = patched_init

    import train_svi
    args = train_svi.parse_args()
    args = train_svi.update_experiment_path(args, short=True)
    print(f"[dump_step1_upstream] starting → {args_cli.output}", flush=True)
    train_svi.train_svi(args)
    print("[dump_step1_upstream] done", flush=True)


if __name__ == "__main__":
    main()

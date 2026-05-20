# 🎞️ Stable-Video-Infinity (SVI) Training

LoRA fine-tune for the SVI flavor of Wan 2.1 I2V 14B 480P, ported into
FastVideo's modular training stack (`fastvideo/train/`). The recipe matches
upstream `Stable-Video-Infinity/train_svi.py` defaults; deviations are
called out below.

## When to use

- You want to train an SVI-style multi-clip I2V LoRA on Wan 2.1 480P.
- You have a directory of raw mp4 clips, optionally grouped by category
  with a per-category CSV (`Filename, Video Description`).
- You're OK with on-the-fly VAE / CLIP / text encoding at every step
  (no parquet preprocessing required).

For inference of the trained LoRA, see
[`examples/inference/basic/basic_svi_i2v.py`](../../examples/inference/basic/basic_svi_i2v.py).

## Dataset layout

```
data/svi_train/
├── Cats/
│   ├── Cats.csv                  # columns: Filename, Video Description (+ extras ignored)
│   ├── mixkit-cat-1535.mp4
│   └── mixkit-cat-1536.mp4
└── sea/
    ├── sea.csv
    ├── mixkit-boat-1941.mp4
    └── mixkit-couple-1040.mp4
```

A toy version of this lives under
`Stable-Video-Infinity/data/toy_train/svi-film-shot/` (4 clips, ~3MB).

## Quick start (single GPU)

```bash
WANDB_MODE=disabled torchrun --nproc_per_node=1 \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/fine_tuning/svi/wan_i2v_svi.yaml \
  --training.data.data_path /path/to/data/svi_train \
  --training.loop.max_train_steps 10000 \
  --training.checkpoint.output_dir outputs/wan_svi_lora
```

The output directory will contain:

- `lora/step_<N>.safetensors` and `lora/last.safetensors` — LoRA-only
  weights consumable by `LoRAPipeline.set_lora_adapter` at inference time.
- `train_log.jsonl` — per-step loss, timestep, weight, and Error Recycling
  diagnostics (one JSON object per row).

## Multi-GPU (FSDP)

```bash
torchrun --nproc_per_node=4 \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/fine_tuning/svi/wan_i2v_svi.yaml \
  --training.data.data_path /path/to/data/svi_train \
  --training.distributed.num_gpus 4 \
  --training.distributed.hsdp_shard_dim 4 \
  --training.loop.max_train_steps 10000 \
  --training.checkpoint.output_dir outputs/wan_svi_lora
```

`SVIRawVideoDataset` shards videos across DP groups via `DistributedSampler`,
so each rank sees a disjoint subset every epoch.

## Key knobs (in YAML)

```yaml
models:
  student:
    _target_: fastvideo.train.models.wan.wan_svi.WanSVIModel
    init_from: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
    flow_shift: 5.0
    lora:
      enable: true
      rank: 128
      alpha: 128
      target_modules: [to_q, to_k, to_v, to_out, ffn.fc_in, ffn.fc_out]

method:
  _target_: fastvideo.train.methods.svi.svi.SVITrainingMethod
  # y-conditioning
  num_motion_frames: 1            # 1 = Shot/Tom, 5 = Film
  ref_pad_num: -1                 # -1=tile ref; 0=zero pad; k>0=ref for first k slots
  ref_pad_cfg: false
  p_motion_threshold: 0.9
  # Error Recycling
  use_error_recycling: true
  error_buffer_k: 500
  num_grids: 50
  buffer_warmup_iter: 50
  buffer_replacement_strategy: random
  noise_prob: 0.99
  y_prob: 0.99
  latent_prob: 0.99
  clean_prob: 0.1
  clean_buffer_update_prob: 0.5
```

| Knob | Default | What it does |
|------|---------|--------------|
| `num_motion_frames` | 1 | How many input frames are encoded into the y-conditioning slab (1 = Shot / Tom, 5 = Film). |
| `ref_pad_num` | -1 | -1 tile the random ref frame across padding; 0 zero-pad; k>0 ref-pad first k then zero. |
| `use_error_recycling` | true | Master switch for the Error Recycling buffers + injection. Turn off to revert to plain LoRA fine-tune. |
| `num_grids` × `error_buffer_k` | 50 × 500 | Per-timestep-grid CPU buffer of past prediction errors. |
| `buffer_warmup_iter` | 50 | First N steps cross-rank `all_gather`-share errors; after that ranks only update local buffers. |
| `noise_prob` / `y_prob` / `latent_prob` | 0.99 each | Per-step probability of injecting each error channel (subject to buffer availability). |
| `clean_prob` | 0.1 | Probability of skipping all injections this step (a clean fine-tune step). |
| `error_modulate_factor` | 0.0 | Uniform jitter on sampled-error intensity (1 ± factor). |

Deviations from upstream:

1. **`target_modules`** uses FV-internal names (`to_q`, `to_k`, …) instead
   of upstream's single-letter suffixes (`q`, `k`, …). The substring match
   in `_is_target_layer` makes `'o'` too greedy (matches `blocks.*`); see
   `.agents/lessons/2026-05-20_lora-target-modules-substring-match.md`.
2. **Distributed strategy** is FSDP/DTensor instead of DeepSpeed. The
   ER warmup `all_gather` lives at object granularity (not parameter
   shards) so it's strategy-agnostic.
3. **Scheduler** is `FlowMatchEulerDiscreteScheduler` (FV-native) instead
   of diffsynth `FlowMatchScheduler`. The `add_noise` formula and the
   shifted-sigma schedule match; `training_weight` is hand-implemented in
   `SVITrainingMethod._compute_bell_weights` to mirror upstream's bell
   curve.

## Checkpoints

LoRA weights are written by `SaveLoRACallback` to
`<output_dir>/lora/step_<N>.safetensors` every `every_steps` iterations
and at train-end. Key format is FV-internal (`blocks.<N>.attn1.to_q.lora_A.weight`,
etc.); the inference loader (`LoRAPipeline.set_lora_adapter`) accepts both
this and the upstream `pipe.dit.blocks.<N>.self_attn.q.lora_A.default.weight`
format via `WanVideoArchConfig.lora_param_names_mapping`.

To round-trip a saved LoRA into inference, see
`.agents/exploration/svi-training/smoke_round_trip.py`.

## Parity check vs upstream

The `train_log.jsonl` schema is intentionally simple so a matching trace can
be captured from upstream `train_svi.py` (via a Lightning callback that
appends `{step, train_loss, ...}` rows). Then:

```bash
python -m fastvideo.tests.svi.parity_check \
  --a outputs/wan_svi_lora/train_log.jsonl \
  --b /path/to/upstream/train_log.jsonl \
  --loss-key total_loss \
  --limit 200 \
  --min-correlation 0.97
```

Reports Pearson correlation on the chosen loss series. The verification
gate is correlation ≥ 0.97 over the first 200 steps under matched
seeds / data / hyperparams.

Capturing the upstream trace is currently a manual step; the SVI repo lives
at `/home/hal-kaiqin/FastVideo_SVI/Stable-Video-Infinity/` on this
workstation and its `train_svi.py` can be hooked with a small
`pl.Callback` that dumps `self.log("train_loss", …)` values to JSONL.

## Troubleshooting

- **Loss is `nan` on step 1**: VAE roundtrip dtype mismatch. Confirm the
  VAE is loaded in float32 (`pipeline_config.vae_precision: fp32` in YAML).
- **`requires_grad` filter returns 0 params**: LoRA target_modules didn't
  match any layer. Check the `[lora.py:248] Enabled LoRA training … on N
  layers` log — `N` should be 400 for Wan I2V with the canonical SVI
  targets.
- **OOM on a single GB200 80GB**: drop `flow_shift` or use 4 GPUs with
  FSDP; the 14B transformer + T5 + CLIP + VAE in fp32 lives under ~50 GB
  total on each rank, but activation memory at 480p×81 frames is
  significant.
- **Slow warmup (~22s/step) vs steady-state (~17s/step)**: expected; the
  first iteration triggers FSDP all-gathers and CUDA graph capture.

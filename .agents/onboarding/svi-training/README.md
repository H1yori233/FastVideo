# SVI Training — Agent Onboarding

Specialized onboarding for agents porting **Stable-Video-Infinity (SVI)**
training into FastVideo's new modular training framework (`fastvideo/train/`).

Read the master onboarding (`.agents/onboarding/README.md`) and the WorldModel
training onboarding (`.agents/onboarding/worldmodel-training/README.md`) first.
This doc only covers the SVI-specific delta.

The work happens on branch `svi-train` (forked from `svi` after `main` was
merged in). The companion inference port lives on `svi`; do **not** edit
inference here.

---

## Domain Context

SVI is a rank-128 LoRA on **Wan 2.1 I2V 14B 480P** that produces arbitrarily
long videos by chaining clips and threading motion-frame conditioning forward.
The upstream repo lives on disk at `Stable-Video-Infinity/` (gitignored).
Inference is already aligned with upstream on the `svi` branch — see
`fastvideo/pipelines/basic/wan/wan_svi_i2v_pipeline.py`.

Training has **two novel pieces** beyond standard flow-match LoRA fine-tune:

1. **`training_weight` bell-curve** on the flow-match MSE loss
   (`diffsynth/schedulers/flow_match.py`).
2. **Error Recycling** (ER) — two per-grid CPU buffers (`latent_error_buffer`,
   `y_error_buffer`) seeded with `x0`/`x1` prediction errors. During the
   forward pass we sample from the buffers and inject the recycled error into
   the corrupted latents and `y` conditioning. A distributed warm-up phase
   `all_gather`s errors across ranks; afterwards each rank updates locally.

Reference: `Stable-Video-Infinity/train_svi.py` and the
`LightningModelForTrain_onestage` class within it.

---

## Guiding Principle

**Default to alignment with the upstream project.** When a design question
has multiple "reasonable" answers, pick the one that matches `train_svi.py`
verbatim. Only deviate when (a) the FV framework forces our hand, or (b) the
upstream choice is provably suboptimal *and* the deviation is documented in a
lesson under `.agents/lessons/`.

This is the same posture we used for inference: ported numerics first, FV-isms
second.

---

## Key Decisions (locked unless re-opened)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **No parquet for SVI** — write a new `RawVideoDataset` in the new train stack | Parquet has no raw pixels; SVI needs CLIP image features. Lossy VAE roundtrip is unacceptable; extending preprocess is heavy. Raw video matches upstream 1:1. Revisit as an optimization later. |
| D2 | **Drop Error Recycling from v1** | Land the plain LoRA fine-tune end-to-end first, then add ER as Stage 5. Smaller PR, easier to A/B against upstream. |
| D3 | **Custom method, not `FineTuneMethod` reuse** | We need the per-timestep `training_weight` multiplier + on-the-fly VAE/CLIP. Wrapping `FineTuneMethod` is messier than a new class. |
| D4 | **LoRA rank from YAML, default 128** | Matches the released `svi-shot.safetensors`. Upstream training script defaults to 4 but the public weights are 128 — go with shipped. |
| D5 | **DTensor/FSDP, not DeepSpeed** | The new train stack is DTensor-native. `all_gather_object` still works for the ER warm-up buffer sync — that's not tied to ZeRO stage. |

If any of these need re-litigation, edit this table and link the reason from
the PR / decision log.

---

## Code Layout (target)

```
fastvideo/train/
├── methods/
│   └── svi/                            ← new
│       ├── __init__.py
│       ├── svi.py                       → SVITrainingMethod(TrainingMethod)
│       └── error_recycling.py           → buffers, sampling, replacement (Stage 5+)
├── datasets/                            ← new directory
│   └── svi_raw_video.py                 → SVIRawVideoDataset (returns video + ref frames)
└── models/wan/
    └── wan.py                           → reuse existing; LoRA already wired

examples/train/configs/fine_tuning/svi/
└── wan_i2v_svi.yaml                     ← new YAML

examples/train/svi/
└── run_svi_training.sh                  ← canonical launch script
```

---

## Staged Plan (do them in order, ship a PR per stage)

Each stage is a green-light gate: **don't start stage N+1 until stage N's
done-criterion is met**.

### Stage 0 — Onboarding & alignment baseline

Goal: this doc; freeze decisions D1–D5; capture a tiny upstream training-loss
trace as the parity target.

Done when:
- `.agents/onboarding/svi-training/README.md` is reviewed.
- `train_svi.py` runs for ~100 steps on a 10-clip toy dataset; loss curve +
  random seed saved under `.agents/exploration/svi-training/upstream_baseline/`.

### Stage 1 — `SVIRawVideoDataset` + YAML schema

Goal: a dataset class that returns exactly what `TextVideoDataset_onestage`
returns, wired into the new train stack via a YAML config block.

Deliverables:
- `fastvideo/train/datasets/svi_raw_video.py` mirroring
  `TextVideoDataset_onestage.__getitem__` (returns
  `{"text", "video"[3,T,H,W], "first_ref_frames", "random_ref_frame"}`).
- Extend `DataConfig` or add an `SVIDataConfig` with: `dataset_path`,
  `csv_path`, `num_ref_frames=12`, `sample_fps`, `max_frames`,
  `random_ref_strategy`. **Match upstream defaults.**
- `wan_i2v_svi.yaml` skeleton with the data block populated.

Verification: a smoke script iterates 5 batches from
`SVIRawVideoDataset` and prints shapes; tensors match upstream
`TextVideoDataset_onestage` on the same indices.

### Stage 2 — `SVITrainingMethod` skeleton (no ER)

Goal: a working LoRA fine-tune method that produces the same per-step loss as
`train_svi.py` minus error recycling.

Deliverables:
- `fastvideo/train/methods/svi/svi.py` subclassing `TrainingMethod`:
  - `single_train_step(batch, iter)` — on-the-fly VAE encode (video, ref
    frames, random ref), CLIP encode (first frame), build the SVI
    `y`-conditioning identical to inference's `SVIImageVAEEncodingStage`,
    sample timestep, add noise, predict, compute MSE loss
    times `training_weight(timestep)`, return `(loss_map, outputs, metrics)`.
  - Reuse `WanModel.prepare_batch` only for the LoRA wiring; otherwise build
    the batch in the method (we need the SVI-specific `y`).
- Hand-implement `training_weight` (linear bell-curve) inside the method —
  FV's `FlowMatchEulerDiscreteScheduler` doesn't expose one.

Verification: single-GPU run, 100 steps, loss curve overlay against the
Stage 0 baseline — should match within ±5% per-step.

### Stage 3 — Single-GPU sanity & checkpoint round-trip

Goal: train for 500 steps on a small dataset, save a LoRA checkpoint, load it
back into `wan_svi_i2v_pipeline` for inference, generate a video, **visually
sane**.

Deliverables:
- Verify `_RoleModuleContainer` saves only `lora_A` / `lora_B` params (it
  already does — confirm key names match the loader on the inference side).
- Add a `save_lora_as_safetensors` callback (or reuse existing) that emits a
  file the inference path can load via `lora_path=`.

Verification: `WanSVIImageToVideoPipeline.from_pretrained(...,
lora_path=<our_ckpt>)` runs end-to-end; output isn't garbage.

### Stage 4 — Multi-GPU (FSDP / DTensor)

Goal: scale to 4-8 GPUs with the existing trainer's distributed primitives.
No ER yet.

Deliverables:
- Test that `WanModel`'s DTensor sharding plays nicely with LoRA-only
  trainable params (it should — `requires_grad` filter is shard-agnostic, but
  verify the optimizer sees the right shards).
- Document any quirks in `.agents/lessons/`.

Verification: 4-GPU run for 200 steps; loss curve matches the single-GPU
Stage 3 run within ±10%.

### Stage 5 — Error Recycling

Goal: bolt ER onto `SVITrainingMethod`. This is the largest single stage.

Deliverables:
- `fastvideo/train/methods/svi/error_recycling.py`:
  - `ErrorBuffer` class — per-grid lists of CPU tensors, configurable
    replacement strategy (`random` / `fifo` / `l2_batch` / `l2_similarity`).
  - Grid-index lookup from timestep.
  - Distributed `all_gather_object` sync for the warm-up phase
    (`buffer_warmup_iter`).
- Hook into `SVITrainingMethod.single_train_step`:
  - Pre-noise: probabilistically sample noise/y/latent errors and inject.
  - Post-loss: with `torch.no_grad`, compute `x0_pred` / `x1_pred` errors and
    push to buffers.
- Expose all ER knobs (`use_error_recycling`, `error_buffer_k`,
  `num_grids`, `timestep_grid_size`, `buffer_warmup_iter`,
  `buffer_replacement_strategy`, `clean_buffer_update_prob`,
  `noise_prob` / `y_prob` / `latent_prob`) in YAML, **defaulting to
  upstream's values**.

Verification: 1k-step run with ER on. Compare loss trajectory & buffer
occupancy histograms against upstream's run with the same seed.

### Stage 6 — Parity gate & PR

Goal: an end-to-end parity check before the PR ships.

Deliverables:
- `tests/svi_training_parity.py` — runs 200 training steps with FV's
  `SVITrainingMethod` and upstream `train_svi.py` from the same seed / data
  / hyperparams. Asserts that the loss-curve correlation is ≥ 0.97 and the
  saved LoRA's tensors are within bf16 tolerance on a fixed set of layers.
- Update `docs/training/finetune.md` (or the SVI-specific doc) with the
  example YAML and the launch command.

Verification: parity test passes; PR opened against `main`.

---

## Open Threads (revisit as needed)

- **Parquet-based path for SVI training data**: parked. If we want this later,
  the cleanest route is a new preprocess pipeline that writes an SVI-specific
  schema with `clip_features_per_frame`. Cost: ~1 hour of compute per 10k
  videos. Don't do this until raw-video training is stable.
- **`ref_pad_cfg` multi-prompt**: SVI-2.0 feature for varying `ref_pad_num`
  across clips. Out of scope for v1 (matches inference scope).
- **Gradio demo / training dashboard**: not in scope.

---

## Quick Pointers

- Upstream training entrypoint: `Stable-Video-Infinity/train_svi.py`
- Upstream Lightning module: `LightningModelForTrain_onestage` (same file)
- Upstream dataset: `TextVideoDataset_onestage` (same file)
- Upstream scheduler: `Stable-Video-Infinity/diffsynth/schedulers/flow_match.py`
- FV inference (already done, source of truth for the `y` build):
  `fastvideo/pipelines/basic/wan/wan_svi_i2v_pipeline.py`
- FV inference's SVI `y` builder:
  `fastvideo/pipelines/stages/image_encoding.py::SVIImageVAEEncodingStage`
- FV LoRA hook: `fastvideo/train/utils/lora.py::enable_lora_training`
- Closest analog method (good template): KD method at
  `fastvideo/train/methods/knowledge_distillation/kd.py` — particularly the
  pattern of stateful method attributes + custom batch prep.

---

## When You're Stuck

- Numerical drift between FV and upstream: the inference work already
  identified the `FlowMatchEulerDiscreteScheduler` family vs `UniPCMultistep`
  as the drift driver. Same trap could bite training — verify schedulers
  first.
- Optimizer not picking up LoRA params: confirm `requires_grad=True` on
  `lora_A` / `lora_B` after `enable_lora_training` and that the optimizer is
  built **after** that call.
- DTensor / FSDP errors with `requires_grad` filtering: see
  `.agents/lessons/` for prior incidents (worldmodel team hit similar).

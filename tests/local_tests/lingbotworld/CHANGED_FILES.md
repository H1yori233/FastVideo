# LingBot-World-Fast + LingBot-World-Act — changed files

Net-new models: LingBot-World-Fast (block-causal, DMD-distilled streaming I2V) and
LingBot-World-Act (full-sequence A14B-MoE I2V with act2cam control, control_dim=7).

## Net-new files (Fast)
- `fastvideo/models/dits/lingbotworld/causal_model.py` — `CausalLingBotWorldTransformer3DModel`
  (Causal-Wan block-causal attention + KV cache + the Cam camera conditioner).
- `fastvideo/pipelines/stages/lingbotworld_fast_denoising.py` —
  `LingBotWorldFastCausalDenoisingStage` (chunked DMD AR loop + clean-context KV update).
- `fastvideo/pipelines/basic/lingbotworld/lingbotworld_fast_pipeline.py` —
  `LingBotWorldCausalDMDPipeline` (matches `model_index.json._class_name`).
- `examples/inference/basic/basic_lingbotworld_fast.py` — runnable example.
- `tests/local_tests/lingbotworld/` — `test_lingbotworld_fast_parity.py`,
  `_fast_parity_runner.py`, `README.md`, `PORT_STATUS.md`, `CHANGED_FILES.md`,
  `_fast_pipeline_runner.py`.
- `tests/local_tests/pipelines/test_lingbotworld_fast_pipeline_smoke.py`,
  `tests/local_tests/pipelines/test_lingbotworld_fast_pipeline_parity.py`.

## Net-new files (Act)
- `fastvideo/models/dits/lingbotworld/act_utils.py` — `prepare_action_embedding`
  (WASD/IJKL action string -> camera trajectory -> rays_d Plücker (3) + WASD (4) =
  control_dim 7 tensor `[1, 7*64, F_lat, H_lat, W_lat]`).
- `scripts/checkpoint_conversion/convert_lingbotworld_act_to_diffusers.py` — symlink
  converter: official MoE preview (`high_noise_model`/`low_noise_model`) ->
  `transformer/` + `transformer_2/` (control_dim=7 config) + reused Wan2.1
  vae/text_encoder/tokenizer/scheduler + `model_index.json` (boundary_ratio 0.947).
- `examples/inference/basic/basic_lingbotworld_base_act.py` — runnable example
  (drives generation from an action string via `prepare_action_embedding`).
- `tests/local_tests/lingbotworld/_act_parity_runner.py` — component-parity runner
  (official base `WanModel(control_type='act')` vs FastVideo Act, MATH SDPA, official-first).
- `tests/local_tests/pipelines/test_lingbotworld_act_pipeline_smoke.py` — preflight
  (registry/preset/config wiring) + real 28B-MoE generate smoke gated on `$LINGBOT_ACT_DIR`.

## Edited (additive) files
- `fastvideo/configs/models/dits/lingbotworld.py` — add `CausalLingBotWorldArchConfig`
  (in_channels=36, sink_size=9, num_frames_per_block=3, sliding_window_num_frames=18)
  and `CausalLingBotWorldVideoConfig`; add `control_dim` (default 6) to the Cam arch
  config; add `LingBotWorldActArchConfig` (control_dim=7, in_channels=36, out_channels=16)
  and `LingBotWorldActVideoConfig`.
- `fastvideo/configs/pipelines/lingbotworld.py` — add `LingBotWorldFastI2V480PConfig`
  (single block-causal DMD transformer, no CLIP, UniPC, flow_shift=3.0) and
  `LingBotWorldActI2V480PConfig` (Wan2.2 A14B MoE base, dit=Act config, boundary 0.947).
- `fastvideo/models/dits/lingbotworld/model.py` — `WanCamControlPatchEmbedding` in_chans
  uses `config.control_dim * 64` (default 6 = Cam unchanged; 7 = Act).
- `fastvideo/models/dits/lingbotworld/__init__.py` — export the causal model.
- `fastvideo/models/registry.py` — register `CausalLingBotWorldTransformer3DModel`.
- `fastvideo/registry.py` — register the Fast + Act pipeline configs + detectors
  (Cam detector excludes fast/act; Fast detector "fast"; Act detector "act") + presets.
- `fastvideo/pipelines/stages/__init__.py` — export the Fast denoising stage.
- `fastvideo/pipelines/basic/lingbotworld/presets.py` — add `lingbotworld_fast_i2v`
  and `lingbotworld_act_i2v` presets.

## Reused (unchanged production code)
- Causal-Wan self-attention (`models/dits/causal_wanvideo.py`),
  the Cam camera conditioner (`models/dits/lingbotworld/model.py`),
  the Wan-2.1 I2V VAE-concat image encoding stage
  (`pipelines/stages/image_encoding.py:MatrixGame2ImageVAEEncodingStage`),
  Wan VAE / UMT5 / UniPC scheduler, and `pred_noise_to_pred_video`.
- Fast: no conversion script (the diffusers repo stores official-named tensors;
  the Cam `param_names_mapping` is inherited verbatim).
- Act: reuses the Cam full-sequence A14B-MoE `LingBotWorldTransformer3DModel` and the
  Cam I2V/MoE pipeline end-to-end — only `control_dim` (6 -> 7) differs. The act2cam
  control rides the existing `c2ws_plucker_emb` plumbing (sampling_param -> stage ->
  `patch_embedding_wancamctrl`); no pipeline/stage code is Act-specific.

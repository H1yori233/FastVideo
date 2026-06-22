# LingBot-World-Fast — changed files

Net-new model: LingBot-World-Fast (block-causal, DMD-distilled streaming I2V).

## Net-new files
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

## Edited (additive) files
- `fastvideo/configs/models/dits/lingbotworld.py` — add `CausalLingBotWorldArchConfig`
  (in_channels=36, sink_size=9, num_frames_per_block=3, sliding_window_num_frames=18)
  and `CausalLingBotWorldVideoConfig`.
- `fastvideo/configs/pipelines/lingbotworld.py` — add `LingBotWorldFastI2V480PConfig`
  (single block-causal DMD transformer, no CLIP, UniPC, flow_shift=3.0).
- `fastvideo/models/dits/lingbotworld/__init__.py` — export the causal model.
- `fastvideo/models/registry.py` — register `CausalLingBotWorldTransformer3DModel`.
- `fastvideo/registry.py` — register the Fast pipeline config + detector
  (mutually exclusive with the Cam detector) + preset.
- `fastvideo/pipelines/stages/__init__.py` — export the Fast denoising stage.
- `fastvideo/pipelines/basic/lingbotworld/presets.py` — add `lingbotworld_fast_i2v` preset.

## Reused (unchanged production code)
- Causal-Wan self-attention (`models/dits/causal_wanvideo.py`),
  the Cam camera conditioner (`models/dits/lingbotworld/model.py`),
  the Wan-2.1 I2V VAE-concat image encoding stage
  (`pipelines/stages/image_encoding.py:MatrixGame2ImageVAEEncodingStage`),
  Wan VAE / UMT5 / UniPC scheduler, and `pred_noise_to_pred_video`.
- No conversion script (the diffusers repo stores official-named tensors;
  the Cam `param_names_mapping` is inherited verbatim).

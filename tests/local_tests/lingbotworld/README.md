# LingBot-World local tests

Reviewer-facing setup + verification log for the LingBot-World family ports
(Cam — already in `main`; Fast — this PR; Act — planned, see `PORT_STATUS.md`).

## Official reference

- Code: https://github.com/Robbyant/lingbot-world (sparse clone at
  `reference/lingbot-world/`, or set `$LINGBOT_WORLD_REPO`). Key files:
  `wan/modules/model_fast.py` (block-causal DiT), `wan/image2video_fast.py`
  (chunked DMD AR loop), `generate_fast.py`.
- Paper: arXiv 2601.20540.

## Weights

| Variant | HF repo | Format | Token |
|---------|---------|--------|-------|
| Cam | `FastVideo/LingBot-World-Base-Cam-Diffusers` | diffusers (official-named tensors) | none (public) |
| Fast | `robbyant/lingbot-world-fast-diffusers` | diffusers (official-named tensors) | none (public) |
| Act | `robbyant/lingbot-world-base-act-preview` | official MoE (`high_noise_model/`, `low_noise_model/`) — needs converter | none (public) |

The Fast diffusers repo stores **official-named** transformer tensors (1421
keys, identical key set to Cam), so the same weights load into both the official
`WanModelFast` and the FastVideo `CausalLingBotWorldTransformer3DModel`. No
conversion script is required for Fast.

Download (cached under the HF hub cache):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('robbyant/lingbot-world-fast-diffusers')"
```

## Environment

Shared env `FastVideo_kaiqin`, `PYTHONPATH=<this repo>`. No special pins.
The official reference repo imports with the shared env (pure torch + flash-attn).

## Tests

```bash
# Component parity (official WanModelFast vs FastVideo causal model, same weights)
DISABLE_SP=1 pytest tests/local_tests/lingbotworld/test_lingbotworld_fast_parity.py -v -s

# Pipeline smoke (load via VideoGenerator + 1-chunk generate)
DISABLE_SP=1 pytest tests/local_tests/pipelines/test_lingbotworld_fast_pipeline_smoke.py -v -s

# Pipeline parity (FastVideo denoising stage vs official-model reference AR loop)
DISABLE_SP=1 pytest tests/local_tests/pipelines/test_lingbotworld_fast_pipeline_parity.py -v -s

# Runnable example
python examples/inference/basic/basic_lingbotworld_fast.py
```

Required-before-handoff: every test above is a non-skip PASS. See `PORT_STATUS.md`
for current status and recorded numbers.

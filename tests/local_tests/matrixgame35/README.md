# Matrix-Game 3.5 Local Tests

Local-only parity and smoke tests for the `matrixgame35` FastVideo port. These
tests compare FastVideo against the pinned official implementation and are not
expected to run in CI unless explicitly promoted later.

Port progress, open questions, issues, and handoff notes live in
`tests/local_tests/matrixgame35/PORT_STATUS.md`.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `matrixgame35` |
| Workload types | camera-conditioned I2V; bidirectional base and causal few-step variants |
| Official reference | `https://github.com/Riemann-Dynamics/Matrix-Game-3.5` |
| Local reference dir | `Matrix-Game-3.5/` |
| Official commit | `fa6d2b628ac9b0f1657dc24689536d74bfeb51da` |
| Base weights | `RiemannDynamics/Matrix-Game-3.5-Base` |
| Base revision | `c3b0c9c541b7754a78b5e2199e9587e003668de9` |
| Distilled weights | `RiemannDynamics/Matrix-Game-3.5-Distilled` |
| Distilled revision | `0b38ca0b0dda2bb994c570e183ad36d1acd53be2` |
| Shared Wan weights | `Wan-AI/Wan2.2-TI2V-5B` / Diffusers equivalent |
| Shared Wan revision | `b8fff7315c768468a5333511427288870b2e9635` |
| Shared depth weights | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` |
| Shared depth revision | `b2359bdf726fb44ef62acca04d629dcf158053e7` |
| Local weights dir | `official_weights/matrixgame35/` (stage on Shifu; not downloaded locally) |
| Source layout | `raw_official` full-DiT safetensors plus shared raw Wan/DA3 components |
| Needs conversion | `yes` |

The public repositories are ungated, so no token is required for correctness.
If Hugging Face rate limiting requires authentication, use only the environment
variable name `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `HF_API_KEY`; never record
the value here.

### Released checkpoint inventory

| Variant | File | Bytes | SHA-256 | Architecture delta |
|---|---:|---:|---|---|
| base first-person | `first-person.safetensors` | `10000464984` | `3d758de69f545c835ad115f50b75719e682a83c18acdf219e6c720c5f3da5ea8` | subject-reference index table has 2 slots |
| base third-person | `third-person.safetensors` | `10000477272` | `3388cf355148355ce216ce18a44bd304574f7eaa8c636fb14c4cbd0b47d777cf` | subject-reference index table has 4 slots |
| distilled first-person | `first-person.safetensors` | `9999659704` | `de476e7fc0bdd756aafb101a2b80040f65b3ad62dafea109e299aafa599b8094` | no subject-reference tensors |

Safetensors header inspection shows the remaining key, shape, and dtype surface
is shared: 30 blocks, hidden width 3072, 24 heads, FFN width 14336, 48 input and
output latent channels, and BF16 tensors throughout.

The released scope is inference-only. `standard`, `hiar-sde`, and
`sink-anchor-context` are distilled runtime profiles, not additional checkpoint
variants. FastVideo keeps one distilled pipeline and selects these policies with
`matrixgame35_distilled_profile`; optional `matrixgame35_distilled_hiar_scales`
preserves the released per-step corruption control. The pinned HiAR rollout
requires CFG=1 even though the public config dataclass retains the generic CFG=3
default; FastVideo validates the executable rollout constraint.

### Frame-count contract

The official README and CLI help still say 80 generated frames per block, but
the pinned executable path and the shipped six-block sample implement:

```text
total RGB frames = 1 anchor + 84 * num_blocks
```

Each block contains 21 noisy latent frames and each noisy latent contributes
four RGB frames. The official six-block preview has 505 frames. FastVideo tests
must assert the current 84-frame behavior and retain a regression note for the
stale upstream documentation.

## Shared Environment Setup

Run from the FastVideo repository root in the same environment used by
FastVideo. Do not create a separate upstream environment for parity tests.

```bash
python ".agents/skills/add-model-01-prep/scripts/clone_reference_repo.py" \
    "https://github.com/Riemann-Dynamics/Matrix-Game-3.5.git" \
    "Matrix-Game-3.5" \
    --commit "fa6d2b628ac9b0f1657dc24689536d74bfeb51da"

# The official repository is source-only; add it to PYTHONPATH for parity.
PYTHONPATH=Matrix-Game-3.5:$PYTHONPATH python -m pytest \
    tests/local_tests/matrixgame35 -v -s
```

Do not change FastVideo's core `torch`, `transformers`, attention, Triton, or
CUDA pins. The official package imports optional download/media dependencies
from its umbrella `diffsynth/__init__.py`; parity tests must import the narrow
official modules through a test-only helper or install only the missing public
packages in the ephemeral Shifu job.

## Official Environment Status

```text
dependency_changes: none
official_env_status: narrow pinned modules load through test-only import helpers
private_dep_stubs: none
blocked_on: the full official umbrella import still reaches unrelated optional packages; parity loads only the exact modules under test
```

Depth-Anything-3 is a public optional preprocessing dependency. Production code
must load it lazily and emit an actionable install error; it must not add or
change FastVideo core dependency pins.

## Weight Setup

The three Matrix-Game checkpoints total about 30 GB before shared Wan2.2 and DA3
assets. Stage them in the queue-managed Shifu job or an existing verified Shifu
cache, not on the local macOS checkout.

```bash
python ".agents/skills/add-model-01-prep/scripts/download_hf_weights.py" \
    "RiemannDynamics/Matrix-Game-3.5-Base" \
    "official_weights/matrixgame35/base" \
    --revision "c3b0c9c541b7754a78b5e2199e9587e003668de9"

python ".agents/skills/add-model-01-prep/scripts/download_hf_weights.py" \
    "RiemannDynamics/Matrix-Game-3.5-Distilled" \
    "official_weights/matrixgame35/distilled" \
    --revision "0b38ca0b0dda2bb994c570e183ad36d1acd53be2"
```

Use the exact Wan2.2 and DA3 revisions recorded above during conversion and
Shifu staging; upstream aliases are not pinned and must not be resolved live.

When the pinned Diffusers snapshot is already staged, the shared VAE/text gates
do not require the raw Wan files:

```bash
export MATRIXGAME35_WAN22_DIFFUSERS_DIR=/path/to/Wan2.2-TI2V-5B-Diffusers
pytest \
  tests/local_tests/matrixgame35/test_matrixgame35_vae_parity.py \
  tests/local_tests/matrixgame35/test_matrixgame35_text_encoder_parity.py \
  -v -s
```

The raw-official tests remain in the same files and continue to use
`MATRIXGAME35_WAN22_RAW_DIR` when those original assets are available.

## Prototype And Conversion Artifacts

```text
official_key_dumps:
  transformer variants: converted_weights/matrixgame35/_mapping/*_official_keys.json
fastvideo_key_dumps:
  transformer variants: converted_weights/matrixgame35/_mapping/*_fastvideo_keys.json
conversion_script: scripts/checkpoint_conversion/matrixgame35_to_diffusers.py
conversion_source_layout: raw_official
converted_weights_dir: converted_weights/matrixgame35
strict_load_status: synthetic_cpu_pass; Base first/third/Distilled real-weight formal Shifu pass
```

The converter must emit one shared Diffusers-style component layout with three
transformer variants, validate the published SHA-256 values, preserve BF16, and
strict-load every variant. Wan VAE, UMT5/tokenizer, scheduler metadata, and the
external DA3 asset are passthrough components with pinned provenance.

`fastvideo/configs/pipelines/matrixgame35.py` owns the shared VAE/text helpers
and the three registry-reachable pipeline configs. The VAE helper delegates to the
existing LucyEdit/DreamX Wan2.2 48-channel config. The text helper delegates to
the DreamX UMT5 config and applies only Matrix-Game's official constructor and
tokenizer deltas: gated GELU, inference-inert dropout `0.1`, whitespace cleanup,
and fixed right-padded length `512`.

## Expected Parity Tests

| Component | Official files / args | Planned test | Main concern | Status |
|---|---|---|---|---|
| Matrix-Game DiT / PRoPE | `diffsynth/models/wan_video_dit.py`; `diffsynth/models/prope_attention.py`; 30 layers, PRoPE every block | `tests/local_tests/matrixgame35/test_matrixgame35_transformer_parity.py`; `test_matrixgame35_noncausal_model_fn_parity.py`; `test_matrixgame35_causal_model_fn_parity.py` | arbitrary-time native RoPE, PRoPE, physical mosaic-hole drop/scatter, causal pre-RoPE K/raw-V cache, per-token timestep, frozen memory prefixes, dtype boundaries | direct pinned noncausal and causal CPU parity pass (`max_abs_diff<=1.67e-06`); Base first/third and Distilled real-weight Shifu pass |
| subject-reference tokens | `diffsynth/pipelines/wan_video.py::_build_subject_ref_memory_tokens` | `tests/local_tests/matrixgame35/test_matrixgame35_subject_ref_parity.py`; `test_matrixgame35_noncausal_model_fn_parity.py` | 2/4-slot shapes, packing, negative-time RoPE, masks, model integration | direct-official CPU helper and integrated model parity pass |
| Wan2.2 VAE | official `WanVideoVAE38`; raw `Wan2.2_VAE.pth`; HF Diffusers `AutoencoderKLWan` | `tests/local_tests/matrixgame35/test_matrixgame35_vae_parity.py` | encode/decode, 3.8 temporal stride, and exact 704x1280 weighted tiling | direct-official CPU config pass; independent Diffusers CUDA and release-resolution tiled Shifu gates pass; raw-weight gate retained separately |
| UMT5 + tokenizer | official Wan text encoder/tokenizer; HF Transformers `UMT5EncoderModel` | `tests/local_tests/matrixgame35/test_matrixgame35_text_encoder_parity.py` | hidden states, fixed padding, whitespace cleaning | direct-official CPU config/cleaning pass; independent snapshot tokenizer/UMT5 Shifu gates pass; raw-weight gate retained separately |
| flow schedules | base FlowMatch shift 5; distilled `[1000, 667, 333]` | `tests/local_tests/matrixgame35/test_matrixgame35_schedule_parity.py` | sigma/timestep and three-step student transition | direct-official CPU parity passes |
| camera preparation / PRoPE | `diffsynth/pipelines/wan_video.py`; camera `.npz` contract | `tests/local_tests/matrixgame35/test_matrixgame35_camera_parity.py` | c2w/w2c, pixel intrinsics, four sub-frame cameras | CPU contract passes |
| DA3 depth adapter | vendored official `depth_anything_3.api.DepthAnything3` | `tests/local_tests/matrixgame35/test_matrixgame35_depth_adapter.py` | lazy loading; Base process resolution 504; Distilled 448; metric-depth output | local source/config pass; standalone pinned-interpreter CUDA gate passes with finite FP32 `[1,350,504]` depth |
| Patch Memory | `frustum/`; `examples/wanvideo/pipeline/mosaic/` | `tests/local_tests/matrixgame35/test_matrixgame35_memory_parity.py`; `test_matrixgame35_distilled_memory_parity.py` | visibility, z-buffer fusion, candidate selection, no cross-block leakage | direct pinned CPU parity passes |
| base first-person pipeline | `infer.py --person first` + `configs/infer_first_person.yaml` | `tests/local_tests/pipelines/test_matrixgame35_base_first_person_pipeline.py` | 25-step 704x1280 rollout; 84 generated RGB frames/block | focused fake-component control-flow pass; full official-case Shifu video gate passes with 505 decoded frames |
| base third-person pipeline | `infer.py --person third` + optional refs | `tests/local_tests/pipelines/test_matrixgame35_base_third_person_pipeline.py` | no-ref and 1-4-ref paths | focused fake-component control-flow pass; full official-case one-ref Shifu video gate passes with 505 decoded frames |
| distilled first-person pipeline | `infer_distilled.py`; `configs/infer_distilled.yaml`; `distilled_config.py` | `tests/local_tests/pipelines/test_matrixgame35_distilled_standard_pipeline.py`; `tests/local_tests/matrixgame35/test_matrixgame35_distilled_profile_parity.py` | shared causal KV/cache-fill path; STANDARD CFG=3; HiAR CFG=1 with per-step rolling/dynamic-context corruption; sink C0 context | direct pinned CPU profile/helper parity and focused fake pipeline paths pass; all three real-weight 505-frame Shifu gates pass |

Local CPU runs may legitimately skip CUDA/weight paths, but a skip is not a
verified pass. Final acceptance requires queue-terminal Shifu jobs and inspected
logs/artifacts for every released variant.

The final targeted local result is `304 passed, 15 skipped`; its exact command
is recorded in `PORT_STATUS.md`. The local skips are CUDA or optional raw-asset
gates whose corresponding real-weight Shifu jobs are recorded there; no skip is
counted as a local pass.

## Review Notes

- The activation order was shared components and tests, base first-person, base
  third-person, then distilled first-person; registry activation stayed last.
- Reuse is accepted only after exact official instantiation arguments and
  non-skip numerical parity are demonstrated.
- Do not copy the official training/validation framework into FastVideo. Port
  only the inference contracts required by the three released checkpoints.
- Keep sequence parallelism explicitly unsupported at first. The official
  release is single-GPU and Matrix camera/token sharding needs separate parity.
- At least one parity boundary must use upstream-produced packing/memory tensors;
  never compare two paths built from the same FastVideo helper.
- Final validation must inspect exact output paths, frame count, resolution,
  FPS, checksums, and deterministic quality evidence on Shifu.

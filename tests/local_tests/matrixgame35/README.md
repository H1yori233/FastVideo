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
| Shared Wan revision | `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` |
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
variants; only `standard` is part of the initial production port.

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
official_env_status: private_deps_need_stubs
private_dep_stubs: planned test-only narrow importer for unused optional modules
blocked_on: full official umbrella import currently reaches missing `modelscope` locally; targeted source modules and public metadata are available
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

## Prototype And Conversion Artifacts

```text
official_key_dumps:
  transformer variants: converted_weights/matrixgame35/_mapping/*_official_keys.json
fastvideo_key_dumps:
  transformer variants: converted_weights/matrixgame35/_mapping/*_fastvideo_keys.json
conversion_script: scripts/checkpoint_conversion/matrixgame35_to_diffusers.py
conversion_source_layout: raw_official
converted_weights_dir: converted_weights/matrixgame35
strict_load_status: not_run
```

The converter must emit one shared Diffusers-style component layout with three
transformer variants, validate the published SHA-256 values, preserve BF16, and
strict-load every variant. Wan VAE, UMT5/tokenizer, scheduler metadata, and the
external DA3 asset are passthrough components with pinned provenance.

## Expected Parity Tests

| Component | Official files / args | Planned test | Main concern | Status |
|---|---|---|---|---|
| Matrix-Game DiT / PRoPE | `diffsynth/models/wan_video_dit.py`; `diffsynth/models/prope_attention.py`; 30 layers, PRoPE every block | `tests/local_tests/matrixgame35/test_matrixgame35_transformer_parity.py` | parameter-free PRoPE, per-token timestep, frozen memory prefixes, dtype boundaries | CPU contracts pass; real-weight Shifu parity pending |
| subject-reference tokens | `diffsynth/pipelines/wan_video.py::_build_subject_ref_memory_tokens` | `tests/local_tests/matrixgame35/test_matrixgame35_subject_ref_parity.py` | 2/4-slot variant shapes, packing, positional embeddings, masks | planned |
| Wan2.2 VAE | official `WanVideoVAE38`; raw `Wan2.2_VAE.pth` | `tests/local_tests/matrixgame35/test_matrixgame35_vae_parity.py` | exact encode/decode and 3.8 temporal stride | planned |
| UMT5 + tokenizer | official Wan text encoder and tokenizer | `tests/local_tests/matrixgame35/test_matrixgame35_text_encoder_parity.py` | hidden states, padding, negative prompt | planned |
| flow schedules | base FlowMatch shift 5; distilled `[1000, 667, 333]` | `tests/local_tests/matrixgame35/test_matrixgame35_schedule_parity.py` | sigma/timestep and three-step student transition | direct-official CPU parity passes |
| camera preparation / PRoPE | `diffsynth/pipelines/wan_video.py`; camera `.npz` contract | `tests/local_tests/matrixgame35/test_matrixgame35_camera_parity.py` | c2w/w2c, pixel intrinsics, four sub-frame cameras | CPU contract passes |
| DA3 depth adapter | vendored official `depth_anything_3.api.DepthAnything3` | `tests/local_tests/matrixgame35/test_matrixgame35_depth_parity.py` | lazy loading, metric depth values and preprocessing | planned |
| Patch Memory | `frustum/`; `examples/wanvideo/pipeline/mosaic/` | `tests/local_tests/matrixgame35/test_matrixgame35_memory_parity.py` | visibility, z-buffer fusion, candidate selection, no cross-block leakage | planned |
| base first-person pipeline | `infer.py --person first` + `configs/infer_first_person.yaml` | `tests/local_tests/pipelines/test_matrixgame35_base_first_pipeline_parity.py` | 25-step 704x1280 rollout; 84 generated RGB frames/block | planned |
| base third-person pipeline | `infer.py --person third` + optional refs | `tests/local_tests/pipelines/test_matrixgame35_base_third_pipeline_parity.py` | no-ref and 1-4-ref paths | planned |
| distilled first-person pipeline | `infer_distilled.py`; `configs/infer_distilled.yaml` | `tests/local_tests/pipelines/test_matrixgame35_distilled_pipeline_parity.py` | causal KV cache, chunking, CFG=3, memory publication | planned |

Local CPU runs may legitimately skip CUDA/weight paths, but a skip is not a
verified pass. Final acceptance requires queue-terminal Shifu jobs and inspected
logs/artifacts for every released variant.

## Review Notes

- The activation order is shared components and tests, base first-person, base
  third-person, then distilled first-person; registry activation is last.
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

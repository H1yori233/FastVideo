# FLUX 3 Image Klein

Native FastVideo inference for `black-forest-labs/flux-3-image-klein`, including
its Qwen3.5 prompt upsampler (PUP), text-to-image, image editing, multiple
references and bounding-box layout. The implementation is independent of the
existing FLUX and FLUX2 families. Port decisions and open gates are recorded in
[PORT_STATUS.md](PORT_STATUS.md).

FastVideo reads the original official checkpoints and
`modular_model_index.json`. The DiT, VAE, text encoder and PUP remain native
FastVideo implementations. Checkpoint names are mapped in memory during loading;
no converted checkpoint, rewritten config or separate checkpoint publication is
required. The official manifest resolves PUP from its separate HF repository.

**Verification status:** the official files have been checked against HF
metadata. Direct official loading passes DiT, VAE and main text parity, plus an
offline HF ID end-to-end smoke. Basic and extended PUP parity and all 22
pipeline cases pass with observed numerical error 0.
Earlier results are retained below as historical evidence.

## Architecture

PUP turns the prompt, optional images and boxes into a caption using native
Qwen3.5 language and vision models. A separate Qwen3-VL language encoder
concatenates eight hidden states (layers 4/8/…/32, each width 2560), producing
text tokens of width 20480. Boxes become caption/object-table conditioning.

References also follow a VAE path: 8× spatial compression to 32 channels, then
channel-major 2×2 packing into 128-dimensional tokens. A 1024×1024 output uses
4096 image tokens. The VAE runs in FP32 and encodes posterior means.

The DiT projects text, reference and noisy-image tokens to width 3072. Each
stream has five separate blocks, followed by 28 shared blocks with 24 attention
heads and four-axis RoPE. Image tokens reach the 128-dimensional output head.
Distilled sampling uses `x_next = x + (t_next - t) * v`, with current and target
time conditioning, 16 steps, alpha 6.93 and no CFG.

Family-specific packing, image preparation, sampling and request state live in
`fastvideo/pipelines/basic/flux3/`. The private state is removed before return
and on failure. Bbox rules live beside PUP in
`fastvideo/models/encoders/flux3_pup/bbox.py`. Numerical code is copied/adapted
from the official sources with copyright and modification notices. Production
uses third-party tokenizers/processors, without third-party model classes or
imports from the reference clone.

## Sources and environment

| Asset | Reproduction revision |
|---|---|
| [Official source](https://github.com/black-forest-labs/flux-3-image-klein) | `bbfe317dd81c825c30d76090a67d3d0db51da3cc` |
| `black-forest-labs/flux-3-image-klein` | `337337621b8526e120f2275d250242f4a6bedd75` |
| `black-forest-labs/flux-3-image-klein-pup` | `3f3edb52b2c55de0d49feab28598b68f7f8e4830` |
| Official Qwen3-VL/Qwen3.5 implementations | Transformers `5.8.1` |

The local reference environment uses Python 3.10.19, Torch 2.8.0+cu128,
Triton 3.4.0 and one RTX 5090 (32 GiB) per process. Optional vLLM, FLA,
causal-conv1d and quantization kernels are not needed. The existing Torch/Triton
stack was retained.

Install FastVideo using its setup guide. For independent reference tests, install
the pinned source into the same environment:

```bash
git clone https://github.com/black-forest-labs/flux-3-image-klein.git
git -C flux-3-image-klein checkout bbfe317dd81c825c30d76090a67d3d0db51da3cc
uv pip install --no-deps -e ./flux-3-image-klein
uv pip install transformers==5.8.1 accelerate==1.13.0 \
    safetensors==0.8.0rc0 tokenizers==0.22.2
uv pip install cloudpickle pytest pre-commit diffusers==0.38.0 peft==0.18.1 \
    ftfy omegaconf==2.3.0 scipy imageio imageio-ffmpeg loguru psutil \
    einops==0.8.1 remote-pdb
uv pip install --no-deps -e .
```

## Official checkpoints

Run commands from the repository root with an authenticated Hugging Face login
or `HF_TOKEN`. The example's default HF ID downloads the official components as
needed. To stage the pinned main snapshot locally:

```bash
python .agents/skills/add-model-01-prep/scripts/download_hf_weights.py \
    black-forest-labs/flux-3-image-klein official_weights/flux3/flux-3-image-klein \
    --revision 337337621b8526e120f2275d250242f4a6bedd75 \
    --allow-pattern config.json --allow-pattern manifest.json \
    --allow-pattern modular_model_index.json --allow-pattern LICENSE.md \
    --allow-pattern NOTICE.md --allow-pattern README.md \
    --allow-pattern 'text_encoder/*' --allow-pattern 'transformer/*' \
    --allow-pattern 'vae/*'

hf download black-forest-labs/flux-3-image-klein-pup \
    --revision 3f3edb52b2c55de0d49feab28598b68f7f8e4830
```

The main snapshot contains the official two-shard transformer, VAE, text
encoder and processor assets. Its entrypoint is `modular_model_index.json`;
there is no generated `model_index.json`.

PUP resolves from the manifest's repository and revision through the shared HF
cache, including when the main model is a local directory. There is no sibling
`../pup` fallback. The pinned manifest has `revision: null`, so production follows
the repository's default branch. For offline execution, resolve that reference
while online as well:

```bash
hf download black-forest-labs/flux-3-image-klein-pup
```

The revision table records the snapshots used for local verification; future HF
default-branch updates may differ. Original files remain unchanged on disk.

Independent official parity additionally needs BFL's raw DiT/VAE files and a
local PUP reference directory. These files are test inputs, separate from
production PUP cache resolution:

```bash
hf download black-forest-labs/flux-3-image-klein model.safetensors ae.safetensors \
    --revision 337337621b8526e120f2275d250242f4a6bedd75 \
    --local-dir official_weights/flux3/flux-3-image-klein
hf download black-forest-labs/flux-3-image-klein-pup \
    --revision 3f3edb52b2c55de0d49feab28598b68f7f8e4830 \
    --local-dir official_weights/flux3/pup
```

Keep checkpoints, generated media and the separate reference clone out of
commits. All 41 selected local source files were checked against HF metadata
after staging the official component weights; the audit is saved in
`outputs/flux3/official_weights/source_audit.json`.

## Generate and edit

The example defaults to `black-forest-labs/flux-3-image-klein`. Use
`--model-path official_weights/flux3/flux-3-image-klein` for the local snapshot
above. PUP runs by default; components load and release sequentially for one
32 GiB GPU. Defaults are 1024×1024, 16 steps and integer seed 1024.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/inference/basic/basic_flux3.py \
    --prompt 'A red bicycle leaning against a white stucco wall, morning sunlight, a small green plant beside it.' \
    --seed 7 --output-path outputs/flux3/image.png

CUDA_VISIBLE_DEVICES=0 python examples/inference/basic/basic_flux3.py \
    --prompt 'Replace the background with a quiet garden.' \
    --reference input.png --auto-size --output-path outputs/flux3/edit.png

CUDA_VISIBLE_DEVICES=0 python examples/inference/basic/basic_flux3.py \
    --prompt 'Put the person from the first image in the room from the second.' \
    --reference person.png --reference room.png \
    --output-path outputs/flux3/multi.png

CUDA_VISIBLE_DEVICES=0 python examples/inference/basic/basic_flux3.py \
    --prompt 'A red fox resting beside a quiet lake.' \
    --boxes '[{"id":"fox","bbox":[0.15,0.25,0.75,0.85],"desc":"a red fox"}]' \
    --output-path outputs/flux3/layout.png
```

`--no-pup` accepts an expanded caption. `--auto-size` adopts the first prepared
reference's size and restores the crop for a single reference. Explicit
width/height must be multiples of 16. Up to ten references are accepted.
Public request extensions are `flux3_boxes`, `flux3_alpha`, `flux3_auto_size`,
`flux3_upsample_prompt` and `flux3_pup_max_new_tokens`.

Supported scope is distilled BF16 inference on one GPU. Base/quantized models,
training, tensor/sequence parallelism and multi-image batches are rejected.
The VAE requires FP32 encode/decode without tiling; unsupported overrides fail
before model loading.
Layerwise offload and FSDP are disabled; deferred component loading manages
memory. The optional BFL demo and classifier wrapper are outside this port.

## Verification commands

The direct official-loading rerun passed. Tests compare independent
official and native execution with real weights. A missing dependency or skipped
test is not a pass. Defaults and overrides are:

| Environment variable | Default | Purpose |
|---|---|---|
| `FLUX3_MODEL_DIR` | `official_weights/flux3/flux-3-image-klein` | Native main-model loading |
| `FLUX3_PUP_WEIGHTS_DIR` | `official_weights/flux3/pup` | Independent PUP component/reference tests |
| `FLUX3_OFFICIAL_REF_DIR` | `flux-3-image-klein` | Pinned official source clone |
| `FLUX3_OFFICIAL_WEIGHTS_DIR` | `official_weights/flux3/flux-3-image-klein` | Raw BFL pipeline reference files |
| `FLUX3_VAE_WEIGHTS` | `official_weights/flux3/flux-3-image-klein/ae.safetensors` | Raw BFL VAE reference |
| `FLUX3_REFERENCE_ROOT` | `flux-3-image-klein` | Sampling helper reference source |

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/local_tests/transformers/test_flux3_parity.py -v -s
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/local_tests/vaes/test_flux3_vae_parity.py -v -s
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/local_tests/encoders/test_flux3_qwen3_vl_parity.py -v -s
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/local_tests/encoders/test_flux3_pup_text_parity.py -v -s
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/local_tests/encoders/test_flux3_pup_vision_parity.py -v -s
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/local_tests/encoders/test_flux3_pup_parity.py -k 'not extended' -v -s
python -m pytest tests/local_tests/flux3/test_flux3_sampling_parity.py \
    tests/local_tests/flux3/test_flux3_pup_policy_parity.py \
    fastvideo/tests/pipelines/test_flux3_contracts.py -q
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/local_tests/pipelines/test_flux3_pipeline_smoke.py -v -s
CUDA_VISIBLE_DEVICES=0 DISABLE_SP=1 python -m pytest tests/local_tests/pipelines/test_flux3_pipeline_parity.py -v -s
```

Run the extended matrices after component and smoke gates:

```bash
CUDA_VISIBLE_DEVICES=0 FLUX3_EXTENDED_PARITY=1 DISABLE_SP=1 \
    python -m pytest tests/local_tests/pipelines/test_flux3_pipeline_parity.py -v -s
CUDA_VISIBLE_DEVICES=0 DISABLE_SP=1 \
    FLUX3_PARITY_CASES=photo_single_512,photo_multi_pup \
    FLUX3_PARITY_OUTPUT_DIR=outputs/flux3/photo_parity \
    python -m pytest tests/local_tests/pipelines/test_flux3_pipeline_parity.py -v -s
CUDA_VISIBLE_DEVICES=1 FLUX3_PUP_EXTENDED=1 \
    python -m pytest tests/local_tests/encoders/test_flux3_pup_parity.py -k extended_generation_matrix -v -s
```

Current official-file results:

| Scope | Result | Evidence |
|---|---|---|
| Source integrity | 41 files verified | Hashes checked against HF metadata |
| DiT | 1 passed, 14.19 s | Two official shards, strict 359 tensors, T2I/editing error 0 |
| VAE | 4 passed, 7.96 s | Strict 251 tensors; all 250 floating tensors equal the raw reference |
| Main text | 5 passed, 30.77 s | Four prompts of 80/80/240/4096 tokens exact; duplicate and mapped-alias collisions rejected |
| HF ID smoke | Passed offline | Native PUP → 256² PNG, four steps; `outputs/flux3/official_weights/smoke.log` |
| Public API smoke | 2 passed, 38.09 s | Typed configuration; 256² and 65×97 PNGs match returned frames exactly |
| PUP language/vision | 5 passed, 14.49 s | Prefill, cached decode, padding, positions and three image grids; `atol=rtol=0` |
| PUP aggregate | 6 passed, 33.34 s | Three CPU tests and three real-weight cases; duplicate keys rejected; logits, tokens and captions exact |
| Sampling, PUP policy and contracts | 157 passed, 4.18 s | 34 sampling, 68 policy and 55 loading/request checks; offline CPU execution |
| API and registry | 66 passed | Config/preset compatibility; `outputs/flux3/official_weights/api_regression.log` |
| Extended PUP | Seven cases passed, 129.10 s | All EOS; 1967 tokens, captions and 41 logit vectors exact; repeated-input isolation passes |
| 22-case pipeline | 22 passed, 223.52 s | 147 latent states / 125 steps, pixels, captions and PUP tokens exact; repeated requests and component release pass |

The official VAE BN counter is 0 versus 400000 in the raw reference; it is
unused during evaluation. All floating checkpoint values match. New PUP metrics
are in `outputs/flux3/official_weights/pup/metrics.json`; every selected logit
vector has maximum and mean absolute error 0. Pipeline results are in
`outputs/flux3/official_weights/pipeline/report.json`, with `atol=rtol=0`.

The final audit reran 175 tests with zero skips; logs and CPU/smoke JUnit reports
are under `outputs/flux3/final_audit/`. The earlier 22-case pipeline and extended
PUP matrices were not rerun for these loading guards. Deleted packaging artifacts,
converter bytecode and stale pytest entries are removed; numerical evidence is
retained. CPU contracts are collected by the existing CI unit lane; real-weight
parity remains local. Empty-cache Hub downloading has not been exercised. Smoke
shutdown still emits a CUDA IPC warning, with no surviving worker processes.

Use `FLUX3_PARITY_OUTPUT_DIR` or `FLUX3_PUP_OUTPUT_DIR` to keep new evidence
separate from the historical reports. For static checks:

```bash
PATH=/.uv/python_bin:$PATH pre-commit run --files <changed files>
```

Respect the configured pre-commit exclusions; this workspace has the required
Python 3.12 hook interpreter installed.

## Historical numerical evidence

These results predate removal of the custom checkpoint layout. They establish
the native numerical implementation's previous behavior, not successful loading
of the newly staged official component files. Those runs staged the raw main
checkpoint at `2a70d1707a1892c5eae29cd5df7ced320edfa174` and the same PUP revision
listed above. The extended pipeline/PUP matrices asserted `atol=rtol=0`;
earlier component tests used their documented tolerances and reported zero
observed error.

| Scope | Previous result | Coverage |
|---|---|---|
| DiT | 2 passed, bit-exact | Native construction and production loading; T2I/editing |
| VAE | 3 CUDA and 2 CPU cases passed | FP32 encode/decode and production loading |
| Main text encoder | 1 passed, bit-exact | 80/80/240/4096 tokens |
| PUP language/vision | 1 language and 4 vision tests passed | Prefill, padding, cached 1/2-token decode, image grids and rotary buffers |
| PUP aggregate | 3 GPU and 1 CPU tests passed | Logits, tokens, captions, bbox and padded generation |
| Sampling and PUP policy | 34 and 68 passed | Flow-map steps, image processing, official messages and processor outputs |
| Public contracts and existing API | 40 and 22 passed | Integer seed, unsupported inputs, request cleanup and API compatibility |
| Pipeline smoke | 2 passed | Native PUP, default seed and 97×65 auto-size crop |
| Extended pipeline | 22 cases passed | 125 denoising iterations, 147 latent states, captions, seeds, tokens and pixels exact |
| Extended PUP | 7 cases passed | 1967 generated tokens and 41 selected logit vectors exact; all reached EOS |
| Full example | Byte-identical PNG | Native PUP, 1024×1024, 16 steps, seed 7 |

The pipeline matrix covers text, multilingual/long context, portrait/landscape,
512²/1024², one-step sampling, non-default alpha, 3/10 references, RGBA/EXIF,
layout boxes, real-photo edits, repeated requests and component release. The PUP
matrix includes a 4860-token context, multilingual prompts, three mixed-aspect
photos and locked/unlocked boxes. All seven generations reached EOS in 229–315
tokens within a 512-token cap. Its 41 logit vectors contain 10,181,120 values;
the maximum absolute difference was zero.

PUP's independent reference is the official model-card Transformers path plus
BFL request policy, greedy generation, generated-token presence penalty (T2I
1.5; I2I 0) and stop tokens. These checks do not claim vLLM equivalence.

Historical artifacts:

- `outputs/flux3/boundary_review/summary.json`, `pipeline/report.json` and `pup/metrics.json`.
- `outputs/flux3/validation_summary.json`, `extended_parity/report.json`, `photo_parity/report.json`.
- `outputs/flux3/extended_pup/{metrics,summary}.json`.
- `outputs/flux3/e2e_native/image.png` and `outputs/flux3/e2e_reference/official.png`.
- `outputs/flux3/e2e_native/comparison.json`: PNG SHA-256 `c8e782b688f57525af68a286883d7dc3093dac641a915bbbab663e40d57059dd`.
- `outputs/flux3/architecture/`: architecture PNG, editable SVG and rendering script.

No model weights or generated reference artifacts have been published.

# FLUX 3 Image Klein Port Status

## Summary

- model_family: `flux3`
- workload_types: `T2I`, `I2I`, multi-reference editing, bounding-box layout
- official_ref: https://github.com/black-forest-labs/flux-3-image-klein
- official_ref_dir: `flux-3-image-klein/`, commit `bbfe317dd81c825c30d76090a67d3d0db51da3cc`
- hf_weights_path: `black-forest-labs/flux-3-image-klein`
- local_weights_dir: `official_weights/flux3/`
- source_layout: official component subdirectories and `modular_model_index.json`; raw BFL files retained for reference tests
- local_tests_readme: `tests/local_tests/flux3/README.md`

## Current Phase

- phase: `handoff`
- status: `complete`
- owner: `orchestrator`
- last_updated: `2026-10-16` (UTC)

The user requested direct official HF loading and removal of the custom
converter/checkpoints. Native FastVideo numerical models are retained. Official
files remain unchanged; key mapping happens in memory. Earlier numerical
results are historical until the corresponding official-file test is rerun.

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| DiT | dit | port | `flux3/dit.py:DiT` | Distilled flow map | `Flux3Transformer2DModel` | 359 native keys | none; map official shards in memory | direct official loader: 1 passed; T2I/editing bit-exact | none |
| Text encoder | encoder | port | `flux3/text_encoder.py:TextEncoder` | Qwen3-VL-4B, eight hidden-state taps | `Flux3TextEncoder` | 398 language keys | none; select official language tensors | direct official loading: 5 passed; 80/80/240/4096 tokens bit-exact; duplicate keys rejected | none |
| VAE | vae | port | `flux3/autoencoder.py:AutoEncoder` | FP32, 32 channels, 2×2 packing | `Flux3AutoEncoder` | 251 native keys | none; map official VAE in memory | direct official loading: 4 passed | none |
| Sampler and preprocessing | generic | port | `flux3/sampling.py`, `pipeline.py`, `bbox.py` | 16 steps, alpha 6.93 | Family-local sampling/image/packing helpers and PUP bbox | pass | not applicable | current 34 sampling and 68 PUP policy tests pass | none |
| Prompt upsampler | encoder | port | `flux3/pup.py:Upsampler`; model-card Transformers path | Qwen3.5-9B language and vision | `Flux3PromptUpsampler` | 760 native keys | none; original PUP checkpoint | direct loading: 6 passed; 5 language/vision tests exact; seven extended generations reached EOS with exact tokens/captions/logits | none |

## Conversion State

- conversion_script: none; removed at user request
- converted_weights_dir: none; custom checkpoint directories removed
- source_layout: original official HF component files, main revision `337337621b8526e120f2275d250242f4a6bedd75`
- strict_load_status: official DiT, VAE, main text and PUP passed; extended PUP and 22-case pipeline passed
- passthrough_components: original files remain unchanged; native arch configs own in-memory key mapping
- retry_history: previous converted workflow retired; 41 selected official files verified against HF metadata

PUP is resolved from the separate repository specified by the official manifest
through the shared HF cache. There is no sibling-directory fallback. Its recorded
reference revision is `3f3edb52b2c55de0d49feab28598b68f7f8e4830`; the manifest's
`revision: null` follows the default branch. `official_weights/flux3/pup` is the
independent component/reference test input, not a production path override.

## Parity Commands

Full commands and environment overrides are in the README.

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| Official source files | Hash selected downloads against HF metadata | 41 verified | Main and PUP source files, with pinned revisions |
| DiT official loading | `pytest tests/local_tests/transformers/test_flux3_parity.py` | 1 passed, 14.19 s | Two official shards; strict 359 tensors; zero error for T2I/editing |
| VAE official loading | `pytest tests/local_tests/vaes/test_flux3_vae_parity.py` | 4 passed, 7.96 s | Strict 251 tensors; 250 floating tensors match raw BFL weights |
| Main text official loading | `pytest tests/local_tests/encoders/test_flux3_qwen3_vl_parity.py` | 5 passed, 30.77 s | Four prompts bit-exact; duplicate keys and mapped-alias collisions rejected |
| PUP official loading | `pytest tests/local_tests/encoders/test_flux3_pup_parity.py -k "not extended"` | 6 passed, 33.34 s | Three CPU tests and three real-weight cases; duplicate keys rejected; logits, tokens and captions exact |
| PUP language/vision | Text and vision parity files | 5 passed, 14.49 s | Prefill, cached decode, padding, positions and three image grids; zero tolerance |
| PUP extended generation | `FLUX3_PUP_EXTENDED=1 pytest tests/local_tests/encoders/test_flux3_pup_parity.py -k extended_generation_matrix` | 1 passed, 129.10 s | Seven complete generations; all EOS; 1967 tokens, captions and 41 logit vectors exact; repeated-input isolation passes |
| HF ID smoke | Public example with official HF ID and offline cache | pass | Native PUP through 256² PNG, four steps; `outputs/flux3/official_weights/smoke.log` |
| Public API smoke | `pytest tests/local_tests/pipelines/test_flux3_pipeline_smoke.py` | 2 passed, 38.09 s | Typed configuration; 256² and 65×97 PNG pixels equal returned frames |
| Sampling, PUP policy and contracts | Three CPU suites listed in README | 157 passed, 4.18 s | 34 sampling, 68 policy and 55 loading/request checks, including VAE override rejection |
| API and registry | Config/preset regression suites | 66 passed | `outputs/flux3/official_weights/api_regression.log` |
| Pipeline official loading | `pytest tests/local_tests/pipelines/test_flux3_pipeline_parity.py` | 22 passed, 223.52 s | 147 latent states / 125 steps, pixels, captions and PUP tokens exact; repeated requests and component release pass |
| Earlier full example | Public example, native PUP, 1024², 16 steps, seed 7 | historical pass | PNG byte-identical to independent official baseline; old checkpoint layout |

The official VAE BN counter is 0; the raw reference counter is 400000. It is
unused during evaluation. All floating checkpoint values match.

The final audit reran 175 tests with zero skips. Evidence is under
`outputs/flux3/final_audit/`; the previous 22-case pipeline and extended PUP
results remain in `outputs/flux3/official_weights/`. Obsolete packaging output,
converter bytecode and deleted-test cache entries were removed. Empty-cache Hub
download remains untested. Public smoke passes but logs a CUDA IPC warning at
shutdown; no worker remains running.

## Open Questions

| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Which checkpoint is required initially? | prep | prep | resolved | Distilled BF16 plus shared encoder/VAE and PUP. |
| Q002 | Can the existing HF login access the gated checkpoint? | prep | prep | resolved | Authenticated config request succeeded; no credential values are recorded. |
| Q003 | Can official files load directly? | pipeline | loading | resolved | Use published official component files and modular manifest with native in-memory mapping; remove custom conversion. Verification is tracked in I012. |
| Q004 | PUP integration boundary | orchestrator | pipeline | resolved | User requires native end-to-end PUP; language, vision and generation are ported. |

## Issues And Blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | prep | environment | medium | Official dependencies were missing. | Initial inventory | prep | resolved | Installed missing imports without replacing Torch/Triton. |
| I002 | prep | PUP reference | medium | vLLM entrypoint requires a different core stack. | Upstream Torch 2.11.0 versus current 2.8.0 | prep | resolved | Use the official model-card Transformers reference and BFL policy; vLLM is not installed. |
| I003 | implementation | text encoder | medium | 4096-token rotary initialization drift. | Four inv_freq values differed by up to 9.31e-10. | encoder | resolved | Initialize frequencies on CPU before moving; real-weight cases are exact. |
| I004 | review | PUP processor | medium | Older supported processors omit multimodal token types by default. | Transformers 5.0.0 default and regression test | encoder | resolved | Request token types explicitly; 68 policy tests passed. |
| I005 | review | historical conversion | medium | Same-schema source replacement could reuse stale output. | Tiny checkpoint replacement test | conversion | resolved | Source/output hashes and tensor-value reuse checks fixed it; the converter is now removed. |
| I006 | review | registry | medium | Earlier raw-only checkout was advertised as runnable without loader support. | No model_index.json in the source repo | pipeline | resolved | Initially removed unsupported advertising; now official modular-manifest support replaces custom layout. New runtime validation is tracked in I012. |
| I007 | review | boundaries | medium | Shared batch exposed model-private tensors and Packed types. | Consumer audit | pipeline | resolved | Family-local helpers and request state; cleanup on success/failure before transport. |
| I008 | review | request schema | medium | Nullable FLUX3 seed violated the existing integer API. | Normalizer/SamplingParam reproduction | pipeline | resolved | Removed preset override and shared nullable changes; default remains 1024. |
| I009 | review | request validation | medium | Unsupported inputs could be silently ignored. | Stage consumer audit | pipeline | resolved | Reject precomputed embeddings, custom sigmas, negative prompts and unsupported length overrides before execution. |
| I010 | review | output metadata | medium | Metadata-only requests retain requested dimensions after dropping pixels. | Auto-size crop can differ from request dimensions | runtime | out_of_scope | Reverted incidental shared worker/generator changes. Saved/returned images retain actual geometry. |
| I011 | review | historical conversion | medium | Converter duplicated native config mappings. | PARAM_NAME_MAP audit | conversion | resolved | Used arch-config mappings and passed 11 tests; the converter is now removed. |
| I012 | loading | official checkpoint path | medium | New official loading path needs complete numerical revalidation. | All components, seven extended PUP cases, HF ID smoke and 22 pipeline cases pass | orchestrator | resolved | Direct official loading preserves exact latents, pixels, captions and tokens; 55 offline loading/request contracts and 66 API checks pass. |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|
| E001 | pipeline | component boundary | External PUP preprocessing or native port? | Native PUP | resolved | User explicitly requires native end-to-end PUP. |

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-10-15 | Retain shared Torch/Triton stack. | Existing reference imports and GPU execution work. | Verify numerics in the shared environment. |
| 2026-10-15 | Initially stage raw BF16 distilled weights and PUP. | BFL reference consumes raw files. | Retain these files for independent parity. |
| 2026-10-15 | Use the official model-card PUP reference during prep. | Avoid replacing Torch for vLLM. | Reference equivalence claims are scoped to Transformers plus BFL policy. |
| 2026-10-16 | Run native PUP and generator in one request with deferred loading. | Required end-to-end behavior on one 32 GiB GPU. | Sequential component lifetime. |
| 2026-10-16 | Compare decoded pixels against the pinned official implementation. | Exact image, latent and component checks isolate numerical drift. | Local regression without remote artifact publication. |
| 2026-10-16 | Load official HF files directly and delete custom conversion. | Explicit user request. | Retain native model code, use modular manifest and in-memory mapping; rerun loading/parity gates. |

## Handoff Notes

- Branch: `feat/kaiqin/flux3-image-klein`.
- Pipeline stages: input preparation → native PUP → text encoding → reference/noise preparation → flow-map denoising → decode/crop.
- Native modules: transformer, VAE, text encoder and PUP; original processor assets remain in their official locations.
- Example default: `black-forest-labs/flux-3-image-klein`; local official snapshot also supported.
- Main tests use `FLUX3_MODEL_DIR`; PUP component/reference tests use `FLUX3_PUP_WEIGHTS_DIR`.
- Current evidence is under `outputs/flux3/official_weights/` and `outputs/flux3/final_audit/`; the former's `pipeline/report.json` records all 22 cases at zero tolerance and `pup/metrics.json` records seven complete generations.
- Historical evidence includes 22 pipeline cases, 147 exact latent states and seven complete PUP generations (1967 tokens and 41 exact logit vectors). See README and `outputs/flux3/boundary_review/`.
- Historical full image: `outputs/flux3/e2e_native/image.png`; PNG SHA-256 `c8e782b688f57525af68a286883d7dc3093dac641a915bbbab663e40d57059dd`.
- Source clone and checkpoint files remain local and outside the port changes. No model or reference publication was requested or performed.
- Unsupported scope remains base/quantized checkpoints, training, TP/SP and multi-image batches. Public seed and shared metadata behavior remain unchanged.

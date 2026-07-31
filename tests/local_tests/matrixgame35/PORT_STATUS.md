# Matrix-Game 3.5 Port Status

## Summary

- model_family: `matrixgame35`
- workload_types: camera-conditioned I2V; bidirectional base and causal few-step variants
- official_ref: `https://github.com/Riemann-Dynamics/Matrix-Game-3.5`
- official_ref_dir: `Matrix-Game-3.5/`
- hf_weights_path: `RiemannDynamics/Matrix-Game-3.5-Base`, `RiemannDynamics/Matrix-Game-3.5-Distilled`, shared Wan2.2 TI2V 5B and DA3
- local_weights_dir: `official_weights/matrixgame35/` planned on Shifu
- source_layout: `raw_official`
- local_tests_readme: `tests/local_tests/matrixgame35/README.md`

## Current Phase

- phase: `Phase 4 shared DiT / conversion; real-weight parity pending`
- status: `in_progress`
- owner: `orchestrator`
- last_updated: `2026-07-31`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| transformer + parameter-free PRoPE | dit | port dedicated subclass over Wan2.2 | `diffsynth/models/wan_video_dit.py::WanModel,DiTBlock,SelfAttention`; `diffsynth/models/prope_attention.py` | `model_configs.py` Wan2.2 TI2V 5B args; `_set_use_prope(... interval=1, layout=full)` | Matrix-scoped DiT/config with no shared-Wan behavior change | cpu_contract_pass | converter_complete | local_direct_official_pass; real_weight_pending | none |
| subject-reference memory tokens | generic/model params | port optional branch | `diffsynth/pipelines/wan_video.py::_build_subject_ref_memory_tokens` | base checkpoints: 2 first-person slots / 4 third-person slots; distilled: disabled | Matrix-Game DiT + conditioning stage | not_started | not_started | not_started | Q002 |
| Wan2.2 VAE 3.8 | vae | reuse_pending | `diffsynth/models/wan_video_vae.py::WanVideoVAE38` | `Wan2.2_VAE.pth` | `fastvideo/models/vaes/wanvae.py` | not_started | passthrough_pending | not_started | none |
| UMT5 text encoder | encoder | reuse_pending | `diffsynth/models/wan_video_text_encoder.py` | Wan2.2 `models_t5_umt5-xxl-enc-bf16.pth` + `google/umt5-xxl` tokenizer | `fastvideo/models/encoders/t5.py` | not_started | passthrough_pending | not_started | none |
| base flow scheduler | scheduler | reuse_pending | `diffsynth/diffusion/flow_match.py` | 25 steps, shift 5, CFG 5 | existing FastVideo flow scheduler | not_started | metadata_pending | not_started | none |
| distilled causal schedule/KV cache | generic | port/reuse_pending | `diffsynth/inference/causal_schedule.py`; `causal_rollout.py` | `[1000,667,333]`, chunk 3, window 21, CFG 3 | causal Wan stages/cache abstractions where parity permits | schedule_complete; cache_pending | not_required | schedule_cpu_pass; cache_pending | none |
| camera preparation | generic | port/reuse_pending | `diffsynth/pipelines/wan_video.py`; `infer.py::load_camera` | c2w matrices + pixel intrinsics; w2c accepted by inversion | model-specific conditioning stage | complete | not_required | cpu_pass | none |
| DA3 metric depth | encoder/preprocessor | lazy external reuse_pending | vendored `third_party/depth-anything-3`; `depth_anything_3.api.DepthAnything3` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | optional lazy preprocessor adapter | not_started | passthrough_pending | not_started | Q001, I004 |
| Patch Memory / frustum reprojection | generic | port minimal inference subset | `frustum/`; `examples/wanvideo/pipeline/mosaic/` | projection-IoU/pose selection and visibility-aware z-buffer fusion | model-specific memory stage/helpers | not_started | not_required | not_started | none |
| pipeline variants | pipeline | port | `infer.py`; `infer_distilled.py`; four YAML configs | base first, base third no-ref/ref, distilled first; 84 generated RGB frames/block | `fastvideo/pipelines/basic/matrixgame35/` | gated | gated | not_started | I002 |

## Conversion State

- conversion_script: `scripts/checkpoint_conversion/matrixgame35_to_diffusers.py`
- converted_weights_dir: `converted_weights/matrixgame35`
- source_layout: `raw_official`
- strict_load_status: `synthetic_cpu_pass; real_weights_pending_shifu`
- passthrough_components: pinned Wan2.2 VAE, UMT5/tokenizer, scheduler metadata, DA3 external weights
- retry_history: `none`

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| transformer | `pytest tests/local_tests/matrixgame35/test_matrixgame35_transformer_parity.py -v -s` | `2 passed, 2 skipped` | skips are real-weight/CUDA gates; run non-skip on Shifu |
| shared components | `pytest tests/local_tests/matrixgame35 -q -p no:cacheprovider` | `37 passed, 2 skipped` | measured before subject/layout helpers; skips are real-weight/CUDA gates |
| base first pipeline | `pytest tests/local_tests/pipelines/test_matrixgame35_base_first_pipeline_parity.py -v -s` | not_created | one block then multi-block |
| base third pipeline | `pytest tests/local_tests/pipelines/test_matrixgame35_base_third_pipeline_parity.py -v -s` | not_created | cover no-ref and refs |
| distilled pipeline | `pytest tests/local_tests/pipelines/test_matrixgame35_distilled_pipeline_parity.py -v -s` | not_created | causal three-step and KV eviction |

## Open Questions

| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Can DA3 remain a lazy optional preprocessing dependency without changing FastVideo core pins? | orchestrator | Phase 3 | open | recommended: lazy `depth-anything-3==0.1.1` adapter with exact parity |
| Q002 | Why does the public base first-person checkpoint retain two subject-reference slots although its CLI exposes no refs? | upstream audit | Phase 1 | open | preserve tensors; do not expose unsupported first-person refs without code evidence |
| Q003 | What does “all variants” mean for the current public release? | orchestrator | Phase 0 | resolved | exactly base first-person, base third-person, and distilled first-person at the pinned revisions |
| Q004 | Should FastVideo follow upstream's documented 80 frames/block or its executable 84 frames/block? | upstream audit | Phase 0 | resolved | preserve executable/sample contract: `1 + 84 * num_blocks`; record upstream text as stale |

## Issues And Blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | prep | official imports | medium | official umbrella import eagerly requires unused optional packages | `PYTHONPATH=Matrix-Game-3.5 python -c 'from diffsynth.models.wan_video_dit import WanModel'` -> missing `modelscope` | parity | resolved | narrow pinned test importer executes the real transformer dependency graph without optional umbrella imports |
| I002 | prep | Shifu | medium | live queue capacity not yet queryable because SSH requires the Tailscale web check | queue profile identified; active list blocked before authentication | orchestrator | open | authenticate, then re-query before any job submission |
| I003 | prep | weights | low | no local 30 GB checkpoint staging and no local CUDA | public headers/digests inspected without full download | orchestrator | open | stage/cache on queue-managed Shifu jobs |
| I004 | scope | DA3 | medium | no native FastVideo DA3 component exists | current-tree search found no DA3/depth preprocessor | component:depth | open | validate smallest lazy adapter boundary; avoid vendoring the DA3 stack |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|
| none | - | - | - | - | - | - |

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-07-31 | Pin GitHub at `fa6d2b62`, base HF at `c3b0c9c5`, and distilled HF at `0b38ca0b` | reproducible source/checkpoint contract | all parity and conversion work |
| 2026-07-31 | Use one shared Matrix-Game DiT implementation and one converter | header comparison shows identical key/shape surface except subject-reference tensors | prevents three copied models |
| 2026-07-31 | Activation order is shared code -> base first -> base third -> distilled -> registry | keeps intermediate steps dead-code or independently runnable | clean stacked review and rollback |
| 2026-07-31 | Do not port upstream training/validation framework | only public inference checkpoints are in scope | limits diff to native inference contracts |
| 2026-07-31 | Keep large weights and GPU truth on Shifu | local macOS skips are not passes and assets exceed 30 GB before dependencies | remote terminal receipts required |
| 2026-07-31 | Pin shared Wan2.2 at `921dbaf3` and DA3 at `b2359bdf` | upstream leaves both aliases mutable | reproducible conversion and Shifu jobs |
| 2026-07-31 | Generate 84 new RGB frames per block | pinned code consumes 21 noisy latents and the official six-block sample is 505 frames; upstream's 80-frame text is stale | frame-count assertions and camera consumption |
| 2026-07-31 | Keep initial execution single-GPU (`sp_size=1`) | official release is one GPU and camera/token sharding is not parity-proven | no implicit SP support claim |

## Handoff Notes

- Shared DiT, PRoPE, camera, schedule, converter, and direct-official CPU contracts are complete; real-weight parity is pending Shifu authentication/staging.
- Shared dependency revisions are pinned; capture local file hashes when Shifu stages them.
- Treat the current `fastvideo/platforms/cuda.py` GPU-UUID incompatibility as a separate cross-cutting infra change if the Matrix-Game Shifu path exercises it.
- Do not activate registry entries until all required component and pipeline parity checks are non-skip passes.

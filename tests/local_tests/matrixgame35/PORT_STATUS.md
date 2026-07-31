# Matrix-Game 3.5 Port Status

## Summary

- model_family: `matrixgame35`
- workload_types: camera-conditioned I2V; bidirectional base and causal few-step variants
- official_ref: `https://github.com/Riemann-Dynamics/Matrix-Game-3.5`
- official_ref_dir: `Matrix-Game-3.5/`
- hf_weights_path: `RiemannDynamics/Matrix-Game-3.5-Base`, `RiemannDynamics/Matrix-Game-3.5-Distilled`, shared Wan2.2 TI2V 5B and DA3
- local_weights_dir: queue-managed Shifu inputs; Base, Wan2.2 Diffusers, and DA3 staged; Distilled pending
- source_layout: `raw_official`
- local_tests_readme: `tests/local_tests/matrixgame35/README.md`

## Current Phase

- phase: `Phase 6 component and pipeline parity; public activation held`
- status: `in_progress`
- owner: `orchestrator`
- last_updated: `2026-07-31`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| transformer + parameter-free PRoPE | dit | port dedicated subclass over Wan2.2 | `diffsynth/models/wan_video_dit.py::WanModel,DiTBlock,SelfAttention`; `diffsynth/models/prope_attention.py` | `model_configs.py` Wan2.2 TI2V 5B args; `_set_use_prope(... interval=1, layout=full)` | Matrix-scoped DiT/config with no shared-Wan behavior change | cpu_contract_pass | converter_complete | local_direct_official_pass including Base physical hole drop/scatter; real_weight_pending | none |
| subject-reference memory tokens | generic/model params | port optional branch | `diffsynth/pipelines/wan_video.py::_build_subject_ref_memory_tokens` | base checkpoints: 2 first-person slots / 4 third-person slots; distilled: disabled | Matrix-Game DiT + conditioning stage | cpu_contract_pass | checkpoint_tensors_mapped | local_direct_official_pass | Q002 |
| Wan2.2 VAE 3.8 | vae | reuse_pending | `diffsynth/models/wan_video_vae.py::WanVideoVAE38`; Diffusers `AutoencoderKLWan` | raw `Wan2.2_VAE.pth` or pinned Diffusers VAE | `fastvideo/models/vaes/wanvae.py`; Matrix helper reuses LucyEdit/DreamX 48-channel config | config_cpu_pass | passthrough_pending | Diffusers real-weight gates ready for Shifu; raw scaffold retained | none |
| UMT5 text encoder | encoder | reuse_pending | `diffsynth/models/wan_video_text_encoder.py`; Transformers `UMT5EncoderModel` | raw Wan2.2 UMT5/tokenizer or pinned Diffusers snapshot | `fastvideo/models/encoders/t5.py`; Matrix helper reuses DreamX config with exact tokenizer deltas | config_cpu_pass | passthrough_pending | Diffusers tokenizer/real-weight gates ready for Shifu; raw scaffold retained | none |
| base flow scheduler | scheduler | port small exact helper | `diffsynth/diffusion/flow_match.py` | 25 steps, shift 5, CFG 5 | Matrix-scoped schedule over the same flow equation | complete | not_required | direct_pinned_cpu_pass | none |
| distilled causal schedule/KV cache | generic | port | `distilled_config.py`; `diffsynth/inference/causal_schedule.py`; `causal_memory.py`; `causal_rollout.py`; `diffsynth/pipelines/wan_video.py::model_fn_causal_kv` | `[1000,667,333]`, chunk 3, window 21; standard/sink CFG 3; HiAR CFG 1 | One Matrix-scoped causal path with `standard`, `hiar-sde`, and `sink-anchor-context` policy selection | schedule_complete; causal_dit_complete; profiles_complete | not_required | direct_pinned_cpu_pass; focused_pipeline_pass; real_weight_pending_shifu | upstream public HiAR config retains generic CFG 3 but executable rollout rejects it; FastVideo follows rollout CFG 1 |
| camera preparation | generic | port/reuse_pending | `diffsynth/pipelines/wan_video.py`; `infer.py::load_camera` | c2w matrices + pixel intrinsics; w2c accepted by inversion | model-specific conditioning stage | complete | not_required | cpu_pass | none |
| DA3 metric depth | encoder/preprocessor | lazy external reuse_pending | vendored `third_party/depth-anything-3`; `depth_anything_3.api.DepthAnything3` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1`; Base process resolution 504, Distilled 448 | lazy Matrix-scoped adapters without dependency-pin changes | adapter_complete | external_pinned_asset | local source/config pass; real CUDA standalone gate ready | I004 |
| Patch Memory / frustum reprojection | generic | port minimal inference subset | `frustum/`; `examples/wanvideo/pipeline/mosaic/` | projection-IoU/pose selection and visibility-aware z-buffer fusion | shared helpers with Base and Distilled policies | complete | not_required | direct_pinned_cpu_pass | none |
| pipeline variants | pipeline | port | `infer.py`; `infer_distilled.py`; four YAML configs | base first, base third no-ref/ref, distilled first; 84 generated RGB frames/block | `fastvideo/pipelines/basic/matrixgame35/` | local_fake_complete | transformer_converter_complete | 45 focused pipeline passes; real-weight and end-to-end Shifu pending | I003 |

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
| noncausal integrated model | `pytest tests/local_tests/matrixgame35/test_matrixgame35_noncausal_model_fn_parity.py -v -s` | `2 passed` | real pinned `model_fn_wan_video`; physical hole drop/scatter with and without subject prefix, disabled/no-hole regression, arbitrary RoPE; `max_abs_diff=7.75e-07` |
| causal integrated model | `pytest tests/local_tests/matrixgame35/test_matrixgame35_causal_model_fn_parity.py -v -s` | `1 passed` | real pinned `model_fn_causal_kv`; bootstrap/read-only denoise/final write/later re-anchored read, chunk IDs, and mosaic hole; `max_abs_diff=1.67e-06` |
| distilled profiles | `MATRIXGAME35_OFFICIAL_REF_DIR=/path/to/pinned/source pytest tests/local_tests/matrixgame35/test_matrixgame35_distilled_profile_parity.py -q` | `5 passed` | direct pinned mapping, HiAR corruption/seeds/latent provenance, and sink C0 selection |
| distilled focused pipeline | `pytest tests/local_tests/pipelines/test_matrixgame35_distilled_standard_pipeline.py -q` | `14 passed` | all three profiles; fake component path only, not real-weight parity |
| Wan2.2 VAE | `pytest tests/local_tests/matrixgame35/test_matrixgame35_vae_parity.py -v -s` | `1 passed, 3 skipped` | local CPU: config pass; CUDA raw, Diffusers component, and 704x1280 tiled gates skip without CUDA/assets |
| UMT5/tokenizer | `pytest tests/local_tests/matrixgame35/test_matrixgame35_text_encoder_parity.py -v -s` | `5 passed, 4 skipped` | local CPU: source contracts pass; raw and Diffusers tokenizer/real-weight gates skip without their assets/CUDA |
| shared components | `pytest tests/local_tests/matrixgame35 -q -rs` | `132 passed, 15 skipped` | measured on local CPU; all skips are CUDA/assets gates and are not passes |
| base first pipeline | `pytest tests/local_tests/pipelines/test_matrixgame35_base_first_person_pipeline.py -q` | included in `45 passed` pipeline suite | fake components; one- and two-block control flow, CFG/no-CFG, memory, offload, and exact inputs |
| base third pipeline | `pytest tests/local_tests/pipelines/test_matrixgame35_base_third_person_pipeline.py -q` | included in `45 passed` pipeline suite | fake components; no-ref/direct-latent and 1-4 image-reference paths |
| distilled pipeline | `pytest tests/local_tests/pipelines/test_matrixgame35_distilled_standard_pipeline.py -q` | `14 passed` | fake components; all three profiles, cache eviction, memory publication, and two-section prompt switching |
| combined local | `pytest tests/local_tests/matrixgame35 tests/local_tests/pipelines/test_matrixgame35_*_pipeline.py fastvideo/tests/api/test_parser.py fastvideo/tests/api/test_compat_translation.py -q -rs` | `201 passed, 15 skipped` | Matrix contracts plus API; no skipped path counted as verified |

## Open Questions

| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Can DA3 remain a lazy optional preprocessing dependency without changing FastVideo core pins? | orchestrator | Phase 3 | resolved | yes; exact adapter imports DA3 lazily and a separate pinned Shifu interpreter runs the standalone real-CUDA gate |
| Q002 | Why does the public base first-person checkpoint retain two subject-reference slots although its CLI exposes no refs? | upstream audit | Phase 1 | open | preserve tensors; do not expose unsupported first-person refs without code evidence |
| Q003 | What does “all variants” mean for the current public release? | orchestrator | Phase 0 | resolved | exactly base first-person, base third-person, and distilled first-person at the pinned revisions |
| Q004 | Should FastVideo follow upstream's documented 80 frames/block or its executable 84 frames/block? | upstream audit | Phase 0 | resolved | preserve executable/sample contract: `1 + 84 * num_blocks`; record upstream text as stale |

## Issues And Blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | prep | official imports | medium | official umbrella import eagerly requires unused optional packages | `PYTHONPATH=Matrix-Game-3.5 python -c 'from diffsynth.models.wan_video_dit import WanModel'` -> missing `modelscope` | parity | resolved | narrow pinned test importer executes the real transformer dependency graph without optional umbrella imports |
| I002 | prep | Shifu | medium | queue authentication/capacity initially unavailable | discovery job `job-20260731T201020Z-8d6ec0ca39` reached terminal success and identified immutable inputs | orchestrator | resolved | formal Base, shared Wan, and DA3 bundles are ready for the stable source snapshot |
| I003 | parity | Distilled weights | high | no immutable Distilled shared input exists and public queue tooling cannot legally materialize an inference-model input | expected content key `385c312df55b0e004c9f18d3ea21ac05662a1e841079328eff9dfdb3690578b2`; only formal training materialization is exposed | orchestrator/operator | open | add or use an operator-supported model-input materialization path; do not fake a training bundle or download 10 GB per run |
| I004 | parity | DA3 environment | medium | FastVideo interpreter lacks DA3 dependencies; DA3 interpreter cannot import the full FastVideo package because it lacks unrelated `cloudpickle` | exact adapter file and pinned vendored DA3 API import successfully in the DA3 interpreter | component:depth | mitigated | standalone gate records exact adapter/source/interpreter provenance; it does not claim package-level FastVideo import |

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
| 2026-07-31 | Represent all distilled profiles as policies on one pipeline | checkpoints, stage order, persistent clean cache-fill, mosaic memory, and output contract are shared; only prefix corruption or context selection changes | no copied profile pipelines; HiAR rebuilds ephemeral per-step KV while sink reuses C0 after context-free chunk 0 |
| 2026-07-31 | Reuse LucyEdit/DreamX shared Wan2.2 VAE and UMT5 configs | Matrix-Game instantiates the same 48-channel VAE and UMT5-XXL weights; only fixed padding, whitespace cleaning, gated GELU, and constructor dropout differ from the existing DreamX helper surface | avoids duplicated 48-value latent statistics and keeps Matrix-specific tokenization explicit |
| 2026-07-31 | Physically drop Base mosaic holes before transformer blocks and scatter zeros after the output head | both released Base YAMLs set `mosaic_drop_holes: true`; mask-only execution preserves values but not the released sequence length or VRAM contract | hidden states, RoPE, timestep, cross-attention mask, and PRoPE carriers use the same kept-token indices; distilled remains unchanged |

## Handoff Notes

- Shared DiT, arbitrary-position RoPE, PRoPE, physical Base hole dropping, subject-reference memory, Base first/third rollouts, all three Distilled profiles, camera, Patch Memory, schedule, converter, VAE/text boundaries, and direct-official CPU contracts are implemented.
- Base first/third checkpoints, pinned Wan2.2 Diffusers components, and pinned DA3 are immutable Shifu inputs. Distilled remains blocked on legal shared-input materialization.
- Formal Base transformer, shared Wan VAE/UMT5/tokenizer, and standalone DA3 bundles are ready. Require terminal success plus inspected JUnit, metrics, provenance, and receipts before changing any `pending` status to `pass`.
- Treat the current `fastvideo/platforms/cuda.py` GPU-UUID incompatibility as a separate cross-cutting infra change if the Matrix-Game Shifu path exercises it.
- Do not activate registry entries until all required component and pipeline parity checks are non-skip passes.

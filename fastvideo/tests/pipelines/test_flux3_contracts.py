# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for FLUX.3 requests, configuration, and reference handling."""

from dataclasses import fields
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.flux3 import Flux3PipelineConfig
from fastvideo.pipelines.basic.flux3.flux3_pipeline import Flux3Pipeline
from fastvideo.pipelines.basic.flux3.presets import FLUX3_KLEIN
from fastvideo.pipelines.basic.flux3 import stages
from fastvideo.pipelines.basic.flux3.state import RUNTIME_STATE_KEY, Flux3RuntimeState, get_state
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.models.vaes.flux3 import latent_ids


OFFICIAL_MODEL_ID = "black-forest-labs/flux-3-image-klein"
PUP_MODEL_ID = "black-forest-labs/flux-3-image-klein-pup"


def write_official_manifest(path):
    path.mkdir(exist_ok=True)
    manifest = {"_class_name": "Flux3DistilledModularPipeline", "_diffusers_version": "0.41.0.dev0"}
    for name, library, architecture, subfolder in (
        ("transformer", "diffusers", "Flux3Transformer2DModel", "transformer"),
        ("vae", "diffusers", "AutoencoderKLFlux2", "vae"),
        ("text_encoder", "transformers", "Qwen3VLForConditionalGeneration", "text_encoder"),
        ("tokenizer", "transformers", "Qwen3VLProcessor", "text_encoder"),
        ("prompt_upsampler", "transformers", "Qwen3_5ForConditionalGeneration", ""),
        ("prompt_upsampler_processor", "transformers", "AutoProcessor", ""),
    ):
        manifest[name] = [library, architecture, {
            "pretrained_model_name_or_path": PUP_MODEL_ID if name.startswith("prompt_upsampler") else OFFICIAL_MODEL_ID,
            "revision": None, "subfolder": subfolder,
        }]
    (path / "modular_model_index.json").write_text(json.dumps(manifest))
    return manifest


def runtime_args():
    return SimpleNamespace(num_gpus=1, tp_size=1, sp_size=1, training_mode=False,
                           pipeline_config=Flux3PipelineConfig(), override_text_encoder_quant=None,
                           lazy_module_load=False, dit_layerwise_offload=True, use_fsdp_inference=True,
                           text_encoder_cpu_offload=True, vae_cpu_offload=True, output_type="pil", model_paths={})


def test_public_request_fields_survive_batch_construction():
    image = Image.new("RGB", (65, 97), "red")
    request = {
        "references": [image],
        "flux3_boxes": [{"id": "red_box", "bbox": [0.1, 0.2, 0.8, 0.9], "desc": "a red box"}],
        "flux3_alpha": 5.5,
        "flux3_auto_size": True,
        "flux3_upsample_prompt": False,
        "flux3_pup_max_new_tokens": 128,
    }
    sampling = SamplingParam(num_frames=1)
    sampling.update(request)
    batch = ForwardBatch(**{item.name: getattr(sampling, item.name) for item in fields(sampling)})
    assert batch.data_type == "image"
    assert batch.references[0] is image
    for name, value in request.items():
        assert getattr(batch, name) == value


def test_registered_sampling_defaults_match_distilled_contract(tmp_path):
    from fastvideo.registry import get_pipeline_config_cls_from_name, get_preset_selection

    model_path = tmp_path / "flux3"
    write_official_manifest(model_path)
    model_id = str(model_path)
    assert get_pipeline_config_cls_from_name(model_id) is Flux3PipelineConfig
    assert get_preset_selection(model_id) == (FLUX3_KLEIN.name, "flux3")
    sampling = SamplingParam.from_pretrained(model_id)
    assert sampling.num_frames == 1
    assert (sampling.height, sampling.width) == (1024, 1024)
    assert sampling.num_inference_steps == 16
    assert sampling.guidance_scale == 1.0
    assert sampling.flux3_alpha == 6.93
    assert sampling.flux3_upsample_prompt is True
    assert sampling.flux3_pup_max_new_tokens == 3584
    assert sampling.negative_prompt == ""
    assert sampling.seed == 1024


@pytest.mark.parametrize("source,seed", [("typed_default", 1024), ("legacy_sampling_param", 1024),
                                        ("typed_explicit", 0), ("legacy_keyword", 7)])
def test_flux3_uses_existing_integer_seed_api(tmp_path, source, seed):
    from fastvideo.api.compat import legacy_generate_call_to_request, normalize_generation_request, request_to_sampling_param

    model_path = tmp_path / "official_model"
    write_official_manifest(model_path)
    model_path = str(model_path)
    if source == "typed_default":
        request = normalize_generation_request({"prompt": "A red bicycle."})
    elif source == "typed_explicit":
        request = normalize_generation_request({"prompt": "A red bicycle.", "sampling": {"seed": seed}})
    elif source == "legacy_sampling_param":
        request = legacy_generate_call_to_request("A red bicycle.", SamplingParam.from_pretrained(model_path))
    else:
        request = legacy_generate_call_to_request("A red bicycle.", None, legacy_kwargs={"seed": seed})
    assert request_to_sampling_param(request, model_path=model_path).seed == seed


def test_official_checkpoint_is_advertised_and_dispatches_native_pipeline(tmp_path, monkeypatch):
    from fastvideo import registry
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.registry import get_model_info, get_registered_models_with_workloads

    manifest = write_official_manifest(tmp_path / "official_model")
    monkeypatch.setattr(registry, "maybe_download_model_index", lambda *args, **kwargs: manifest)
    models = get_registered_models_with_workloads()
    assert any(model["id"] == OFFICIAL_MODEL_ID for model in models)
    for workload in (WorkloadType.T2I, WorkloadType.I2I):
        info = get_model_info(OFFICIAL_MODEL_ID, workload_type=workload)
        assert info.pipeline_cls is Flux3Pipeline
        assert info.pipeline_config_cls is Flux3PipelineConfig


def test_official_modular_manifest_resolves_without_flux3_directory_name(tmp_path):
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.registry import get_model_info, get_pipeline_config_cls_from_name, get_preset_selection

    model_path = tmp_path / "official_image_model"
    write_official_manifest(model_path)
    assert get_pipeline_config_cls_from_name(str(model_path)) is Flux3PipelineConfig
    assert get_preset_selection(str(model_path)) == (FLUX3_KLEIN.name, "flux3")
    assert get_model_info(str(model_path), workload_type=WorkloadType.T2I).pipeline_cls is Flux3Pipeline


def test_config_instances_do_not_share_mutable_component_state():
    first, second = Flux3PipelineConfig(), Flux3PipelineConfig()
    first.dit_config.arch_config.flow_map = False
    first.vae_config.load_encoder = False
    first.text_encoder_configs[0].arch_config.output_layers = (4,)
    first.text_encoder_configs[1].arch_config.text_config.num_hidden_layers = 1
    assert second.dit_config.flow_map is True
    assert second.vae_config.load_encoder is True
    assert second.text_encoder_configs[0].output_layers == (4, 8, 12, 16, 20, 24, 28, 32)
    assert second.text_encoder_configs[1].arch_config.text_config.num_hidden_layers > 1
    assert second.dit_precision == "bf16" and second.vae_precision == "fp32"
    assert second.text_encoder_precisions == ("bf16", "bf16")


@pytest.mark.parametrize("field,value", [("num_gpus", 2), ("tp_size", 2), ("sp_size", 2)])
def test_parallel_modes_rejected_before_loading(monkeypatch, field, value):
    args = runtime_args()
    setattr(args, field, value)
    monkeypatch.setattr(ComposedPipelineBase, "load_modules", lambda *args, **kwargs: pytest.fail("loaded weights"))
    with pytest.raises(ValueError, match="one GPU"):
        Flux3Pipeline.load_modules(object.__new__(Flux3Pipeline), args)


@pytest.mark.parametrize("component", ["transformer", "text_encoder", "prompt_upsampler", "override"])
def test_quantization_rejected_before_loading(monkeypatch, component):
    args = runtime_args()
    if component == "transformer":
        args.pipeline_config.dit_config.quant_config = object()
    elif component == "override":
        args.override_text_encoder_quant = "fp8"
    else:
        index = 0 if component == "text_encoder" else 1
        args.pipeline_config.text_encoder_configs[index].quant_config = object()
    monkeypatch.setattr(ComposedPipelineBase, "load_modules", lambda *args, **kwargs: pytest.fail("loaded weights"))
    with pytest.raises(ValueError, match="unquantized"):
        Flux3Pipeline.load_modules(object.__new__(Flux3Pipeline), args)


@pytest.mark.parametrize("field,value", [("dit_precision", "fp32"), ("vae_precision", "bf16"),
                                        ("vae_decode_precision", "bf16"), ("vae_decode_precision", "fp16"),
                                        ("text_encoder_precisions", ("bf16", "fp32"))])
def test_wrong_component_precision_rejected_before_loading(monkeypatch, field, value):
    args = runtime_args()
    setattr(args.pipeline_config, field, value)
    monkeypatch.setattr(ComposedPipelineBase, "load_modules", lambda *args, **kwargs: pytest.fail("loaded weights"))
    with pytest.raises(ValueError, match="BF16.*FP32"):
        Flux3Pipeline.load_modules(object.__new__(Flux3Pipeline), args)


@pytest.mark.parametrize("mode", ["training", "base_checkpoint", "vae_tiling"])
def test_unsupported_execution_rejected_before_loading(monkeypatch, mode):
    args = runtime_args()
    if mode == "training":
        args.training_mode = True
    elif mode == "vae_tiling":
        args.pipeline_config.vae_tiling = True
    else:
        args.pipeline_config.dit_config.arch_config.flow_map = False
    monkeypatch.setattr(ComposedPipelineBase, "load_modules", lambda *args, **kwargs: pytest.fail("loaded weights"))
    with pytest.raises(ValueError, match="inference only|distilled flow-map|VAE tiling"):
        Flux3Pipeline.load_modules(object.__new__(Flux3Pipeline), args)


def test_external_text_encoder_override_rejected_before_loading():
    args = runtime_args()
    args.override_text_encoder_safetensors = "/unused/alternate.safetensors"
    with pytest.raises(ValueError, match="official component repositories"):
        Flux3Pipeline.load_modules(object.__new__(Flux3Pipeline), args)


def prepare_official_pipeline(tmp_path):
    path = tmp_path / "official_image_model"
    manifest = write_official_manifest(path)
    for name in ("transformer", "vae", "text_encoder"):
        (path / name).mkdir()
    (path / "transformer/config.json").write_text(json.dumps({"flow_map": True}))
    args = runtime_args()
    pipeline = object.__new__(Flux3Pipeline)
    pipeline.model_path = str(path)
    pipeline.fastvideo_args = args
    return pipeline, args, manifest


@pytest.mark.parametrize("invalid", ["foreign_family", "base_transformer"])
def test_foreign_checkpoint_rejected_before_component_loading(tmp_path, monkeypatch, invalid):
    from fastvideo.pipelines.basic.flux3 import flux3_pipeline as module

    pipeline, args, manifest = prepare_official_pipeline(tmp_path)
    if invalid == "foreign_family":
        manifest["_class_name"] = "Flux2Pipeline"
        (Path(pipeline.model_path) / "modular_model_index.json").write_text(json.dumps(manifest))
    else:
        (Path(pipeline.model_path) / "transformer/config.json").write_text(json.dumps({"flow_map": False}))
    monkeypatch.setattr(module, "maybe_download_model", lambda *args, **kwargs: pytest.fail("resolved PUP"))
    with pytest.raises(ValueError, match="official distilled|distilled flow-map"):
        pipeline.load_modules(args)


@pytest.mark.parametrize("vae_decode_precision", [None, "fp32"])
def test_official_component_paths_and_native_encoder_dispatch(tmp_path, monkeypatch, vae_decode_precision):
    from fastvideo.models.registry import ModelRegistry
    from fastvideo.pipelines.basic.flux3 import flux3_pipeline as module
    from fastvideo.pipelines.lazy_module import is_lazy_module

    pipeline, args, manifest = prepare_official_pipeline(tmp_path)
    args.pipeline_config.vae_decode_precision = vae_decode_precision
    main = Path(pipeline.model_path)
    pup_repo = tmp_path / "upstream_pup"
    pup = pup_repo / "component"
    pup.mkdir(parents=True)
    for name in ("prompt_upsampler", "prompt_upsampler_processor"):
        manifest[name][2].update(revision="pinned-pup-revision", subfolder="component")
    (main / "modular_model_index.json").write_text(json.dumps(manifest))
    (main / "text_encoder/config.json").write_text(json.dumps({
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "text_config": {"hidden_size": 64, "rope_theta": 7000000,
                        "rope_scaling": {"mrope_section": [8, 8, 16], "rope_type": "default"}},
    }))
    (pup / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"hidden_size": 128}, "vision_config": {"out_hidden_size": 128},
    }))
    downloads, loaded, encoders = [], [], []

    def resolve_pup(repo, revision):
        downloads.append((repo, revision))
        return str(pup_repo)

    def load_generic(name, path, library, fastvideo_args):
        loaded.append((name, path, library))
        return object()

    def load_encoder(self, path, config, device, fastvideo_args, dtype):
        cls, _ = ModelRegistry.resolve_model_cls(config.architectures)
        encoders.append((path, cls, config, device, dtype))
        return object()

    def load_vae(cls, path, **kwargs):
        loaded.append(("vae", path, kwargs))
        return object()

    monkeypatch.setattr(module, "maybe_download_model", resolve_pup)
    monkeypatch.setattr(module, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module, "record_resolved_attention_backend", lambda config: None)
    monkeypatch.setattr(module.PipelineComponentLoader, "load_module", load_generic)
    monkeypatch.setattr(module.TextEncoderLoader, "load_model", load_encoder)
    monkeypatch.setattr(module.ProcessorLoader, "load",
                        lambda self, path, args: load_generic("prompt_upsampler_processor", path, "transformers", args))
    monkeypatch.setattr(module.Flux3AutoEncoder, "from_pretrained", classmethod(load_vae))
    modules = pipeline.load_modules(args)

    assert downloads == [(PUP_MODEL_ID, "pinned-pup-revision")]
    assert loaded == [("tokenizer", str(main / "text_encoder"), "transformers"),
                      ("prompt_upsampler_processor", str(pup), "transformers")]
    assert set(modules) == {"transformer", "vae", "text_encoder", "tokenizer",
                            "prompt_upsampler", "prompt_upsampler_processor"}
    for name in ("transformer", "vae", "text_encoder", "prompt_upsampler"):
        assert is_lazy_module(modules[name]) and not modules[name].is_materialized
        modules[name].materialize()
    assert loaded[2] == ("transformer", str(main / "transformer"), "diffusers")
    assert loaded[3][0:2] == ("vae", str(main / "vae"))
    assert loaded[3][2]["dtype"] is torch.float32
    assert loaded[3][2]["device"] == "cpu"
    assert [(path, cls.__name__, dtype) for path, cls, _, _, dtype in encoders] == [
        (str(main / "text_encoder"), "Flux3TextEncoder", "bf16"),
        (str(pup), "Flux3PromptUpsampler", "bf16"),
    ]
    assert all(cls.__module__.startswith("fastvideo.models.encoders.") for _, cls, _, _, _ in encoders)
    assert encoders[0][2].hidden_size == 64
    assert encoders[0][2].rope_theta == 7000000
    assert encoders[0][2].mrope_section == (8, 8, 16)
    assert encoders[1][2].text_config.hidden_size == encoders[1][2].vision_config.out_hidden_size == 128
    assert args.pipeline_config.text_encoder_configs[0].hidden_size == 2560
    assert args.pipeline_config.text_encoder_configs[1].text_config.hidden_size == 4096
    assert args.lazy_module_load and not args.dit_layerwise_offload and not args.use_fsdp_inference


class UnusedUpsampler:
    def to(self, *args, **kwargs):
        pytest.fail("A bypassed prompt must not load the upsampler")


class ShapeVAE:
    def to(self, *args, **kwargs):
        return self

    def encode(self, image):
        height, width = image.shape[-2] // 16, image.shape[-1] // 16
        return torch.zeros(height * width, 128), latent_ids(height, width)

    def decode(self, tokens, ids):
        height, width = int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1
        assert tokens.shape == (height * width, 128)
        return torch.zeros(3, height * 16, width * 16)


def request_batch(**overrides):
    values = {"data_type": "image", "prompt": "A red balloon.", "num_frames": 1,
              "height": 64, "width": 96, "seed": None, "flux3_upsample_prompt": False}
    values.update(overrides)
    return ForwardBatch(**values)


def prepare_without_weights(batch, args):
    batch.extra[RUNTIME_STATE_KEY] = Flux3RuntimeState()
    stages.Flux3InputPreparationStage().forward(batch, args)
    stages.Flux3PromptUpsamplingStage(UnusedUpsampler(), None, {}).forward(batch, args)
    batch.prompt_embeds = [torch.zeros(2, 20480, dtype=torch.bfloat16)]
    get_state(batch).text_ids = torch.zeros(2, 4, dtype=torch.long)
    return batch


@pytest.mark.parametrize("auto_size,expected_size,expected_canvas", [
    (True, (97, 65), (112, 80)),
    (False, (64, 96), (64, 96)),
])
def test_single_reference_auto_size_crop_is_explicit(monkeypatch, auto_size, expected_size, expected_canvas):
    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))
    args = runtime_args()
    image = Image.new("RGB", (65, 97), "red")
    batch = prepare_without_weights(request_batch(references=[image], flux3_auto_size=auto_size), args)
    vae = ShapeVAE()
    stages.Flux3LatentPreparationStage(vae).forward(batch, args)
    assert (batch.height, batch.width) == expected_canvas
    assert get_state(batch).image_ids.dtype == torch.int64
    assert get_state(batch).image_ids[:, 1].max() == expected_canvas[0] // 16 - 1
    assert get_state(batch).image_ids[:, 2].max() == expected_canvas[1] // 16 - 1
    assert batch.latents.dtype == torch.float32
    stages.Flux3DecodingStage(vae).forward(batch, args)
    assert batch.output.shape == (1, 3, 1, *expected_size)
    assert image.size == (65, 97)


def test_reference_paths_and_pil_inputs_share_auto_size_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))
    args = runtime_args()
    path = tmp_path / "reference.png"
    Image.new("RGB", (65, 97), "blue").save(path)
    with Image.open(path) as source:
        image = source.copy()
    batches = []
    for inputs in ({"image_path": str(path)}, {"pil_image": image}, {"references": [path]}):
        batch = prepare_without_weights(request_batch(flux3_auto_size=True, **inputs), args)
        stages.Flux3LatentPreparationStage(ShapeVAE()).forward(batch, args)
        batches.append(batch)
    assert {(batch.width, batch.height, get_state(batch).crop) for batch in batches} == {(80, 112, (7, 7, 72, 104))}
    for batch in batches[1:]:
        torch.testing.assert_close(batch.latents, batches[0].latents, atol=0, rtol=0)
        torch.testing.assert_close(get_state(batch).packed.ids["image_cond"], get_state(batches[0]).packed.ids["image_cond"],
                                   atol=0, rtol=0)


def test_reused_stages_do_not_keep_previous_reference_or_crop(monkeypatch):
    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))
    args = runtime_args()
    stage = stages.Flux3LatentPreparationStage(ShapeVAE())
    first = prepare_without_weights(request_batch(references=[Image.new("RGB", (65, 97))], flux3_auto_size=True), args)
    stage.forward(first, args)
    assert get_state(first).crop is not None
    assert len(get_state(first).packed.data["image_cond"]) > 0
    second = prepare_without_weights(request_batch(seed=7), args)
    stage.forward(second, args)
    fresh = prepare_without_weights(request_batch(seed=7), args)
    stages.Flux3LatentPreparationStage(ShapeVAE()).forward(fresh, args)
    assert get_state(second).crop is None
    assert len(get_state(second).packed.data["image_cond"]) == 0
    assert (second.height, second.width) == (64, 96)
    torch.testing.assert_close(second.latents, fresh.latents, atol=0, rtol=0)


def test_bbox_caption_and_hash_seed_do_not_leak_between_requests():
    args = runtime_args()
    stage = stages.Flux3PromptUpsamplingStage(UnusedUpsampler(), None, {})
    box = {"id": "balloon", "bbox": [0.1, 0.2, 0.4, 0.5], "desc": "red balloon"}
    first = request_batch(flux3_boxes=[box], references=[])
    stage.forward(first, args)
    caption, encoded = first.prompt.split(" [", 1)
    assert caption == "A red balloon."
    assert json.loads("[" + encoded) == [{"id": "balloon", "bbox": [100, 200, 400, 500], "desc": "red balloon"}]
    assert first.seed == int(hashlib.sha256(first.prompt.encode()).hexdigest(), 16) % 2**32
    second = request_batch(prompt="A blue boat.", references=[])
    stage.forward(second, args)
    assert second.prompt == "A blue boat."
    assert second.seed == int(hashlib.sha256(b"A blue boat.").hexdigest(), 16) % 2**32


def test_finished_bbox_caption_bypasses_prompt_upsampling():
    prompt = 'A red <balloon>. [{"id": "balloon", "bbox": [100, 200, 400, 500], "desc": "red balloon"}]'
    batch = request_batch(prompt=prompt, references=[], flux3_upsample_prompt=True, seed=17)
    stages.Flux3PromptUpsamplingStage(UnusedUpsampler(), None, {}).forward(batch, runtime_args())
    assert batch.prompt == prompt
    assert batch.seed == 17


@pytest.mark.parametrize("fails", [False, True])
def test_pipeline_cleans_private_state_on_success_and_failure(monkeypatch, fails):
    pipeline = object.__new__(Flux3Pipeline)
    pipeline.post_init_called = True
    pipeline._stage_name_mapping = {}
    released = []
    monkeypatch.setattr(pipeline, "_release_all_lazy_modules", lambda: released.append(True))
    seen = []

    def run_stage(batch, args):
        state = get_state(batch)
        assert state.packed is None and state.crop is None
        assert all(state is not previous for previous in seen)
        seen.append(state)
        state.text_ids = torch.zeros(2, 4)
        state.crop = (0, 0, 16, 16)
        if fails:
            raise ValueError("stage failed")
        batch.output = torch.zeros(1, 3, 1, 16, 16)
        return batch

    pipeline._stages = [run_stage]
    for _ in range(2):
        batch = request_batch()
        batch.extra["user_metadata"] = "keep"
        if fails:
            with pytest.raises(ValueError, match="stage failed"):
                pipeline.forward(batch, runtime_args())
        else:
            assert pipeline.forward(batch, runtime_args()) is batch
            assert batch.output.shape == (1, 3, 1, 16, 16)
        assert batch.extra == {"user_metadata": "keep"}
    assert len(released) == (2 if fails else 0)


def test_pipeline_cleans_private_state_when_initialization_fails(monkeypatch):
    pipeline = object.__new__(Flux3Pipeline)
    pipeline.post_init_called = False

    def fail_init():
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(pipeline, "post_init", fail_init)
    batch = request_batch()
    with pytest.raises(RuntimeError, match="initialization failed"):
        pipeline.forward(batch, runtime_args())
    assert batch.extra == {}


@pytest.mark.parametrize("overrides,error", [
    ({"references": [Image.new("RGB", (64, 64))] * 11}, "at most 10"),
    ({"references": [], "pil_image": Image.new("RGB", (64, 64))}, "only one"),
    ({"num_frames": 2}, "one image"),
    ({"num_videos_per_prompt": 2}, "one image"),
    ({"num_inference_steps": 0}, "num_inference_steps must be a positive integer"),
    ({"flux3_alpha": 0}, "flux3_alpha must be positive and finite"),
    ({"flux3_alpha": float("nan")}, "flux3_alpha must be positive and finite"),
    ({"flux3_pup_max_new_tokens": 0}, "flux3_pup_max_new_tokens must be a positive integer"),
    ({"return_trajectory_latents": True}, "trajectory output"),
    ({"return_trajectory_decoded": True}, "trajectory output"),
    ({"guidance_scale": 3.5}, "without CFG"),
    ({"width": 65}, "multiples of 16"),
    ({"flux3_boxes": [{"bbox": [0.1, 0.2, 0.3]}]}, "a box is"),
    ({"prompt_embeds": [torch.zeros(2, 20480)]}, "precomputed prompt_embeds"),
    ({"sigmas": [1.0, 0.5, 0.0]}, "custom sigmas"),
    ({"negative_prompt": "blurred"}, "negative_prompt"),
    ({"max_sequence_length": 512}, "fixed max_sequence_length"),
])
def test_invalid_requests_fail_before_model_execution(overrides, error):
    with pytest.raises(ValueError, match=error):
        stages.Flux3InputPreparationStage().forward(request_batch(**overrides), runtime_args())

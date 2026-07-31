# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.pipelines.matrixgame35 import MatrixGame35BaseFirstPersonPipelineConfig
from fastvideo.pipelines.basic.matrixgame35 import base_first_person_stages
from fastvideo.pipelines.basic.matrixgame35.base_first_person_pipeline import (
    MatrixGame35BaseFirstPersonPipeline,
)
from fastvideo.pipelines.basic.matrixgame35.base_first_person_stages import (
    MatrixGame35BaseInputValidationStage,
    MatrixGame35BaseRolloutStage,
    matrixgame35_base_block_seed,
)
from fastvideo.pipelines.basic.matrixgame35.prompts import (
    MATRIXGAME35_NEGATIVE_PROMPT,
    MatrixGame35TextEncodingStage,
)
from fastvideo.pipelines.basic.matrixgame35.schedule import (
    base_flow_step,
    build_base_schedule,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from tests.local_tests.matrixgame35._upstream import PINNED_OFFICIAL_REVISION


_REPO_ROOT = Path(__file__).resolve().parents[3]
_OFFICIAL_DIR = _REPO_ROOT / "Matrix-Game-3.5"


@pytest.fixture(autouse=True)
def _use_cpu_as_local_execution_device(monkeypatch) -> None:
    monkeypatch.setattr(base_first_person_stages, "get_local_torch_device", lambda: torch.device("cpu"))


class _Posterior:

    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def mode(self) -> torch.Tensor:
        return self._value


class _FakeVAE(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(latents_mean=(0.0, 0.0), latents_std=(1.0, 1.0))
        self.encode_batch_sizes: list[int] = []
        self.decode_latent_lengths: list[int] = []
        self.encode_dtypes: list[torch.dtype] = []
        self.decode_dtypes: list[torch.dtype] = []

    def encode(self, video: torch.Tensor) -> _Posterior:
        self.encode_batch_sizes.append(int(video.shape[0]))
        self.encode_dtypes.append(video.dtype)
        latents = F.avg_pool3d(video[:, :2], kernel_size=(1, 2, 2))
        return _Posterior(latents)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.decode_latent_lengths.append(int(latents.shape[2]))
        self.decode_dtypes.append(latents.dtype)
        spatial = F.interpolate(latents[:, :1].float(), scale_factor=(1, 2, 2), mode="nearest")
        if spatial.shape[2] == 1:
            temporal = spatial
        else:
            temporal = torch.cat((spatial[:, :, :1], spatial[:, :, 1:].repeat_interleave(4, dim=2)), dim=2)
        return torch.cat((temporal, temporal * 0.5, -temporal), dim=1).clamp(-1.0, 1.0)


class _FakeTransformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.in_channels = 2
        self.calls: list[dict] = []

    def forward(self, hidden_states, encoder_hidden_states, timestep, **kwargs):
        layout = kwargs["latent_layout"]
        camera_info = kwargs["camera_info"]
        self.calls.append({
            "layout_counts": (
                layout.first_frame_count,
                layout.mosaic_frame_count,
                layout.noisy_frame_count,
            ),
            "mosaic_indices": layout.mosaic_frame_indices.detach().cpu(),
            "drop_mosaic_holes": layout.drop_mosaic_holes,
            "camera_shape": tuple(camera_info[0].shape),
            "clean_w2c": camera_info[0][:, 0].detach().cpu().clone(),
            "text_sum": float(encoder_hidden_states.sum()),
            "timestep_shape": tuple(timestep.shape),
            "subject_ref_latents_id": id(kwargs["subject_ref_latents"])
            if kwargs.get("subject_ref_latents") is not None else None,
        })
        return torch.zeros_like(hidden_states)


class _TextValueTransformer(_FakeTransformer):

    def forward(self, hidden_states, encoder_hidden_states, timestep, **kwargs):
        super().forward(hidden_states, encoder_hidden_states, timestep, **kwargs)
        return torch.full_like(hidden_states, float(encoder_hidden_states.sum()))


class _RecordingFakeVAE(_FakeVAE):

    def __init__(self) -> None:
        super().__init__()
        self.to_calls: list[object] = []

    def to(self, *args, **kwargs):
        self.to_calls.append(args[0] if args else kwargs.get("device"))
        return super().to(*args, **kwargs)


class _RecordingFakeTransformer(_FakeTransformer):

    def __init__(self) -> None:
        super().__init__()
        self.to_calls: list[object] = []

    def to(self, *args, **kwargs):
        self.to_calls.append(args[0] if args else kwargs.get("device"))
        return super().to(*args, **kwargs)


class _FakeDepthAdapter:

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def estimate_depth(self, frames) -> np.ndarray:
        self.batch_sizes.append(len(frames))
        height, width = frames[0].shape[:2]
        return np.ones((len(frames), height, width), dtype=np.float32)


def _camera_file(tmp_path, frame_count: int, *, translation_step: float = 0.0) -> str:
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
    c2w[:, 0, 3] = np.arange(frame_count, dtype=np.float32) * translation_step
    intrinsic = np.array(((16.0, 0.0, 8.0), (0.0, 16.0, 8.0), (0.0, 0.0, 1.0)), dtype=np.float32)
    path = tmp_path / "camera.npz"
    np.savez(path, extrinsics_c2w=c2w, intrinsics=intrinsic)
    return str(path)


def _args(config: MatrixGame35BaseFirstPersonPipelineConfig):
    return SimpleNamespace(
        pipeline_config=config,
        disable_autocast=True,
        sp_size=1,
        vae_cpu_offload=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
    )


def test_base_schedule_and_seed_match_pinned_upstream_formula() -> None:
    actual = build_base_schedule()
    unshifted = torch.linspace(1.0, 0.0, 26, dtype=torch.float32)[:-1]
    expected_sigmas = 5.0 * unshifted / (1.0 + 4.0 * unshifted)

    torch.testing.assert_close(actual.sigmas, expected_sigmas, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.timesteps, expected_sigmas * 1000.0, rtol=0.0, atol=0.0)
    assert matrixgame35_base_block_seed(3407, batch_index=2, block_index=3) == 5410

    sample = torch.ones(1, dtype=torch.bfloat16)
    velocity = torch.full((1, ), 2.0, dtype=torch.bfloat16)
    stepped = base_flow_step(sample, velocity, actual.sigmas[0], actual.sigmas[1])
    expected = sample + velocity * (actual.sigmas[1].to(torch.bfloat16) - actual.sigmas[0].to(torch.bfloat16))
    torch.testing.assert_close(stepped, expected, rtol=0.0, atol=0.0)


def test_base_schedule_matches_pinned_upstream_implementation() -> None:
    source = _OFFICIAL_DIR / "diffsynth" / "diffusion" / "flow_match.py"
    if not source.is_file():
        pytest.skip(f"Pinned Matrix-Game 3.5 source is missing: {source}")
    revision = subprocess.run(
        ["git", "-C", str(_OFFICIAL_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == PINNED_OFFICIAL_REVISION
    spec = importlib.util.spec_from_file_location("_matrixgame35_base_schedule", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    expected_sigmas, expected_timesteps = module.FlowMatchScheduler.set_timesteps_wan(
        num_inference_steps=25,
        shift=5.0,
    )
    actual = build_base_schedule()
    torch.testing.assert_close(actual.sigmas, expected_sigmas, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.timesteps, expected_timesteps, rtol=0.0, atol=0.0)


def test_base_config_and_composition_are_first_person_only() -> None:
    config = MatrixGame35BaseFirstPersonPipelineConfig()
    assert config.dit_config.arch_config.subject_ref_memory_max_refs == 2
    assert config.dit_config.arch_config.causal is False
    assert (config.matrixgame35_height, config.matrixgame35_width) == (704, 1280)
    assert config.flow_shift == 5.0
    assert config.vae_tiling is False
    assert config.vae_precision == "bf16"
    assert config.matrixgame35_da3_model_ref == "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    assert MatrixGame35BaseFirstPersonPipeline._required_config_modules == [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
    ]

    pipe = object.__new__(MatrixGame35BaseFirstPersonPipeline)
    pipe.modules = {
        "text_encoder": object(),
        "tokenizer": object(),
        "vae": object(),
        "transformer": object(),
    }
    pipe._stages = []
    pipe._stage_name_mapping = {}
    config.matrixgame35_da3_model_ref = "/models/da3-pinned"
    pipe.initialize_pipeline(_args(config))
    assert pipe._depth_adapter._model_ref == "/models/da3-pinned"
    pipe._depth_adapter = _FakeDepthAdapter()
    pipe.create_pipeline_stages(_args(config))

    assert list(pipe._stage_name_mapping) == [
        "input_validation_stage",
        "prompt_encoding_stage",
        "conditioning_stage",
        "base_rollout_stage",
    ]
    assert isinstance(pipe._stage_name_mapping["prompt_encoding_stage"], MatrixGame35TextEncodingStage)


def test_input_validation_accepts_only_released_resolution_and_frame_formula(tmp_path) -> None:
    config = MatrixGame35BaseFirstPersonPipelineConfig()
    camera_path = _camera_file(tmp_path, 85)
    batch = ForwardBatch(
        data_type="video",
        prompt="move forward",
        negative_prompt="",
        pil_image=Image.new("RGB", (1280, 704)),
        camera_trajectory=camera_path,
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=25,
        guidance_scale=5.0,
        seed=3407,
    )

    output = MatrixGame35BaseInputValidationStage().forward(batch, _args(config))

    assert output.pil_image.shape == (1, 3, 1, 704, 1280)
    assert output.seeds == [3407]
    assert isinstance(output.generator, list) and output.generator[0].initial_seed() == 3407
    assert output.negative_prompt == MATRIXGAME35_NEGATIVE_PROMPT
    assert output.section_prompts == ["move forward"]

    batch.seed = None
    batch.pil_image = Image.new("RGB", (1280, 704))
    output = MatrixGame35BaseInputValidationStage().forward(batch, _args(config))
    assert output.seed == 3407
    assert output.seeds == [3407]
    assert output.generator[0].initial_seed() == 3407

    batch.num_frames = 84
    with pytest.raises(ValueError, match=r"1 \+ 84 \* num_blocks"):
        MatrixGame35BaseInputValidationStage().forward(batch, _args(config))

    batch.num_frames = 85
    batch.subject_ref_source = str(tmp_path)
    with pytest.raises(ValueError, match="Base first-person does not accept subject_ref_source"):
        MatrixGame35BaseInputValidationStage().forward(batch, _args(config))

    batch.subject_ref_source = None
    batch.subject_ref_latents = torch.zeros(1, 2, 1, 4, 4)
    with pytest.raises(ValueError, match="Base first-person does not accept direct subject_ref_latents"):
        MatrixGame35BaseInputValidationStage().forward(batch, _args(config))


def test_input_validation_accepts_official_sampling_overrides(tmp_path) -> None:
    config = MatrixGame35BaseFirstPersonPipelineConfig()
    batch = ForwardBatch(
        data_type="video",
        prompt="move forward",
        negative_prompt="",
        pil_image=Image.new("RGB", (1280, 704)),
        camera_trajectory=_camera_file(tmp_path, 85),
        camera_convention="w2c",
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=7,
        guidance_scale=0.5,
        seed=123,
    )

    output = MatrixGame35BaseInputValidationStage().forward(batch, _args(config))

    assert output.num_inference_steps == 7
    assert output.guidance_scale == 0.5
    assert output.seed == 123
    assert output.do_classifier_free_guidance is True

    batch.guidance_scale = 1.0
    batch.pil_image = Image.new("RGB", (1280, 704))
    output = MatrixGame35BaseInputValidationStage().forward(batch, _args(config))
    assert output.do_classifier_free_guidance is False

    batch.num_inference_steps = 0
    with pytest.raises(ValueError, match="num_inference_steps must be a positive integer"):
        MatrixGame35BaseInputValidationStage().forward(batch, _args(config))


def test_input_validation_materializes_official_caption_json(tmp_path) -> None:
    caption_path = tmp_path / "caption.json"
    caption_path.write_text(
        json.dumps({
            "detailed": {
                "0": {
                    "dynamic": "forward"
                },
                "85": {
                    "dynamic": "turn"
                },
            }
        }),
        encoding="utf-8",
    )
    batch = ForwardBatch(
        data_type="video",
        prompt=None,
        caption_path=str(caption_path),
        negative_prompt="",
        pil_image=Image.new("RGB", (1280, 704)),
        camera_trajectory=_camera_file(tmp_path, 169),
        height=704,
        width=1280,
        num_frames=169,
        num_inference_steps=25,
        guidance_scale=5.0,
        seed=3407,
    )

    output = MatrixGame35BaseInputValidationStage().forward(
        batch,
        _args(MatrixGame35BaseFirstPersonPipelineConfig()),
    )

    assert output.prompt is None
    assert output.caption_path == str(caption_path)
    assert output.section_prompts == ["forward", "turn"]


def test_input_validation_crops_before_resizing_anchor(tmp_path) -> None:
    config = MatrixGame35BaseFirstPersonPipelineConfig()
    pixels = np.zeros((1080, 1920, 3), dtype=np.uint8)
    pixels[:160, :, 1] = 255
    pixels[-160:, :, 1] = 255
    pixels[:, :250, 0] = 255
    pixels[:, -250:, 0] = 255
    image = Image.fromarray(pixels)
    batch = ForwardBatch(
        data_type="video",
        prompt="move forward",
        negative_prompt="",
        pil_image=image,
        camera_trajectory=_camera_file(tmp_path, 85),
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=25,
        guidance_scale=5.0,
        seed=3407,
    )

    output = MatrixGame35BaseInputValidationStage().forward(batch, _args(config))

    assert output.pil_image.shape == (1, 3, 1, 704, 1280)
    # The released dataset crops the oversized source before any resize, so
    # all four colored outer bands disappear. A cover resize would retain them.
    torch.testing.assert_close(output.pil_image, torch.full_like(output.pil_image, -1.0))


def test_anchor_preprocess_matches_released_small_image_resize() -> None:
    pixels = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
    batch = ForwardBatch(data_type="video", pil_image=Image.fromarray(pixels))

    base_first_person_stages.preprocess_matrixgame35_anchor(batch, height=16, width=32)

    source = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float()
    source.mul_(2.0 / 255.0).sub_(1.0)
    expected = F.interpolate(
        source,
        size=(16, 32),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).unsqueeze(2)
    torch.testing.assert_close(batch.pil_image, expected, rtol=0.0, atol=0.0)


def test_rollout_decodes_each_section_and_never_registers_the_final_block(tmp_path, monkeypatch) -> None:
    config = MatrixGame35BaseFirstPersonPipelineConfig()
    config.dit_precision = "fp32"
    config.vae_precision = "fp32"
    config.vae_decode_precision = "fp32"
    transformer = _FakeTransformer()
    vae = _FakeVAE()
    depth_adapter = _FakeDepthAdapter()
    stage = MatrixGame35BaseRolloutStage(transformer, vae, depth_adapter)

    load_camera_trajectory = base_first_person_stages.load_camera_trajectory
    observed_camera_args = {}

    def _capturing_load_camera_trajectory(path, *, convention, frame_count):
        observed_camera_args.update(convention=convention, frame_count=frame_count)
        return load_camera_trajectory(path, convention=convention, frame_count=frame_count)

    monkeypatch.setattr(base_first_person_stages, "load_camera_trajectory", _capturing_load_camera_trajectory)
    batch = ForwardBatch(
        data_type="video",
        pil_image=torch.zeros(1, 3, 1, 8, 8),
        camera_trajectory=_camera_file(tmp_path, 169),
        camera_convention="w2c",
        prompt_embeds=[torch.ones(2, 4, 4)],
        negative_prompt_embeds=[torch.zeros(1, 4, 4)],
        height=704,
        width=1280,
        num_frames=169,
        num_inference_steps=25,
        guidance_scale=5.0,
        seed=3407,
        subject_ref_latents=torch.ones(1, 2, 1, 4, 4),
    )

    output = stage.forward(batch, _args(config))

    assert output.output.shape == (1, 3, 169, 8, 8)
    assert output.output.device.type == "cpu"
    assert output.latents.shape == (1, 2, 43, 4, 4)
    assert observed_camera_args == {"convention": "w2c", "frame_count": 169}
    assert vae.decode_latent_lengths == [1, 22, 22, 43]
    assert vae.encode_batch_sizes == [1] * 86
    assert depth_adapter.batch_sizes == [1, 84]
    assert len(transformer.calls) == 2 * 25 * 2
    assert all(call["layout_counts"] == (1, 21, 21) for call in transformer.calls)
    assert all(call["drop_mosaic_holes"] is True for call in transformer.calls)
    assert all(call["camera_shape"] == (1, 43, 4, 4, 4) for call in transformer.calls)
    assert all(torch.equal(call["mosaic_indices"], torch.arange(21)) for call in transformer.calls)
    assert {call["subject_ref_latents_id"] for call in transformer.calls} == {id(batch.subject_ref_latents)}

    first_noise = torch.randn(
        (1, 2, 21, 4, 4),
        generator=torch.Generator("cpu").manual_seed(3407),
    )
    second_noise = torch.randn(
        (1, 2, 21, 4, 4),
        generator=torch.Generator("cpu").manual_seed(3408),
    )
    torch.testing.assert_close(output.latents[:, :, 1:22], first_noise, rtol=0.0, atol=0.0)
    torch.testing.assert_close(output.latents[:, :, 22:], second_noise, rtol=0.0, atol=0.0)


def test_camera_info_builds_inverses_before_target_device_transfer(monkeypatch) -> None:
    stage = MatrixGame35BaseRolloutStage(_FakeTransformer(), _FakeVAE(), _FakeDepthAdapter())
    observed_inverse_devices: list[str] = []
    original_inverse = torch.linalg.inv

    def _recording_inverse(value):
        observed_inverse_devices.append(value.device.type)
        return original_inverse(value)

    def _stub_viewmats(c2w, _intrinsics, **_kwargs):
        assert c2w.device.type == "meta"
        placeholder = torch.empty_like(c2w)
        return placeholder, placeholder, placeholder

    monkeypatch.setattr(torch.linalg, "inv", _recording_inverse)
    monkeypatch.setattr(base_first_person_stages, "build_prope_viewmats", _stub_viewmats)
    target_w2c = np.repeat(np.eye(4, dtype=np.float32)[None], 84, axis=0)
    target_intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 84, axis=0)

    full_w2c, _ = stage._build_camera_info(
        clean_w2c=np.eye(4, dtype=np.float32),
        clean_intrinsics=np.eye(3, dtype=np.float32),
        target_w2c=target_w2c,
        target_intrinsics=target_intrinsics,
        mosaic_indices=torch.arange(21),
        image_height=704,
        image_width=1280,
        device=torch.device("meta"),
        dtype=torch.float32,
    )

    assert full_w2c.device.type == "meta"
    assert observed_inverse_devices == ["cpu", "cpu"]


def test_two_block_rollout_uses_distinct_previous_rgb_cameras_for_clean_latent(tmp_path) -> None:
    config = MatrixGame35BaseFirstPersonPipelineConfig()
    config.dit_precision = "fp32"
    config.vae_precision = "fp32"
    config.vae_decode_precision = "fp32"
    transformer = _FakeTransformer()
    stage = MatrixGame35BaseRolloutStage(transformer, _FakeVAE(), _FakeDepthAdapter())
    batch = ForwardBatch(
        data_type="video",
        pil_image=torch.zeros(1, 3, 1, 8, 8),
        camera_trajectory=_camera_file(tmp_path, 169, translation_step=0.001),
        prompt_embeds=[torch.stack((torch.ones(4, 4), torch.full((4, 4), 2.0)))],
        negative_prompt_embeds=[],
        height=704,
        width=1280,
        num_frames=169,
        num_inference_steps=1,
        guidance_scale=1.0,
        seed=3407,
    )

    stage.forward(batch, _args(config))

    assert len(transformer.calls) == 2
    assert [call["text_sum"] for call in transformer.calls] == [16.0, 32.0]
    torch.testing.assert_close(
        transformer.calls[0]["clean_w2c"][0, :, 0, 3],
        torch.zeros(4),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        transformer.calls[1]["clean_w2c"][0, :, 0, 3],
        -torch.arange(81, 85, dtype=torch.float32) * 0.001,
        rtol=0.0,
        atol=1e-6,
    )


def test_guidance_scale_one_runs_only_positive_branch_and_uses_positive_output(tmp_path) -> None:
    config = MatrixGame35BaseFirstPersonPipelineConfig()
    config.dit_precision = "fp32"
    config.vae_precision = "fp32"
    config.vae_decode_precision = "fp32"
    transformer = _TextValueTransformer()
    stage = MatrixGame35BaseRolloutStage(transformer, _FakeVAE(), _FakeDepthAdapter())
    batch = ForwardBatch(
        data_type="video",
        pil_image=torch.zeros(1, 3, 1, 8, 8),
        camera_trajectory=_camera_file(tmp_path, 85),
        prompt_embeds=[torch.full((1, 1, 1), 2.0)],
        negative_prompt_embeds=[],
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=1,
        guidance_scale=1.0,
        seed=3407,
    )

    output = stage.forward(batch, _args(config))

    assert len(transformer.calls) == 1
    assert transformer.calls[0]["text_sum"] == 2.0
    expected_noise = torch.randn(
        (1, 2, 21, 4, 4),
        generator=torch.Generator("cpu").manual_seed(3407),
    )
    torch.testing.assert_close(output.latents[:, :, 1:], expected_noise - 2.0, rtol=0.0, atol=0.0)


def test_rollout_moves_modules_for_use_and_offloads_them_with_dtype_safe_inputs(tmp_path) -> None:
    config = MatrixGame35BaseFirstPersonPipelineConfig()
    config.dit_precision = "fp32"
    config.vae_precision = "fp32"
    config.vae_decode_precision = "bf16"
    transformer = _RecordingFakeTransformer()
    vae = _RecordingFakeVAE()
    stage = MatrixGame35BaseRolloutStage(transformer, vae, _FakeDepthAdapter())
    fastvideo_args = _args(config)
    fastvideo_args.vae_cpu_offload = True
    fastvideo_args.dit_cpu_offload = True
    batch = ForwardBatch(
        data_type="video",
        pil_image=torch.zeros(1, 3, 1, 8, 8),
        camera_trajectory=_camera_file(tmp_path, 85),
        prompt_embeds=[torch.ones(1, 4, 4)],
        negative_prompt_embeds=[],
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=1,
        guidance_scale=1.0,
        seed=3407,
    )

    stage.forward(batch, fastvideo_args)

    assert vae.to_calls[::2] == [torch.device("cpu")] * 5
    assert vae.to_calls[1::2] == ["cpu"] * 5
    assert transformer.to_calls == [torch.device("cpu"), "cpu"]
    assert vae.encode_dtypes == [torch.float32, torch.float32]
    assert vae.decode_dtypes == [torch.float32, torch.float32, torch.float32]


def test_rollout_fails_before_patch_memory_when_depth_adapter_is_missing(tmp_path) -> None:
    config = MatrixGame35BaseFirstPersonPipelineConfig()
    stage = MatrixGame35BaseRolloutStage(_FakeTransformer(), _FakeVAE(), None)
    batch = ForwardBatch(
        data_type="video",
        pil_image=torch.zeros(1, 3, 1, 8, 8),
        camera_trajectory=_camera_file(tmp_path, 85),
        prompt_embeds=[torch.ones(1, 4, 4)],
        negative_prompt_embeds=[torch.zeros(1, 4, 4)],
        num_frames=85,
    )

    with pytest.raises(RuntimeError, match="requires a configured depth adapter"):
        stage.forward(batch, _args(config))


def test_base_validation_rejects_negative_prompt_override(tmp_path) -> None:
    config = MatrixGame35BaseFirstPersonPipelineConfig()
    batch = ForwardBatch(
        data_type="video",
        prompt="move forward",
        negative_prompt="custom negative",
        pil_image=Image.new("RGB", (1280, 704)),
        camera_trajectory=_camera_file(tmp_path, 85),
        height=704,
        width=1280,
        num_frames=85,
        num_inference_steps=25,
        guidance_scale=5.0,
        seed=3407,
    )

    with pytest.raises(ValueError, match="released fixed negative prompt"):
        MatrixGame35BaseInputValidationStage().forward(batch, _args(config))

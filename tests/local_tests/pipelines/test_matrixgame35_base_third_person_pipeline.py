# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from fastvideo.configs.pipelines.matrixgame35 import (
    MatrixGame35BaseFirstPersonPipelineConfig,
    MatrixGame35BaseThirdPersonPipelineConfig,
)
from fastvideo.pipelines.basic.matrixgame35.base_first_person_pipeline import (
    MatrixGame35BaseFirstPersonPipeline,
)
from fastvideo.pipelines.basic.matrixgame35.base_third_person_pipeline import (
    MatrixGame35BaseThirdPersonPipeline,
)
from fastvideo.pipelines.basic.matrixgame35.base_third_person_stages import (
    MatrixGame35BaseThirdPersonInputValidationStage,
    MatrixGame35BaseThirdPersonSubjectReferenceStage,
)
from fastvideo.pipelines.basic.matrixgame35 import base_third_person_stages
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


class _Posterior:

    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def mode(self) -> torch.Tensor:
        return self._value


class _FakeVAE(nn.Module):

    def __init__(self, latent_shape: tuple[int, ...] = (1, 48, 1, 44, 80)) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(latents_mean=(0.0, ) * 48, latents_std=(1.0, ) * 48)
        self.latent_shape = latent_shape
        self.encode_shapes: list[tuple[int, ...]] = []
        self.to_calls: list[object] = []

    def to(self, *args, **kwargs):
        self.to_calls.append(args[0] if args else kwargs.get("device"))
        return super().to(*args, **kwargs)

    def encode(self, video: torch.Tensor) -> _Posterior:
        self.encode_shapes.append(tuple(video.shape))
        value = torch.full(self.latent_shape, float(len(self.encode_shapes)), device=video.device, dtype=video.dtype)
        return _Posterior(value)


def _args(config: MatrixGame35BaseThirdPersonPipelineConfig):
    return SimpleNamespace(
        pipeline_config=config,
        disable_autocast=True,
        sp_size=1,
        vae_cpu_offload=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
    )


@pytest.fixture(autouse=True)
def _force_cpu_runtime(monkeypatch):
    monkeypatch.setattr(base_third_person_stages, "get_local_torch_device", lambda: torch.device("cpu"))


def _canvases(count: int, *, channels: int = 3) -> torch.Tensor:
    return torch.zeros(count, channels, 1, 1).expand(count, channels, 704, 1280)


def test_third_person_config_and_pipeline_are_thin_base_extensions() -> None:
    config = MatrixGame35BaseThirdPersonPipelineConfig()
    assert isinstance(config, MatrixGame35BaseFirstPersonPipelineConfig)
    assert config.dit_config.arch_config.subject_ref_memory_max_refs == 4
    assert config.vae_precision == "bf16"
    assert issubclass(MatrixGame35BaseThirdPersonPipeline, MatrixGame35BaseFirstPersonPipeline)
    assert MatrixGame35BaseThirdPersonInputValidationStage.allow_subject_refs is True

    pipe = object.__new__(MatrixGame35BaseThirdPersonPipeline)
    pipe.modules = {
        "text_encoder": object(),
        "tokenizer": object(),
        "vae": object(),
        "transformer": object(),
    }
    pipe._stages = []
    pipe._stage_name_mapping = {}
    pipe._depth_adapter = object()
    pipe.create_pipeline_stages(_args(config))

    assert list(pipe._stage_name_mapping) == [
        "input_validation_stage",
        "prompt_encoding_stage",
        "conditioning_stage",
        "subject_reference_stage",
        "base_rollout_stage",
    ]
    assert type(pipe.input_validation_stage) is MatrixGame35BaseThirdPersonInputValidationStage
    assert type(pipe.subject_reference_stage) is MatrixGame35BaseThirdPersonSubjectReferenceStage
    assert sum(type(stage) is MatrixGame35BaseThirdPersonSubjectReferenceStage for stage in pipe._stages) == 1


def test_zero_subject_refs_skip_loading_and_vae(monkeypatch) -> None:
    from fastvideo.pipelines.basic.matrixgame35 import base_third_person_stages

    def _unexpected_load(*args, **kwargs):
        raise AssertionError("subject-reference loading must not run without a source")

    monkeypatch.setattr(base_third_person_stages, "load_subject_reference_canvases", _unexpected_load)
    vae = _FakeVAE()
    batch = ForwardBatch(data_type="video", subject_ref_source=None)

    output = MatrixGame35BaseThirdPersonSubjectReferenceStage(vae).forward(
        batch,
        _args(MatrixGame35BaseThirdPersonPipelineConfig()),
    )

    assert output.subject_ref_latents is None
    assert vae.encode_shapes == []


def test_direct_subject_ref_latents_are_preserved_and_cannot_mix_with_source(tmp_path) -> None:
    vae = _FakeVAE()
    stage = MatrixGame35BaseThirdPersonSubjectReferenceStage(vae)
    direct = torch.zeros(2, 48, 1, 44, 80)
    batch = ForwardBatch(data_type="video", subject_ref_latents=direct)

    output = stage.forward(batch, _args(MatrixGame35BaseThirdPersonPipelineConfig()))

    assert output.subject_ref_latents is direct
    assert vae.encode_shapes == []

    batch.subject_ref_source = str(tmp_path)
    with pytest.raises(ValueError, match="either subject_ref_source or subject_ref_latents"):
        stage.forward(batch, _args(MatrixGame35BaseThirdPersonPipelineConfig()))


@pytest.mark.parametrize(
    "latents",
    (
        torch.zeros(0, 48, 1, 44, 80),
        torch.zeros(5, 48, 1, 44, 80),
        torch.zeros(1, 47, 1, 44, 80),
        torch.zeros(1, 48, 2, 44, 80),
        torch.zeros(1, 48, 1, 43, 80),
    ),
)
def test_direct_subject_ref_latents_require_the_released_shape(latents) -> None:
    stage = MatrixGame35BaseThirdPersonSubjectReferenceStage(_FakeVAE())
    batch = ForwardBatch(data_type="video", subject_ref_latents=latents)

    with pytest.raises(ValueError, match="Direct subject_ref_latents must have shape"):
        stage.forward(batch, _args(MatrixGame35BaseThirdPersonPipelineConfig()))


@pytest.mark.parametrize("ref_count", [1, 2, 3, 4])
def test_each_subject_reference_is_loaded_and_vae_encoded_once(ref_count, monkeypatch, tmp_path) -> None:
    from fastvideo.pipelines.basic.matrixgame35 import base_third_person_stages

    observed = {}

    def _load(source, **kwargs):
        observed.update(source=source, **kwargs)
        return _canvases(ref_count)

    monkeypatch.setattr(base_third_person_stages, "load_subject_reference_canvases", _load)
    vae = _FakeVAE()
    batch = ForwardBatch(data_type="video", subject_ref_source=str(tmp_path))

    output = MatrixGame35BaseThirdPersonSubjectReferenceStage(vae).forward(
        batch,
        _args(MatrixGame35BaseThirdPersonPipelineConfig()),
    )

    assert observed == {
        "source": str(tmp_path),
        "height": 704,
        "width": 1280,
        "max_refs": 4,
    }
    assert vae.encode_shapes == [(1, 3, 1, 704, 1280)] * ref_count
    assert output.subject_ref_latents is not None
    assert output.subject_ref_latents.shape == (ref_count, 48, 1, 44, 80)
    for index in range(ref_count):
        torch.testing.assert_close(
            output.subject_ref_latents[index],
            torch.full_like(output.subject_ref_latents[index], float(index + 1)),
        )


def test_subject_references_share_one_vae_device_session_and_honor_cpu_offload(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(base_third_person_stages, "load_subject_reference_canvases", lambda *args, **kwargs: _canvases(2))
    config = MatrixGame35BaseThirdPersonPipelineConfig()
    config.vae_precision = "fp32"
    vae = _FakeVAE()
    fastvideo_args = _args(config)
    fastvideo_args.vae_cpu_offload = True

    MatrixGame35BaseThirdPersonSubjectReferenceStage(vae).forward(
        ForwardBatch(data_type="video", subject_ref_source=str(tmp_path)),
        fastvideo_args,
    )

    assert vae.to_calls == [torch.device("cpu"), "cpu"]
    assert vae.encode_shapes == [(1, 3, 1, 704, 1280)] * 2


def test_subject_reference_stage_rejects_more_than_four_refs(monkeypatch, tmp_path) -> None:
    from fastvideo.pipelines.basic.matrixgame35 import base_third_person_stages

    monkeypatch.setattr(base_third_person_stages, "load_subject_reference_canvases", lambda *args, **kwargs: _canvases(5))
    vae = _FakeVAE()
    batch = ForwardBatch(data_type="video", subject_ref_source=str(tmp_path))

    with pytest.raises(ValueError, match="at most 4 subject references"):
        MatrixGame35BaseThirdPersonSubjectReferenceStage(vae).forward(
            batch,
            _args(MatrixGame35BaseThirdPersonPipelineConfig()),
        )
    assert vae.encode_shapes == []


def test_subject_reference_stage_rejects_invalid_source_and_canvas(monkeypatch, tmp_path) -> None:
    config = MatrixGame35BaseThirdPersonPipelineConfig()
    stage = MatrixGame35BaseThirdPersonSubjectReferenceStage(_FakeVAE())
    missing = ForwardBatch(data_type="video", subject_ref_source=str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="subject_ref_source must be a directory"):
        stage.forward(missing, _args(config))

    from fastvideo.pipelines.basic.matrixgame35 import base_third_person_stages

    monkeypatch.setattr(
        base_third_person_stages,
        "load_subject_reference_canvases",
        lambda *args, **kwargs: _canvases(1, channels=4),
    )
    malformed = ForwardBatch(data_type="video", subject_ref_source=str(tmp_path))
    with pytest.raises(ValueError, match=r"must produce \[R,3,704,1280\]"):
        stage.forward(malformed, _args(config))


def test_subject_reference_stage_rejects_wrong_vae_shape(monkeypatch, tmp_path) -> None:
    from fastvideo.pipelines.basic.matrixgame35 import base_third_person_stages

    monkeypatch.setattr(base_third_person_stages, "load_subject_reference_canvases", lambda *args, **kwargs: _canvases(1))
    stage = MatrixGame35BaseThirdPersonSubjectReferenceStage(_FakeVAE((1, 48, 1, 43, 80)))
    batch = ForwardBatch(data_type="video", subject_ref_source=str(tmp_path))

    with pytest.raises(ValueError, match="must encode to"):
        stage.forward(batch, _args(MatrixGame35BaseThirdPersonPipelineConfig()))

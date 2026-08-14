# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for generic and causal consistency distillation."""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.consistency_model import (
    ConsistencyDistillationMethod,
    ConsistencyDistillationValidationCallback,
)
from fastvideo.train.models.base import CausalModelBase, ModelBase
from fastvideo.train.utils.config import load_run_config

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BIDIRECTIONAL_CONFIG = _REPO_ROOT / "examples/train/configs/consistency_model/wan/cd_t2v.yaml"


class _ScaleTransformer(torch.nn.Module):

    def __init__(self, scale: float, *, trainable: bool) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale), requires_grad=trainable)


class _BidirectionalModel(ModelBase):

    def __init__(
        self,
        scale: float,
        *,
        trainable: bool,
        training_config: Any | None = None,
    ) -> None:
        del training_config
        super().__init__(trainable=trainable)
        self.transformer = _ScaleTransformer(scale, trainable=trainable)
        self.noise_scheduler = SimpleNamespace(num_train_timesteps=1000)
        self.calls: list[dict[str, Any]] = []

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        del generator, latents_source
        latents = raw_batch["latents"]
        return TrainingBatch(
            latents=latents,
            timesteps=torch.zeros(latents.shape[0]),
            conditional_dict={},
            unconditional_dict={},
        )

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del timestep
        return clean_latents + noise

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        del cfg_uncond, attn_kind
        assert batch.timesteps is not None
        self.calls.append({
            "conditional": conditional,
            "timestep": timestep.detach().clone(),
            "context_timestep": batch.timesteps.detach().clone(),
            "clean_x": None,
            "scale": self.transformer.scale.detach().clone(),
        })
        return noisy_latents * self.transformer.scale

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        del ctx
        (loss / grad_accum_rounds).backward()


class _CausalModel(_BidirectionalModel, CausalModelBase):

    def clear_caches(self, *, cache_tag: str = "pos") -> None:
        del cache_tag

    def predict_noise_streaming(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cache_tag: str = "pos",
        store_kv: bool = False,
        cur_start_frame: int = 0,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        del cache_tag, store_kv, cur_start_frame
        return self.predict_noise(noisy_latents,
                                  timestep,
                                  batch,
                                  conditional=conditional,
                                  cfg_uncond=cfg_uncond,
                                  attn_kind=attn_kind)

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
        clean_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cfg_uncond, attn_kind
        assert batch.timesteps is not None
        self.calls.append({
            "conditional": conditional,
            "timestep": timestep.detach().clone(),
            "context_timestep": batch.timesteps.detach().clone(),
            "clean_x": clean_x,
            "scale": self.transformer.scale.detach().clone(),
        })
        return noisy_latents * self.transformer.scale


def _config(
    model_type: type[_BidirectionalModel] = _BidirectionalModel,
) -> SimpleNamespace:
    return SimpleNamespace(
        models={
            "student": {
                "_target_": (f"{__name__}.{model_type.__name__}"),
                "scale": 0.25,
                "trainable": True,
            },
            "teacher": {},
        },
        method={
            "discrete_cd_N": 4,
            "guidance_scale": 3.0,
            "ema_decay": 0.9,
        },
        training=SimpleNamespace(
            pipeline_config=SimpleNamespace(flow_shift=5.0),
            optimizer=SimpleNamespace(learning_rate=1e-3, betas=(0.9, 0.999), lr_scheduler="constant"),
            loop=SimpleNamespace(),
        ),
    )


def _fake_optimizer_builder(**kwargs: Any) -> tuple[torch.optim.Optimizer, Any]:
    optimizer = torch.optim.SGD(kwargs["params"], lr=kwargs["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    return optimizer, scheduler


def _role_models(model_type: type[_BidirectionalModel]) -> dict[str, ModelBase]:
    return {
        "student": model_type(0.25, trainable=True),
        "teacher": model_type(0.5, trainable=False),
    }


def _run_step(
    method_type: type[ConsistencyDistillationMethod],
    model_type: type[_BidirectionalModel],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ConsistencyDistillationMethod, dict[str, ModelBase], dict[str, torch.Tensor]]:
    monkeypatch.setattr(
        "fastvideo.train.methods.consistency_model.consistency_distillation.build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    role_models = _role_models(model_type)
    method = method_type(cfg=_config(model_type), role_models=role_models)
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(7)
    batch = {"latents": torch.randn(2, 3, 1, 2, 2)}

    loss_map, outputs, _ = method.single_train_step(batch, iteration=0)
    method.backward(loss_map, outputs)

    assert torch.isfinite(loss_map["total_loss"])
    assert role_models["student"].transformer.scale.grad is not None
    return method, role_models, loss_map


def test_bidirectional_cd_uses_batch_timesteps_and_standard_model_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, models, loss_map = _run_step(ConsistencyDistillationMethod, _BidirectionalModel, monkeypatch)

    assert loss_map["cd_loss"] is loss_map["total_loss"]
    ema_target = method.get_ema_target_model()
    assert isinstance(ema_target, _BidirectionalModel)
    calls = models["teacher"].calls + models["student"].calls + ema_target.calls
    assert all(call["timestep"].shape == (2, ) for call in calls)
    assert all(torch.equal(call["timestep"], call["context_timestep"]) for call in calls)
    assert all(call["clean_x"] is None for call in calls)
    assert not torch.equal(models["student"].calls[0]["timestep"], ema_target.calls[0]["timestep"])


def test_causal_cd_keeps_per_frame_timesteps_and_clean_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, models, loss_map = _run_step(ConsistencyDistillationMethod, _CausalModel, monkeypatch)

    assert loss_map["cd_loss"] is loss_map["total_loss"]
    ema_target = method.get_ema_target_model()
    assert isinstance(ema_target, _CausalModel)
    calls = models["teacher"].calls + models["student"].calls + ema_target.calls
    assert all(call["timestep"].shape == (2, 3) for call in calls)
    assert all(torch.equal(call["timestep"], call["context_timestep"]) for call in calls)
    assert all(call["clean_x"] is not None for call in calls)


def test_bidirectional_wan_config_selects_generic_cd() -> None:
    cfg = load_run_config(str(_BIDIRECTIONAL_CONFIG))

    assert cfg.models["student"]["_target_"] == "fastvideo.train.models.wan.WanModel"
    assert "ema" not in cfg.models
    assert cfg.method["_target_"] == ("fastvideo.train.methods.consistency_model.ConsistencyDistillationMethod")
    assert "ema_start_step" not in cfg.method


def test_cd_rejects_ema_update_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fastvideo.train.methods.consistency_model.consistency_distillation.build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    cfg = _config()
    cfg.method["ema_start_step"] = 200

    with pytest.raises(ValueError, match="online target EMA must update after every"):
        ConsistencyDistillationMethod(cfg=cfg, role_models=_role_models(_BidirectionalModel))


@pytest.mark.parametrize("ema_decay", [-0.1, 1.0, 1.1])
def test_cd_rejects_invalid_ema_decay(ema_decay: float, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fastvideo.train.methods.consistency_model.consistency_distillation.build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    cfg = _config()
    cfg.method["ema_decay"] = ema_decay

    with pytest.raises(ValueError, match=r"method\.ema_decay must be in \[0, 1\)"):
        ConsistencyDistillationMethod(cfg=cfg, role_models=_role_models(_BidirectionalModel))


def test_online_target_ema_is_included_in_checkpoint_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fastvideo.train.methods.consistency_model.consistency_distillation.build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    method = ConsistencyDistillationMethod(cfg=_config(), role_models=_role_models(_BidirectionalModel))

    state = method.checkpoint_state()

    assert "consistency_distillation.ema" in state
    assert "roles.ema.transformer" not in state


def test_cd_rejects_public_ema_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fastvideo.train.methods.consistency_model.consistency_distillation.build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    role_models = _role_models(_BidirectionalModel)
    role_models["ema"] = _BidirectionalModel(0.25, trainable=False)

    with pytest.raises(ValueError, match=r"remove models\.ema"):
        ConsistencyDistillationMethod(cfg=_config(), role_models=role_models)


@pytest.mark.parametrize("iteration", [0, 1, 199, 200])
@pytest.mark.parametrize("model_type", [_BidirectionalModel, _CausalModel])
def test_online_target_ema_updates_after_every_optimizer_step(
    iteration: int,
    model_type: type[_BidirectionalModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fastvideo.train.methods.consistency_model.consistency_distillation.build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    role_models = _role_models(model_type)
    method = ConsistencyDistillationMethod(cfg=_config(model_type), role_models=role_models)

    student_param = next(role_models["student"].transformer.parameters())
    ema_param = next(method.get_ema_target_model().transformer.parameters())
    with torch.no_grad():
        student_param.fill_(2.0)

    # Keep this test focused on the method-specific post-optimizer behavior.
    method._student_optimizer.step = lambda: None  # type: ignore[method-assign]
    method._student_lr_scheduler.step = lambda: None  # type: ignore[method-assign]
    method.optimizers_schedulers_step(iteration)

    assert method._ema_update_count == 1
    assert torch.allclose(method._target_ema.shadow["scale"], torch.tensor(0.425))

    # The callable target is synchronized lazily immediately before use.
    assert not torch.allclose(ema_param, method._target_ema.shadow["scale"])
    method._sync_ema_target()
    assert torch.allclose(ema_param, method._target_ema.shadow["scale"])


def test_target_forward_uses_authoritative_ema_shadow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fastvideo.train.methods.consistency_model.consistency_distillation.build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    role_models = _role_models(_BidirectionalModel)
    method = ConsistencyDistillationMethod(cfg=_config(), role_models=role_models)
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(7)

    with torch.no_grad():
        method._target_ema.shadow["scale"].fill_(1.25)
        method._ema_target_model.transformer.scale.fill_(9.0)
    method._ema_target_dirty = True

    method.single_train_step({"latents": torch.randn(2, 3, 1, 2, 2)}, iteration=0)

    assert torch.allclose(method._ema_target_model.calls[-1]["scale"], torch.tensor(1.25))


def test_cd_owns_frozen_ema_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fastvideo.train.methods.consistency_model.consistency_distillation.build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    role_models = _role_models(_BidirectionalModel)
    method = ConsistencyDistillationMethod(cfg=_config(), role_models=role_models)
    with torch.no_grad():
        method._target_ema.shadow["scale"].fill_(1.5)
        method._ema_target_model.transformer.scale.fill_(8.0)
    method._ema_target_dirty = True

    target = method.get_ema_target_model()

    assert target is method._ema_target_model
    assert torch.allclose(target.transformer.scale, torch.tensor(1.5))
    assert not target.transformer.training
    assert all(not parameter.requires_grad for parameter in target.transformer.parameters())


def test_cd_validation_callback_uses_internal_ema_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fastvideo.train.methods.consistency_model.consistency_distillation.build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    method = ConsistencyDistillationMethod(cfg=_config(), role_models=_role_models(_BidirectionalModel))
    callback = ConsistencyDistillationValidationCallback(
        pipeline_target="example.Pipeline",
        dataset_file="validation.json",
    )
    seen: list[torch.nn.Module] = []
    monkeypatch.setattr(
        callback,
        "_validation_memory_context",
        lambda _method, *, validation_transformer: contextlib.nullcontext(),
    )
    monkeypatch.setattr(callback, "_attn_qat_infer_context", lambda _transformer: contextlib.nullcontext())
    monkeypatch.setattr(callback, "_run_validation_inner", lambda _method, _step, transformer: seen.append(transformer))

    callback._run_validation(method, step=100)

    assert seen == [method._ema_target_model.transformer]


def test_ema_checkpoint_round_trip_restores_shadow_and_callable_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fastvideo.train.methods.consistency_model.consistency_distillation.build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    method = ConsistencyDistillationMethod(cfg=_config(), role_models=_role_models(_BidirectionalModel))
    state_wrapper = method.checkpoint_state()["consistency_distillation.ema"]
    with torch.no_grad():
        method._target_ema.shadow["scale"].fill_(1.75)
    method._ema_update_count = 12
    state = state_wrapper.state_dict()

    with torch.no_grad():
        method._target_ema.shadow["scale"].zero_()
        method._ema_target_model.transformer.scale.fill_(9.0)
    method._ema_update_count = 0
    state_wrapper.load_state_dict(state)

    assert method._ema_update_count == 12
    assert torch.allclose(method._target_ema.shadow["scale"], torch.tensor(1.75))
    assert torch.allclose(method._ema_target_model.transformer.scale, torch.tensor(1.75))


@pytest.mark.parametrize(
    "callbacks",
    [
        {"ema": {"decay": 0.99}},
        {"other": {"_target_": "fastvideo.train.callbacks.ema.EMACallback"}},
    ],
)
def test_cd_rejects_second_generic_ema(
    callbacks: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fastvideo.train.methods.consistency_model.consistency_distillation.build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    cfg = _config()
    cfg.callbacks = callbacks

    with pytest.raises(ValueError, match="owns its target EMA"):
        ConsistencyDistillationMethod(cfg=cfg, role_models=_role_models(_BidirectionalModel))


def test_cd_rejects_generic_validation_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fastvideo.train.methods.consistency_model.consistency_distillation.build_optimizer_and_scheduler",
        _fake_optimizer_builder,
    )
    cfg = _config()
    cfg.callbacks = {"validation": {}}

    with pytest.raises(ValueError, match="ConsistencyDistillationValidationCallback"):
        ConsistencyDistillationMethod(cfg=cfg, role_models=_role_models(_BidirectionalModel))

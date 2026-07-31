# SPDX-License-Identifier: Apache-2.0
"""Small state/forward contracts for the shared Matrix-Game 3.5 DiT."""

from collections.abc import Iterator

import pytest
import torch

import fastvideo.attention.layer as attention_layer
import fastvideo.models.dits.matrixgame35 as matrixgame35
from fastvideo.attention.backends.sdpa import SDPABackend
from fastvideo.configs.models.dits.matrixgame35 import (
    MatrixGame35WanVideoArchConfig,
    MatrixGame35WanVideoConfig,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.utils import get_param_names_mapping


@pytest.fixture
def cpu_sdpa(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """FastVideo has no production CPU platform; bind its Torch SDPA backend for contract tests."""

    monkeypatch.setattr(
        attention_layer,
        "get_attn_backend",
        lambda *args, **kwargs: SDPABackend,
    )
    yield


def _tiny_config(
    *,
    max_refs: int = 0,
    causal: bool = False,
) -> MatrixGame35WanVideoConfig:
    arch = MatrixGame35WanVideoArchConfig(
        patch_size=(1, 2, 2),
        num_attention_heads=1,
        attention_head_dim=128,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=256,
        num_layers=1,
        subject_ref_memory_max_refs=max_refs,
        causal=causal,
    )
    return MatrixGame35WanVideoConfig(arch_config=arch)


def _official_raw_keys(*, max_refs: int) -> set[str]:
    keys = {
        "patch_embedding.weight",
        "patch_embedding.bias",
        "text_embedding.0.weight",
        "text_embedding.0.bias",
        "text_embedding.2.weight",
        "text_embedding.2.bias",
        "time_embedding.0.weight",
        "time_embedding.0.bias",
        "time_embedding.2.weight",
        "time_embedding.2.bias",
        "time_projection.1.weight",
        "time_projection.1.bias",
        "head.head.weight",
        "head.head.bias",
        "head.modulation",
        "blocks.0.modulation",
        "blocks.0.norm3.weight",
        "blocks.0.norm3.bias",
    }
    for attention in ("self_attn", "cross_attn"):
        for projection in ("q", "k", "v", "o"):
            keys.add(f"blocks.0.{attention}.{projection}.weight")
            keys.add(f"blocks.0.{attention}.{projection}.bias")
        keys.add(f"blocks.0.{attention}.norm_q.weight")
        keys.add(f"blocks.0.{attention}.norm_k.weight")
    for layer in (0, 2):
        keys.add(f"blocks.0.ffn.{layer}.weight")
        keys.add(f"blocks.0.ffn.{layer}.bias")
    if max_refs:
        keys.update({
            "subject_ref_index_embedding",
            "subject_ref_type_embedding",
            "subject_ref_local_h_embedding",
            "subject_ref_local_w_embedding",
        })
    return keys


def _identity_camera_info(
    *,
    batch_size: int,
    frames: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    matrix = torch.eye(4, dtype=dtype).reshape(1, 1, 1, 4, 4)
    matrix = matrix.expand(batch_size, frames, 4, 4, 4).clone()
    return matrix, (matrix, matrix, matrix)


@pytest.mark.parametrize("max_refs", [0, 2, 4])
def test_matrixgame35_meta_state_surface_matches_mapped_official_keys(
    cpu_sdpa: None,
    max_refs: int,
) -> None:
    del cpu_sdpa
    with torch.device("meta"):
        model = matrixgame35.MatrixGame35Transformer3DModel(
            _tiny_config(max_refs=max_refs),
            hf_config={},
        )

    mapping = get_param_names_mapping(model.param_names_mapping)
    mapped_official_keys = {mapping(key)[0] for key in _official_raw_keys(max_refs=max_refs)}
    assert set(model.state_dict()) == mapped_official_keys
    assert model.subject_ref_memory_enabled is (max_refs > 0)
    assert matrixgame35.EntryClass is matrixgame35.MatrixGame35Transformer3DModel

    if max_refs == 0:
        assert not hasattr(model, "subject_ref_index_embedding")
    else:
        assert model.subject_ref_index_embedding.shape == (max_refs, 128)
        assert model.subject_ref_type_embedding.shape == (1, 128)
        assert model.subject_ref_local_h_embedding.shape == (64, 128)
        assert model.subject_ref_local_w_embedding.shape == (64, 128)


def test_matrixgame35_cpu_strict_load_and_forward(cpu_sdpa: None) -> None:
    del cpu_sdpa
    torch.manual_seed(20260731)
    model = matrixgame35.MatrixGame35Transformer3DModel(
        _tiny_config(),
        hf_config={},
    ).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.normal_(mean=0.0, std=0.02)

    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    incompatible = model.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    hidden_states = torch.randn(1, 4, 2, 4, 4)
    sequence_length = 2 * 2 * 2
    encoder_hidden_states = torch.randn(1, 3, 16)
    timestep = torch.linspace(0, 999, sequence_length).reshape(1, sequence_length)
    camera_info = _identity_camera_info(batch_size=1, frames=2)

    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            camera_info=camera_info,
        )
    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert torch.isfinite(output).all()


def test_matrixgame35_rejects_unreleased_execution_contracts(
    cpu_sdpa: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cpu_sdpa
    causal_model = matrixgame35.MatrixGame35Transformer3DModel(
        _tiny_config(causal=True),
        hf_config={},
    )
    incompatible = causal_model.load_state_dict(causal_model.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    monkeypatch.setattr(matrixgame35, "get_sp_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        matrixgame35.MatrixGame35Transformer3DModel(_tiny_config(), hf_config={})
    monkeypatch.setattr(matrixgame35, "get_sp_world_size", lambda: 1)

    model = matrixgame35.MatrixGame35Transformer3DModel(_tiny_config(), hf_config={})
    hidden_states = torch.zeros(1, 4, 2, 4, 4)
    encoder_hidden_states = torch.zeros(1, 3, 16)
    camera_info = _identity_camera_info(batch_size=1, frames=2)

    with pytest.raises(NotImplementedError, match="causal forward"):
        causal_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=torch.zeros(1, 8),
            camera_info=camera_info,
        )

    with pytest.raises(ValueError, match="one value per packed latent token"):
        model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=torch.zeros(1),
            camera_info=camera_info,
        )
    with pytest.raises(ValueError, match="camera_info="):
        model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=torch.zeros(1, 8),
        )
    with pytest.raises(NotImplementedError, match="kv_cache"):
        model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=torch.zeros(1, 8),
            camera_info=camera_info,
            kv_cache={},
        )

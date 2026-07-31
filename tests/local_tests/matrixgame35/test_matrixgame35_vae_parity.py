# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 3.5 WanVideoVAE38 reuse contracts and parity.

Coverage scope: both. The CUDA path loads the pinned raw Wan2.2 VAE through
the official Matrix-Game class and the Diffusers-style VAE through FastVideo's
production component loader, then compares deterministic encode and decode.
"""

from __future__ import annotations

import ast
import gc
import inspect
import os
from pathlib import Path
import textwrap

import pytest
import torch
from diffusers import AutoencoderKLWan as DiffusersAutoencoderKLWan
from torch.testing import assert_close

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.matrixgame35 import make_matrixgame35_vae_config
from fastvideo.configs.pipelines.wan import LucyEditDevConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import VAELoader
from fastvideo.pipelines.basic.matrixgame35.codec import (
    decode_matrixgame35_tiled_video,
    encode_matrixgame35_tiled_video,
)
from tests.local_tests.matrixgame35._shared_upstream import load_upstream_wan_vae


REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY_SCOPE = "both"
OFFICIAL_REF_DIR = Path(
    os.getenv("MATRIXGAME35_OFFICIAL_REF_DIR", REPO_ROOT / "Matrix-Game-3.5")
)
WAN22_RAW_DIR = Path(
    os.getenv(
        "MATRIXGAME35_WAN22_RAW_DIR",
        REPO_ROOT / "official_weights" / "Wan2.2-TI2V-5B",
    )
)
WAN22_DIFFUSERS_DIR = Path(
    os.getenv(
        "MATRIXGAME35_WAN22_DIFFUSERS_DIR",
        REPO_ROOT / "official_weights" / "Wan2.2-TI2V-5B-Diffusers",
    )
)
RAW_VAE_PATH = WAN22_RAW_DIR / "Wan2.2_VAE.pth"
FASTVIDEO_VAE_DIR = WAN22_DIFFUSERS_DIR / "vae"


class _DiffusersWan38Adapter(torch.nn.Module):
    """Expose the pinned upstream wrapper's normalized codec boundary."""

    def __init__(self, vae: DiffusersAutoencoderKLWan) -> None:
        super().__init__()
        self.vae = vae

    @staticmethod
    def _stats(
        scale: list[torch.Tensor],
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, inverse_std = scale
        return (
            mean.to(device=value.device, dtype=value.dtype).view(1, -1, 1, 1, 1),
            inverse_std.to(device=value.device, dtype=value.dtype).view(1, -1, 1, 1, 1),
        )

    def encode(self, video: torch.Tensor, scale: list[torch.Tensor]) -> torch.Tensor:
        posterior = self.vae.encode(video).latent_dist.mode()
        mean, inverse_std = self._stats(scale, posterior)
        return (posterior - mean) * inverse_std

    def decode(self, latents: torch.Tensor, scale: list[torch.Tensor]) -> torch.Tensor:
        mean, inverse_std = self._stats(scale, latents)
        return _diffusers_decode_unclamped(self.vae, latents / inverse_std + mean)


def _diffusers_decode_unclamped(
    vae: DiffusersAutoencoderKLWan,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Run the independent Diffusers decoder before its per-call clamp."""
    vae.clear_cache()
    hidden = vae.post_quant_conv(latents)
    chunks = []
    for frame_index in range(latents.shape[2]):
        vae._conv_idx = [0]
        chunks.append(
            vae.decoder(
                hidden[:, :, frame_index:frame_index + 1],
                feat_cache=vae._feat_map,
                feat_idx=vae._conv_idx,
                first_chunk=frame_index == 0,
            ))
    output = torch.cat(chunks, dim=2)
    patch_size = vae.config.patch_size
    if patch_size is not None and patch_size != 1:
        batch, channels, frames, height, width = output.shape
        channels //= patch_size * patch_size
        output = output.view(
            batch,
            channels,
            patch_size,
            patch_size,
            frames,
            height,
            width,
        )
        output = output.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
        output = output.view(
            batch,
            channels,
            frames,
            height * patch_size,
            width * patch_size,
        )
    vae.clear_cache()
    return output


def _make_diffusers_backed_upstream_wrapper(
    vae: DiffusersAutoencoderKLWan,
) -> torch.nn.Module:
    """Bind the pinned Matrix-Game tiler to a Diffusers-weighted codec."""
    module = load_upstream_wan_vae(OFFICIAL_REF_DIR)
    wrapper = module.WanVideoVAE38.__new__(module.WanVideoVAE38)
    torch.nn.Module.__init__(wrapper)
    wrapper.mean = torch.tensor(vae.config.latents_mean)
    wrapper.std = torch.tensor(vae.config.latents_std)
    wrapper.scale = [wrapper.mean, wrapper.std.reciprocal()]
    wrapper.model = _DiffusersWan38Adapter(vae)
    wrapper.upsampling_factor = 16
    wrapper.z_dim = 48
    return wrapper


def _assert_vae_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    diff = (actual - expected).abs()
    print(
        f"{name}: diff_max={diff.max().item():.6f} "
        f"diff_mean={diff.mean().item():.6f} "
        f"reference_abs_mean={expected.abs().mean().item():.6f}"
    )
    assert diff.mean().item() <= 1e-2
    assert_close(actual, expected, atol=5e-2, rtol=5e-2)


def _official_init_constants(init) -> tuple[tuple[float, ...], tuple[float, ...]]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(init)))
    values: dict[str, tuple[float, ...]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {"mean", "std"}:
            values[target.id] = tuple(ast.literal_eval(node.value))
    return values["mean"], values["std"]


def _load_official_vae(device: torch.device) -> torch.nn.Module:
    if not RAW_VAE_PATH.is_file():
        pytest.skip(f"Raw Wan2.2 VAE weights are absent: {RAW_VAE_PATH}")
    module = load_upstream_wan_vae(OFFICIAL_REF_DIR)
    model = module.WanVideoVAE38().to(device=device, dtype=torch.float32)
    state = torch.load(RAW_VAE_PATH, map_location="cpu", weights_only=True)
    if "model_state" in state:
        state = state["model_state"]
    prefixed = [key.startswith("model.") for key in state]
    if any(prefixed) and not all(prefixed):
        raise AssertionError("Raw Wan2.2 VAE mixes model.-prefixed and unprefixed keys")
    if not any(prefixed):
        state = {f"model.{key}": value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def _load_diffusers_vae(
    device: torch.device,
    dtype: torch.dtype,
) -> DiffusersAutoencoderKLWan:
    if not (FASTVIDEO_VAE_DIR / "config.json").is_file():
        pytest.skip(f"Diffusers-style Wan2.2 VAE is absent: {FASTVIDEO_VAE_DIR}")
    return (
        DiffusersAutoencoderKLWan.from_pretrained(
            FASTVIDEO_VAE_DIR,
            local_files_only=True,
            torch_dtype=dtype,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )


def _load_fastvideo_vae(*, precision: str = "fp32") -> torch.nn.Module:
    if not (FASTVIDEO_VAE_DIR / "config.json").is_file():
        pytest.skip(f"Diffusers-style Wan2.2 VAE is absent: {FASTVIDEO_VAE_DIR}")
    config = make_matrixgame35_vae_config()
    args = FastVideoArgs(
        model_path=str(WAN22_DIFFUSERS_DIR),
        pipeline_config=PipelineConfig(vae_config=config, vae_precision=precision),
        pin_cpu_memory=False,
    )
    args.vae_cpu_offload = False
    return VAELoader().load(str(FASTVIDEO_VAE_DIR), args)


def test_matrixgame35_vae_config_is_exact_lucy_wan22_reuse() -> None:
    config = make_matrixgame35_vae_config()
    shared = LucyEditDevConfig().vae_config
    module = load_upstream_wan_vae(OFFICIAL_REF_DIR)
    signature = inspect.signature(module.WanVideoVAE38.__init__)
    official_mean, official_std = _official_init_constants(
        module.WanVideoVAE38.__init__
    )

    assert signature.parameters["z_dim"].default == 48
    assert signature.parameters["dim"].default == 160
    assert type(config) is type(shared)
    assert type(config.arch_config) is type(shared.arch_config)
    assert config.load_encoder is True
    assert config.load_decoder is True
    assert config.z_dim == 48
    assert config.base_dim == 160
    assert config.decoder_base_dim == 256
    assert config.in_channels == 12
    assert config.out_channels == 12
    assert config.patch_size == 2
    assert config.scale_factor_temporal == 4
    assert config.scale_factor_spatial == 16
    assert config.is_residual is True
    assert config.clip_output is False
    assert tuple(config.latents_mean) == official_mean
    assert tuple(config.latents_std) == official_std
    assert_close(
        config.shift_factor,
        torch.tensor(official_mean).view(1, 48, 1, 1, 1),
    )
    assert_close(
        config.scaling_factor,
        torch.tensor(official_std).reciprocal().view(1, 48, 1, 1, 1),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for VAE parity")
def test_matrixgame35_wan22_vae_encode_decode_parity() -> None:
    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu").manual_seed(3535)
    video_cpu = torch.randn(1, 3, 5, 64, 64, generator=generator).clamp(-1, 1)
    latent_cpu = torch.randn(1, 48, 2, 4, 4, generator=generator)

    official = _load_official_vae(device)
    with torch.inference_mode():
        official_latent = official.encode(video_cpu, device).float().cpu()
        official_video = official.decode(latent_cpu, device).float().cpu()
    del official
    torch.cuda.empty_cache()

    fastvideo = _load_fastvideo_vae()
    config = make_matrixgame35_vae_config()
    shift = config.shift_factor.to(device=device, dtype=torch.float32)
    scale = config.scaling_factor.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        fastvideo_latent = (
            fastvideo.encode(video_cpu.to(device)).mode() - shift
        ) * scale
        fastvideo_video = fastvideo.decode(latent_cpu.to(device) / scale + shift)
    fastvideo_latent = fastvideo_latent.float().cpu()
    fastvideo_video = fastvideo_video.float().cpu()

    assert official_latent.shape == fastvideo_latent.shape == (1, 48, 2, 4, 4)
    assert official_video.shape == fastvideo_video.shape == (1, 3, 5, 64, 64)
    assert_close(fastvideo_latent, official_latent, atol=5e-2, rtol=5e-2)
    assert_close(fastvideo_video, official_video, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for VAE parity")
def test_matrixgame35_diffusers_wan22_vae_encode_decode_parity() -> None:
    """Compare production loading to the independent Diffusers component."""
    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu").manual_seed(3536)
    video_cpu = torch.randn(1, 3, 5, 64, 64, generator=generator).clamp(-1, 1)
    latent_cpu = torch.randn(1, 48, 2, 4, 4, generator=generator)

    official = _load_diffusers_vae(device, torch.float32)
    with torch.inference_mode():
        official_latent = official.encode(video_cpu.to(device)).latent_dist.mode().cpu()
        official_video = official.decode(latent_cpu.to(device)).sample.cpu()
    del official
    gc.collect()
    torch.cuda.empty_cache()

    fastvideo = _load_fastvideo_vae()
    with torch.inference_mode():
        fastvideo_latent = fastvideo.encode(video_cpu.to(device)).mode().cpu()
        fastvideo_video = fastvideo.decode(latent_cpu.to(device)).cpu()

    assert official_latent.shape == fastvideo_latent.shape == (1, 48, 2, 4, 4)
    assert official_video.shape == fastvideo_video.shape == (1, 3, 5, 64, 64)
    _assert_vae_close("diffusers encode mode", fastvideo_latent, official_latent)
    _assert_vae_close("diffusers decode", fastvideo_video, official_video)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for VAE parity")
def test_matrixgame35_diffusers_wan22_vae_release_resolution_tiled_parity() -> None:
    """Compare the 704x1280 codec to the pinned upstream tiler and Diffusers VAE."""
    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu").manual_seed(3537)
    video_cpu = torch.randn(1, 3, 1, 704, 1280, generator=generator).clamp(-1, 1).to(torch.bfloat16)
    latent_cpu = (torch.randn(1, 48, 1, 44, 80, generator=generator) * 0.25).to(torch.bfloat16)

    official = _load_diffusers_vae(device, torch.bfloat16)
    upstream = _make_diffusers_backed_upstream_wrapper(official)
    with torch.inference_mode():
        reference_latent = upstream.encode(
            [video_cpu[0]],
            device=device,
            tiled=True,
            tile_size=(34, 34),
            tile_stride=(18, 16),
        )
        reference_video = upstream.decode(
            latent_cpu,
            device=device,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
        reference_video = reference_video.float().mul(0.5).add(0.5)
    del upstream, official
    gc.collect()
    torch.cuda.empty_cache()

    fastvideo = _load_fastvideo_vae(precision="bf16")
    with torch.inference_mode():
        fastvideo_latent = encode_matrixgame35_tiled_video(
            fastvideo,
            video_cpu.to(device),
        )
        fastvideo_video = decode_matrixgame35_tiled_video(
            fastvideo,
            latent_cpu.to(device),
        )

    assert reference_latent.shape == fastvideo_latent.shape == (1, 48, 1, 44, 80)
    assert reference_video.shape == fastvideo_video.shape == (1, 3, 1, 704, 1280)
    _assert_vae_close("release-resolution tiled encode", fastvideo_latent, reference_latent)
    _assert_vae_close("release-resolution tiled decode", fastvideo_video, reference_video)

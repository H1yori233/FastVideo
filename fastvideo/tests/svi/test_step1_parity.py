# SPDX-License-Identifier: Apache-2.0
"""Step-1 numerical parity gate for the SVI training port.

Reads the upstream fixture produced by
``.agents/exploration/svi-training/dump_step1_upstream.py`` and runs one
``SVITrainingMethod.single_train_step`` on the FV side with matched inputs.
Asserts each intermediate tensor matches within the documented tolerance.

Launch (single GPU is enough; torchrun is required only because the FV
training method touches ``get_world_group()``):

    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29503 \\
        -m pytest fastvideo/tests/svi/test_step1_parity.py -v -s

The fixture is skipped if absent (so the test is safe in CI without GPUs).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from fastvideo.tests.svi._step1_fixture_io import (
    inputs_path,
    load_inputs,
    load_upstream,
    upstream_path,
)

REPO = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO / "outputs/parity_fixture/step01"
YAML_CONFIG = REPO / "examples/train/configs/fine_tuning/svi/wan_i2v_svi.yaml"

# Tolerance table. Both sides run identical algorithms but call different
# library implementations (HF UMT5 / CLIP / AutoencoderKL on FV; diffsynth's
# Wan* reimplementations on upstream), so we bound mean-drift per tensor (the
# meaningful signal) and let abs_max drift be looser to absorb per-element
# fp noise across non-identical kernel stacks.
TOLERANCES: dict[str, dict[str, float]] = {
    "prompt_emb_context": {"atol": 1.0e-1, "rtol": 1.0e-1, "max_mean_drift_pct": 5.0},
    "clip_feature":       {"atol": 6.0,    "rtol": 1.0e-1, "max_mean_drift_pct": 10.0},
    "clean_latents":      {"atol": 2.0e-1, "rtol": 1.0e-1, "max_mean_drift_pct": 2.0},
    "y":                  {"atol": 2.0e-1, "rtol": 1.0e-1, "max_mean_drift_pct": 2.0},
    "noisy_latents":      {"atol": 2.0e-1, "rtol": 1.0e-1, "max_mean_drift_pct": 2.0},
    "target":             {"atol": 2.0e-1, "rtol": 1.0e-1, "max_mean_drift_pct": 2.0},
    "pred":               {"atol": 5.0e-1, "rtol": 1.0e-1, "max_mean_drift_pct": 5.0},
    # Loss is a derived scalar; small per-element pred/target drifts amplify
    # through MSE so we only track magnitude alignment, not bit-exactness.
    "loss":               {"atol": 1.0e-1, "rtol": 1.0e-1, "max_mean_drift_pct": 50.0},
    "timestep_actual":    {"atol": 1.0e-3, "rtol": 0.0,    "max_mean_drift_pct": 0.01},
}


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    den = b.abs().mean().clamp(min=1e-12)
    return {
        "abs_max": float(diff.max().item()),
        "abs_mean": float(diff.mean().item()),
        "mean_drift_pct": float((diff.mean() / den * 100.0).item()),
        "a_mean": float(a.mean().item()),
        "b_mean": float(b.mean().item()),
    }


# ----------------------------------------------------------------------
# Session-scoped fixture: run FV's single_train_step once
# ----------------------------------------------------------------------

@pytest.fixture(scope="session")
def _captures() -> dict[str, dict[str, Any]]:
    """Run FV's single_train_step once with the upstream fixture; return both
    the FV-captured tensors and the upstream-captured tensors.
    """
    if not Path(inputs_path(str(FIXTURE_ROOT))).exists() or not Path(upstream_path(str(FIXTURE_ROOT))).exists():
        pytest.skip(
            f"parity fixture missing under {FIXTURE_ROOT}; "
            "run .agents/exploration/svi-training/dump_step1_upstream.py first")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — step-1 parity needs a GPU")

    inputs = load_inputs(str(FIXTURE_ROOT))
    upstream_dump = load_upstream(str(FIXTURE_ROOT))

    # Distributed init (single rank). The FV training stack expects a
    # process group to exist.
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29504")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    # Build FV method via the standard YAML loader.
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.config import load_run_config

    cfg = load_run_config(str(YAML_CONFIG))
    # Force ER off + single GPU + correct data path.
    cfg.method["use_error_recycling"] = False
    cfg.training.distributed.num_gpus = 1
    cfg.training.distributed.hsdp_shard_dim = 1
    cfg.training.distributed.tp_size = 1
    cfg.training.distributed.sp_size = 1
    cfg.training.data.data_path = str(REPO / "Stable-Video-Infinity/data/toy_train/svi-film-shot")
    cfg.training.checkpoint.output_dir = str(REPO / "outputs/parity_step1_fv")
    cfg.training.loop.max_train_steps = 1

    _, method, _, _ = build_from_config(cfg)
    method.on_train_start()

    # Load upstream LoRA init state into FV's transformer where keys map.
    _load_upstream_lora_state(method, inputs["lora_init_state"])

    # Build a synthetic batch from inputs.pt (skip the dataloader). Upstream's
    # default DataLoader added a batch dim on the per-frame reference tensors;
    # FV's collate expects per-sample list[Tensor[H,W,3]] so we strip it here.
    first_refs_unbatched = [t.squeeze(0) if t.dim() == 4 else t for t in inputs["first_ref_frames"]]
    random_ref = inputs["random_ref_frame"]
    if random_ref.dim() == 4:
        random_ref = random_ref.squeeze(0)
    batch = {
        "text": [inputs["text"]],
        "video": inputs["video"],
        "first_ref_frames": [first_refs_unbatched],
        "random_ref_frame": random_ref.unsqueeze(0),  # FV collate expects [B, H, W, 3]
        "path": [inputs.get("path", "")],
    }

    fv_capture: dict[str, Any] = {}
    method.single_train_step(
        batch,
        iteration=0,
        _test_fixture={
            "noise": inputs["noise"],
            "timestep_id": inputs["timestep_id"] if isinstance(inputs["timestep_id"], torch.Tensor) else torch.tensor(
                int(inputs["timestep_id"]), dtype=torch.int64),
        },
        _test_capture=fv_capture,
    )

    # Persist for offline inspection (the run is ~1 min, so re-running is cheap
    # but having the .pt around makes it easy to diff individual tensors).
    fv_dump_path = REPO / "outputs/parity_fixture/step01/fv_intermediates.pt"
    torch.save({k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in fv_capture.items()},
               fv_dump_path)
    print(f"[parity] wrote {fv_dump_path}", flush=True)

    return {"fv": fv_capture, "upstream": upstream_dump}


def _load_upstream_lora_state(method, lora_state: dict[str, torch.Tensor]) -> None:
    """Best-effort load of upstream's PEFT-style LoRA state into FV's transformer.

    Upstream keys look like ``pipe.dit.blocks.X.self_attn.q.lora_A.default.weight``;
    FV layer params live at ``blocks.X.attn1.to_q.lora_A`` (no ``.weight``
    suffix on the Parameter, FV stores them as ``nn.Parameter`` directly).
    """
    transformer = method.student.transformer
    fv_params = dict(transformer.named_parameters())

    # Build a mapping upstream → FV via the same regex table used by the
    # inference LoRA loader.
    import re
    rules = [
        # Self-attention: FV's transformer collapses ``attn1.*`` into ``blocks.X.to_{q,k,v,out}``.
        (re.compile(r"^pipe\.dit\.blocks\.(\d+)\.self_attn\.q\.(lora_[AB])\.default\.weight$"),
         r"blocks.\1.to_q.\2"),
        (re.compile(r"^pipe\.dit\.blocks\.(\d+)\.self_attn\.k\.(lora_[AB])\.default\.weight$"),
         r"blocks.\1.to_k.\2"),
        (re.compile(r"^pipe\.dit\.blocks\.(\d+)\.self_attn\.v\.(lora_[AB])\.default\.weight$"),
         r"blocks.\1.to_v.\2"),
        (re.compile(r"^pipe\.dit\.blocks\.(\d+)\.self_attn\.o\.(lora_[AB])\.default\.weight$"),
         r"blocks.\1.to_out.\2"),
        # Cross-attention keeps the ``attn2.*`` prefix.
        (re.compile(r"^pipe\.dit\.blocks\.(\d+)\.cross_attn\.q\.(lora_[AB])\.default\.weight$"),
         r"blocks.\1.attn2.to_q.\2"),
        (re.compile(r"^pipe\.dit\.blocks\.(\d+)\.cross_attn\.k\.(lora_[AB])\.default\.weight$"),
         r"blocks.\1.attn2.to_k.\2"),
        (re.compile(r"^pipe\.dit\.blocks\.(\d+)\.cross_attn\.v\.(lora_[AB])\.default\.weight$"),
         r"blocks.\1.attn2.to_v.\2"),
        (re.compile(r"^pipe\.dit\.blocks\.(\d+)\.cross_attn\.o\.(lora_[AB])\.default\.weight$"),
         r"blocks.\1.attn2.to_out.\2"),
        (re.compile(r"^pipe\.dit\.blocks\.(\d+)\.ffn\.0\.(lora_[AB])\.default\.weight$"),
         r"blocks.\1.ffn.fc_in.\2"),
        (re.compile(r"^pipe\.dit\.blocks\.(\d+)\.ffn\.2\.(lora_[AB])\.default\.weight$"),
         r"blocks.\1.ffn.fc_out.\2"),
    ]

    # FV's actual param names may include a `_checkpoint_wrapped_module.`
    # segment from activation-checkpointing wrappers.
    def _normalize(name: str) -> str:
        return name.replace("_checkpoint_wrapped_module.", "")

    fv_by_normalized = {_normalize(n): n for n in fv_params}

    matched = 0
    skipped = []
    for up_key, tensor in lora_state.items():
        fv_key = None
        for pat, repl in rules:
            m = pat.match(up_key)
            if m:
                fv_key = pat.sub(repl, up_key)
                break
        if fv_key is None:
            skipped.append(up_key)
            continue
        fv_actual = fv_by_normalized.get(fv_key)
        if fv_actual is None:
            skipped.append(f"{up_key} → {fv_key} (no FV target)")
            continue
        with torch.no_grad():
            fv_param = fv_params[fv_actual]
            src = tensor.to(device=fv_param.device, dtype=fv_param.dtype)
            if hasattr(fv_param, "to_local"):
                fv_param.to_local().copy_(src)
            else:
                fv_param.copy_(src)
        matched += 1
    print(f"[parity] loaded {matched} upstream LoRA tensors; "
          f"skipped {len(skipped)} (first few: {skipped[:3]})", flush=True)


# ----------------------------------------------------------------------
# Per-tensor sub-tests
# ----------------------------------------------------------------------

@pytest.mark.parametrize("key", list(TOLERANCES.keys()))
def test_step1_intermediate_matches_upstream(_captures, key: str) -> None:
    fv = _captures["fv"]
    up = _captures["upstream"]
    if key not in fv:
        pytest.skip(f"FV capture missing key: {key}")
    if key not in up:
        pytest.skip(f"upstream capture missing key: {key}")

    fv_t = fv[key]
    up_t = up[key]
    if not isinstance(fv_t, torch.Tensor) or not isinstance(up_t, torch.Tensor):
        pytest.skip(f"non-tensor value at key={key}: fv={type(fv_t).__name__} up={type(up_t).__name__}")
    if fv_t.shape != up_t.shape:
        pytest.fail(f"shape mismatch at {key}: fv={tuple(fv_t.shape)} upstream={tuple(up_t.shape)}")

    stats = _diff_stats(fv_t, up_t)
    tol = TOLERANCES[key]
    print(
        f"  {key:>22s}  abs_max={stats['abs_max']:.3e}  abs_mean={stats['abs_mean']:.3e}  "
        f"drift={stats['mean_drift_pct']:.2f}%  fv_mean={stats['a_mean']:.3e}  "
        f"up_mean={stats['b_mean']:.3e}  (atol={tol['atol']}, rtol={tol['rtol']}, "
        f"max_drift={tol['max_mean_drift_pct']}%)",
        flush=True,
    )
    torch.testing.assert_close(fv_t.float(), up_t.float(), atol=tol["atol"], rtol=tol["rtol"])
    assert stats["mean_drift_pct"] <= tol["max_mean_drift_pct"], (
        f"{key} mean drift {stats['mean_drift_pct']:.2f}% exceeds {tol['max_mean_drift_pct']}%")


def test_pred_mean_drift_under_5pct(_captures) -> None:
    fv = _captures["fv"]
    up = _captures["upstream"]
    if "pred" not in fv or "pred" not in up:
        pytest.skip("pred missing")
    stats = _diff_stats(fv["pred"], up["pred"])
    print(f"  pred mean_drift = {stats['mean_drift_pct']:.2f}%", flush=True)
    assert stats["mean_drift_pct"] < 5.0, (f"pred abs-mean drift {stats['mean_drift_pct']:.2f}% exceeds 5%")

# SPDX-License-Identifier: Apache-2.0
"""Pipeline parity for LingBot-World-Fast: the real denoising stage vs the
official model driven through the identical chunked-DMD AR recipe.

Delegates to ``_fast_pipeline_runner.py`` (a clean subprocess): the official
``WanModelFast`` is numerically unstable on GB200/Blackwell once a FastVideo
model has run in the same process, so the runner runs the official AR reference
first and forces the MATH SDPA backend on both sides. It compares the denoised
latents of the real ``LingBotWorldFastCausalDenoisingStage`` against the official
AR loop (same scheduler, inputs, and re-noise draws) over two block-causal chunks.

Usage:
    DISABLE_SP=1 pytest tests/local_tests/pipelines/test_lingbotworld_fast_pipeline_parity.py -v -s
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
REF_REPO = Path(os.getenv("LINGBOT_WORLD_REPO", REPO_ROOT / "reference" / "lingbot-world"))
RUNNER = str(REPO_ROOT / "tests" / "local_tests" / "lingbotworld" / "_fast_pipeline_runner.py")

requires_runtime = pytest.mark.skipif(not torch.cuda.is_available(),
                                      reason="CUDA required")


def _weights_available() -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("robbyant/lingbot-world-fast-diffusers")
        return True
    except Exception:
        return False


@requires_runtime
def test_lingbotworld_fast_pipeline_parity():
    if not REF_REPO.exists() or not _weights_available():
        pytest.skip("fast-diffusers weights or reference repo unavailable")
    env = dict(os.environ, DISABLE_SP="1", PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run([sys.executable, RUNNER], env=env, cwd=str(REPO_ROOT),
                          capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
    assert "RESULT pipeline" in proc.stdout, f"runner produced no result:\n{proc.stderr[-4000:]}"
    assert proc.returncode == 0, "pipeline parity outside bounds (see RESULT line above)"

# SPDX-License-Identifier: Apache-2.0
"""Numerical parity test for the LingBot-World-Fast block-causal transformer.

Coverage scope: both (production ``param_names_mapping`` + forward implementation).

The official ``WanModelFast`` and the FastVideo
``CausalLingBotWorldTransformer3DModel`` are loaded from the *same*
``robbyant/lingbot-world-fast-diffusers`` transformer weights (the diffusers repo
stores official-named tensors), fed identical inputs through a per-layer KV cache,
and their denoised-noise outputs compared for chunk 0 (forward) and chunk 1
(block-causal KV-cache reuse).

The numerics run in a CLEAN SUBPROCESS (``_fast_parity_runner.py``): the official
``WanModelFast`` is numerically unstable on GB200/Blackwell once a FastVideo model
has run in the same process (its bf16 attention flips to ~2.5x-off garbage), so
the runner builds and runs the official first and forces the MATH SDPA backend
on both sides (flash / mem-efficient SDPA are the unstable path here). This keeps
the official reference faithful while still exercising the real FastVideo model.

Requires CUDA and the official reference repo at ``reference/lingbot-world``
(or ``$LINGBOT_WORLD_REPO``).

Usage:
    DISABLE_SP=1 pytest tests/local_tests/lingbotworld/test_lingbotworld_fast_parity.py -v -s
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
RUNNER = str(Path(__file__).with_name("_fast_parity_runner.py"))

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
def test_lingbotworld_fast_transformer_parity():
    if not REF_REPO.exists() or not _weights_available():
        pytest.skip("fast-diffusers weights or reference repo unavailable")
    env = dict(os.environ, DISABLE_SP="1", PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run([sys.executable, RUNNER], env=env, cwd=str(REPO_ROOT),
                          capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
    # The runner exercises both chunk 0 (forward) and chunk 1 (KV-cache causality)
    # and exits non-zero if either comparison falls outside the bounds.
    assert "RESULT chunk0" in proc.stdout and "RESULT chunk1" in proc.stdout, \
        f"runner did not produce results:\n{proc.stderr[-4000:]}"
    assert proc.returncode == 0, "component parity outside bounds (see RESULT lines above)"

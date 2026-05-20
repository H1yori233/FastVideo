"""Lightning callback that captures one full upstream training step into a
parity fixture (``inputs.pt`` + ``upstream_intermediates.pt``).

Strategy:
  - ``on_train_start``: capture LoRA init state from the LightningModule.
  - ``on_train_batch_start``: capture the dataloader batch verbatim, then
    install one-shot taps on ``torch.randn_like``/``torch.randint`` so the
    very first call inside ``training_step`` (noise + timestep sampling)
    gets saved to the fixture.
  - Method wrappers (installed in ``__init__``) tap the OUTPUT of
    ``pipe_VAE.encode_prompt``, ``.encode_video``, ``.encode_images_adaptive``,
    ``pipe.scheduler.add_noise``, ``.training_target``, ``.training_weight``,
    and ``pipe.denoising_model().__call__``. Each wrapped fn stashes its
    output into ``_upstream`` then returns unchanged.
  - ``on_train_batch_end``: dump the two .pt files, restore RNG patches,
    set ``trainer.should_stop = True``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import lightning as pl
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from fastvideo.tests.svi._step1_fixture_io import save_inputs, save_upstream  # noqa: E402


def _to_cpu(t: Any) -> Any:
    if isinstance(t, torch.Tensor):
        return t.detach().to("cpu").contiguous()
    return t


class _OneShotTap:
    """Wrap a callable so its first call captures into ``store[key]``.

    After the first call, the wrapper passes through unchanged. Used for
    ``torch.randn_like`` (one noise tensor per step) and ``torch.randint``
    (one timestep_id per step).
    """

    def __init__(self, original, store: dict[str, Any], key: str):
        self._original = original
        self._store = store
        self._key = key
        self._fired = False

    def __call__(self, *args, **kwargs):
        out = self._original(*args, **kwargs)
        if not self._fired:
            self._store[self._key] = _to_cpu(out)
            self._fired = True
        return out


class UpstreamStep1DumpCallback(pl.Callback):

    def __init__(self, root: str) -> None:
        super().__init__()
        self._root = root
        self._inputs: dict[str, Any] = {}
        self._upstream: dict[str, Any] = {}
        self._installed_method_patches: list[tuple[Any, str, Any]] = []
        self._installed_rng_patches: list[tuple[str, Any]] = []
        self._step_done = False

    # ------------------------------------------------------------------
    # Method wrapping helpers
    # ------------------------------------------------------------------

    def _wrap_method(self, obj: Any, name: str, fn) -> None:
        """Replace ``obj.<name>`` with ``fn``; remember the original."""
        original = getattr(obj, name)
        self._installed_method_patches.append((obj, name, original))
        setattr(obj, name, fn)

    def _restore_method_patches(self) -> None:
        for obj, name, original in self._installed_method_patches:
            setattr(obj, name, original)
        self._installed_method_patches.clear()

    def _install_rng_taps(self) -> None:
        # RNG draws go into ``_inputs`` so the FV side replays them as the
        # source of truth for ``noise`` / ``timestep_id``.
        for attr, key in (("randn_like", "noise"), ("randint", "timestep_id")):
            original = getattr(torch, attr)
            tap = _OneShotTap(original, self._inputs, key)
            self._installed_rng_patches.append((attr, original))
            setattr(torch, attr, tap)

    def _restore_rng_taps(self) -> None:
        for attr, original in self._installed_rng_patches:
            setattr(torch, attr, original)
        self._installed_rng_patches.clear()

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if int(trainer.global_rank) != 0:
            return
        print(f"[step1_dump] root={self._root}", flush=True)

        # 1. LoRA init state — every trainable parameter on the LightningModule.
        lora_state: dict[str, torch.Tensor] = {}
        for n, p in pl_module.named_parameters():
            if p.requires_grad:
                lora_state[n] = p.detach().to("cpu").clone()
        self._inputs["lora_init_state"] = lora_state
        print(f"[step1_dump] captured {len(lora_state)} LoRA tensors", flush=True)

        # 2. Method taps.
        pipe_VAE = pl_module.pipe_VAE
        pipe = pl_module.pipe

        original_encode_prompt = pipe_VAE.encode_prompt

        def wrapped_encode_prompt(text, *a, **kw):
            out = original_encode_prompt(text, *a, **kw)
            if "prompt_emb_context" not in self._upstream and isinstance(out, dict) and "context" in out:
                self._upstream["prompt_emb_context"] = _to_cpu(out["context"])
            return out

        self._wrap_method(pipe_VAE, "encode_prompt", wrapped_encode_prompt)

        original_encode_video = pipe_VAE.encode_video

        def wrapped_encode_video(video, *a, **kw):
            out = original_encode_video(video, *a, **kw)
            if "clean_latents" not in self._upstream:
                latents = out if isinstance(out, torch.Tensor) else out[0]
                self._upstream["clean_latents"] = _to_cpu(latents)
            return out

        self._wrap_method(pipe_VAE, "encode_video", wrapped_encode_video)

        original_encode_images = pipe_VAE.encode_images_adaptive

        def wrapped_encode_images(*a, **kw):
            out = original_encode_images(*a, **kw)
            if isinstance(out, dict):
                if "clip_feature" not in self._upstream and "clip_feature" in out:
                    self._upstream["clip_feature"] = _to_cpu(out["clip_feature"])
                if "y" not in self._upstream and "y" in out:
                    self._upstream["y"] = _to_cpu(out["y"])
            return out

        self._wrap_method(pipe_VAE, "encode_images_adaptive", wrapped_encode_images)

        original_add_noise = pipe.scheduler.add_noise

        def wrapped_add_noise(latents, noise, timestep, *a, **kw):
            out = original_add_noise(latents, noise, timestep, *a, **kw)
            if "noisy_latents" not in self._upstream:
                self._upstream["noisy_latents"] = _to_cpu(out)
                self._upstream["timestep_actual"] = _to_cpu(timestep if isinstance(timestep, torch.Tensor)
                                                            else torch.tensor(float(timestep)))
            return out

        self._wrap_method(pipe.scheduler, "add_noise", wrapped_add_noise)

        original_training_target = pipe.scheduler.training_target

        def wrapped_training_target(sample, noise, timestep, *a, **kw):
            out = original_training_target(sample, noise, timestep, *a, **kw)
            if "target" not in self._upstream:
                self._upstream["target"] = _to_cpu(out)
            return out

        self._wrap_method(pipe.scheduler, "training_target", wrapped_training_target)

        original_training_weight = pipe.scheduler.training_weight

        def wrapped_training_weight(timestep, *a, **kw):
            out = original_training_weight(timestep, *a, **kw)
            if "training_weight" not in self._upstream:
                self._upstream["training_weight"] = _to_cpu(out)
            return out

        self._wrap_method(pipe.scheduler, "training_weight", wrapped_training_weight)

        # The DiT — wrap its forward to capture the model output (pred).
        dit = pipe.denoising_model()
        original_dit_call = dit.__class__.__call__

        def wrapped_dit_call(module_self, *a, **kw):
            out = original_dit_call(module_self, *a, **kw)
            if module_self is dit and "pred" not in self._upstream:
                self._upstream["pred"] = _to_cpu(out if isinstance(out, torch.Tensor) else out[0])
            return out

        self._wrap_method(dit.__class__, "__call__", wrapped_dit_call)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch_idx
        if int(trainer.global_rank) != 0 or self._step_done:
            return

        # Capture the dataloader batch verbatim.
        self._inputs["text"] = batch["text"][0] if isinstance(batch["text"], list) else batch["text"]
        self._inputs["video"] = _to_cpu(batch["video"])
        self._inputs["first_ref_frames"] = [_to_cpu(t) for t in batch["first_ref_frames"]]
        self._inputs["random_ref_frame"] = _to_cpu(batch["random_ref_frame"][0])
        self._inputs["path"] = batch["path"][0] if isinstance(batch.get("path"), list) else batch.get("path")

        # Install one-shot RNG taps so the noise + timestep_id sampled inside
        # training_step are captured. Restored in on_train_batch_end.
        self._install_rng_taps()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        if int(trainer.global_rank) != 0 or self._step_done:
            return
        self._step_done = True

        # Capture loss (Lightning passes the training_step return value here).
        loss_val: Any = outputs
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_val = outputs["loss"]
        if isinstance(loss_val, torch.Tensor):
            self._upstream["loss"] = _to_cpu(loss_val)
        # If the unweighted loss is reconstructible, compute it post-hoc.
        if "training_weight" in self._upstream and isinstance(self._upstream["training_weight"], torch.Tensor):
            tw = float(self._upstream["training_weight"].item())
            if tw != 0.0 and isinstance(self._upstream.get("loss"), torch.Tensor):
                self._upstream["loss_unweighted"] = self._upstream["loss"] / tw

        # Fixture metadata
        self._inputs["seed"] = 42

        # Persist & restore RNG hooks before requesting stop.
        save_inputs(self._inputs, root=self._root)
        save_upstream(self._upstream, root=self._root)
        print(f"[step1_dump] wrote inputs.pt ({len(self._inputs)} keys) "
              f"and upstream_intermediates.pt ({len(self._upstream)} keys) under {self._root}", flush=True)

        self._restore_rng_taps()
        self._restore_method_patches()
        trainer.should_stop = True

# SVI CLIP Image Preprocessing Mismatch

**Severity:** High (algorithmic divergence; corrupts both training LoRA semantics and inference feature space).
**Scope:** Wan SVI I2V — both `fastvideo/train/methods/svi/svi.py::_clip_encode` (training) and `fastvideo/pipelines/stages/image_encoding.py::ImageEncodingStage` (inference, reused by `WanSVIImageToVideoPipeline`).
**Surfaced by:** Stage 6 single-step numerical parity gate (`fastvideo/tests/svi/test_step1_parity.py`).

## The bug

The SVI inference + training port feeds the CLIP image encoder a different set of pixels than upstream does. The released `svi-shot.safetensors` LoRA was trained against upstream's CLIP features, so FV training learns on the wrong feature space and FV inference applies the released LoRA to features it never saw.

### Upstream (correct reference)

`Stable-Video-Infinity/diffsynth/models/wan_video_image_encoder.py:864-880`:

```python
def encode_image(self, videos):
    size = (self.model.image_size,) * 2     # (224, 224)
    videos = torch.cat([
        F.interpolate(u, size=size, mode='bicubic', align_corners=False)
        for u in videos
    ])
    videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))  # [-1,1] → [0,1] then CLIP-Normalize
    out = self.model.visual(videos, use_31_block=True)                   # truncate at block 31
    return out
```

Key properties:
1. Bicubic **squash** straight to 224×224 — the original 832×480 aspect ratio is destroyed, no cropping, the full frame survives.
2. Manual `[-1, 1] → [0, 1] → Normalize` with OpenCLIP mean/std.
3. Returns block-31 hidden state.

### FV training before the fix

`fastvideo/train/methods/svi/svi.py::_clip_encode` (the pre-fix body):

```python
inputs = student.image_processor(images=image, return_tensors="pt").to(device)  # HF CLIPImageProcessor
outputs = student.image_encoder(**inputs)
return outputs.last_hidden_state.to(dtype)
```

`CLIPImageProcessor` defaults: resize shortest edge to 224, then **center-crop** 224×224. On 832×480 input:

- Upstream sees: squashed full frame (224×224, distorted aspect).
- FV training sees: cropped middle 480×480 region resized to 224×224 (aspect preserved, sides dropped).

The two CLIP encoders therefore see fundamentally different content.

### Inference

`fastvideo/pipelines/stages/image_encoding.py::ImageEncodingStage.forward` (lines 67-82 on `svi` branch) is the same `CLIPImageProcessor` code path. The released SVI LoRA expects upstream-style features; we feed it FV-style features. Visual quality survives because LoRAs are inherently robust to mild distribution shift, but it's not what the LoRA was trained for.

## Numerical evidence (step-1 parity)

With the upstream-aligned fixture loaded into FV and a matched LoRA init:

| `clip_feature` measure | Before fix | After fix |
|---|---|---|
| mean drift (`abs_mean(diff) / abs_mean(upstream)`) | 44.23% | 7.28% |
| `abs_max(diff)` | 8.94 | 5.81 |
| per-token cosine similarity (sampled tokens 0 / 128 / 256) | n/a | 0.9999 / 0.9994 / 0.9992 |

The remaining 7% drift after the fix is cross-implementation fp noise (HF `CLIPVisionModelWithProjection` 31-layer truncation vs diffsynth's `WanImageEncoder` 31-of-32 implementation — same weights, slightly different forward arithmetic).

Downstream effects of fixing the preprocessing alone:

- `pred` mean drift: 1.55% → **0.91%** (already under the 5% gate but cleaner).
- `loss` absolute drift unchanged (it's a derived scalar that amplifies any pred/target drift).

## Proposed fix

### Training side (`fastvideo/train/methods/svi/svi.py`)

Already in this branch on the `svi-train` working tree as part of the Stage 6 parity work. The new `_clip_encode` body:

```python
arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
t = F.interpolate(t, size=(224, 224), mode="bicubic", align_corners=False)
t = t.mul(0.5).add(0.5)
mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device, dtype=t.dtype).view(1, 3, 1, 1)
std  = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device, dtype=t.dtype).view(1, 3, 1, 1)
t = (t - mean) / std
outputs = student.image_encoder(pixel_values=t)
return outputs.last_hidden_state.to(dtype=dtype)
```

(`student.image_processor` is no longer read by `_clip_encode`. The PR can also drop the `image_processor` load from `WanSVIModel.init_preprocessors` if nothing else uses it.)

### Inference side (`fastvideo/pipelines/stages/image_encoding.py`)

Two reasonable options:

1. **SVI-only fix**: create a new stage `SVIImageEncodingStage(ImageEncodingStage)` that does the `F.interpolate` preprocessing and wire it into `WanSVIImageToVideoPipeline`. Stock Wan I2V keeps using `CLIPImageProcessor`. Minimal blast radius, but the rest of Wan I2V is still technically inconsistent with how its CLIP weights were originally fed at training time (Wan's own training presumably also used the same `F.interpolate` path that diffsynth exposes).
2. **Wan-wide fix**: change `ImageEncodingStage` itself to the `F.interpolate` path for Wan-family models, gated on the encoder config. Pulls in non-SVI Wan I2V too, so requires an SSIM regression check against existing reference outputs.

Recommendation: **start with option 1** for the SVI PR. Reuse the same helper as the training-side fix (put it in `fastvideo/pipelines/stages/image_encoding.py` or a shared utility) so training + inference share one preprocessing implementation. Re-validate end-to-end inference of the released `svi-shot.safetensors` after the swap — the new preprocessing should match what the LoRA was trained against, so output quality should be ≥ current.

## Verification plan

1. **Numerical** (training side): step-1 parity test stays green.
   ```
   CUDA_VISIBLE_DEVICES=0 python .agents/exploration/svi-training/dump_step1_upstream.py --output outputs/parity_fixture/step01
   CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29504 \
       -m pytest fastvideo/tests/svi/test_step1_parity.py -v -s
   ```
   `clip_feature` mean drift ≤ 10% and `pred` mean drift ≤ 5%.

2. **Visual** (inference side, post-fix): run the existing SVI inference smoke (`.agents/exploration/svi-training/smoke_round_trip.py` with the released `svi-shot.safetensors`) and compare frame 0 / mid / final SSIM against the pre-fix baseline output. With matched preprocessing the SSIM should hold or improve.

3. **Regression** (if pursuing option 2 above): run any existing SSIM regression suite that touches Wan I2V at 480P (`pytest fastvideo/tests/ssim/ -vs` if applicable) before/after.

## What this PR does NOT fix

- The remaining ~7% `clip_feature` drift comes from HF vs diffsynth implementations of the CLIP ViT-H/14 transformer (different layer norm placement / attention impl). Out of scope.
- The 1-2% UMT5 / VAE drift in `prompt_emb_context`, `clean_latents`, `y`, `target` is the same kind of cross-implementation fp noise and is not bug-grade.

## Related files for the PR

- `fastvideo/train/methods/svi/svi.py` (the fix, already on `svi-train`)
- `fastvideo/pipelines/stages/image_encoding.py` (new `SVIImageEncodingStage` to add)
- `fastvideo/pipelines/basic/wan/wan_svi_i2v_pipeline.py` (swap to the new stage)
- `fastvideo/tests/svi/test_step1_parity.py` (already exists; verifies the training-side fix)
- Optional: a small inference parity test for the SVI pipeline that asserts `clip_feature` after preprocessing matches a captured upstream fixture.

# LoRA `target_modules` substring match is too greedy with single letters

**Filed: 2026-05-20**  
**Subsystem: `fastvideo/train/utils/lora.py`**

## What happened

While porting Stable-Video-Infinity training, I copied upstream's
`target_modules: [q, k, v, o, ffn.0, ffn.2]` into a FastVideo training YAML.
Training succeeded but the saved LoRA had **960 tensors** (480 layers × 2)
instead of the upstream-matching 800 (400 layers × 2).

## Why

`fastvideo/train/utils/lora.py::_is_target_layer` matches by `any(target in
module_name for target in target_modules)` — plain Python substring `in`.
Upstream uses single-letter targets because their model's module names end
in `.q`, `.k`, `.v`, `.o`. FastVideo's internal names are
`blocks.<N>.attn1.to_q`, `attn2.add_k_proj`, etc. — and crucially, every
block name contains `blocks` which contains the letter `'o'`. So `'o' in
'blocks.0.attn1.to_q'` is **True**, and every linear layer in every block
gets LoRA-wrapped.

## How to apply

Use FV-internal layer names spelled out in full:

```yaml
target_modules:
  - to_q
  - to_k
  - to_v
  - to_out
  - ffn.fc_in
  - ffn.fc_out
```

Substring `to_q` only matches `…to_q`, never spurious bits of unrelated
layer names. Drop the single-letter style entirely — anything one or two
characters long is unsafe with the current matcher.

Long-term fix would be to make `_is_target_layer` do a word-boundary match
(suffix-after-dot, or anchored regex). Until that lands, treat the YAML
field as suffix-style and verify the LoRA layer count after the run
(`logger.info("Enabled LoRA training … on %d layers", …)`).

## How we caught it

`SaveLoRACallback` dumped 960 tensors instead of 800. Comparing patterns
revealed `attn2.add_k_proj`, `attn2.add_v_proj` were wrapped too, plus
`ffn.fc_in/fc_out` (because `'0'` matched any digit, including `blocks.0`).

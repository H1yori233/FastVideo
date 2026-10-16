# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 PUP real-weight logits and cached generation parity; scope: production_loader.

Requires one CUDA device with 24 GiB free and the staged official PUP checkpoint.
"""
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = Path(os.getenv("FLUX3_PUP_WEIGHTS_DIR", REPO_ROOT / "official_weights/flux3/pup"))
PARITY_SCOPE = "production_loader"
EXTENDED_OUTPUT = Path(os.getenv("FLUX3_PUP_OUTPUT_DIR", REPO_ROOT / "outputs/flux3/extended_pup"))


def test_official_pup_config_selects_native_architecture():
    from fastvideo.configs.models.encoders.flux3_pup import Flux3PromptUpsamplerConfig

    source = {"architectures": ["Qwen3_5ForConditionalGeneration"],
              "text_config": {"hidden_size": 32, "num_hidden_layers": 4},
              "vision_config": {"hidden_size": 16, "depth": 1}}
    config = Flux3PromptUpsamplerConfig()
    config.update_model_arch(source)
    assert config.architectures == ["Flux3PromptUpsampler"]
    assert config.hidden_size == 32 and config.text_config.num_hidden_layers == 4
    assert config.vision_config.hidden_size == 16 and config.vision_config.depth == 1
    assert source["architectures"] == ["Qwen3_5ForConditionalGeneration"]


def _tiny_pup_config():
    text = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16, pad_token_id=0,
                linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=1,
                linear_num_value_heads=2, rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0, "mrope_section": [3, 3, 2], "mrope_interleaved": True})
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                  out_hidden_size=32, patch_size=2, temporal_patch_size=2, num_position_embeddings=16)
    return dict(text_config=text, vision_config=vision, pad_token_id=0, eos_token_id=63)


def test_flux3_pup_rejects_duplicate_checkpoint_weights():
    from fastvideo.configs.models.encoders.flux3_pup import Flux3PromptUpsamplerConfig
    from fastvideo.models.encoders.flux3_pup.model import Flux3PromptUpsampler

    config = Flux3PromptUpsamplerConfig()
    config.update_model_arch(_tiny_pup_config())
    model = Flux3PromptUpsampler(config)
    parameters = dict(model.named_parameters())
    weights = [(name, torch.zeros_like(parameter)) for name, parameter in parameters.items()]
    assert model.load_weights(weights) == set(parameters)
    name, value = weights[0]
    with pytest.raises(ValueError, match="Duplicate PUP weight"):
        model.load_weights(weights + [(name, torch.ones_like(value))])
    assert torch.count_nonzero(parameters[name]) == 0


def test_flux3_pup_padded_generation_positions(monkeypatch):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from fastvideo.configs.models.encoders.flux3_pup import Flux3PromptUpsamplerConfig
    from fastvideo.models.encoders.flux3_pup.model import Flux3PromptUpsampler

    torch.manual_seed(817)
    official_config = Qwen3_5Config(**_tiny_pup_config())
    official_config._attn_implementation = "sdpa"
    reference = Qwen3_5ForConditionalGeneration(official_config).eval()
    native_config = Flux3PromptUpsamplerConfig()
    native_config.update_model_arch(official_config.to_dict())
    native = Flux3PromptUpsampler(native_config).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    input_ids = torch.tensor([[0, 0, 17, 21, 8]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1]])
    expected_positions = reference._prepare_position_ids_for_generation(input_ids, {"attention_mask": attention_mask})
    observed_positions = []
    original_forward = native.forward

    def capture_positions(*args, **kwargs):
        observed_positions.append(kwargs["position_ids"].clone())
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(native, "forward", capture_positions)
    with torch.inference_mode():
        expected = reference.generate(input_ids, attention_mask=attention_mask, max_new_tokens=3, do_sample=False)
        actual = native.generate(input_ids, attention_mask=attention_mask, max_new_tokens=3, stop_token_ids=[63])
    torch.testing.assert_close(observed_positions[0], expected_positions, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class PresencePenalty:
    def __init__(self, prompt_length, penalty):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        for batch, tokens in enumerate(input_ids[:, self.prompt_length:]):
            scores[batch, tokens.unique()] -= self.penalty
        return scores


@pytest.mark.parametrize("n_images,boxes", [
    (0, None),
    (2, None),
    (0, [{"id": "lake", "bbox": [0.2, 0.1, 0.8, 0.9], "desc": "a quiet lake"}]),
])
def test_flux3_pup_parity(n_images, boxes):
    if not torch.cuda.is_available() or not (OFFICIAL / "model.safetensors").exists():
        pytest.skip("CUDA and local FLUX 3 PUP weights required")
    from transformers import AutoProcessor, LogitsProcessorList, Qwen3_5ForConditionalGeneration
    from fastvideo.configs.models.encoders.flux3_pup import Flux3PromptUpsamplerConfig
    from fastvideo.models.encoders.flux3_pup.model import Flux3PromptUpsampler
    from fastvideo.models.loader.component_loader import TextEncoderLoader
    from fastvideo.models.encoders.flux3_pup import bbox

    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(OFFICIAL, local_files_only=True)
    prompts = json.loads((OFFICIAL / "system_prompts.json").read_text())
    images = [Image.new("RGB", (64, 96), color) for color in ("red", "blue")][:n_images]
    request, locked = Flux3PromptUpsampler.prepare_inputs(
        "A quiet lake at sunset.", images, boxes, processor, prompts)
    request = {key: value.to(device) for key, value in request.items()}
    penalty = 0.0 if n_images else 1.5
    reference = Qwen3_5ForConditionalGeneration.from_pretrained(
        OFFICIAL, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa", local_files_only=True).eval()
    with torch.inference_mode():
        expected_logits = reference(**request, logits_to_keep=1, use_cache=False).logits.cpu()
        expected_tokens = reference.generate(
            **request, do_sample=False, max_new_tokens=8,
            eos_token_id=[processor.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")],
            logits_processor=LogitsProcessorList([PresencePenalty(request["input_ids"].shape[1], penalty)]),
        ).cpu()
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    config = Flux3PromptUpsamplerConfig()
    config.update_model_arch(json.loads((OFFICIAL / "config.json").read_text()))
    args = SimpleNamespace(
        text_encoder_cpu_offload=False, pin_cpu_memory=False, override_text_encoder_quant=None,
        override_text_encoder_safetensors=None, disable_offload_on_unified_memory=lambda *a, **k: False)
    native = TextEncoderLoader().load_model(str(OFFICIAL), config, device, args, dtype="bf16")
    assert isinstance(native, Flux3PromptUpsampler)
    with torch.inference_mode():
        actual_logits = native(**request, logits_to_keep=1, use_cache=False).logits.cpu()
        actual_tokens = native.generate(**request, max_new_tokens=8, presence_penalty=penalty,
                                       stop_token_ids=[processor.tokenizer.convert_tokens_to_ids(t)
                                                       for t in ("<|im_end|>", "<|endoftext|>")]).cpu()
    drift = (actual_logits.float() - expected_logits.float()).abs()
    print(f"PUP images={n_images} logits max={drift.max().item():.6g} mean={drift.mean().item():.6g}")
    torch.testing.assert_close(actual_logits, expected_logits, atol=0, rtol=0)
    torch.testing.assert_close(actual_tokens, expected_tokens, atol=0, rtol=0)
    expected_caption = processor.decode(expected_tokens[0, request["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip()
    if locked:
        expected_caption = bbox.reconcile(expected_caption, locked)
    assert native.upsample("A quiet lake at sunset.", images=images, boxes=boxes, max_new_tokens=8,
                           processor=processor, system_prompts=prompts) == expected_caption
    del native
    gc.collect()
    torch.cuda.empty_cache()


def _extended_cases(processor, official):
    names = [
        "mixkit-motorboat-on-a-large-lake-with-turquoise-blue-waters-4996.png",
        "mixkit-a-rancher-riding-a-horse-at-sunset-1143.png",
        "mixkit-birds-flying-low-over-a-lake-in-a-meadow-41417.png",
    ]
    images = []
    for name in names:
        path = REPO_ROOT / "assets/images" / name
        with Image.open(path) as image:
            images.append(image.copy())
    mixed = [images[0], images[1].crop((216, 0, 616, 480)), images[2].crop((176, 0, 656, 480))]
    notes = []
    while len(processor.tokenizer.encode(" ".join(notes), add_special_tokens=False)) < 4096:
        notes.append(f"Scene note {len(notes):03d}: the small wooden cabin faces a still alpine lake; "
                     "pine trees frame the left shore, snow covers distant peaks, "
                     "and warm morning light reflects on water.")
    long_prompt = ("Create one coherent photographic scene from these repeated reference notes, "
                   "without displaying any writing. Summarize the shared details rather than repeating the notes. "
                   + " ".join(notes))
    locked = [{"id": "balloon", "bbox": [0.12, 0.10, 0.55, 0.38], "desc": "a red balloon"}]
    edit_boxes = [{"id": "boat", "from": "ref_image_0", "src_bbox": [0.2, 0.2, 0.7, 0.6],
                   "tgt_bbox": [0.25, 0.5, 0.75, 0.9], "desc": "the same motorboat"}]
    multilingual = dict(name="multilingual_t2i", prompt="雨后的京都小巷，暖色灯笼映在石板路上，一只白猫坐在木门旁。",
                        images=[], boxes=None)
    return [
        multilingual,
        dict(name="long_context_4k", prompt=long_prompt, images=[], boxes=None),
        dict(name="png_edit_spanish", prompt="Cambia la escena a la hora azul y añade luces cálidas en el barco, "
             "manteniendo su forma y la composición del lago.", images=[images[0]], boxes=None),
        dict(name="three_mixed_aspect_references", prompt="Create a cohesive illustrated travel poster using the "
             "lake from the first image, the horse and rider from the second, and the birds from the third. "
             "Keep all three subjects recognizable and do not add text.", images=mixed, boxes=None),
        dict(name="locked_bbox_add_unlocked_object", prompt="A red balloon on the left and a blue ceramic cup on "
             "the right, both above a wooden desk. Preserve the balloon's locked location and add a box for the cup.",
             images=[], boxes=locked),
        dict(name="embedded_i2i_bbox", prompt="Move the motorboat to the right side while preserving the water "
             "and mountain landscape. " + json.dumps(official.bbox.to_canvas(edit_boxes)),
             images=[images[0]], boxes=None),
        {**multilingual, "name": "multilingual_t2i_repeat"},
    ], names


def _official_pup_request(case, processor, prompts, official, monkeypatch):
    class CaptureGeneration:
        def generate(self, requests, params, use_tqdm):
            self.request, self.params = requests[0], params
            return [SimpleNamespace(outputs=[SimpleNamespace(text="captured request")])]

    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
        upsampler = object.__new__(official.Upsampler)
        upsampler.processor, upsampler.tokenizer = processor, processor.tokenizer
        upsampler.prompts, upsampler.llm = prompts, CaptureGeneration()
        upsampler(case["prompt"], case["images"], case["boxes"])
        request, params = upsampler.llm.request, upsampler.llm.params
    if case["images"]:
        inputs = dict(processor(text=[request["prompt"]], images=request["multi_modal_data"]["image"],
                                return_tensors="pt", add_special_tokens=False, return_mm_token_type_ids=True))
    else:
        ids = torch.tensor([request["prompt_token_ids"]])
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    return inputs, params, official.bbox.plan(case["prompt"], case["boxes"], len(case["images"]))[2]


def _capture_generation(model, request, generate_kwargs):
    selected, count = {}, 0
    last = None

    def capture(module, inputs, output):
        nonlocal count, last
        last = output.logits[:, -1].detach().cpu()
        if count in {0, 1, 31, 127, 255, 383}:
            selected[count] = last
        count += 1

    handle = model.register_forward_hook(capture)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            tokens = model.generate(**request, **generate_kwargs).cpu()
        torch.cuda.synchronize()
    finally:
        handle.remove()
    assert count == tokens.shape[1] - request["input_ids"].shape[1]
    selected[count - 1] = last
    return tokens, selected, time.perf_counter() - started, torch.cuda.max_memory_allocated()


@pytest.mark.skipif(os.getenv("FLUX3_PUP_EXTENDED") != "1",
                    reason="Set FLUX3_PUP_EXTENDED=1 for extended real-weight parity")
def test_flux3_pup_extended_generation_matrix(monkeypatch):
    """Seven requests, up to 512 new tokens each; EOS versus cap is recorded per case."""
    if not torch.cuda.is_available() or not (OFFICIAL / "model.safetensors").is_file():
        pytest.skip("CUDA and local FLUX 3 PUP weights required")
    from transformers import AutoProcessor, LogitsProcessorList, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from fastvideo.configs.models.encoders.flux3_pup import Flux3PromptUpsamplerConfig
    from fastvideo.models.encoders.flux3_pup import bbox
    from fastvideo.models.encoders.flux3_pup.model import Flux3PromptUpsampler
    from fastvideo.models.loader.component_loader import TextEncoderLoader

    for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                 "fused_recurrent_gated_delta_rule", "FusedRMSNormGated"):
        monkeypatch.setattr(modeling_qwen3_5, name, None)
    source = Path(os.getenv("FLUX3_OFFICIAL_REF_DIR", REPO_ROOT / "flux-3-image-klein"))
    monkeypatch.syspath_prepend(str(source / "src"))
    official = importlib.import_module("flux3.pup")
    processor = AutoProcessor.from_pretrained(OFFICIAL, local_files_only=True)
    prompts = json.loads((OFFICIAL / "system_prompts.json").read_text())
    cases, assets = _extended_cases(processor, official)
    prepared = [_official_pup_request(case, processor, prompts, official, monkeypatch) for case in cases]
    device = torch.device("cuda:0")
    limit = 512
    metrics = {"scope": PARITY_SCOPE, "dtype": "bfloat16", "attention": "sdpa", "generation_cap": limit,
               "source_request_policy": "official flux3.pup.Upsampler.__call__", "source_assets": assets,
               "cases": [], "status": "running"}
    EXTENDED_OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics_path = EXTENDED_OUTPUT / "metrics.json"

    def save_metrics():
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    save_metrics()
    reference = Qwen3_5ForConditionalGeneration.from_pretrained(
        OFFICIAL, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa", local_files_only=True).eval()
    expected = []
    try:
        for case, (inputs, params, locked) in zip(cases, prepared, strict=True):
            request = {key: value.to(device) for key, value in inputs.items()}
            length = inputs["input_ids"].shape[1]
            assert length + limit <= 8192
            if case["name"] == "long_context_4k":
                assert length >= 4096
            tokens, logits, duration, peak = _capture_generation(reference, request, dict(
                do_sample=False, use_cache=True, max_new_tokens=limit, eos_token_id=params["stop_token_ids"],
                logits_processor=LogitsProcessorList([PresencePenalty(length, params["presence_penalty"])])))
            caption = processor.decode(tokens[0, length:], skip_special_tokens=True).strip()
            caption = official.bbox.reconcile(caption, locked) if locked else caption
            expected.append((tokens, logits, caption))
            generated = tokens.shape[1] - length
            item = {"name": case["name"], "input_tokens": length,
                    "image_sizes": [list(im.size) for im in case["images"]], "generated_tokens": generated,
                    "termination": "eos" if int(tokens[0, -1]) in params["stop_token_ids"] else "512_token_cap",
                    "presence_penalty": params["presence_penalty"], "locked_rows": locked,
                    "reference_seconds": duration, "reference_peak_gib": peak / 1024**3,
                    "caption": caption, "generated_token_ids": tokens[0, length:].tolist(),
                    "final_token_id": int(tokens[0, -1]), "stop_token_ids": params["stop_token_ids"],
                    "image_grid_thw": inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None,
                    "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest()}
            metrics["cases"].append(item)
            save_metrics()
            print(f"reference {case['name']}: input={length} generated={generated} stop={item['termination']} "
                  f"seconds={duration:.2f}", flush=True)
    finally:
        del reference
        gc.collect()
        torch.cuda.empty_cache()

    native_config = Flux3PromptUpsamplerConfig()
    native_config.update_model_arch(json.loads((OFFICIAL / "config.json").read_text()))
    args = SimpleNamespace(text_encoder_cpu_offload=False, pin_cpu_memory=False, override_text_encoder_quant=None,
                           override_text_encoder_safetensors=None,
                           disable_offload_on_unified_memory=lambda *a, **k: False)
    native = TextEncoderLoader().load_model(str(OFFICIAL), native_config, device, args,
                                           dtype="bf16")
    try:
        for index, (case, (inputs, params, locked), (target_tokens, target_logits, target_caption)) in enumerate(
                zip(cases, prepared, expected, strict=True)):
            native_inputs, native_locked = native.prepare_inputs(case["prompt"], case["images"], case["boxes"],
                                                                 processor, prompts)
            assert native_locked == locked
            assert native_inputs.keys() == inputs.keys()
            for key in inputs:
                torch.testing.assert_close(native_inputs[key], inputs[key], atol=0, rtol=0)
            request = {key: value.to(device) for key, value in native_inputs.items()}
            tokens, logits, duration, peak = _capture_generation(native, request, dict(
                max_new_tokens=limit, stop_token_ids=params["stop_token_ids"],
                presence_penalty=params["presence_penalty"]))
            item = metrics["cases"][index]
            item.update(native_seconds=duration, native_peak_gib=peak / 1024**3,
                        token_ids_exact=torch.equal(tokens, target_tokens), selected_logits={})
            for step in sorted(logits.keys() & target_logits.keys()):
                drift = (logits[step].float() - target_logits[step].float()).abs()
                item["selected_logits"][step] = {"max_abs": drift.max().item(), "mean_abs": drift.mean().item(),
                                                  "exact": torch.equal(logits[step], target_logits[step])}
            caption = processor.decode(tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            caption = bbox.reconcile(caption, native_locked) if native_locked else caption
            item["caption_exact"] = caption == target_caption
            save_metrics()
            torch.testing.assert_close(tokens, target_tokens, atol=0, rtol=0)
            assert logits.keys() == target_logits.keys()
            for step in logits:
                torch.testing.assert_close(logits[step], target_logits[step], atol=0, rtol=0)
            assert caption == target_caption
            if locked:
                _, rows = bbox.split(caption)
                for original in locked:
                    restored = next(row for row in rows if row["id"] == original["id"])
                    for key, value in original.items():
                        if key != "desc":
                            assert restored[key] == value
            if index == len(cases) - 1:
                torch.testing.assert_close(tokens, expected[0][0], atol=0, rtol=0)
                for step in logits:
                    torch.testing.assert_close(logits[step], expected[0][1][step], atol=0, rtol=0)
                metrics["repeated_request_isolated"] = True
            item["status"] = "passed"
            save_metrics()
            print(f"native {case['name']}: {tokens.shape[1] - inputs['input_ids'].shape[1]} tokens exact; "
                  f"logit_steps={sorted(logits)} seconds={duration:.2f}", flush=True)
    finally:
        del native
        gc.collect()
        torch.cuda.empty_cache()
    assert any(case["termination"] == "eos" for case in metrics["cases"])
    assert any(case["generated_tokens"] >= 256 for case in metrics["cases"])
    metrics["status"] = "passed"
    save_metrics()

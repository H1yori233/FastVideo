# SPDX-License-Identifier: Apache-2.0
"""Real FLUX.3 pipeline parity; scope: production_loader.

Run FLUX3_EXTENDED_PARITY=1 for the 20-case matrix on one 32 GiB CUDA GPU.
FLUX3_PARITY_CASES selects named cases, including two additional photo cases.
Both pipelines run sequentially; every Euler state and decoded pixel must match.
PUP uses official BFL policy with Transformers Qwen3.5 instead of a vLLM server.
"""

import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(os.getenv("FLUX3_OFFICIAL_REF_DIR", ROOT / "flux-3-image-klein"))
WEIGHTS = Path(os.getenv("FLUX3_OFFICIAL_WEIGHTS_DIR", ROOT / "official_weights/flux3/flux-3-image-klein"))
PUP_WEIGHTS = Path(os.getenv("FLUX3_PUP_WEIGHTS_DIR", ROOT / "official_weights/flux3/pup"))
MODEL = Path(os.getenv("FLUX3_MODEL_DIR", ROOT / "official_weights/flux3/flux-3-image-klein"))
ARTIFACTS = Path(os.getenv("FLUX3_PARITY_OUTPUT_DIR", ROOT / "outputs/flux3/official_weights/pipeline"))
PARITY_SCOPE = "production_loader"
EXTENDED = os.getenv("FLUX3_EXTENDED_PARITY", "0") == "1"
DEFAULT_CASES = ("t2i", "single_auto", "multi", "bbox", "pup_t2i", "pup_i2i_bbox")
EXTENDED_CASES = (
    "seed0_landscape", "seed42_portrait", "square512_16steps", "square1024_16steps",
    "one_step_alpha2", "long_multilingual", "single_fixed", "three_references", "ten_references",
    "rgba_exif_path", "multiple_boxes", "finished_caption", "pup_full_t2i", "pup_full_multi_bbox",
)
PHOTO_CASES = ("photo_single_512", "photo_multi_pup")
ALL_CASES = DEFAULT_CASES + EXTENDED_CASES + PHOTO_CASES
CASE_FILTER = tuple(name.strip() for name in os.getenv("FLUX3_PARITY_CASES", "").split(",") if name.strip())
if unknown := set(CASE_FILTER) - set(ALL_CASES):
    raise ValueError(f"Unknown FLUX3_PARITY_CASES: {sorted(unknown)}")
CASE_NAMES = (tuple(name for name in ALL_CASES if name in CASE_FILTER) if CASE_FILTER
              else DEFAULT_CASES + EXTENDED_CASES if EXTENDED else DEFAULT_CASES)


def _image(width, height, phase=0):
    y, x = np.indices((height, width))
    pixels = np.stack(((x * 7 + phase) % 256, (y * 11 + phase) % 256,
                       ((x + y) * 3 + phase) % 256), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels)


def _cases():
    prompt = "A red fox resting beside a quiet lake."
    boxes = [{"id": "fox", "bbox": [0.15, 0.25, 0.75, 0.85], "desc": "a red fox"}]
    base = {"prompt": prompt, "references": [], "width": 256, "height": 256,
            "boxes": None, "auto_size": False, "upsample": False, "seed": 123,
            "steps": 4, "alpha": 6.93, "max_new_tokens": 16}
    cases = {
        "t2i": dict(base),
        "single_auto": base | {"references": [_image(247, 183)], "auto_size": True},
        "multi": base | {"references": [_image(97, 81), _image(113, 65, 37)], "height": 384},
        "bbox": base | {"boxes": boxes},
        "pup_t2i": base | {"upsample": True},
        "pup_i2i_bbox": base | {"references": [_image(64, 96)], "boxes": boxes, "upsample": True},
    }
    if all(name in DEFAULT_CASES for name in CASE_NAMES):
        return {name: cases[name] for name in CASE_NAMES}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rgba = _image(111, 73).convert("RGBA")
    rgba.putalpha(Image.fromarray(np.indices((73, 111))[1].astype(np.uint8) * 2))
    exif = rgba.getexif()
    exif[274] = 6
    path = ARTIFACTS / "reference_rgba_exif.png"
    if "rgba_exif_path" in CASE_NAMES:
        rgba.save(path, exif=exif)
    extra_boxes = boxes + [{"id": "boat", "bbox": [0.05, 0.65, 0.3, 0.95], "desc": "a small blue boat"}]
    finished = ('A red <fox> rests beside a lake. '
                '[{"id": "fox", "bbox": [150, 250, 750, 850], "desc": "a red fox"}]')
    cases.update({
        "seed0_landscape": base | {"seed": 0, "width": 384, "alpha": 2.0},
        "seed42_portrait": base | {"seed": 42, "height": 512},
        "square512_16steps": base | {"width": 512, "height": 512, "steps": 16, "seed": 2026},
        "square1024_16steps": base | {"width": 1024, "height": 1024, "steps": 16, "seed": 7},
        "one_step_alpha2": base | {"steps": 1, "alpha": 2.0, "seed": 99},
        "long_multilingual": base | {
            "prompt": "清晨的湖边，一只红狐狸坐在石头上。雪山倒映在水面。 " * 70
                      + "Un phare ancien au bord du lac. 夜空に星が輝く。 Detailed watercolor, soft light.",
            "seed": 17},
        "single_fixed": base | {"references": [_image(247, 183)], "height": 384},
        "three_references": base | {"references": [_image(97 + i * 16, 81, i * 37) for i in range(3)]},
        "ten_references": base | {"references": [_image(64 + i * 3, 64 + i * 2, i * 23) for i in range(10)]},
        "rgba_exif_path": base | {"references": [str(path)], "auto_size": True, "seed": 0},
        "multiple_boxes": base | {"boxes": extra_boxes, "width": 384},
        "finished_caption": base | {"prompt": finished, "upsample": True},
        "pup_full_t2i": base | {"prompt": "An old lighthouse on a rocky coast at sunrise.",
                                 "upsample": True, "max_new_tokens": 3584, "seed": None},
        "pup_full_multi_bbox": base | {
            "prompt": "Combine the colored patterns from both references into a single abstract poster.",
            "references": [_image(96, 64), _image(64, 96, 83)], "upsample": True,
            "max_new_tokens": 3584,
            "boxes": [{"id": "pattern", "from": "ref_image_0", "src_bbox": [0, 0, 1, 1],
                       "tgt_bbox": [0.1, 0.1, 0.9, 0.9], "desc": "the colored pattern"}]},
    })
    photo_boat = str(ROOT / "assets/images/mixkit-motorboat-on-a-large-lake-with-turquoise-blue-waters-4996.png")
    photo_horse = str(ROOT / "assets/images/mixkit-a-rancher-riding-a-horse-at-sunset-1143.png")
    cases.update({
        "photo_single_512": base | {
            "prompt": "Change the boat to bright red while preserving the lake and surrounding mountains.",
            "references": [photo_boat], "width": 512, "height": 512, "steps": 16, "seed": 41},
        "photo_multi_pup": base | {
            "prompt": "Place the horse and rider from the second reference beside the lake in the first reference.",
            "references": [photo_boat, photo_horse], "width": 512, "height": 384, "steps": 8,
            "upsample": True, "max_new_tokens": 3584, "seed": 31},
    })
    return {name: cases[name] for name in CASE_NAMES}


def _reference_images(case):
    images = []
    for item in case["references"]:
        if isinstance(item, str | Path):
            with Image.open(item) as image:
                images.append(image.copy())
        else:
            images.append(item.copy())
    return images


class _PresencePenalty:
    def __init__(self, prompt_length, penalty):
        self.prompt_length, self.penalty = prompt_length, penalty

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        for batch, tokens in enumerate(input_ids[:, self.prompt_length:]):
            scores[batch, tokens.unique()] -= self.penalty
        return scores


def _generation_record(tokens, prompt_length, stop_ids):
    generated = tokens[0, prompt_length:].detach().cpu().tolist()
    return {"tokens": generated, "generated_tokens": len(generated),
            "ended_with_stop": bool(generated and generated[-1] in stop_ids)}


def _official_captions(cases):
    from transformers import AutoProcessor, LogitsProcessorList, Qwen3_5ForConditionalGeneration

    pup = importlib.import_module("flux3.pup")
    bbox = importlib.import_module("flux3.bbox")
    processor = AutoProcessor.from_pretrained(PUP_WEIGHTS, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        PUP_WEIGHTS, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
        local_files_only=True).eval()
    generation = {}
    active = {}

    class ReferenceLLM:
        def generate(self, requests, params, use_tqdm):
            request = requests[0]
            if "multi_modal_data" in request:
                inputs = processor(text=[request["prompt"]], images=request["multi_modal_data"]["image"],
                                   return_tensors="pt", add_special_tokens=False)
            else:
                ids = torch.tensor([request["prompt_token_ids"]])
                inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
            assert params.temperature == 0.0
            prompt_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                tokens = model.generate(
                    **inputs, do_sample=False, max_new_tokens=active["max_new_tokens"],
                    eos_token_id=params.stop_token_ids,
                    logits_processor=LogitsProcessorList([_PresencePenalty(prompt_length, params.presence_penalty)]),
                )
            generation[active["name"]] = _generation_record(tokens, prompt_length, params.stop_token_ids)
            caption = processor.decode(tokens[0, prompt_length:], skip_special_tokens=True).strip()
            return [SimpleNamespace(outputs=[SimpleNamespace(text=caption)])]

    upsampler = object.__new__(pup.Upsampler)
    upsampler.processor, upsampler.tokenizer = processor, processor.tokenizer
    upsampler.prompts = json.loads((PUP_WEIGHTS / "system_prompts.json").read_text())
    upsampler.llm = ReferenceLLM()
    captions = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))
        for name, case in cases.items():
            finished = not case["boxes"] and bbox.is_finished(case["prompt"])
            if case["upsample"] and not finished:
                active.update(name=name, max_new_tokens=case["max_new_tokens"])
                captions[name] = upsampler(case["prompt"], _reference_images(case), case["boxes"])
                assert captions[name]
                print(f"official PUP/{name}: {generation[name]['generated_tokens']} tokens, "
                      f"stop={generation[name]['ended_with_stop']}", flush=True)
    del model, upsampler
    gc.collect()
    torch.cuda.empty_cache()
    return captions, generation


def _capture_denoise(original, captured):
    def capture(dit, packed, schedule):
        states = []

        def record_forward(value, **kwargs):
            states.append(value.data["image"].detach().cpu().clone())
            return dit(value, **kwargs)

        result = original(record_forward, packed, schedule)
        states.append(result.detach().cpu().clone())
        captured.update(latents=states[-1], trajectory=states)
        return result

    return capture


def _save_image(name, side, pixels):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(ARTIFACTS / f"{name}_{side}.png")


def _official_outputs(cases, captions, generation):
    official = importlib.import_module("flux3.pipeline")
    bbox = importlib.import_module("flux3.bbox")
    pipeline = official.Pipeline({"distilled": str(WEIGHTS)}, str(WEIGHTS / "text_encoder"),
                                 str(WEIGHTS / "ae.safetensors"), device="cuda:0")
    outputs, captured = {}, {}
    try:
        with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
            patch.setattr(official, "denoise_flow_map", _capture_denoise(official.denoise_flow_map, captured))
            for name, case in cases.items():
                caption = captions.get(name, case["prompt"])
                if case["boxes"] and not case["upsample"]:
                    caption = f'{caption.rstrip()} {json.dumps(bbox.to_canvas(case["boxes"]), ensure_ascii=False)}'
                references, crop = [], None
                images = _reference_images(case)
                if len(images) > 1:
                    references = [official.prepare_multi_reference(image) for image in images]
                elif images:
                    reference, crop = official.prepare_reference(images[0])
                    references = [reference]
                width, height = (official.default_size(references) if case["auto_size"]
                                 else (case["width"], case["height"]))
                seed = case["seed"]
                if seed is None:
                    seed = int(hashlib.sha256(caption.encode()).hexdigest(), 16) % 2**32
                image = pipeline.generate("distilled", caption, references, width, height, case["steps"], None,
                                          seed, alpha=case["alpha"])
                if crop and case["auto_size"]:
                    image = image.crop(crop)
                outputs[name] = {**captured, "pixels": np.asarray(image).copy(), "caption": caption,
                                 "generation": generation.get(name), "resolved_seed": seed}
                _save_image(name, "official", outputs[name]["pixels"])
                print(f"official/{name}: pixels={image.size} states={len(captured['trajectory'])}", flush=True)
    finally:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    return outputs


def _native_outputs(cases):
    from fastvideo.configs.pipelines.flux3 import Flux3PipelineConfig
    from fastvideo.distributed import cleanup_dist_env_and_memory
    from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
    from fastvideo.models.encoders.flux3_pup.model import Flux3PromptUpsampler
    from fastvideo.pipelines.basic.flux3 import stages
    from fastvideo.pipelines.basic.flux3.flux3_pipeline import Flux3Pipeline
    from fastvideo.pipelines.lazy_module import is_lazy_module
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29689")
    args = FastVideoArgs(model_path=str(MODEL), pipeline_config=Flux3PipelineConfig(), workload_type=WorkloadType.T2I,
                         num_gpus=1, sp_size=1, tp_size=1, hsdp_shard_dim=1, pin_cpu_memory=False,
                         lazy_module_load=True, use_fsdp_inference=False, dit_layerwise_offload=False,
                         dit_cpu_offload=False, vae_cpu_offload=False, text_encoder_cpu_offload=False)
    pipeline = Flux3Pipeline(str(MODEL), args)
    outputs, captured, generation = {}, {}, {}
    repeat_case = "t2i" if "t2i" in cases else next(iter(cases))
    repeat_name = f"repeat_{repeat_case}"
    original_generate = Flux3PromptUpsampler.generate

    def capture_generate(model, input_ids, **kwargs):
        tokens = original_generate(model, input_ids, **kwargs)
        generation.update(_generation_record(tokens, input_ids.shape[1], kwargs["stop_token_ids"]))
        return tokens

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(stages, "denoise_flow_map", _capture_denoise(stages.denoise_flow_map, captured))
            patch.setattr(Flux3PromptUpsampler, "generate", capture_generate)
            for name in (*cases, repeat_name):
                case = cases[repeat_case if name == repeat_name else name]
                generation.clear()
                batch = ForwardBatch(data_type="image", prompt=case["prompt"], references=case["references"],
                                     height=case["height"], width=case["width"], num_frames=1, seed=case["seed"],
                                     num_inference_steps=case["steps"], guidance_scale=1.0, save_video=False,
                                     flux3_boxes=case["boxes"], flux3_alpha=case["alpha"],
                                     flux3_auto_size=case["auto_size"], flux3_upsample_prompt=case["upsample"],
                                     flux3_pup_max_new_tokens=case["max_new_tokens"])
                result = pipeline.forward(batch, args)
                pixels = (result.output[0, :, 0].float() * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                outputs[name] = {**captured, "pixels": pixels, "caption": result.prompt,
                                 "generation": dict(generation) if generation else None, "resolved_seed": result.seed}
                torch.testing.assert_close(result.latents.cpu(), captured["latents"], atol=0, rtol=0)
                assert "_flux3_runtime" not in result.extra
                for component in ("transformer", "vae", "text_encoder", "prompt_upsampler"):
                    module = pipeline.modules[component]
                    assert is_lazy_module(module), component
                    assert not module.is_materialized, f"{component} remained resident after {name}"
                _save_image(name, "native", pixels)
                print(f"native/{name}: pixels={pixels.shape} states={len(captured['trajectory'])}", flush=True)
        for before, after in zip(outputs[repeat_case]["trajectory"], outputs[repeat_name]["trajectory"], strict=True):
            torch.testing.assert_close(after, before, atol=0, rtol=0)
        np.testing.assert_array_equal(outputs[repeat_name]["pixels"], outputs[repeat_case]["pixels"])
        assert outputs[repeat_name]["caption"] == outputs[repeat_case]["caption"]
    finally:
        pipeline.close()
        del pipeline
        cleanup_dist_env_and_memory()
    return outputs


def _tensor_metrics(expected, actual):
    delta = (actual.float() - expected.float()).abs()
    return {"exact": torch.equal(expected, actual), "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item()}


def _report(cases, expected, actual):
    import transformers

    report = {"scope": PARITY_SCOPE, "extended": EXTENDED, "selected_cases": list(CASE_NAMES), "atol": 0, "rtol": 0,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "transformers": transformers.__version__, "cuda": torch.version.cuda,
              "official_source": str(SOURCE), "official_weights": str(WEIGHTS),
              "native_model_path": str(MODEL), "repeat_request_exact": True,
              "lazy_release_verified": True, "cases": {}}
    for name, case in cases.items():
        left, right = expected[name], actual[name]
        states = [_tensor_metrics(a, b) for a, b in zip(left["trajectory"], right["trajectory"], strict=True)]
        pixels_exact = np.array_equal(left["pixels"], right["pixels"])
        generation_exact = left["generation"] == right["generation"]
        entry = {key: value for key, value in case.items() if key != "references"}
        entry.update(references=len(case["references"]),
                     reference_paths=[str(item) for item in case["references"] if isinstance(item, str | Path)],
                     resolved_seed=right["resolved_seed"],
                     seed_exact=left["resolved_seed"] == right["resolved_seed"], output_shape=list(right["pixels"].shape),
                     latent_shape=list(right["latents"].shape), states=states,
                     latents=_tensor_metrics(left["latents"], right["latents"]),
                     caption_exact=left["caption"] == right["caption"], caption=right["caption"],
                     generation_exact=generation_exact, official_generation=left["generation"],
                     native_generation=right["generation"], pixels_exact=pixels_exact,
                     official_pixel_sha256=hashlib.sha256(left["pixels"].tobytes()).hexdigest(),
                     native_pixel_sha256=hashlib.sha256(right["pixels"].tobytes()).hexdigest())
        entry["passed"] = all(state["exact"] for state in states) and pixels_exact and entry["caption_exact"]
        entry["passed"] &= generation_exact and entry["seed_exact"]
        if case["max_new_tokens"] == 3584:
            entry["passed"] &= bool(right["generation"] and right["generation"]["ended_with_stop"])
        report["cases"][name] = entry
    report["passed"] = sum(case["passed"] for case in report["cases"].values())
    report["total"] = len(cases)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


@pytest.fixture(scope="module")
def pipeline_outputs():
    if not torch.cuda.is_available():
        pytest.skip("A 32 GiB CUDA GPU is required")
    required = (SOURCE / "src/flux3/pipeline.py", WEIGHTS / "model.safetensors",
                PUP_WEIGHTS / "model.safetensors", MODEL / "modular_model_index.json")
    if any(not path.exists() for path in required):
        pytest.skip("Official source and FLUX.3/PUP checkpoints are required")
    sys.path.insert(0, str(SOURCE / "src"))
    cases = _cases()
    captions, generation = _official_captions(cases)
    expected = _official_outputs(cases, captions, generation)
    actual = _native_outputs(cases)
    report = _report(cases, expected, actual)
    return expected, actual, report


@pytest.mark.parametrize("name", CASE_NAMES)
def test_flux3_pipeline_official_parity(pipeline_outputs, name):
    expected, actual, report = pipeline_outputs
    left, right = expected[name], actual[name]
    assert right["caption"] == left["caption"]
    assert right["generation"] == left["generation"]
    assert right["resolved_seed"] == left["resolved_seed"]
    assert len(right["trajectory"]) == report["cases"][name]["steps"] + 1
    for step, (before, after) in enumerate(zip(left["trajectory"], right["trajectory"], strict=True)):
        torch.testing.assert_close(after, before, atol=0, rtol=0, msg=f"{name}: Euler state {step}")
    torch.testing.assert_close(right["latents"], left["latents"], atol=0, rtol=0)
    np.testing.assert_array_equal(right["pixels"], left["pixels"])
    if report["cases"][name]["max_new_tokens"] == 3584:
        assert right["generation"]["ended_with_stop"], "Full PUP case must reach an official stop token"
    assert report["cases"][name]["passed"]
    print(f"{name}: {len(right['trajectory'])} Euler states, caption, tokens and pixels exact")

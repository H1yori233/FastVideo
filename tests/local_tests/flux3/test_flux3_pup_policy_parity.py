# SPDX-License-Identifier: Apache-2.0
"""CPU parity for PUP image and message policies; scope: implementation_subcomponent."""

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


@pytest.fixture(scope="module")
def modules():
    root = Path(__file__).resolve().parents[3]
    reference = Path(os.getenv("FLUX3_OFFICIAL_REF_DIR", root / "flux-3-image-klein"))
    if not (reference / "src/flux3/pup.py").is_file():
        pytest.skip(f"Official FLUX3 reference missing: {reference}")
    sys.path.insert(0, str(reference / "src"))
    return (importlib.import_module("flux3.pup"),
            importlib.import_module("fastvideo.models.encoders.flux3_pup.policy"))


def pattern(width=83, height=113, mode="RGB"):
    y, x = np.indices((height, width))
    array = np.stack((x * 11 % 256, y * 7 % 256, (x + y) * 19 % 256), axis=-1).astype(np.uint8)
    image = Image.fromarray(array)
    if mode in ("RGBA", "LA"):
        image = image.convert(mode)
        image.putalpha(Image.fromarray(((x + y) * 3 % 256).astype(np.uint8)))
    elif mode in ("P", "PA"):
        image = image.convert("P")
        if mode == "P":
            image.info["transparency"] = 0
        else:
            image = image.convert("PA")
    return image


def test_policy_constants(modules):
    official, native = modules
    for name in ("STOP_TOKENS", "MAX_REFERENCES", "PRESENCE_PENALTY", "I2I_SCAFFOLD",
                 "REFERENCE_MULTIPLE", "MAX_ASPECT_RATIO", "MAX_PIXELS_BY_REFS", "JPEG_QUALITY"):
        assert getattr(native, name) == getattr(official, name)
    assert native.MAX_NEW_TOKENS == official.SAMPLING["max_tokens"]


@pytest.mark.parametrize("n_refs", [1, 2, 10])
@pytest.mark.parametrize("width,height", [(1024, 1024), (2049, 1025), (17, 111), (1, 200), (200, 1), (64, 64)])
def test_reference_size(modules, n_refs, width, height):
    official, native = modules
    actual = native.reference_size(width, height, n_refs)
    assert actual == official.reference_size(width, height, n_refs)
    assert actual[0] % 32 == actual[1] % 32 == 0
    assert actual[0] * actual[1] <= official.MAX_PIXELS_BY_REFS[n_refs]


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "LA", "P", "PA"])
@pytest.mark.parametrize("n_refs", [1, 2, 10])
def test_policy_reference_pixels(modules, mode, n_refs):
    official, native = modules
    image = pattern(mode=mode)
    actual, expected = native.policy_reference(image, n_refs), official.policy_reference(image, n_refs)
    assert actual.mode == expected.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("size", [(1024, 1024), (1537, 1025)])
@pytest.mark.parametrize("n_refs", [1, 2, 10])
def test_policy_reference_resize_budget(modules, size, n_refs):
    official, native = modules
    image = pattern(*size)
    actual, expected = native.policy_reference(image, n_refs), official.policy_reference(image, n_refs)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("orientation", [6, 8])
def test_policy_reference_exif(modules, orientation):
    official, native = modules
    image = pattern(mode="RGBA")
    image.getexif()[274] = orientation
    actual, expected = native.policy_reference(image, 2), official.policy_reference(image, 2)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert actual.width > actual.height


def test_policy_reference_white_alpha_and_jpeg(modules):
    official, native = modules
    transparent = Image.new("RGBA", (64, 64), (211, 13, 27, 0))
    for policy in (official, native):
        np.testing.assert_array_equal(np.asarray(policy.policy_reference(transparent, 1)), np.full((64, 64, 3), 255))
    image = pattern(64, 64)
    actual, expected = native.policy_reference(image, 1), official.policy_reference(image, 1)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert np.any(np.asarray(actual) != np.asarray(image))


@pytest.mark.parametrize("size", [(1, 201), (201, 1)])
def test_aspect_errors(modules, size):
    official, native = modules
    for policy in (official, native):
        with pytest.raises(ValueError, match="exceeds 200:1"):
            policy.reference_size(*size, 1)
        with pytest.raises(ValueError, match="exceeds 200:1"):
            policy.policy_reference(pattern(*size), 1)


class CaptureTemplate:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "rendered prompt"

    def encode(self, text, add_special_tokens):
        assert text == "rendered prompt" and not add_special_tokens
        return [1, 2, 3]

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 4, "<|endoftext|>": 5}[token]


class CaptureLLM:
    def generate(self, requests, params, use_tqdm):
        self.requests = requests
        self.params = params
        assert not use_tqdm
        return [SimpleNamespace(outputs=[SimpleNamespace(text="a detailed caption")])]


@pytest.mark.parametrize("n_refs", [0, 1, 2, 10])
@pytest.mark.parametrize("box_mode", ["none", "explicit", "embedded"])
def test_prepare_messages(modules, monkeypatch, n_refs, box_mode):
    official, native = modules
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    prompt = "Move the red balloon above the table."
    boxes = [{"id": "balloon", "bbox": [0.1, 0.2, 0.4, 0.5], "desc": "red balloon"}]
    if box_mode == "embedded":
        prompt += " " + json.dumps(official.bbox.to_canvas(boxes))
    supplied = boxes if box_mode == "explicit" else None
    images = [pattern(64 + i, 96 + i) for i in range(n_refs)]
    prompts = {key: f"system {key}" for key in ("t2i", "i2i", "t2i_bbox", "i2i_bbox")}
    upstream = object.__new__(official.Upsampler)
    upstream.prompts = prompts
    upstream.processor = CaptureTemplate()
    upstream.tokenizer = CaptureTemplate()
    upstream.llm = CaptureLLM()
    upstream(prompt, images, supplied)
    messages, references, locked = native.prepare_messages(prompt, images, supplied, prompts)
    expected_messages = (upstream.processor if images else upstream.tokenizer).messages
    assert messages == expected_messages
    assert locked == official.bbox.plan(prompt, supplied, n_refs)[2]
    expected_references = upstream.llm.requests[0].get("multi_modal_data", {}).get("image", [])
    assert len(references) == len(expected_references) == n_refs
    for actual, expected in zip(references, expected_references, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_prepare_messages_reference_limit(modules):
    _, native = modules
    with pytest.raises(ValueError, match="at most 10 references"):
        native.prepare_messages("edit", [pattern(32, 32)] * 11, None, {})


@pytest.fixture(scope="module")
def processor():
    root = Path(__file__).resolve().parents[3]
    weights = Path(os.getenv("FLUX3_PUP_WEIGHTS_DIR", root / "official_weights/flux3/pup"))
    if not (weights / "processor_config.json").is_file():
        pytest.skip(f"Official PUP processor missing: {weights}")
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(weights, local_files_only=True)


@pytest.mark.parametrize("n_refs", [0, 1, 2, 10])
@pytest.mark.parametrize("with_boxes", [False, True])
def test_prepare_inputs_official_request_tokens(modules, processor, monkeypatch, n_refs, with_boxes):
    official, _ = modules
    from fastvideo.models.encoders.flux3_pup.generation import prepare_inputs

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    prompt = "把红色气球移到桌子的上方。"
    boxes = [{"id": "balloon", "bbox": [0.1, 0.2, 0.4, 0.5], "desc": "red balloon"}] if with_boxes else None
    images = [pattern(64 + i, 96 + i, mode="RGBA") for i in range(n_refs)]
    prompts = {key: f"system {key}" for key in ("t2i", "i2i", "t2i_bbox", "i2i_bbox")}
    upstream = object.__new__(official.Upsampler)
    upstream.prompts, upstream.processor, upstream.tokenizer = prompts, processor, processor.tokenizer
    upstream.llm = CaptureLLM()
    upstream(prompt, images, boxes)
    request = upstream.llm.requests[0]
    if images:
        expected = dict(processor(text=[request["prompt"]], images=request["multi_modal_data"]["image"],
                                  return_tensors="pt", add_special_tokens=False, return_mm_token_type_ids=True))
    else:
        ids = torch.tensor([request["prompt_token_ids"]])
        expected = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    actual, locked = prepare_inputs(prompt, images, boxes, processor, prompts)
    assert locked == official.bbox.plan(prompt, boxes, n_refs)[2]
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


@pytest.mark.parametrize("n_refs", [1, 2])
def test_prepare_inputs_requests_image_token_types_with_legacy_processor_default(processor, monkeypatch, n_refs):
    from fastvideo.models.encoders.flux3_pup.generation import prepare_inputs

    processor_call = type(processor).__call__

    def legacy_default(self, *args, **kwargs):
        kwargs.setdefault("return_mm_token_type_ids", False)
        return processor_call(self, *args, **kwargs)

    monkeypatch.setattr(type(processor), "__call__", legacy_default)
    inputs, _ = prepare_inputs("Describe the reference images.", [pattern(64, 96)] * n_refs, None,
                               processor, {"i2i": "Describe the requested edit."})
    image_tokens = inputs["input_ids"] == processor.image_token_id
    torch.testing.assert_close(inputs["mm_token_type_ids"], image_tokens.long(), atol=0, rtol=0)
    expected_count = (inputs["image_grid_thw"].prod(-1) // processor.image_processor.merge_size**2).sum()
    assert image_tokens.sum() == expected_count > 0

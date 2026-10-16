"""CPU parity against the independent official implementation; scope: implementation_subcomponent."""

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture(scope="module")
def modules():
    reference = Path(os.environ.get("FLUX3_REFERENCE_ROOT", Path(__file__).resolve().parents[3] / "flux-3-image-klein"))
    if not (reference / "src/flux3/pipeline.py").is_file():
        pytest.skip(f"Official FLUX3 reference missing: {reference}")
    sys.path.insert(0, str(reference / "src"))
    return {
        "upstream": importlib.import_module("flux3.pipeline"),
        "native": importlib.import_module("fastvideo.pipelines.basic.flux3.packing"),
        "native_images": importlib.import_module("fastvideo.pipelines.basic.flux3.image_processing"),
        "upstream_sampling": importlib.import_module("flux3.sampling"),
        "native_sampling": importlib.import_module("fastvideo.pipelines.basic.flux3.sampling"),
        "upstream_bbox": importlib.import_module("flux3.bbox"),
        "native_bbox": importlib.import_module("fastvideo.models.encoders.flux3_pup.bbox"),
        "upstream_vae": importlib.import_module("flux3.autoencoder"),
        "native_vae": importlib.import_module("fastvideo.models.vaes.flux3"),
    }


def samples(pipeline, vae, *, conditioned=True, cfg=False):
    generator = torch.Generator().manual_seed(147)
    image = torch.randn(15, 128, generator=generator).to(torch.bfloat16).float()
    image_ids = pipeline.scaled_ids(vae.latent_ids(3, 5, "cpu"))
    cond = None
    if conditioned:
        refs = []
        for index, (height, width) in enumerate(((2, 3), (4, 2))):
            ids = pipeline.scaled_ids(vae.latent_ids(height, width, "cpu"))
            ids[:, 3] = -(index + 1)
            refs.append((torch.randn(height * width, 128, generator=generator).bfloat16(), ids))
        cond = (torch.cat([ref[0] for ref in refs]), torch.cat([ref[1] for ref in refs]))
    result = []
    for length in ([2, 7] if cfg else [7]):
        text = torch.randn(length, 20480, generator=generator).bfloat16()
        text_ids = torch.zeros(length, 4)
        text_ids[:, 3] = torch.arange(length)
        result.append([(text, text_ids), cond, (image, image_ids)])
    return pipeline.pack(result)


def assert_packed_equal(actual, expected):
    assert actual.num_samples == expected.num_samples
    assert actual.mode_indices == expected.mode_indices
    for attr in ("data", "ids", "timesteps"):
        actual_tensors, expected_tensors = getattr(actual, attr), getattr(expected, attr)
        assert actual_tensors.keys() == expected_tensors.keys()
        for mode in expected_tensors:
            torch.testing.assert_close(actual_tensors[mode], expected_tensors[mode], atol=0, rtol=0)


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("cfg", [False, True])
def test_non_square_mode_major_pack(modules, conditioned, cfg):
    actual = samples(modules["native"], modules["native_vae"], conditioned=conditioned, cfg=cfg)
    expected = samples(modules["upstream"], modules["upstream_vae"], conditioned=conditioned, cfg=cfg)
    assert_packed_equal(actual, expected)
    assert actual.mode_indices["image"][0] == (0, 15)
    if conditioned:
        assert actual.ids["image_cond"][:6, 3].tolist() == [-1.0] * 6
        assert actual.ids["image_cond"][6:14, 3].tolist() == [-2.0] * 8


@pytest.mark.parametrize("steps,alpha", [(0, 6.93), (1, 1.0), (4, 6.93), (16, 6.93), (50, 2.5)])
def test_schedule(modules, steps, alpha):
    actual = modules["native_sampling"].timesteps(steps, alpha)
    expected = modules["upstream_sampling"].timesteps(steps, alpha)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class Velocity:
    def __init__(self):
        self.calls = []

    def __call__(self, packed, timesteps_ref=None):
        self.calls.append((packed.data["image"].clone(), packed.timesteps["image"].clone(),
                           None if timesteps_ref is None else timesteps_ref.clone()))
        output = packed.data["image"] * 0.125 + packed.timesteps["image"].float()[:, None]
        for sample, (start, end) in enumerate(packed.mode_indices["image"]):
            first, last = packed.mode_indices["text"][sample]
            output[start:end] += packed.data["text"][first:last].float().mean()
            first, last = packed.mode_indices["image_cond"][sample]
            if last > first:
                output[start:end] += packed.data["image_cond"][first:last].float().mean()
        if timesteps_ref is not None:
            output += timesteps_ref.float().reshape(1, 1) * 0.375
        return output


@pytest.mark.parametrize("steps", [0, 1, 4, 16])
def test_flow_map_denoising_trajectory(modules, steps):
    outputs, packs, histories = [], [], []
    for prefix in ("upstream", "native"):
        packed = samples(modules[prefix], modules[f"{prefix}_vae"])
        sampling = modules[f"{prefix}_sampling"]
        model = Velocity()
        schedule = sampling.timesteps(steps, 6.93)
        output = sampling.denoise_flow_map(model, packed, schedule)
        outputs.append(output)
        packs.append(packed)
        histories.append(model.calls)
    torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
    assert_packed_equal(packs[1], packs[0])
    assert len(histories[0]) == len(histories[1]) == steps
    for actual, expected in zip(histories[1], histories[0], strict=True):
        for a, e in zip(actual, expected, strict=True):
            if e is None:
                assert a is None
            else:
                torch.testing.assert_close(a, e, atol=0, rtol=0)


def image_pattern(width, height):
    y, x = np.indices((height, width))
    return Image.fromarray(np.stack((x % 256, y % 256, (x + 3 * y) % 256), axis=-1).astype(np.uint8))


@pytest.mark.parametrize("width,height", [(11, 19), (64, 64), (65, 97), (2051, 2051), (8193, 201), (10000, 1)])
def test_reference_canvas_and_crop_pixels(modules, width, height):
    upstream, native = modules["upstream"], modules["native_images"]
    assert native.reference_canvas(width, height) == upstream.reference_canvas(width, height)
    source = image_pattern(width, height)
    expected, expected_crop = upstream.prepare_reference(source)
    actual, actual_crop = native.prepare_reference(source)
    assert actual_crop == expected_crop
    assert actual.mode == expected.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    if actual_crop is not None:
        np.testing.assert_array_equal(np.asarray(actual.crop(actual_crop)), np.asarray(expected.crop(expected_crop)))


def test_exif_reference_orientation(modules):
    source = image_pattern(81, 113)
    source.getexif()[274] = 6
    actual, actual_crop = modules["native_images"].prepare_reference(source)
    expected, expected_crop = modules["upstream"].prepare_reference(source)
    assert actual_crop == expected_crop
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_multi_reference_crop_and_default_size(modules):
    images = [image_pattern(65, 97), image_pattern(1037, 2023)]
    actual = [modules["native_images"].prepare_multi_reference(image) for image in images]
    expected = [modules["upstream"].prepare_multi_reference(image) for image in images]
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(e))
        assert a.width % 16 == a.height % 16 == 0
    assert modules["native_images"].default_size(actual) == modules["upstream"].default_size(expected)
    assert modules["native_images"].default_size([]) == modules["upstream"].default_size([])


def test_pixel_tensor_conversion(modules):
    image = image_pattern(65, 97)
    actual = modules["native_images"].to_tensor(image)
    expected = modules["upstream"].to_tensor(image)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual[:, :2] *= 2
    expected[:, :2] *= 2
    np.testing.assert_array_equal(np.asarray(modules["native_images"].to_pil(actual)),
                                  np.asarray(modules["upstream"].to_pil(expected)))


BOXES = [
    {"id": "obj_1", "bbox": [-0.1, 0.1235, 0.55, 1.2], "desc": "红色 [balloon]"},
    {"id": "item_2", "from": "ref_image_1", "src_bbox": [0, 0.1, 0.3, 0.4], "tgt_bbox": None},
    {"id": "item_3", "from": "target", "src_bbox": None, "tgt_bbox": [0.5, 0.6, 0.8, 0.9]},
]


@pytest.mark.parametrize("images", [0, 2])
def test_bbox_plan_detection_and_reconcile(modules, images):
    upstream, native = modules["upstream_bbox"], modules["native_bbox"]
    actual, expected = native.to_canvas(BOXES), upstream.to_canvas(BOXES)
    assert actual == expected
    prompt = "Place the red balloon above the table."
    assert native.plan(prompt, BOXES, images) == upstream.plan(prompt, BOXES, images)
    request = upstream.upsampling_input(prompt, expected)
    for text in (prompt, request, f"{prompt} {json.dumps(expected)}", "A shop [SALE]"):
        assert native.detect(text) == upstream.detect(text)
        assert native.split(text) == upstream.split(text)
        assert native.is_finished(text) == upstream.is_finished(text)
        assert native.plan(text, None, images) == upstream.plan(text, None, images)
    rows = [{"id": "balloon", "bbox": expected[0]["bbox"], "desc": 'red [balloon] with "string"'},
            {"id": "table", "bbox": [700, 200, 900, 800], "desc": "table"},
            {"id": "unused", "bbox": [100, 100, 200, 200], "desc": "uncited"}]
    answer = f"The <balloon> floats above <table>. {json.dumps(rows)}"
    assert native.reconcile(answer, actual) == upstream.reconcile(answer, expected)


@pytest.mark.parametrize("box", [None, [], [0, 0, 1, 1], [0, 0, 100, 100], [0, 0, 1001, 100],
                                 [0, 0, float("nan"), 100], [0, False, 100, 100], [100, 0, 0, 100]])
def test_bbox_validation(modules, box):
    assert modules["native_bbox"].valid_box(box) == modules["upstream_bbox"].valid_box(box)


@pytest.mark.parametrize("boxes", [[{"bbox": [0, 1, 2]}], [{}]])
def test_bbox_invalid_request_errors(modules, boxes):
    for prefix in ("upstream", "native"):
        with pytest.raises(ValueError):
            modules[f"{prefix}_bbox"].to_canvas(boxes)

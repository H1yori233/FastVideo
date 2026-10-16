# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Black Forest Labs, Inc.
# Modified for FastVideo imports and stateless request preparation.
"""FLUX3 prompt-upsample reference and message policies."""

import io
import math

from PIL import Image, ImageOps

from fastvideo.models.encoders.flux3_pup import bbox

STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
MAX_REFERENCES = 10
MAX_NEW_TOKENS = 3584
PRESENCE_PENALTY = {"t2i": 1.5, "i2i": 0.0}
I2I_SCAFFOLD = {
    False: ("Reference images follow in order (the first image, the second image, ...).\n"
            "User request: {instruction}\n"
            "Output the caption."),
    True: ("Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
           "User editing instruction: {instruction}\n"
           "Decide the edit, imagine the target image, and output the annotation record."),
}
REFERENCE_MULTIPLE = 32
MAX_ASPECT_RATIO = 200
MAX_PIXELS_BY_REFS = {1: 1024 * 1024, 2: 724 * 724, **{n: 512 * 512 for n in range(3, MAX_REFERENCES + 1)}}
JPEG_QUALITY = 92


def reference_size(width: int, height: int, n_refs: int) -> tuple[int, int]:
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        raise ValueError(f"aspect ratio of {width}x{height} exceeds {MAX_ASPECT_RATIO}:1")
    floor = lambda v: max(REFERENCE_MULTIPLE, int(math.floor(v / REFERENCE_MULTIPLE)) * REFERENCE_MULTIPLE)
    cap = MAX_PIXELS_BY_REFS[n_refs]
    w = max(REFERENCE_MULTIPLE, round(width / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    h = max(REFERENCE_MULTIPLE, round(height / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = floor(width / beta), floor(height / beta)
    if w * h > cap:
        if w == REFERENCE_MULTIPLE:
            h = floor(cap / w)
        else:
            w = floor(cap / h)
    return w, h


def policy_reference(image: Image.Image, n_refs: int) -> Image.Image:
    """Reference preprocessing of the upsampler: RGB over white, LANCZOS to the size budget, JPEG q92."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = reference_size(image.width, image.height, n_refs)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def prepare_messages(prompt: str, images: list[Image.Image], boxes: list[dict] | None,
                     prompts: dict[str, str]) -> tuple[list[dict], list[Image.Image], list[dict] | None]:
    if len(images) > MAX_REFERENCES:
        raise ValueError(f"the upsampler takes at most {MAX_REFERENCES} references")
    system_key, prompt, locked = bbox.plan(prompt, boxes, len(images))
    references = []
    if images:
        references = [policy_reference(image, len(images)) for image in images]
        scaffold = I2I_SCAFFOLD[locked is not None].format(instruction=prompt)
        content = [{"type": "image"} for _ in references] + [{"type": "text", "text": scaffold}]
        messages = [{"role": "system", "content": prompts[system_key]}, {"role": "user", "content": content}]
    else:
        messages = [{"role": "system", "content": prompts[system_key]}, {"role": "user", "content": prompt}]
    return messages, references, locked

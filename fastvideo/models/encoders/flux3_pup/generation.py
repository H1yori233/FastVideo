# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Black Forest Labs, Inc.
# Modified for FastVideo processor input preparation.
"""Tokenize FLUX3 prompt-upsample requests."""

from fastvideo.models.encoders.flux3_pup.policy import prepare_messages


def prepare_inputs(prompt, images, boxes, processor, system_prompts):
    messages, references, locked = prepare_messages(prompt, images, boxes, system_prompts)
    if references:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        inputs = processor(text=[text], images=references, return_tensors="pt", add_special_tokens=False,
                           return_mm_token_type_ids=True)
    else:
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
        inputs = processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return dict(inputs), locked

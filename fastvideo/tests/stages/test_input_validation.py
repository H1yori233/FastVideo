# SPDX-License-Identifier: Apache-2.0

from fastvideo.pipelines.stages.input_validation import _wan_ti2v_output_size


def test_wan_ti2v_output_size_preserves_default_floor() -> None:
    assert _wan_ti2v_output_size(1920, 1080, 16, 16, 448, 256) == (832, 480)


def test_wan_ti2v_output_size_accepts_720p_request() -> None:
    assert _wan_ti2v_output_size(1920, 1080, 16, 16, 1280, 720) == (1280, 720)


def test_wan_ti2v_output_size_keeps_explicit_size_for_different_input_aspect_ratio() -> None:
    assert _wan_ti2v_output_size(1280, 704, 16, 16, 1280, 720) == (1280, 720)

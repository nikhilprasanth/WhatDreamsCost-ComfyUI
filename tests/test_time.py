"""Frame/time/geometry invariants.

These are the constraints LTX imposes, so they are the constraints the whole
system rests on. Anything that weakens one of these assertions breaks the
compiler somewhere far from here.
"""

from __future__ import annotations

import pytest

from director.core import time as t


# --------------------------------------------------------------------------
# frame counts: 1 + 8k
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (1, 1), (2, 9), (8, 9), (9, 9), (10, 17),
        (97, 97), (120, 121), (121, 121), (122, 129),
        (0, 1), (-5, 1),
    ],
)
def test_snap_frames(raw: int, expected: int) -> None:
    assert t.snap_frames(raw) == expected


def test_snap_frames_is_idempotent_and_valid() -> None:
    for n in range(-3, 400):
        once = t.snap_frames(n)
        assert t.snap_frames(once) == once, f"not idempotent at {n}"
        assert t.is_valid_frame_count(once), f"{once} is not 1 + 8k"


def test_snap_frames_never_shortens() -> None:
    # Rounding up matters: "5 seconds" must never silently become less.
    for n in range(1, 400):
        assert t.snap_frames(n) >= n


def test_nearest_valid_frames_brackets_the_input() -> None:
    down, up = t.nearest_valid_frames(120)
    assert (down, up) == (113, 121)
    assert t.is_valid_frame_count(down) and t.is_valid_frame_count(up)

    # Already valid: both sides collapse onto the value itself.
    assert t.nearest_valid_frames(121) == (121, 121)


def test_nearest_valid_frames_clamps_at_one() -> None:
    down, up = t.nearest_valid_frames(2)
    assert down == 1 and up == 9


@pytest.mark.parametrize("frames,latent", [(1, 1), (9, 2), (97, 13), (121, 16)])
def test_latent_frames_matches_empty_ltxv_latent_video(frames: int, latent: int) -> None:
    # Mirrors ((length - 1) // 8) + 1 in comfy_extras/nodes_lt.py.
    assert t.latent_frames_for(frames) == latent


# --------------------------------------------------------------------------
# spatial: multiples of 32
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [(0, 32), (1, 32), (31, 32), (32, 32), (47, 32), (48, 64), (700, 704), (1280, 1280)],
)
def test_snap_dim(raw: int, expected: int) -> None:
    assert t.snap_dim(raw) == expected


def test_snap_dim_is_idempotent_and_valid() -> None:
    for n in range(0, 2048, 7):
        once = t.snap_dim(n)
        assert t.snap_dim(once) == once
        assert t.is_valid_dim(once)


# --------------------------------------------------------------------------
# unit conversion
# --------------------------------------------------------------------------

def test_seconds_round_trip_through_frames() -> None:
    for fps in (24.0, 25.0, 30.0, 50.0):
        for frame in range(0, 200, 13):
            seconds = t.frames_to_seconds(frame, fps)
            assert t.seconds_to_frames(seconds, fps) == frame


def test_percent_maps_endpoints_exactly() -> None:
    assert t.percent_to_frames(0, 121) == 0
    assert t.percent_to_frames(100, 121) == 120
    assert t.frames_to_percent(0, 121) == 0.0
    assert t.frames_to_percent(120, 121) == 100.0


def test_percent_is_clamped() -> None:
    assert t.percent_to_frames(-50, 121) == 0
    assert t.percent_to_frames(500, 121) == 120


def test_single_frame_timeline_has_no_percent_range() -> None:
    # Guards a divide-by-zero that is easy to reintroduce.
    assert t.percent_to_frames(50, 1) == 0
    assert t.frames_to_percent(0, 1) == 0.0


def test_zero_fps_does_not_divide_by_zero() -> None:
    assert t.frames_to_seconds(100, 0) == 0.0
    assert t.seconds_to_frames(4.0, 0) == 0


@pytest.mark.parametrize("unit", ["seconds", "frames", "percent"])
def test_to_and_from_frames_are_inverse(unit: str) -> None:
    fps, total = 24.0, 121
    for frame in (0, 1, 48, 120):
        value = t.from_frames(frame, unit, fps, total)
        assert t.to_frames(value, unit, fps, total) == frame


def test_unknown_unit_is_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="Unknown time unit"):
        t.to_frames(1.0, "furlongs", 24.0, 121)  # type: ignore[arg-type]


def test_format_timecode() -> None:
    assert t.format_timecode(0, 24.0) == "00:00.00"
    assert t.format_timecode(36, 24.0) == "00:01.50"
    assert t.format_timecode(24 * 65, 24.0) == "01:05.00"


# --------------------------------------------------------------------------
# guide index legality
# --------------------------------------------------------------------------

def test_single_frame_guides_accept_any_index() -> None:
    # LTXVAddGuide: "For single-frame images … any frame_idx value is acceptable."
    for idx in (0, 1, 7, 13, 119):
        assert t.snap_guide_index(idx, guide_frames=1) == idx


def test_multi_frame_guides_snap_down_to_a_multiple_of_eight() -> None:
    assert t.snap_guide_index(13, guide_frames=25) == 8
    assert t.snap_guide_index(16, guide_frames=25) == 16
    assert t.snap_guide_index(7, guide_frames=25) == 0


def test_negative_guide_index_is_preserved() -> None:
    # -1 is the documented way to anchor a last frame; snapping it would break that.
    assert t.snap_guide_index(-1, guide_frames=25) == -1
    assert t.snap_iclora_index(-1, guide_frames=25) == -1


def test_iclora_guides_snap_to_one_mod_eight() -> None:
    # LTXAddVideoICLoRAGuide uses a start_end patchifier, hence 1 (mod 8).
    assert t.snap_iclora_index(1, guide_frames=25) == 1
    assert t.snap_iclora_index(9, guide_frames=25) == 9
    assert t.snap_iclora_index(12, guide_frames=25) == 9
    assert t.snap_iclora_index(0, guide_frames=25) == 1
    assert t.snap_iclora_index(8, guide_frames=25) == 1

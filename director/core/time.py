"""Time and geometry arithmetic for LTX.

Everything here is pure integer/float maths with no dependencies, because the
same rules have to hold in three places: the browser (optimistically, for
dragging), the validator, and the compiler.

The two constraints that drive all of it, verified against ComfyUI's
``comfy_extras/nodes_lt.py``:

* **Frame count must be ``1 + 8k``.** ``EmptyLTXVLatentVideo`` declares
  ``length`` with ``step=8`` and allocates ``(length - 1) // 8 + 1`` latent
  frames; ``LTXVAddGuide`` rounds multi-frame guide indices down to a multiple
  of 8.
* **Width and height must be multiples of 32.** ``EmptyLTXVLatentVideo``
  declares both with ``step=32`` and allocates ``height // 32`` by
  ``width // 32``.
"""

from __future__ import annotations

import math
from typing import Literal

__all__ = [
    "TEMPORAL_STRIDE",
    "SPATIAL_STRIDE",
    "TimeUnit",
    "snap_frames",
    "nearest_valid_frames",
    "is_valid_frame_count",
    "snap_dim",
    "is_valid_dim",
    "latent_frames_for",
    "frames_to_seconds",
    "seconds_to_frames",
    "percent_to_frames",
    "frames_to_percent",
    "to_frames",
    "from_frames",
    "format_timecode",
    "snap_guide_index",
    "snap_iclora_index",
]

#: Pixel frames per latent frame for the LTX video VAE.
TEMPORAL_STRIDE = 8

#: Pixel columns/rows per latent cell for the LTX video VAE.
SPATIAL_STRIDE = 32

TimeUnit = Literal["seconds", "frames", "percent"]


# --------------------------------------------------------------------------
# frame counts
# --------------------------------------------------------------------------

def is_valid_frame_count(frames: int) -> bool:
    """True when ``frames`` is ``1 + 8k`` for some ``k >= 0``."""
    return frames >= 1 and (frames - 1) % TEMPORAL_STRIDE == 0


def snap_frames(frames: int) -> int:
    """Round ``frames`` **up** to the next valid ``1 + 8k`` count.

    Rounding up rather than to-nearest means "5 seconds" never silently becomes
    less than the user asked for. Idempotent on already-valid counts.

    >>> [snap_frames(n) for n in (1, 2, 9, 120, 121)]
    [1, 9, 9, 121, 121]
    """
    if frames <= 1:
        return 1
    return int(math.ceil((frames - 1) / TEMPORAL_STRIDE) * TEMPORAL_STRIDE) + 1


def nearest_valid_frames(frames: int) -> tuple[int, int]:
    """The valid frame counts immediately below and above ``frames``.

    Used to offer the user a choice in a diagnostic rather than silently
    picking one. Both values are valid; they are equal when ``frames`` already
    is.
    """
    up = snap_frames(frames)
    if up == frames:
        return frames, frames
    down = max(1, up - TEMPORAL_STRIDE)
    return down, up


def latent_frames_for(frames: int) -> int:
    """Latent temporal size for a pixel frame count, matching ``EmptyLTXVLatentVideo``."""
    return ((max(1, frames) - 1) // TEMPORAL_STRIDE) + 1


# --------------------------------------------------------------------------
# spatial dimensions
# --------------------------------------------------------------------------

def is_valid_dim(value: int) -> bool:
    """True when ``value`` is a positive multiple of 32."""
    return value >= SPATIAL_STRIDE and value % SPATIAL_STRIDE == 0


def snap_dim(value: int, *, minimum: int = SPATIAL_STRIDE) -> int:
    """Round a pixel dimension to the nearest multiple of 32, never below ``minimum``."""
    if value <= minimum:
        return minimum
    snapped = int(round(value / SPATIAL_STRIDE)) * SPATIAL_STRIDE
    return max(minimum, snapped)


# --------------------------------------------------------------------------
# unit conversion
#
# Frames are the storage unit everywhere. Seconds and percent are display
# conveniences that are converted on entry, so changing fps never drifts a
# stored position.
# --------------------------------------------------------------------------

def frames_to_seconds(frames: float, fps: float) -> float:
    return float(frames) / fps if fps > 0 else 0.0


def seconds_to_frames(seconds: float, fps: float) -> int:
    return int(round(float(seconds) * fps)) if fps > 0 else 0


def percent_to_frames(percent: float, total_frames: int) -> int:
    """``percent`` in 0..100 mapped onto ``0 .. total_frames - 1``."""
    if total_frames <= 1:
        return 0
    ratio = min(max(float(percent), 0.0), 100.0) / 100.0
    return int(round(ratio * (total_frames - 1)))


def frames_to_percent(frames: float, total_frames: int) -> float:
    if total_frames <= 1:
        return 0.0
    return min(max(float(frames) / (total_frames - 1), 0.0), 1.0) * 100.0


def to_frames(value: float, unit: TimeUnit, fps: float, total_frames: int) -> int:
    """Convert a user-facing value in ``unit`` to a stored frame index."""
    if unit == "frames":
        return int(round(value))
    if unit == "seconds":
        return seconds_to_frames(value, fps)
    if unit == "percent":
        return percent_to_frames(value, total_frames)
    raise ValueError(f"Unknown time unit {unit!r}; expected seconds, frames or percent.")


def from_frames(frames: int, unit: TimeUnit, fps: float, total_frames: int) -> float:
    """Convert a stored frame index back to a user-facing value in ``unit``."""
    if unit == "frames":
        return float(frames)
    if unit == "seconds":
        return frames_to_seconds(frames, fps)
    if unit == "percent":
        return frames_to_percent(frames, total_frames)
    raise ValueError(f"Unknown time unit {unit!r}; expected seconds, frames or percent.")


def format_timecode(frames: int, fps: float) -> str:
    """``mm:ss.cc`` for timeline rulers and diagnostics."""
    seconds = frames_to_seconds(frames, fps)
    minutes, rem = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes):02d}:{rem:05.2f}"


# --------------------------------------------------------------------------
# guide index legality
# --------------------------------------------------------------------------

def snap_guide_index(frame_idx: int, guide_frames: int) -> int:
    """Snap a ``LTXVAddGuide`` frame index the way the node itself would.

    Single frames and short clips (1..8 frames) accept any index. Longer guides
    are rounded **down** to a multiple of 8. Negative indices count from the end
    and are passed through untouched, because ``-1`` is the documented way to
    anchor a last frame.
    """
    if frame_idx < 0 or guide_frames <= TEMPORAL_STRIDE:
        return frame_idx
    return (frame_idx // TEMPORAL_STRIDE) * TEMPORAL_STRIDE


def snap_iclora_index(frame_idx: int, guide_frames: int) -> int:
    """Snap a ``LTXAddVideoICLoRAGuide`` frame index.

    IC-LoRA guides use a start-end patchifier, so multi-frame guides must sit at
    ``1 (mod 8)`` rather than ``0 (mod 8)``. The node rounds down; this mirrors
    it so the UI can show the position the model will actually use.
    """
    if frame_idx < 0 or guide_frames <= 1:
        return frame_idx
    if frame_idx % TEMPORAL_STRIDE == 1:
        return frame_idx
    snapped = ((frame_idx - 1) // TEMPORAL_STRIDE) * TEMPORAL_STRIDE + 1
    return max(1, snapped)

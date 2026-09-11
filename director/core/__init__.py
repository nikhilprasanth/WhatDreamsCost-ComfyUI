"""Director core — project state, time maths, prompt compilation, validation.

Nothing in this package imports ``torch``, ``comfy``, ``folder_paths`` or
``server``. That is a hard rule, not a preference: it is what lets the compiler
and the whole test suite run on a machine with no ComfyUI, no GPU and no model
weights installed.
"""

from __future__ import annotations

from .ids import digest_obj, new_id, sha256_bytes, sha256_file
from .migrate import MigrationError, migrate
from .prompt import compile_camera_phrase, compile_lens_phrase, compile_prompt
from .spec import (
    DEFAULT_SIGMAS_STAGE1,
    DEFAULT_SIGMAS_STAGE2,
    MODES,
    REF_ROLES,
    AudioClip,
    AudioSettings,
    Generation,
    LoraRef,
    MediaEntry,
    Models,
    Reference,
    RelaySettings,
    Segment,
    Spec,
    Take,
    TimePoint,
    new_spec,
)
from .time import (
    SPATIAL_STRIDE,
    TEMPORAL_STRIDE,
    format_timecode,
    from_frames,
    is_valid_dim,
    is_valid_frame_count,
    latent_frames_for,
    nearest_valid_frames,
    snap_dim,
    snap_frames,
    snap_guide_index,
    snap_iclora_index,
    to_frames,
)
from .validate import Diagnostic, Report, validate

__all__ = [
    "AudioClip",
    "AudioSettings",
    "DEFAULT_SIGMAS_STAGE1",
    "DEFAULT_SIGMAS_STAGE2",
    "Diagnostic",
    "Generation",
    "LoraRef",
    "MODES",
    "MediaEntry",
    "MigrationError",
    "Models",
    "REF_ROLES",
    "Reference",
    "RelaySettings",
    "Report",
    "SPATIAL_STRIDE",
    "Segment",
    "Spec",
    "TEMPORAL_STRIDE",
    "Take",
    "TimePoint",
    "compile_camera_phrase",
    "compile_lens_phrase",
    "compile_prompt",
    "digest_obj",
    "format_timecode",
    "from_frames",
    "is_valid_dim",
    "is_valid_frame_count",
    "latent_frames_for",
    "migrate",
    "nearest_valid_frames",
    "new_id",
    "new_spec",
    "sha256_bytes",
    "sha256_file",
    "snap_dim",
    "snap_frames",
    "snap_guide_index",
    "snap_iclora_index",
    "to_frames",
    "validate",
]

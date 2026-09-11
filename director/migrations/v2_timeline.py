"""Import a Director 2.x timeline into a Director Next Spec.

The v2 ``timeline_data`` widget held a browser-side editor's state: three
parallel segment arrays plus a pile of UI flags. This module reads that shape
and produces a Spec, reporting what could not be carried over rather than
dropping it silently.

The v2 shape, as written by ``js/ltx_director.js``::

    {
      "global_prompt": str,
      "segments":       [ {id, start, length, type: text|image|video,
                           prompt, imageFile, imageB64, isEndFrame,
                           trimStart, fileName} ],
      "motionSegments": [ {id, start, length, videoFile, trimStart} ],
      "audioSegments":  [ {id, start, length, audioFile, trimStart,
                           audioDurationFrames} ],
      "retakeMode": bool, "retakeStart": int, "retakeLength": int,
      "retakeStrength": float, "retakeVideo": {...} | null,
      "normalStartFrame": int, "normalDurationFrames": int,
      ...UI flags...
    }

What is *not* carried over, and why:

* ``imageB64`` — inline base64 media. The Spec stores file references only, so
  a segment that only ever existed as base64 produces a diagnostic asking the
  user to re-add the file. Keeping it would reintroduce the megabyte-scale
  workflow JSON the rebuild exists to remove.
* Retake's hand-built temporal mask. The region is preserved as a Prompt Relay
  segment plus a note; the masking itself is replaced by the in/outpainting
  IC-LoRA path, which is what upstream actually supports.
* Track visibility, panel heights and other editor chrome.
"""

from __future__ import annotations

import os
from typing import Any

from ..core import time as dtime
from ..core.ids import new_id
from ..core.spec import (
    DEFAULT_SIGMAS_STAGE1,
    DEFAULT_SIGMAS_STAGE2,
    AudioClip,
    MediaEntry,
    Reference,
    Segment,
    Spec,
)
from ..core.validate import Diagnostic

__all__ = ["convert", "convert_with_report", "V2ImportResult"]

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_VIDEO_EXT = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v", ".flv", ".wmv"}
_AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus"}


class V2ImportResult:
    """A converted Spec plus what the conversion could not bring across."""

    def __init__(self, spec: Spec, diagnostics: list[Diagnostic]) -> None:
        self.spec = spec
        self.diagnostics = diagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def convert(data: dict[str, Any], **kwargs: Any) -> Spec:
    """Convert a v2 timeline to a Spec, discarding the diagnostics.

    Used by :func:`director.core.migrate.migrate`, which has nowhere to put a
    report. Call :func:`convert_with_report` when the messages matter — the
    import action in the UI does.
    """
    return convert_with_report(data, **kwargs).spec


def convert_with_report(
    data: dict[str, Any],
    *,
    fps: float = 24.0,
    width: int = 0,
    height: int = 0,
    guide_strength: str = "",
    img_compression: int = 18,
) -> V2ImportResult:
    """Convert a v2 timeline, reporting anything that needed the user's attention.

    The extra arguments come from the v2 node's sibling widgets, which lived
    outside ``timeline_data``. Passing them is optional; sensible defaults are
    used when a caller only has the blob.
    """
    diagnostics: list[Diagnostic] = []
    spec = Spec()
    spec.meta.name = "Imported from Director 2"
    spec.meta.note = "Converted from a Director 2.x timeline."

    # -- project ---------------------------------------------------------
    spec.project.fps = float(fps) if fps else 24.0
    start_frame = int(data.get("normalStartFrame") or 0)
    duration = int(data.get("normalDurationFrames") or 121)
    spec.project.frames = dtime.snap_frames(max(1, duration))
    if spec.project.frames != duration:
        diagnostics.append(Diagnostic(
            "info", "v2.frames_snapped",
            f"The v2 timeline was {duration} frames long; LTX needs 1 plus a multiple of 8, "
            f"so the shot is now {spec.project.frames} frames.",
            "Adjust the duration if you need a different length.",
            "project.frames",
        ))
    if width:
        spec.project.width = dtime.snap_dim(int(width))
    if height:
        spec.project.height = dtime.snap_dim(int(height))

    spec.generation.img_compression = int(img_compression)
    spec.generation.sigmas_stage1 = list(DEFAULT_SIGMAS_STAGE1)
    spec.generation.sigmas_stage2 = list(DEFAULT_SIGMAS_STAGE2)

    # -- prompts ---------------------------------------------------------
    spec.prompt.mode = "simple"
    spec.prompt.raw = (data.get("global_prompt") or "").strip()

    strengths = _parse_strengths(guide_strength)

    # -- media registry --------------------------------------------------
    # Built first so segments and clips can reference by id. Keyed by the v2
    # filename so the same file used twice becomes one entry.
    media_by_file: dict[str, MediaEntry] = {}

    def register(filename: str | None, kind: str) -> str | None:
        if not filename:
            return None
        existing = media_by_file.get(filename)
        if existing is not None:
            return existing.id
        subfolder, base = _split_input_path(filename)
        entry = MediaEntry(
            id=new_id("media"),
            filename=base,
            subfolder=subfolder,
            kind=_kind_for(base, kind),
        )
        media_by_file[filename] = entry
        spec.media[entry.id] = entry
        return entry.id

    # -- main track: prompt regions and keyframes -------------------------
    guide_index = 0
    for raw in _sorted_segments(data.get("segments")):
        seg_type = raw.get("type") or "image"
        seg_start = max(0, int(raw.get("start") or 0) - start_frame)
        seg_length = max(1, int(raw.get("length") or 1))

        # Anything entirely before the old render window is not part of the shot.
        if int(raw.get("start") or 0) + seg_length <= start_frame:
            continue

        prompt_text = (raw.get("prompt") or "").strip()
        if prompt_text:
            spec.segments.append(Segment(
                id=new_id("segment"),
                start=seg_start,
                length=min(seg_length, spec.project.frames - seg_start) or 1,
                text=prompt_text,
            ))

        if seg_type == "text":
            continue

        image_file = raw.get("imageFile")
        if not image_file:
            if raw.get("imageB64"):
                diagnostics.append(Diagnostic(
                    "warning", "v2.base64_media",
                    "A keyframe was stored inline as base64 rather than as a file, so it could "
                    "not be imported.",
                    "Drag the image back onto the timeline; Director Next uploads it once and "
                    "references it by name.",
                    "references",
                ))
            guide_index += 1
            continue

        media_id = register(image_file, "video" if seg_type == "video" else "image")
        is_end = bool(raw.get("isEndFrame"))
        frame = seg_start + seg_length - 1 if is_end else seg_start

        ref = Reference(
            id=new_id("reference"),
            role="keyframe",
            media=media_id,
            label=_label_for(raw, image_file),
            enabled=True,
            anchor="end" if is_end else ("start" if frame <= 0 else "index"),
            strength=_strength_at(strengths, guide_index),
        )
        ref.at.frame = max(0, min(frame, spec.project.frames - 1))
        spec.references.append(ref)
        guide_index += 1

    # -- IC-LoRA / motion track ------------------------------------------
    for raw in _sorted_segments(data.get("motionSegments")):
        video_file = raw.get("videoFile") or raw.get("imageFile")
        if not video_file:
            continue
        media_id = register(video_file, "video")
        seg_start = max(0, int(raw.get("start") or 0) - start_frame)
        ref = Reference(
            id=new_id("reference"),
            role="motion",
            media=media_id,
            label=_label_for(raw, video_file),
            anchor="start" if seg_start <= 0 else "index",
            strength=1.0,
        )
        ref.at.frame = max(0, min(seg_start, spec.project.frames - 1))
        spec.references.append(ref)

    # -- audio track ------------------------------------------------------
    for raw in _sorted_segments(data.get("audioSegments")):
        audio_file = raw.get("audioFile")
        if not audio_file:
            continue
        media_id = register(audio_file, "audio")
        seg_start = max(0, int(raw.get("start") or 0) - start_frame)
        spec.audio.clips.append(AudioClip(
            id=new_id("audio"),
            media=media_id,
            start=seg_start,
            length=max(1, int(raw.get("length") or 1)),
            trim_start=float(raw.get("trimStart") or 0.0) / (spec.project.fps or 24.0),
        ))

    if spec.audio.clips:
        spec.audio.enabled = True
        spec.audio.mode = "inpaint" if data.get("inpaint_audio", True) else "import"
    else:
        spec.audio.mode = "generate"

    # -- retake ----------------------------------------------------------
    if data.get("retakeMode"):
        retake_prompt = (data.get("retakePrompt") or "").strip()
        r_start = max(0, int(data.get("retakeStart") or 0) - start_frame)
        r_len = max(1, int(data.get("retakeLength") or 1))
        if retake_prompt:
            spec.segments.append(Segment(
                id=new_id("segment"),
                start=min(r_start, spec.project.frames - 1),
                length=min(r_len, spec.project.frames - r_start) or 1,
                text=retake_prompt,
            ))
        diagnostics.append(Diagnostic(
            "warning", "v2.retake_not_ported",
            "This timeline used Retake mode. Its region and prompt were imported, but the "
            "re-generation itself was not — v2 built the mask by hand, which is why it was "
            "never potent enough.",
            "Use IC-LoRA mode with the in/outpainting LoRA for region edits.",
            "project.mode",
        ))

    if data.get("overrideAudio"):
        diagnostics.append(Diagnostic(
            "info", "v2.override_audio",
            "“Override audio” took the soundtrack from the IC-LoRA video. Director Next does "
            "that by loading the clip onto the audio track instead.",
            "Add the video's audio as an audio clip if you want it kept.",
            "audio",
        ))

    # -- mode inference ---------------------------------------------------
    spec.project.mode = _infer_mode(spec)
    spec.relay.enabled = len([s for s in spec.segments if s.text.strip()]) > 1

    spec.normalise()
    return V2ImportResult(spec, diagnostics)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _infer_mode(spec: Spec) -> str:
    """Pick the mode the imported timeline was really doing."""
    keys = [r for r in spec.references if r.role == "keyframe" and r.media]
    has_motion = any(r.role == "motion" and r.media for r in spec.references)
    if has_motion:
        return "iclora"
    anchors = {r.anchor for r in keys}
    if "start" in anchors and "end" in anchors:
        return "fflf"
    if len(keys) > 1:
        return "keyframes"
    if keys:
        return "i2v" if "start" in anchors else "keyframes"
    return "t2v"


def _sorted_segments(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items = [v for v in value if isinstance(v, dict)]
    return sorted(items, key=lambda s: int(s.get("start") or 0))


def _parse_strengths(text: str) -> list[float]:
    """v2 kept guide strengths in a separate comma-separated widget, by position."""
    out: list[float] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def _strength_at(strengths: list[float], index: int) -> float:
    if 0 <= index < len(strengths):
        return max(0.0, min(1.0, strengths[index]))
    return 1.0


def _split_input_path(filename: str) -> tuple[str, str]:
    """Split a v2 path into (subfolder, basename).

    v2 wrote either a bare filename or ``whatdreamscost/name.png``, and resolved
    the fallback at load time. Normalising separators here means the Spec never
    carries a Windows path.
    """
    normalised = filename.replace("\\", "/").lstrip("/")
    head, _, tail = normalised.rpartition("/")
    return head, tail or normalised


def _kind_for(basename: str, fallback: str) -> str:
    ext = os.path.splitext(basename)[1].lower()
    if ext in _IMAGE_EXT:
        return "image"
    if ext in _VIDEO_EXT:
        return "video"
    if ext in _AUDIO_EXT:
        return "audio"
    return fallback if fallback in ("image", "video", "audio") else "image"


def _label_for(raw: dict[str, Any], filename: str) -> str:
    label = (raw.get("label") or raw.get("fileName") or "").strip()
    if label:
        return label
    return _split_input_path(filename)[1]

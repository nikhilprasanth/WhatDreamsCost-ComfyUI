"""Spec validation, in sentences a filmmaker can act on.

The rule this module exists to enforce: a user should never see
``KeyError: conditioning_1``. They should see *"Reference frame at 4.2 s could
not be compiled because the selected model does not support this conditioning
mode"* — and, where one exists, the fix.

Every rule here is traceable to a constraint in ``docs/RESEARCH.md`` §7, which
in turn came from reading the node sources rather than from folklore.

The same function backs three surfaces: live editing (over HTTP), compile time
and execute time. One rule set, so they cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from .time import (
    TEMPORAL_STRIDE,
    format_timecode,
    is_valid_dim,
    is_valid_frame_count,
    latent_frames_for,
    nearest_valid_frames,
    snap_dim,
    snap_frames,
)

__all__ = ["Level", "Diagnostic", "Report", "validate"]

Level = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class Diagnostic:
    """One problem, one sentence, and — where possible — one way out."""

    level: Level
    code: str
    message: str
    fix: str = ""
    path: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "fix": self.fix,
            "path": self.path,
        }

    def __str__(self) -> str:
        text = f"{self.level.upper()} {self.code}: {self.message}"
        return f"{text} {self.fix}" if self.fix else text


@dataclass
class Report:
    diagnostics: list[Diagnostic] = field(default_factory=list)

    # -- building ----------------------------------------------------------

    def error(self, code: str, message: str, fix: str = "", path: str = "") -> None:
        self.diagnostics.append(Diagnostic("error", code, message, fix, path))

    def warn(self, code: str, message: str, fix: str = "", path: str = "") -> None:
        self.diagnostics.append(Diagnostic("warning", code, message, fix, path))

    def info(self, code: str, message: str, fix: str = "", path: str = "") -> None:
        self.diagnostics.append(Diagnostic("info", code, message, fix, path))

    def extend(self, others: Iterable[Diagnostic]) -> None:
        self.diagnostics.extend(others)

    # -- querying ----------------------------------------------------------

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.level == "error"]

    @property
    def ok(self) -> bool:
        """True when nothing blocks compilation. Warnings do not block."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "diagnostics": [d.to_dict() for d in self.diagnostics]}

    def raise_if_failed(self, context: str = "") -> None:
        """Turn accumulated errors into one readable exception.

        Used at execute time, where ComfyUI can only surface an exception. The
        message is the joined sentences, not a traceback the user must decode.
        """
        errs = self.errors
        if not errs:
            return
        head = f"LTX Director cannot {context}:" if context else "LTX Director found a problem:"
        body = "\n".join(f"  • {d.message}{(' ' + d.fix) if d.fix else ''}" for d in errs)
        raise ValueError(f"{head}\n{body}")

    def __len__(self) -> int:
        return len(self.diagnostics)

    def __iter__(self):
        return iter(self.diagnostics)


# --------------------------------------------------------------------------
# the rules
# --------------------------------------------------------------------------

#: Which model components each mode cannot run without.
_REQUIRED_MODELS: dict[str, tuple[str, ...]] = {
    "t2v": ("unet", "clip", "vae"),
    "i2v": ("unet", "clip", "vae"),
    "fflf": ("unet", "clip", "vae"),
    "keyframes": ("unet", "clip", "vae"),
    "continue": ("unet", "clip", "vae"),
    "a2v": ("unet", "clip", "vae", "audio_vae"),
    "t2a": ("unet", "clip", "audio_vae"),
    "iclora": ("unet", "clip", "vae"),
}

_MODEL_LABEL = {
    "unet": "a transformer checkpoint",
    "clip": "a text encoder",
    "vae": "the video VAE",
    "audio_vae": "the audio VAE",
}

#: Feature flag each mode depends on, when it depends on one.
_MODE_FEATURE: dict[str, str] = {
    "a2v": "audio",
    "t2a": "audio_only",
    "iclora": "iclora",
    "keyframes": "mid_timeline_guides",
    "fflf": "mid_timeline_guides",
}

_MODE_LABEL = {
    "t2v": "Text to Video",
    "i2v": "Image to Video",
    "fflf": "First / Last Frame",
    "keyframes": "Keyframes",
    "continue": "Continue Shot",
    "a2v": "Audio to Video",
    "t2a": "Text to Audio",
    "iclora": "IC-LoRA Control",
}


def validate(spec: Any, caps: Any = None) -> Report:
    """Check a Spec against LTX's constraints and the installed capabilities.

    ``caps`` is an optional object exposing ``has(flag) -> bool`` and
    ``reason(flag) -> str``. When omitted, capability rules are skipped rather
    than assumed false — so the validator stays usable in tests and in the
    frontend before the probe has answered.
    """
    report = Report()
    _check_project(spec, report)
    _check_models(spec, report)
    _check_prompt(spec, report)
    _check_references(spec, report)
    _check_segments(spec, report)
    _check_audio(spec, report)
    _check_generation(spec, report)
    if caps is not None:
        _check_capabilities(spec, caps, report)
    return report


# -- project ---------------------------------------------------------------

def _check_project(spec: Any, report: Report) -> None:
    p = spec.project

    if p.fps <= 0:
        report.error(
            "fps.invalid",
            f"Frame rate is {p.fps}, which cannot be used.",
            "Set a frame rate between 1 and 60; 24 is the usual choice.",
            "project.fps",
        )
    elif p.fps > 60:
        report.warn(
            "fps.high",
            f"{p.fps:g} fps is above the 50 fps the LTX models are documented for.",
            "Consider 24, 25, 30 or 50.",
            "project.fps",
        )

    if not is_valid_frame_count(p.frames):
        down, up = nearest_valid_frames(p.frames)
        fps = p.fps or 24.0
        report.error(
            "frames.not_8k1",
            f"{p.frames} frames is not a length LTX can generate — it needs "
            f"1 plus a multiple of {TEMPORAL_STRIDE}.",
            f"Use {up} frames ({up / fps:.2f} s) or {down} frames ({down / fps:.2f} s).",
            "project.frames",
        )

    if p.frames < 9 and p.mode != "t2a":
        report.warn(
            "frames.very_short",
            f"{p.frames} frames is under half a second of video.",
            "Most shots want at least 49 frames (about 2 s at 24 fps).",
            "project.frames",
        )

    if p.mode != "t2a":
        for axis, value in (("width", p.width), ("height", p.height)):
            if not is_valid_dim(value):
                report.error(
                    f"{axis}.not_multiple_32",
                    f"{axis.capitalize()} {value} is not a multiple of 32, which the LTX "
                    f"video VAE requires.",
                    f"Use {snap_dim(value)}.",
                    f"project.{axis}",
                )

        pixels = p.width * p.height
        if pixels > 1920 * 1088:
            report.warn(
                "resolution.large",
                f"{p.width}×{p.height} is above 1080p; a single stage at this size is slow "
                f"and memory-hungry.",
                "Generate at half size and enable the refine stage instead.",
                "project.width",
            )

    if p.mode not in _MODE_LABEL:
        report.error(
            "mode.unknown",
            f"Mode {p.mode!r} is not one the Director knows how to compile.",
            f"Pick one of: {', '.join(_MODE_LABEL)}.",
            "project.mode",
        )


# -- models ----------------------------------------------------------------

def _check_models(spec: Any, report: Report) -> None:
    required = _REQUIRED_MODELS.get(spec.project.mode, ())
    for name in required:
        if not getattr(spec.models, name, ""):
            report.error(
                f"model.missing.{name}",
                f"{_MODE_LABEL.get(spec.project.mode, spec.project.mode)} needs "
                f"{_MODE_LABEL.get(name) or _MODEL_LABEL.get(name, name)}, and none is selected.",
                "Choose one in the Models section.",
                f"models.{name}",
            )

    if spec.generation.stages > 1 and not spec.models.upscaler:
        report.error(
            "model.missing.upscaler",
            "The refine stage needs a latent spatial upscaler, and none is selected.",
            "Choose an upscaler, or set the shot to a single stage.",
            "models.upscaler",
        )

    if spec.prompt.enhance.enabled and not spec.models.enhancer_clip:
        report.error(
            "model.missing.enhancer",
            "Prompt enhancement is on but no enhancer text encoder is selected.",
            "Choose the small Gemma-4 encoder, or turn enhancement off.",
            "models.enhancer_clip",
        )

    if spec.project.mode == "iclora":
        if not any(lr.enabled and lr.kind == "iclora" for lr in spec.models.loras):
            report.error(
                "iclora.none_selected",
                "IC-LoRA mode is selected but no IC-LoRA is enabled.",
                "Add an IC-LoRA in the Models section, or switch to another mode.",
                "models.loras",
            )

    seen: set[str] = set()
    for lora in spec.models.loras:
        if not lora.name:
            continue
        if lora.name in seen:
            report.warn(
                "lora.duplicate",
                f"LoRA {lora.name!r} is listed more than once; its strength will stack.",
                "Remove the duplicate, or merge them into one entry.",
                "models.loras",
            )
        seen.add(lora.name)


# -- prompt ----------------------------------------------------------------

def _check_prompt(spec: Any, report: Report) -> None:
    from .prompt import compile_prompt  # local import: prompt imports nothing heavy

    text = compile_prompt(spec)
    if not text.strip():
        level = report.error if spec.project.mode in ("t2v", "t2a") else report.warn
        level(
            "prompt.empty",
            "There is no prompt text.",
            "Describe the shot in the prompt box."
            if spec.prompt.mode != "director"
            else "Fill in at least the Scene or Action section.",
            "prompt",
        )
    elif len(text) > 6000:
        report.warn(
            "prompt.very_long",
            f"The compiled prompt is {len(text)} characters; the text encoder will truncate it.",
            "Trim it, or move detail into the per-segment prompts.",
            "prompt",
        )


# -- references ------------------------------------------------------------

def _check_references(spec: Any, report: Report) -> None:
    total = spec.project.frames
    mode = spec.project.mode

    for ref in spec.references:
        path = f"references.{ref.id}"

        if ref.media and ref.media not in spec.media:
            report.error(
                "ref.media_missing",
                f"Reference {_ref_name(ref)} points at media that is no longer in the project.",
                "Re-add the image or remove the reference.",
                path,
            )
            continue

        if ref.enabled and not ref.media:
            report.warn(
                "ref.no_media",
                f"Reference {_ref_name(ref)} is enabled but empty.",
                "Drop an image onto it, or disable it.",
                path,
            )
            continue

        if not ref.enabled:
            continue

        if ref.anchor == "index" and ref.at.frame >= total:
            report.error(
                "ref.past_end",
                f"Reference {_ref_name(ref)} sits at "
                f"{format_timecode(ref.at.frame, spec.project.fps)}, past the end of a "
                f"{format_timecode(total, spec.project.fps)} shot.",
                "Move it inside the shot, or extend the duration.",
                path,
            )

        if not 0.0 <= ref.strength <= 1.0:
            report.error(
                "ref.strength_range",
                f"Reference {_ref_name(ref)} has strength {ref.strength:g}; guides accept 0 to 1.",
                "Use a value between 0 and 1.",
                path,
            )

    keyframes = [r for r in spec.references if r.enabled and r.media and r.role == "keyframe"]

    # Collisions: two guides landing on the same latent frame fight each other.
    by_frame: dict[int, list[Any]] = {}
    for ref in keyframes:
        by_frame.setdefault(ref.resolved_frame(total), []).append(ref)
    for frame, group in by_frame.items():
        if len(group) > 1:
            where = "the last frame" if frame < 0 else format_timecode(frame, spec.project.fps)
            report.warn(
                "ref.duplicate_position",
                f"{len(group)} keyframes land on {where}: "
                f"{', '.join(_ref_name(r) for r in group)}.",
                "Move all but one, or disable the extras.",
                "references",
            )

    # Mode expectations.
    if mode == "i2v" and not _has_anchor(keyframes, "start"):
        report.error(
            "mode.i2v_no_first_frame",
            "Image to Video needs a first-frame image, and none is anchored to the start.",
            "Add a keyframe reference and set its anchor to Start.",
            "references",
        )
    if mode == "fflf":
        if not _has_anchor(keyframes, "start"):
            report.error(
                "mode.fflf_no_first",
                "First / Last Frame needs an image anchored to the start.",
                "Add a keyframe reference anchored to Start.",
                "references",
            )
        if not _has_anchor(keyframes, "end"):
            report.error(
                "mode.fflf_no_last",
                "First / Last Frame needs an image anchored to the end.",
                "Add a keyframe reference anchored to End.",
                "references",
            )
    if mode == "keyframes" and not keyframes:
        report.error(
            "mode.keyframes_none",
            "Keyframes mode needs at least one keyframe reference.",
            "Add one, or switch to Text to Video.",
            "references",
        )
    if mode == "iclora" and not [
        r for r in spec.references if r.enabled and r.media and r.role in ("motion", "control")
    ]:
        report.error(
            "mode.iclora_no_guide",
            "IC-LoRA mode needs a motion or control reference to follow.",
            "Drop a video or annotation clip onto the IC-LoRA track.",
            "references",
        )

    if mode in ("t2v", "t2a") and keyframes:
        report.info(
            "ref.ignored_in_t2v",
            f"{len(keyframes)} keyframe reference(s) will be ignored in "
            f"{_MODE_LABEL[mode]} mode.",
            "Switch to Image to Video or Keyframes to use them.",
            "references",
        )

    # Multi-frame video guides must sit on the stride grid.
    for ref in keyframes:
        media = spec.media.get(ref.media or "")
        if media is None or media.kind != "video":
            continue
        frame = ref.resolved_frame(total)
        if frame > 0 and frame % TEMPORAL_STRIDE != 0:
            snapped = (frame // TEMPORAL_STRIDE) * TEMPORAL_STRIDE
            report.warn(
                "ref.video_index_snapped",
                f"Video guide {_ref_name(ref)} at frame {frame} will be moved to frame "
                f"{snapped}: multi-frame guides must sit on a multiple of {TEMPORAL_STRIDE}.",
                f"Place it at frame {snapped} to see exactly what the model will use.",
                f"references.{ref.id}",
            )


def _has_anchor(refs: Iterable[Any], anchor: str) -> bool:
    return any(r.anchor == anchor for r in refs)


def _ref_name(ref: Any) -> str:
    if ref.label:
        return f"“{ref.label}”"
    return f"{ref.role} {ref.id.split('_')[-1][:4]}"


# -- relay segments --------------------------------------------------------

def _check_segments(spec: Any, report: Report) -> None:
    segments = spec.segments
    if not segments:
        return

    total = spec.project.frames
    latent = latent_frames_for(total)

    populated = [s for s in segments if s.text.strip()]
    if len(populated) == 1 and spec.relay.enabled:
        report.info(
            "relay.single_segment",
            "Only one prompt region has text, so Prompt Relay will not be applied.",
            "Add a second region to vary the prompt over time.",
            "segments",
        )

    for seg in segments:
        if seg.length < 1:
            report.error(
                "segment.empty",
                f"Prompt region starting at frame {seg.start} has no length.",
                "Drag its edge to give it at least one frame.",
                f"segments.{seg.id}",
            )
        if seg.start < 0:
            report.error(
                "segment.negative_start",
                f"A prompt region starts at frame {seg.start}, before the shot begins.",
                "Move it to frame 0 or later.",
                f"segments.{seg.id}",
            )

    ordered = sorted(segments, key=lambda s: s.start)
    for a, b in zip(ordered, ordered[1:]):
        if b.start < a.end:
            report.warn(
                "segment.overlap",
                f"Prompt regions overlap between frames {b.start} and {a.end}.",
                "Both prompts will influence the overlap. Trim one if that is not intended.",
                f"segments.{b.id}",
            )

    if ordered and ordered[-1].end > total:
        report.warn(
            "segment.past_end",
            f"The last prompt region ends at frame {ordered[-1].end}, past the shot's "
            f"{total} frames. It will be clipped.",
            "Shorten it, or extend the duration.",
            "segments",
        )

    if len(populated) > latent:
        report.error(
            "relay.too_many_segments",
            f"{len(populated)} prompt regions cannot fit in {latent} latent frames — each "
            f"region needs at least one.",
            f"Reduce to {latent} regions or fewer, or lengthen the shot.",
            "segments",
        )

    if spec.relay.enabled and not 0 < spec.relay.epsilon < 1:
        report.error(
            "relay.epsilon_range",
            f"Prompt Relay epsilon is {spec.relay.epsilon:g}; it must be between 0 and 1.",
            "0.001 gives sharp boundaries; 0.5 and above blend them.",
            "relay.epsilon",
        )


# -- audio -----------------------------------------------------------------

def _check_audio(spec: Any, report: Report) -> None:
    audio = spec.audio
    total = spec.project.frames
    mode = spec.project.mode

    if mode == "a2v" and not [c for c in audio.clips if c.enabled and c.media]:
        report.error(
            "audio.a2v_no_clip",
            "Audio to Video needs an audio clip to follow, and none is loaded.",
            "Drop an audio file onto the audio track.",
            "audio.clips",
        )

    if audio.mode in ("import", "inpaint") and not audio.clips:
        report.error(
            "audio.no_clips",
            f"Audio mode is “{audio.mode}” but the audio track is empty.",
            "Add a clip, or set audio to Generate.",
            "audio.clips",
        )

    for clip in audio.clips:
        path = f"audio.clips.{clip.id}"
        if clip.media and clip.media not in spec.media:
            report.error(
                "audio.media_missing",
                "An audio clip points at media that is no longer in the project.",
                "Re-add the audio file or remove the clip.",
                path,
            )
            continue
        if not clip.enabled:
            continue
        if clip.length < 1:
            report.error(
                "audio.zero_length",
                f"An audio clip at frame {clip.start} has no length.",
                "Drag its edge to give it duration.",
                path,
            )
        if clip.start >= total:
            report.warn(
                "audio.past_end",
                f"An audio clip starts at "
                f"{format_timecode(clip.start, spec.project.fps)}, after the shot ends.",
                "Move it inside the shot, or remove it.",
                path,
            )
        media = spec.media.get(clip.media or "")
        if media and media.duration > 0:
            needed = clip.trim_start + clip.length / (spec.project.fps or 24.0)
            if needed > media.duration + 0.05:
                report.warn(
                    "audio.trim_past_end",
                    f"An audio clip asks for {needed:.2f} s from a file that is only "
                    f"{media.duration:.2f} s long; the tail will be silent.",
                    "Shorten the clip, or reduce its trim-in point.",
                    path,
                )

    if audio.reference and audio.reference not in spec.media:
        report.error(
            "audio.reference_missing",
            "The speaker-identity reference points at media that is no longer in the project.",
            "Re-add the reference clip, or clear it.",
            "audio.reference",
        )
    elif audio.reference:
        media = spec.media[audio.reference]
        if media.duration and not 2.0 <= media.duration <= 15.0:
            report.warn(
                "audio.reference_duration",
                f"The speaker-identity reference is {media.duration:.1f} s; identity transfer "
                f"is trained around 5 s.",
                "Trim it to roughly 5 seconds for the best voice match.",
                "audio.reference",
            )


# -- generation ------------------------------------------------------------

def _check_generation(spec: Any, report: Report) -> None:
    gen = spec.generation

    if gen.stages not in (1, 2):
        report.error(
            "stages.unsupported",
            f"{gen.stages} stages is not something the Director compiles.",
            "Use 1 (fast) or 2 (generate then refine).",
            "generation.stages",
        )

    for name, sigmas, needed in (
        ("sigmas_stage1", gen.sigmas_stage1, True),
        ("sigmas_stage2", gen.sigmas_stage2, gen.stages > 1),
    ):
        if not needed:
            continue
        if len(sigmas) < 2:
            report.error(
                f"{name}.too_short",
                f"The {'first' if name.endswith('1') else 'refine'} stage has no usable sigma "
                f"schedule ({len(sigmas)} value(s)).",
                "Pick a quality preset to restore the schedule.",
                f"generation.{name}",
            )
            continue
        if sigmas[-1] != 0.0:
            report.warn(
                f"{name}.no_zero",
                f"The {'first' if name.endswith('1') else 'refine'} stage's sigma schedule does "
                f"not end at 0, so sampling will stop early.",
                "Append 0.0 to the schedule.",
                f"generation.{name}",
            )
        if any(b > a for a, b in zip(sigmas, sigmas[1:])):
            report.error(
                f"{name}.not_descending",
                f"The {'first' if name.endswith('1') else 'refine'} stage's sigma schedule is "
                f"not descending.",
                "Sigmas must fall from high noise to 0.",
                f"generation.{name}",
            )

    if gen.seed_mode == "list" and not gen.seed_list:
        report.error(
            "seed.list_empty",
            "Seed mode is “list” but no seeds are listed.",
            "Add seeds, or switch to Fixed or Random.",
            "generation.seed_list",
        )

    if not 0.0 <= gen.first_frame_strength <= 1.0:
        report.error(
            "strength.first_frame_range",
            f"First-frame strength is {gen.first_frame_strength:g}; it must be between 0 and 1.",
            "0.7 is the value the reference graphs use.",
            "generation.first_frame_strength",
        )

    if not 0 <= gen.img_compression <= 100:
        report.error(
            "img_compression.range",
            f"Guide compression is {gen.img_compression}; it is a CRF value from 0 to 100.",
            "18 is the value the reference graphs use.",
            "generation.img_compression",
        )

    dec = gen.decode
    if dec.tile_size <= dec.overlap:
        report.error(
            "decode.overlap_too_large",
            f"Decode tile size ({dec.tile_size}) must be larger than the overlap ({dec.overlap}).",
            "Try tile 512 with overlap 64.",
            "generation.decode",
        )
    if dec.temporal_size <= dec.temporal_overlap:
        report.error(
            "decode.temporal_overlap_too_large",
            f"Decode temporal size ({dec.temporal_size}) must be larger than the temporal "
            f"overlap ({dec.temporal_overlap}).",
            "Try temporal size 64 with overlap 8.",
            "generation.decode",
        )

    if spec.project.mode != "t2a" and gen.stages > 1:
        stage1_w = snap_dim(spec.project.width // max(1, gen.stage1_divisor))
        stage1_h = snap_dim(spec.project.height // max(1, gen.stage1_divisor))
        if stage1_w < 64 or stage1_h < 64:
            report.error(
                "stage1.too_small",
                f"The first stage would run at {stage1_w}×{stage1_h}, which is too small to "
                f"hold composition.",
                "Raise the output resolution, or use a single stage.",
                "generation.stage1_divisor",
            )


# -- capabilities ----------------------------------------------------------

def _check_capabilities(spec: Any, caps: Any, report: Report) -> None:
    """Refuse what the installation genuinely cannot do — with the reason."""
    mode = spec.project.mode
    flag = _MODE_FEATURE.get(mode)
    if flag and not caps.has(flag):
        report.error(
            f"caps.mode.{mode}",
            f"{_MODE_LABEL.get(mode, mode)} is not available in this installation: "
            f"{caps.reason(flag)}",
            "Install the missing component, or choose another mode.",
            "project.mode",
        )

    if spec.generation.stages > 1 and not caps.has("two_stage"):
        report.error(
            "caps.two_stage",
            f"The refine stage is not available: {caps.reason('two_stage')}",
            "Use a single stage until the upscaler is installed.",
            "generation.stages",
        )

    if spec.prompt.enhance.enabled and not caps.has("prompt_enhancer"):
        report.error(
            "caps.prompt_enhancer",
            f"Prompt enhancement is not available: {caps.reason('prompt_enhancer')}",
            "Turn enhancement off, or install the enhancer text encoder.",
            "prompt.enhance",
        )

    if spec.audio.reference and not caps.has("reference_audio"):
        report.error(
            "caps.reference_audio",
            f"Speaker-identity reference audio is not available: "
            f"{caps.reason('reference_audio')}",
            "Clear the reference clip, or update ComfyUI.",
            "audio.reference",
        )

    if spec.relay.enabled and spec.relay_active and not caps.has("prompt_relay"):
        report.warn(
            "caps.prompt_relay",
            f"Prompt Relay cannot run here: {caps.reason('prompt_relay')} "
            f"Per-region prompts will be merged into one.",
            "",
            "relay",
        )

    family = spec.models.family
    families = getattr(caps, "families", ())
    if families and family not in families:
        report.error(
            "caps.family",
            f"No compiler is available for model family {family!r}.",
            f"Available: {', '.join(families)}.",
            "models.family",
        )

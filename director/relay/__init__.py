"""Prompt Relay — per-region prompts in a single sampler pass.

One entry point, :func:`apply_relay`, which takes a model, a CLIP and a Spec and
returns a patched model plus conditioning. Everything it can fail at, it fails
softly: an unsupported model, attention another node has claimed, a tokenizer it
cannot read — each of those costs the user their per-region prompts and nothing
more.

The maths is in :mod:`director.relay.mask`, the prompt assembly in
:mod:`director.relay.tokens`, and the model patching in
:mod:`director.relay.patch`. See ``mask.py`` for provenance and licence.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any

from ..core.spec import Spec
from ..core.time import latent_frames_for
from ..core.validate import Report

log = logging.getLogger(__name__)

__all__ = ["apply_relay", "RelayResult", "clear_cache"]


class RelayResult:
    """What the relay produced, and what it had to give up doing so."""

    __slots__ = ("model", "conditioning", "report", "applied", "segments")

    def __init__(
        self,
        model: Any,
        conditioning: Any,
        report: Report,
        applied: bool,
        segments: int = 0,
    ) -> None:
        self.model = model
        self.conditioning = conditioning
        self.report = report
        self.applied = applied
        self.segments = segments


#: Patched model clones, keyed by the source model then by ``(node id, digest)``.
#: A WeakKeyDictionary so unloading the model drops the clones with it, and a
#: digest key so editing the timeline produces a new clone instead of silently
#: reusing a stale mask.
_CLONE_CACHE: "weakref.WeakKeyDictionary[Any, dict[tuple[str, str], Any]]" = (
    weakref.WeakKeyDictionary()
)


def clear_cache() -> None:
    """Drop every cached model clone. For tests and for the reload button."""
    _CLONE_CACHE.clear()


def apply_relay(
    model: Any,
    clip: Any,
    spec: Spec,
    *,
    latent: dict[str, Any] | None = None,
    node_id: str | None = None,
) -> RelayResult:
    """Encode the shot's prompts and, when there is more than one region, relay them.

    ``latent`` is optional. When given, the latent's own shape decides the token
    geometry, which is exact. When absent the geometry is derived from the Spec's
    frame count — right for every graph the Director compiles, and the reason
    this node does not require a latent connection just to encode text.
    """
    from .mask import build_segments, create_mask_fn, distribute_segment_lengths
    from .support import inspect_model
    from .tokens import get_raw_tokenizer, map_token_indices

    report = Report()
    from ..core.prompt import compile_prompt

    global_prompt = compile_prompt(spec)
    segments = [s for s in spec.segments if s.text.strip()]

    # -- the ordinary case: one prompt, no patching, no cost ---------------
    if not spec.relay.enabled or len(segments) < 2:
        text = _merge_single(global_prompt, segments)
        return RelayResult(model, _encode(clip, text), report, applied=False)

    support = inspect_model(model)
    if not support.ok:
        report.warn(
            "relay.model_unsupported",
            f"Per-region prompts could not be applied: {support.reason}",
            "The regions were merged into one prompt.",
            "relay",
        )
        return RelayResult(
            model, _encode(clip, _merge_all(global_prompt, segments)), report, applied=False
        )

    # -- token ranges ------------------------------------------------------
    locals_list = [s.text.strip() or global_prompt.strip() or "video" for s in segments]
    try:
        tokenizer = get_raw_tokenizer(clip)
        full_prompt, token_ranges = map_token_indices(tokenizer, global_prompt, locals_list)
    except Exception as exc:
        log.debug("[Director] Prompt Relay tokenisation failed: %s", exc)
        report.warn(
            "relay.tokenizer_unreadable",
            "Per-region prompts need to know where each region's words land in the "
            "encoded prompt, and this text encoder does not expose that.",
            "The regions were merged into one prompt.",
            "relay",
        )
        return RelayResult(
            model, _encode(clip, _merge_all(global_prompt, segments)), report, applied=False
        )

    conditioning = _encode(clip, full_prompt)

    # -- latent geometry ---------------------------------------------------
    latent_frames, tokens_per_frame = _geometry(spec, latent, support.patch_size)
    if latent_frames < len(segments):
        report.warn(
            "relay.too_short",
            f"{len(segments)} prompt regions do not fit in {latent_frames} latent frames.",
            "Lengthen the shot, or use fewer regions.",
            "segments",
        )
        return RelayResult(
            model, _encode(clip, _merge_all(global_prompt, segments)), report, applied=False
        )

    lengths = distribute_segment_lengths(
        len(segments),
        latent_frames,
        _latent_lengths([s.length for s in segments], support.temporal_stride, latent_frames),
    )

    mask_fn = create_mask_fn(
        build_segments(
            token_ranges,
            lengths,
            spec.relay.epsilon,
            {
                "video_strength": spec.relay.video_strength,
                "video_window_scale": spec.relay.video_window_scale,
                "audio_strength": spec.relay.audio_strength,
                "audio_window_scale": spec.relay.audio_window_scale,
            },
        ),
        tokens_per_frame,
        latent_frames,
    )

    # -- patch -------------------------------------------------------------
    # Imported here rather than at module scope: this is the first point that
    # genuinely needs ComfyUI, and everything above it must work without one.
    from .patch import apply_patches, get_current_node_id

    key = (node_id or get_current_node_id(), spec.digest())
    per_model = _CLONE_CACHE.setdefault(model, {})
    patched = per_model.get(key)
    if patched is None:
        patched = model.clone()
        failure = apply_patches(patched, support.arch, mask_fn)
        if failure:
            report.warn(
                "relay.patch_refused",
                f"Per-region prompts could not be applied: {failure}",
                "The regions were merged into one prompt.",
                "relay",
            )
            return RelayResult(
                model, _encode(clip, _merge_all(global_prompt, segments)), report,
                applied=False,
            )
        per_model[key] = patched

    # The mask is set on every call, not only on a cache miss: the clone is
    # reused across runs but the schedule may have moved.
    patched.model_options.setdefault("transformer_options", {})["promptrelay_mask_fn"] = mask_fn

    log.info(
        "[Director] Prompt Relay: %d regions over %d latent frames, lengths %s.",
        len(segments), latent_frames, lengths,
    )
    return RelayResult(patched, conditioning, report, applied=True, segments=len(segments))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _encode(clip: Any, text: str) -> Any:
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


def _merge_single(global_prompt: str, segments: list[Any]) -> str:
    """The prompt for a shot with at most one region."""
    local = segments[0].text.strip() if segments else ""
    if global_prompt.strip() and local:
        return f"{global_prompt.strip()}, {local}"
    return local or global_prompt.strip()


def _merge_all(global_prompt: str, segments: list[Any]) -> str:
    """Fallback when relay cannot run: one prompt carrying every region's words.

    Order is preserved so the sequence of events still reads correctly, even
    though the timing no longer binds.
    """
    parts = [global_prompt.strip()] + [s.text.strip() for s in segments]
    return ", ".join(p for p in parts if p)


def _geometry(
    spec: Spec, latent: dict[str, Any] | None, patch_size: tuple[int, int, int]
) -> tuple[int, int]:
    """``(latent_frames, tokens_per_frame)``.

    Taken from the latent when one is connected — that is exact. Otherwise
    derived from the Spec, using the same arithmetic ``EmptyLTXVLatentVideo``
    uses, so the two agree.
    """
    samples = (latent or {}).get("samples")
    if samples is not None and getattr(samples, "ndim", 0) == 5:
        _, _, frames, height, width = samples.shape
        tokens = (height // patch_size[1]) * (width // patch_size[2])
        return int(frames), max(1, int(tokens))

    from ..core.time import SPATIAL_STRIDE

    frames = latent_frames_for(spec.project.frames)
    tokens = (spec.project.height // SPATIAL_STRIDE) * (spec.project.width // SPATIAL_STRIDE)
    return frames, max(1, tokens)


def _latent_lengths(
    pixel_lengths: list[int], temporal_stride: int, latent_frames: int
) -> list[int]:
    """Pixel-space region lengths as whole latent frames.

    Largest-remainder apportionment, so the parts sum to the whole: rounding
    each region independently would leave the last one short or long, and the
    penalty matrix is indexed by cumulative position.
    """
    if not pixel_lengths:
        return []
    total = sum(pixel_lengths)
    if total <= 0:
        return [1] * len(pixel_lengths)

    naive = max(1, round(total / temporal_stride))
    target = min(latent_frames, naive)
    # Within one frame of full coverage means the user meant full coverage.
    if target >= latent_frames - 1:
        target = latent_frames

    exact = [p * target / total for p in pixel_lengths]
    result = [int(e) for e in exact]
    shortfall = target - sum(result)
    if shortfall > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(shortfall):
            result[order[k % len(order)]] += 1

    # Every region needs at least one latent frame; take from the largest.
    for i, value in enumerate(result):
        if value >= 1:
            continue
        biggest = max(range(len(result)), key=lambda j: result[j])
        if result[biggest] > 1:
            result[biggest] -= 1
            result[i] = 1

    return result

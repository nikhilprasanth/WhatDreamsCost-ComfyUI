"""Capabilities as feature flags, each with a reason when it is off.

The rule: a feature the installation cannot provide is **hidden with an
explanation**, never offered and then crashed into. The reason string is what
the UI shows on hover and what the validator quotes back, so it is written for a
filmmaker, not for a stack trace.

Each flag names the nodes or models that would turn it on, so "install this" is
always actionable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .probe import Probe, probe as run_probe

__all__ = ["Feature", "Capabilities", "capabilities", "FEATURES"]


@dataclass(frozen=True)
class Feature:
    """One capability: how to test for it, and what to say when it is absent."""

    name: str
    label: str
    #: Predicate over a probe.
    test: Callable[[Probe], bool]
    #: Sentence explaining the absence. Ends with a full stop; no node names
    #: unless a user would type them into a search box.
    reason: str


def _nodes(*names: str) -> Callable[[Probe], bool]:
    return lambda p: p.has_node(*names)


def _either(*names: str) -> Callable[[Probe], bool]:
    return lambda p: any(p.has_node(name) for name in names)


def _both(*tests: Callable[[Probe], bool]) -> Callable[[Probe], bool]:
    return lambda p: all(test(p) for test in tests)


#: Every feature the Director gates on, in the order the UI shows them.
FEATURES: tuple[Feature, ...] = (
    Feature(
        "core_ltx", "LTX core nodes",
        _nodes("EmptyLTXVLatentVideo", "LTXVConditioning", "SamplerCustomAdvanced"),
        "ComfyUI's built-in LTX nodes were not found. Update ComfyUI.",
    ),
    Feature(
        "audio", "Audio",
        _nodes("LTXVEmptyLatentAudio", "LTXVAudioVAEDecode", "LTXVConcatAVLatent"),
        "This ComfyUI build has no LTX audio nodes, so video is generated silently. "
        "Update ComfyUI to generate sound.",
    ),
    Feature(
        "audio_only", "Text to audio",
        _nodes("LTXVAudioOnlyModel", "LTXVAudioOnlyEmptyVideoLatent"),
        "Audio-only generation needs the LTXVideo custom nodes from Lightricks.",
    ),
    Feature(
        "audio_import", "Import audio",
        _nodes("LTXVAudioVAEEncode", "TrimAudioDuration"),
        "Importing a soundtrack needs ComfyUI's LTX audio encode nodes. Update ComfyUI.",
    ),
    Feature(
        "freeze_latent", "Freeze latents",
        _nodes("LTXVFreezeLatent"),
        "Holding a supplied soundtrack unchanged needs a newer ComfyUI.",
    ),
    Feature(
        "mid_timeline_guides", "Keyframes",
        _nodes("LTXVAddGuide", "LTXVCropGuides"),
        "Pinning frames other than the first needs ComfyUI's LTX guide nodes. Update ComfyUI.",
    ),
    Feature(
        "first_frame", "First frame",
        _nodes("LTXVImgToVideoInplace"),
        "Starting from an image needs ComfyUI's LTX image-to-video node. Update ComfyUI.",
    ),
    Feature(
        "iclora", "IC-LoRA control",
        _nodes("LTXICLoRALoaderModelOnly", "LTXAddVideoICLoRAGuide"),
        "IC-LoRA control needs the LTXVideo custom nodes from Lightricks.",
    ),
    Feature(
        "dual_cfg", "Separate audio and video guidance",
        _nodes("LTXVDualCFGGuider"),
        "Separate guidance for picture and sound needs a newer ComfyUI; one scale "
        "is used for both.",
    ),
    Feature(
        "modality_guidance", "Lip-sync guidance",
        _nodes("LTXVModalityGuidance"),
        "Stronger audio-video coupling needs a newer ComfyUI.",
    ),
    Feature(
        "reference_audio", "Voice matching",
        _nodes("LTXVReferenceAudio"),
        "Matching a speaker's voice from a reference clip needs a newer ComfyUI.",
    ),
    Feature(
        "dfr_keyframes", "Detail keyframes",
        _nodes("LTXVAddGeneratedKeyframes", "LTXVSeparateGeneratedKeyframes"),
        "Generated detail keyframes need a newer ComfyUI and a checkpoint trained for them.",
    ),
    Feature(
        "duration_predictor", "Suggest duration",
        _both(_nodes("LTXVDurationPredictor"), lambda p: bool(p.models.model_patch)),
        "Suggesting a shot length needs the LTX duration head in ComfyUI/models/model_patches.",
    ),
    Feature(
        "prompt_enhancer", "Prompt enhancement",
        _both(_nodes("TextGenerateLTX2Prompt"), lambda p: bool(p.enhancers())),
        "Prompt enhancement needs the small Gemma-4 encoder in "
        "ComfyUI/models/text_encoders.",
    ),
    Feature(
        "two_stage", "Refine pass",
        _both(
            _nodes("LTXVLatentUpsampler", "LatentUpscaleModelLoader"),
            lambda p: bool(p.upscalers()),
        ),
        "The refine pass needs a latent spatial upscaler in "
        "ComfyUI/models/latent_upscale_models.",
    ),
    Feature(
        "tiled_decode", "Tiled decode",
        _nodes("VAEDecodeTiled"),
        "Tiled decoding needs a newer ComfyUI; decoding will use more memory.",
    ),
    Feature(
        "prompt_relay", "Prompt Relay",
        lambda p: p.available,
        "Prompt Relay needs a running ComfyUI with an LTX model loaded.",
    ),
    Feature(
        "sparse_tracks", "Motion paths",
        _nodes("LTXVSparseTrackEditor", "LTXVDrawTracks"),
        "Drawing motion paths needs the LTXVideo custom nodes from Lightricks.",
    ),
)

_BY_NAME = {f.name: f for f in FEATURES}


@dataclass
class Capabilities:
    """The answer the UI and the validator both read."""

    probe: Probe
    flags: dict[str, bool] = field(default_factory=dict)
    #: Families a compiler is registered for *and* models exist for.
    families: tuple[str, ...] = ()

    def has(self, name: str) -> bool:
        """True when the feature is usable. Unknown names are False, loudly logged."""
        if name not in self.flags:
            raise KeyError(
                f"Unknown capability {name!r}. Known: {', '.join(sorted(self.flags))}."
            )
        return self.flags[name]

    def reason(self, name: str) -> str:
        feature = _BY_NAME.get(name)
        if feature is None:
            return f"{name} is not a capability this build knows about."
        if not self.probe.available:
            return "ComfyUI is not running, so what is installed could not be checked."
        return feature.reason

    def missing(self) -> dict[str, str]:
        """Every unavailable feature, mapped to why."""
        return {n: self.reason(n) for n, ok in self.flags.items() if not ok}

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.probe.available,
            "flags": dict(sorted(self.flags.items())),
            "labels": {f.name: f.label for f in FEATURES},
            "missing": self.missing(),
            "families": list(self.families),
            "models": self.probe.to_dict()["grouped"],
        }


def capabilities(*, refresh: bool = False) -> Capabilities:
    """Probe the installation and turn it into feature flags."""
    from ..compiler.base import available_families  # local: avoids an import cycle

    p = run_probe(refresh=refresh)
    flags = {f.name: bool(f.test(p)) for f in FEATURES}

    registered = set(available_families())
    installed = set(p.families())
    # Offer a family only when both a compiler and a model exist for it. When
    # nothing can be classified by filename, fall back to what we can compile —
    # a user with oddly-named files should still be able to work.
    families = tuple(sorted(registered & installed, reverse=True)) or tuple(
        sorted(registered, reverse=True)
    )

    return Capabilities(probe=p, flags=flags, families=families)

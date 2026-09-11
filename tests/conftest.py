"""Test configuration.

The whole suite must run on a machine with no ComfyUI, no torch and no model
weights. Nothing here imports any of them; anything that would is skipped with
a reason rather than erroring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The package under test lives at the repo root, which is this file's grandparent.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from director.core import new_spec  # noqa: E402
from director.core.spec import (  # noqa: E402
    AudioClip,
    LoraRef,
    MediaEntry,
    Reference,
    Segment,
    Spec,
)


def _image(name: str = "frame.png", **kw) -> MediaEntry:
    return MediaEntry(filename=name, kind="image", width=1280, height=704, **kw)


def _video(name: str = "clip.mp4", **kw) -> MediaEntry:
    return MediaEntry(
        filename=name, kind="video", width=1280, height=704, duration=6.0, fps=24.0, **kw
    )


def _audio(name: str = "voice.wav", **kw) -> MediaEntry:
    return MediaEntry(filename=name, kind="audio", duration=8.0, **kw)


def add_media(spec: Spec, entry: MediaEntry) -> str:
    """Register a media entry and return its id."""
    spec.media[entry.id] = entry
    return entry.id


def add_keyframe(spec: Spec, *, anchor: str = "start", frame: int = 0, **kw) -> Reference:
    ref = Reference(
        role="keyframe", media=add_media(spec, _image()), anchor=anchor, **kw
    )
    ref.at.frame = frame
    spec.references.append(ref)
    return ref


@pytest.fixture
def base_spec() -> Spec:
    """A valid, complete LTX-2.5 text-to-video spec.

    Every mode fixture builds on this, so a rule that only fires on a broken
    spec cannot be masked by unrelated errors.
    """
    spec = new_spec()
    spec.prompt.raw = "A lighthouse beam sweeps across a storm-lit harbour."
    spec.prompt.negative = "blurry, low contrast"
    spec.models.family = "ltx2.5"
    spec.models.unet = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    spec.models.clip = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    spec.models.vae = "ltx-2.5-video-vae-bf16.safetensors"
    spec.models.audio_vae = "ltx-2.5-audio-vae-bf16.safetensors"
    return spec


@pytest.fixture
def spec_i2v(base_spec: Spec) -> Spec:
    base_spec.project.mode = "i2v"
    add_keyframe(base_spec, anchor="start")
    return base_spec


@pytest.fixture
def spec_fflf(base_spec: Spec) -> Spec:
    base_spec.project.mode = "fflf"
    add_keyframe(base_spec, anchor="start")
    add_keyframe(base_spec, anchor="end")
    return base_spec


@pytest.fixture
def spec_keyframes(base_spec: Spec) -> Spec:
    base_spec.project.mode = "keyframes"
    add_keyframe(base_spec, anchor="start")
    add_keyframe(base_spec, anchor="index", frame=48)
    add_keyframe(base_spec, anchor="end")
    return base_spec


@pytest.fixture
def spec_a2v(base_spec: Spec) -> Spec:
    base_spec.project.mode = "a2v"
    base_spec.audio.mode = "import"
    media_id = add_media(base_spec, _audio())
    base_spec.audio.clips.append(AudioClip(media=media_id, start=0, length=121))
    return base_spec


@pytest.fixture
def spec_t2a(base_spec: Spec) -> Spec:
    base_spec.project.mode = "t2a"
    base_spec.prompt.raw = "Rain on a tin roof, distant thunder."
    return base_spec


@pytest.fixture
def spec_iclora(base_spec: Spec) -> Spec:
    base_spec.project.mode = "iclora"
    ref = Reference(role="motion", media=add_media(base_spec, _video()), anchor="start")
    base_spec.references.append(ref)
    base_spec.models.loras.append(
        LoraRef(name="ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors", kind="iclora")
    )
    return base_spec


@pytest.fixture
def spec_two_stage(base_spec: Spec) -> Spec:
    base_spec.generation.stages = 2
    base_spec.models.upscaler = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    return base_spec


@pytest.fixture
def spec_relay(base_spec: Spec) -> Spec:
    base_spec.segments = [
        Segment(start=0, length=40, text="the beam sweeps left"),
        Segment(start=40, length=40, text="rain intensifies"),
        Segment(start=80, length=41, text="the beam returns"),
    ]
    return base_spec

"""Golden-file tests for compiled workflows.

The compiler is pure, so its output is a fact that can be committed. That turns
any change in what the Director emits into a reviewable diff rather than
something you discover in ComfyUI three weeks later.

Regenerate deliberately, and read the diff::

    DIRECTOR_UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py

Fixtures use fixed ids so the output is stable; the compiler itself never
generates a random id.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from director.compiler import compile_spec
from director.core.spec import (
    AudioClip,
    LoraRef,
    MediaEntry,
    Reference,
    Segment,
    Spec,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("DIRECTOR_UPDATE_GOLDEN") == "1"


# --------------------------------------------------------------------------
# deterministic fixtures
#
# Built here rather than reused from conftest so the ids are fixed and the
# golden files do not churn.
# --------------------------------------------------------------------------

def _spec(mode: str) -> Spec:
    spec = Spec()
    spec.meta.name = f"golden-{mode}"
    spec.project.mode = mode  # type: ignore[assignment]
    spec.project.fps = 24.0
    spec.project.width, spec.project.height = 1280, 704
    spec.project.frames = 121
    spec.prompt.raw = "A lighthouse beam sweeps across a storm-lit harbour."
    spec.prompt.negative = "blurry, low contrast, washed out"
    spec.generation.seed = 42
    spec.generation.sigmas_stage1 = [
        1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
    ]
    spec.generation.sigmas_stage2 = [0.85, 0.725, 0.4219, 0.0]
    spec.models.family = "ltx2.5"
    spec.models.unet = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    spec.models.clip = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    spec.models.vae = "ltx-2.5-video-vae-bf16.safetensors"
    spec.models.audio_vae = "ltx-2.5-audio-vae-bf16.safetensors"
    return spec


def _image(spec: Spec, media_id: str, filename: str) -> str:
    spec.media[media_id] = MediaEntry(
        id=media_id, filename=filename, kind="image", width=1280, height=704,
    )
    return media_id


def _keyframe(spec: Spec, ref_id: str, media_id: str, anchor: str, frame: int = 0) -> None:
    ref = Reference(id=ref_id, role="keyframe", media=media_id, anchor=anchor)  # type: ignore[arg-type]
    ref.at.frame = frame
    spec.references.append(ref)


def golden_specs() -> dict[str, Spec]:
    specs: dict[str, Spec] = {}

    specs["t2v"] = _spec("t2v")

    i2v = _spec("i2v")
    _keyframe(i2v, "ref_000000000a", _image(i2v, "med_000000000a", "first.png"), "start")
    specs["i2v"] = i2v

    fflf = _spec("fflf")
    _keyframe(fflf, "ref_000000000a", _image(fflf, "med_000000000a", "first.png"), "start")
    _keyframe(fflf, "ref_000000000b", _image(fflf, "med_000000000b", "last.png"), "end")
    specs["fflf"] = fflf

    keys = _spec("keyframes")
    _keyframe(keys, "ref_000000000a", _image(keys, "med_000000000a", "first.png"), "start")
    _keyframe(keys, "ref_000000000b", _image(keys, "med_000000000b", "mid.png"), "index", 48)
    _keyframe(keys, "ref_000000000c", _image(keys, "med_000000000c", "last.png"), "end")
    specs["keyframes"] = keys

    two_stage = _spec("t2v")
    two_stage.generation.stages = 2
    two_stage.models.upscaler = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    specs["t2v_two_stage"] = two_stage

    i2v_two = _spec("i2v")
    _keyframe(i2v_two, "ref_000000000a", _image(i2v_two, "med_000000000a", "first.png"), "start")
    i2v_two.generation.stages = 2
    i2v_two.models.upscaler = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    specs["i2v_two_stage"] = i2v_two

    a2v = _spec("a2v")
    a2v.media["med_000000000d"] = MediaEntry(
        id="med_000000000d", filename="score.wav", kind="audio", duration=12.0,
    )
    a2v.audio.mode = "import"
    a2v.audio.clips.append(
        AudioClip(id="aud_000000000a", media="med_000000000d", start=0, length=121)
    )
    specs["a2v"] = a2v

    t2a = _spec("t2a")
    t2a.prompt.raw = "Rain on a tin roof, distant thunder rolling in."
    specs["t2a"] = t2a

    iclora = _spec("iclora")
    iclora.media["med_000000000e"] = MediaEntry(
        id="med_000000000e", filename="drive.mp4", kind="video",
        width=1280, height=704, duration=6.0, fps=24.0,
    )
    iclora.references.append(Reference(
        id="ref_000000000d", role="motion", media="med_000000000e", anchor="start",
    ))
    iclora.models.loras.append(LoraRef(
        name="ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors", kind="iclora",
    ))
    specs["iclora"] = iclora

    relay = _spec("t2v")
    relay.segments = [
        Segment(id="seg_000000000a", start=0, length=40, text="the beam sweeps left"),
        Segment(id="seg_000000000b", start=40, length=40, text="rain intensifies"),
        Segment(id="seg_000000000c", start=80, length=41, text="the beam returns"),
    ]
    specs["relay"] = relay

    enhanced = _spec("t2v")
    enhanced.prompt.enhance.enabled = True
    enhanced.models.enhancer_clip = "gemma4_e2b_it_bf16.safetensors"
    specs["prompt_enhanced"] = enhanced

    return specs


GOLDEN = golden_specs()


# --------------------------------------------------------------------------
# the tests
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(GOLDEN))
@pytest.mark.parametrize("layout", ["flat", "subgraphs"])
def test_compiled_workflow_matches_its_golden_file(name: str, layout: str) -> None:
    result = compile_spec(GOLDEN[name], layout=layout)  # type: ignore[arg-type]
    assert result.ok, [str(d) for d in result.report.errors]

    path = GOLDEN_DIR / f"{name}.{layout}.json"
    actual = json.dumps(result.workflow, indent=1, sort_keys=True, ensure_ascii=False) + "\n"

    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        pytest.skip(f"regenerated {path.name}")

    assert path.exists(), (
        f"{path.name} is missing. Regenerate with "
        f"DIRECTOR_UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py"
    )
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{path.name} differs from what the compiler now emits. If that is "
        f"intended, regenerate with DIRECTOR_UPDATE_GOLDEN=1 and review the diff."
    )


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_specs_round_trip(name: str) -> None:
    spec = GOLDEN[name]
    assert Spec.from_json(spec.to_json()).to_dict() == spec.to_dict()


def test_every_mode_has_a_golden_file() -> None:
    from director.compiler.ltx25 import LTX25Compiler

    covered = {GOLDEN[name].project.mode for name in GOLDEN}
    missing = set(LTX25Compiler.modes) - covered - {"continue"}
    assert not missing, f"modes with no golden file: {sorted(missing)}"

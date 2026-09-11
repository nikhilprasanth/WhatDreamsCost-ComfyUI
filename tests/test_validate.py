"""Validation rules.

Two things are asserted throughout: that the rule fires on the right input, and
that what the user is shown is a *sentence with a way out*. The second matters
as much as the first — the whole point of this layer is that nobody ever sees
``KeyError: conditioning_1`` again.
"""

from __future__ import annotations

import pytest

from director.core import validate
from director.core.spec import AudioClip, LoraRef, MediaEntry, Reference, Segment, Spec
from director.core.validate import Report


def codes(report: Report) -> set[str]:
    return {d.code for d in report}


def find(report: Report, code: str):
    for d in report:
        if d.code == code:
            return d
    raise AssertionError(f"expected diagnostic {code!r}, got {sorted(codes(report))}")


# --------------------------------------------------------------------------
# the contract every diagnostic must honour
# --------------------------------------------------------------------------

_PYTHONISMS = ("KeyError", "TypeError", "ValueError", "Traceback", "None", "NoneType",
               "__", "self.", "list index")


@pytest.mark.parametrize(
    "fixture",
    ["base_spec", "spec_i2v", "spec_fflf", "spec_keyframes", "spec_a2v",
     "spec_t2a", "spec_iclora", "spec_two_stage", "spec_relay"],
)
def test_valid_specs_produce_no_errors(fixture: str, request: pytest.FixtureRequest) -> None:
    report = validate(request.getfixturevalue(fixture))
    assert report.ok, [str(d) for d in report.errors]


def test_every_diagnostic_reads_as_english() -> None:
    """Sweep a deliberately broken spec and check the wording of everything raised."""
    spec = Spec()  # empty: trips a large number of rules at once
    spec.project.frames = 120
    spec.project.width = 700
    spec.project.mode = "fflf"
    spec.generation.stages = 2
    spec.generation.sigmas_stage1 = []
    spec.generation.seed_mode = "list"
    spec.segments = [Segment(start=-5, length=0, text="x"), Segment(start=0, length=8, text="y")]
    spec.audio.mode = "import"
    report = validate(spec)

    assert report.errors, "the sweep spec should be broken"
    for d in report:
        assert d.message, f"{d.code} has no message"
        assert d.message[0].isupper() or d.message[0].isdigit(), f"{d.code} is not a sentence"
        assert d.message.rstrip()[-1] in ".!?", f"{d.code} has no terminal punctuation"
        for term in _PYTHONISMS:
            assert term not in d.message, f"{d.code} leaks {term!r} at the user"
        if d.level == "error":
            assert d.fix, f"{d.code} tells the user what is wrong but not what to do"


def test_errors_block_and_warnings_do_not(base_spec: Spec) -> None:
    base_spec.project.fps = 90  # warning only
    report = validate(base_spec)
    assert "fps.high" in codes(report)
    assert report.ok


def test_raise_if_failed_joins_sentences() -> None:
    report = validate(Spec())
    with pytest.raises(ValueError) as excinfo:
        report.raise_if_failed("build this shot")
    text = str(excinfo.value)
    assert "cannot build this shot" in text
    assert "•" in text
    assert "Traceback" not in text


def test_raise_if_failed_is_silent_when_clean(base_spec: Spec) -> None:
    validate(base_spec).raise_if_failed("build this shot")


# --------------------------------------------------------------------------
# project geometry
# --------------------------------------------------------------------------

def test_frame_count_must_be_8k_plus_one(base_spec: Spec) -> None:
    base_spec.project.frames = 120
    d = find(validate(base_spec), "frames.not_8k1")
    # The fix must name both neighbours, so the user chooses rather than being surprised.
    assert "121" in d.fix and "113" in d.fix


def test_dimensions_must_be_multiples_of_32(base_spec: Spec) -> None:
    base_spec.project.width = 1281
    d = find(validate(base_spec), "width.not_multiple_32")
    assert "1280" in d.fix


def test_audio_only_mode_ignores_video_geometry(spec_t2a: Spec) -> None:
    spec_t2a.project.width = 1281
    spec_t2a.project.height = 3
    assert "width.not_multiple_32" not in codes(validate(spec_t2a))


def test_unusable_fps_is_an_error_not_a_crash(base_spec: Spec) -> None:
    base_spec.project.fps = 0
    assert "fps.invalid" in codes(validate(base_spec))


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def test_missing_components_are_named_individually(base_spec: Spec) -> None:
    base_spec.models.unet = ""
    base_spec.models.vae = ""
    found = codes(validate(base_spec))
    assert "model.missing.unet" in found and "model.missing.vae" in found


def test_audio_vae_is_not_required_for_plain_video(base_spec: Spec) -> None:
    base_spec.models.audio_vae = ""
    assert "model.missing.audio_vae" not in codes(validate(base_spec))


def test_audio_vae_is_required_for_audio_to_video(spec_a2v: Spec) -> None:
    spec_a2v.models.audio_vae = ""
    assert "model.missing.audio_vae" in codes(validate(spec_a2v))


def test_refine_stage_requires_an_upscaler(spec_two_stage: Spec) -> None:
    spec_two_stage.models.upscaler = None
    d = find(validate(spec_two_stage), "model.missing.upscaler")
    assert "single stage" in d.fix


def test_prompt_enhancement_requires_its_encoder(base_spec: Spec) -> None:
    base_spec.prompt.enhance.enabled = True
    assert "model.missing.enhancer" in codes(validate(base_spec))


def test_iclora_mode_requires_an_iclora(spec_iclora: Spec) -> None:
    spec_iclora.models.loras.clear()
    assert "iclora.none_selected" in codes(validate(spec_iclora))


def test_duplicate_loras_warn_about_stacking(base_spec: Spec) -> None:
    base_spec.models.loras = [LoraRef(name="a.safetensors"), LoraRef(name="a.safetensors")]
    assert "lora.duplicate" in codes(validate(base_spec))


# --------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------

def test_reference_to_deleted_media_is_reported(spec_i2v: Spec) -> None:
    spec_i2v.references[0].media = "med_gone"
    d = find(validate(spec_i2v), "ref.media_missing")
    assert "Re-add" in d.fix


def test_reference_past_the_end_names_the_time(base_spec: Spec) -> None:
    entry = MediaEntry(filename="x.png", kind="image")
    base_spec.media[entry.id] = entry
    ref = Reference(role="keyframe", media=entry.id, anchor="index")
    ref.at.frame = 500
    base_spec.references.append(ref)
    d = find(validate(base_spec), "ref.past_end")
    assert ":" in d.message, "the message should quote a timecode"


def test_two_keyframes_on_one_frame_warn(base_spec: Spec) -> None:
    for _ in range(2):
        entry = MediaEntry(filename="x.png", kind="image")
        base_spec.media[entry.id] = entry
        ref = Reference(role="keyframe", media=entry.id, anchor="index")
        ref.at.frame = 40
        base_spec.references.append(ref)
    assert "ref.duplicate_position" in codes(validate(base_spec))


def test_i2v_without_a_first_frame_is_an_error(spec_i2v: Spec) -> None:
    spec_i2v.references.clear()
    d = find(validate(spec_i2v), "mode.i2v_no_first_frame")
    assert "anchor to Start" in d.fix


def test_fflf_reports_each_missing_end_separately(spec_fflf: Spec) -> None:
    spec_fflf.references = [r for r in spec_fflf.references if r.anchor == "start"]
    found = codes(validate(spec_fflf))
    assert "mode.fflf_no_last" in found and "mode.fflf_no_first" not in found


def test_iclora_without_a_guide_clip_is_an_error(spec_iclora: Spec) -> None:
    spec_iclora.references.clear()
    assert "mode.iclora_no_guide" in codes(validate(spec_iclora))


def test_keyframes_in_t2v_are_reported_as_ignored_not_wrong(spec_keyframes: Spec) -> None:
    spec_keyframes.project.mode = "t2v"
    d = find(validate(spec_keyframes), "ref.ignored_in_t2v")
    assert d.level == "info"


def test_video_guides_off_the_stride_grid_warn_with_the_snapped_frame(base_spec: Spec) -> None:
    base_spec.project.mode = "keyframes"
    entry = MediaEntry(filename="clip.mp4", kind="video", duration=2.0, fps=24.0)
    base_spec.media[entry.id] = entry
    ref = Reference(role="keyframe", media=entry.id, anchor="index")
    ref.at.frame = 13
    base_spec.references.append(ref)
    d = find(validate(base_spec), "ref.video_index_snapped")
    assert "frame 8" in d.message


def test_enabled_but_empty_reference_warns(base_spec: Spec) -> None:
    base_spec.references.append(Reference(role="character", media=None))
    assert "ref.no_media" in codes(validate(base_spec))


# --------------------------------------------------------------------------
# prompt relay segments
# --------------------------------------------------------------------------

def test_overlapping_regions_warn_without_blocking(spec_relay: Spec) -> None:
    spec_relay.segments[1].start = 20
    report = validate(spec_relay)
    assert "segment.overlap" in codes(report)
    assert report.ok


def test_more_regions_than_latent_frames_is_an_error(base_spec: Spec) -> None:
    base_spec.project.frames = 9  # 2 latent frames
    base_spec.segments = [Segment(start=i, length=1, text=f"p{i}") for i in range(5)]
    d = find(validate(base_spec), "relay.too_many_segments")
    assert "latent frames" in d.message


def test_single_populated_region_is_explained_not_flagged(base_spec: Spec) -> None:
    base_spec.segments = [Segment(start=0, length=121, text="one")]
    d = find(validate(base_spec), "relay.single_segment")
    assert d.level == "info"


def test_epsilon_out_of_range_is_an_error(spec_relay: Spec) -> None:
    spec_relay.relay.epsilon = 5.0
    assert "relay.epsilon_range" in codes(validate(spec_relay))


def test_region_past_the_end_warns_that_it_will_be_clipped(spec_relay: Spec) -> None:
    spec_relay.segments[-1].length = 400
    d = find(validate(spec_relay), "segment.past_end")
    assert "clipped" in d.message


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def test_a2v_without_a_clip_is_an_error(spec_a2v: Spec) -> None:
    spec_a2v.audio.clips.clear()
    assert "audio.a2v_no_clip" in codes(validate(spec_a2v))


def test_clip_longer_than_its_file_warns_about_silence(spec_a2v: Spec) -> None:
    spec_a2v.audio.clips[0].length = 24 * 30  # 30 s from an 8 s file
    d = find(validate(spec_a2v), "audio.trim_past_end")
    assert "silent" in d.message


def test_identity_reference_duration_is_advisory(base_spec: Spec) -> None:
    entry = MediaEntry(filename="voice.wav", kind="audio", duration=45.0)
    base_spec.media[entry.id] = entry
    base_spec.audio.reference = entry.id
    d = find(validate(base_spec), "audio.reference_duration")
    assert d.level == "warning" and "5" in d.fix


def test_dangling_audio_media_is_reported(spec_a2v: Spec) -> None:
    spec_a2v.audio.clips[0].media = "med_gone"
    assert "audio.media_missing" in codes(validate(spec_a2v))


def test_import_mode_with_no_clips_is_an_error(base_spec: Spec) -> None:
    base_spec.audio.mode = "import"
    assert "audio.no_clips" in codes(validate(base_spec))


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def test_sigma_schedule_must_descend_to_zero(base_spec: Spec) -> None:
    base_spec.generation.sigmas_stage1 = [0.2, 0.9, 0.0]
    assert "sigmas_stage1.not_descending" in codes(validate(base_spec))

    base_spec.generation.sigmas_stage1 = [1.0, 0.5, 0.2]
    assert "sigmas_stage1.no_zero" in codes(validate(base_spec))


def test_refine_sigmas_are_ignored_on_a_single_stage_shot(base_spec: Spec) -> None:
    base_spec.generation.sigmas_stage2 = []
    assert "sigmas_stage2.too_short" not in codes(validate(base_spec))


def test_refine_sigmas_are_required_on_a_two_stage_shot(spec_two_stage: Spec) -> None:
    spec_two_stage.generation.sigmas_stage2 = []
    assert "sigmas_stage2.too_short" in codes(validate(spec_two_stage))


def test_empty_seed_list_is_an_error(base_spec: Spec) -> None:
    base_spec.generation.seed_mode = "list"
    assert "seed.list_empty" in codes(validate(base_spec))


def test_decode_overlap_cannot_exceed_the_tile(base_spec: Spec) -> None:
    base_spec.generation.decode.overlap = 999
    assert "decode.overlap_too_large" in codes(validate(base_spec))


def test_stage_one_cannot_be_divided_into_nothing(spec_two_stage: Spec) -> None:
    spec_two_stage.generation.stage1_divisor = 32
    d = find(validate(spec_two_stage), "stage1.too_small")
    assert "composition" in d.message


def test_unsupported_stage_count_is_rejected(base_spec: Spec) -> None:
    base_spec.generation.stages = 7
    assert "stages.unsupported" in codes(validate(base_spec))


# --------------------------------------------------------------------------
# capabilities
# --------------------------------------------------------------------------

class FakeCaps:
    """Minimal stand-in for the probe, so validation is testable without ComfyUI."""

    families = ("ltx2.5",)

    def __init__(self, **flags: bool) -> None:
        self._flags = flags

    def has(self, flag: str) -> bool:
        return self._flags.get(flag, True)

    def reason(self, flag: str) -> str:
        return f"{flag} is not installed."


def test_capability_rules_are_skipped_when_no_probe_is_supplied(spec_a2v: Spec) -> None:
    # The validator must stay usable in tests and before the probe has answered.
    assert validate(spec_a2v, None).ok


def test_unavailable_mode_is_refused_with_the_reason(spec_a2v: Spec) -> None:
    d = find(validate(spec_a2v, FakeCaps(audio=False)), "caps.mode.a2v")
    assert "is not installed" in d.message
    assert "another mode" in d.fix


def test_missing_upscaler_capability_blocks_the_refine_stage(spec_two_stage: Spec) -> None:
    assert "caps.two_stage" in codes(validate(spec_two_stage, FakeCaps(two_stage=False)))


def test_prompt_relay_degrades_with_a_warning_rather_than_failing(spec_relay: Spec) -> None:
    report = validate(spec_relay, FakeCaps(prompt_relay=False))
    d = find(report, "caps.prompt_relay")
    assert d.level == "warning"
    assert report.ok, "losing relay should not stop the shot rendering"


def test_unknown_model_family_lists_what_is_available(base_spec: Spec) -> None:
    base_spec.models.family = "ltx9"
    d = find(validate(base_spec, FakeCaps()), "caps.family")
    assert "ltx2.5" in d.fix

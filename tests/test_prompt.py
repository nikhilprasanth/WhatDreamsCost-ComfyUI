"""Prompt compilation.

The rule being defended: LTX-2.5's encoder wants prose, and the user's own
words are never rewritten. Director mode may order and punctuate; it may not
paraphrase.
"""

from __future__ import annotations

import pytest

from director.core.prompt import (
    CAMERA_MOVES,
    SHOT_SIZES,
    compile_camera_phrase,
    compile_lens_phrase,
    compile_prompt,
)
from director.core.spec import CameraDirection, LensSetup, Spec


# --------------------------------------------------------------------------
# simple and expert modes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["simple", "expert"])
def test_raw_text_passes_through_untouched(base_spec: Spec, mode: str) -> None:
    base_spec.prompt.mode = mode
    base_spec.prompt.raw = "A lighthouse beam, unembellished."
    assert compile_prompt(base_spec) == "A lighthouse beam, unembellished."


def test_simple_and_expert_compile_identically(base_spec: Spec) -> None:
    # The two modes differ in what the UI exposes, never in the output — so
    # switching between them can never change the render.
    base_spec.prompt.raw = "Same text either way"
    base_spec.prompt.mode = "simple"
    simple = compile_prompt(base_spec)
    base_spec.prompt.mode = "expert"
    assert compile_prompt(base_spec) == simple


def test_camera_settings_are_ignored_outside_director_mode(base_spec: Spec) -> None:
    base_spec.prompt.mode = "simple"
    base_spec.prompt.raw = "Just this."
    base_spec.prompt.camera = CameraDirection(move="dolly_in", intensity="strong")
    assert compile_prompt(base_spec) == "Just this."


# --------------------------------------------------------------------------
# camera phrasing
# --------------------------------------------------------------------------

def test_camera_phrases_are_prose_not_tags() -> None:
    phrase = compile_camera_phrase(CameraDirection(move="dolly_in", intensity="moderate"))
    assert phrase == "the camera slowly dollies in toward the subject"
    assert "," not in phrase.split(" toward")[0], "reads as a clause, not a tag list"


def test_intensity_changes_the_adverb() -> None:
    subtle = compile_camera_phrase(CameraDirection(move="pan_left", intensity="subtle"))
    strong = compile_camera_phrase(CameraDirection(move="pan_left", intensity="strong"))
    assert "almost imperceptibly" in subtle
    assert "quickly and decisively" in strong


def test_static_shots_say_so_rather_than_saying_nothing() -> None:
    # "holds still" is information the model uses; silence is not the same thing.
    phrase = compile_camera_phrase(CameraDirection(move="static", intensity="moderate"))
    assert "holds still" in phrase
    assert "slowly" not in phrase, "an adverb on a locked-off camera is nonsense"


def test_custom_move_uses_the_users_words_verbatim() -> None:
    phrase = compile_camera_phrase(
        CameraDirection(move="custom", custom="the camera spirals through a doorway")
    )
    assert phrase == "the camera spirals through a doorway"


def test_custom_text_is_appended_to_a_preset_move() -> None:
    phrase = compile_camera_phrase(
        CameraDirection(move="dolly_in", intensity="moderate", custom="ending on her hands")
    )
    assert phrase.endswith("ending on her hands")
    assert "dollies in" in phrase


@pytest.mark.parametrize("move", sorted(m for m in CAMERA_MOVES if m not in ("", "custom")))
def test_every_camera_move_produces_something_usable(move: str) -> None:
    phrase = compile_camera_phrase(CameraDirection(move=move, intensity="moderate"))
    assert phrase and phrase[0].islower(), "phrases must compose mid-sentence"


@pytest.mark.parametrize("move", ["", "custom"])
def test_unset_and_empty_custom_moves_say_nothing(move: str) -> None:
    assert compile_camera_phrase(CameraDirection(move=move, intensity="moderate")) == ""


def test_an_untouched_camera_control_adds_nothing(base_spec: Spec) -> None:
    # The default must not assert a locked-off camera the user never chose.
    base_spec.prompt.mode = "director"
    base_spec.prompt.sections.scene = "a harbour at night"
    assert compile_prompt(base_spec) == "A harbour at night."


# --------------------------------------------------------------------------
# lens / framing
# --------------------------------------------------------------------------

def test_framing_sentence_builds_only_from_filled_fields() -> None:
    assert compile_lens_phrase(LensSetup()) == ""
    assert compile_lens_phrase(LensSetup(shot_size="CU")) == "a close-up"


def test_framing_combines_size_angle_and_optics() -> None:
    phrase = compile_lens_phrase(
        LensSetup(shot_size="MS", angle="low", focal_mm=35, dof="shallow")
    )
    assert "a medium shot from a low angle" in phrase
    assert "35mm lens" in phrase
    assert "shallow depth of field" in phrase


def test_angle_alone_still_reads_as_english() -> None:
    assert compile_lens_phrase(LensSetup(angle="dutch")).startswith("shot with the horizon")


@pytest.mark.parametrize("size", sorted(SHOT_SIZES))
def test_every_shot_size_has_a_phrase(size: str) -> None:
    assert compile_lens_phrase(LensSetup(shot_size=size))


# --------------------------------------------------------------------------
# director mode assembly
# --------------------------------------------------------------------------

def test_sections_are_ordered_subject_first(base_spec: Spec) -> None:
    base_spec.prompt.mode = "director"
    base_spec.prompt.sections.action = "she turns to the window"
    base_spec.prompt.sections.subject = "a lighthouse keeper in oilskins"
    base_spec.prompt.sections.scene = "a storm-lit harbour"
    out = compile_prompt(base_spec)
    assert out.index("keeper") < out.index("harbour") < out.index("turns")


def test_empty_sections_leave_no_trace(base_spec: Spec) -> None:
    base_spec.prompt.mode = "director"
    base_spec.prompt.sections.subject = "a lighthouse keeper"
    out = compile_prompt(base_spec)
    assert out == "A lighthouse keeper."
    assert ".." not in out and "  " not in out


def test_sentences_are_terminated_and_capitalised(base_spec: Spec) -> None:
    base_spec.prompt.mode = "director"
    base_spec.prompt.sections.scene = "a harbour at night"   # no full stop
    base_spec.prompt.sections.action = "rain falls."          # already has one
    assert compile_prompt(base_spec) == "A harbour at night. Rain falls."


def test_user_wording_is_never_paraphrased(base_spec: Spec) -> None:
    # Capitalising an opening letter and closing a sentence is typography.
    # Changing, adding or reordering words is not allowed.
    base_spec.prompt.mode = "director"
    base_spec.prompt.sections.dialogue = "she says “we should have left an hour ago”"
    out = compile_prompt(base_spec)
    assert "“we should have left an hour ago”" in out
    assert out == "She says “we should have left an hour ago”."


def test_framing_leads_and_camera_closes(base_spec: Spec) -> None:
    base_spec.prompt.mode = "director"
    base_spec.prompt.sections.action = "she turns to the window"
    base_spec.prompt.lens = LensSetup(shot_size="MCU", angle="eye")
    base_spec.prompt.camera = CameraDirection(move="push_in", intensity="subtle")
    out = compile_prompt(base_spec)
    assert out.startswith("A medium close-up")
    assert out.rstrip().endswith("tightening on the subject.")


def test_camera_clause_is_dropped_when_the_user_already_wrote_one(base_spec: Spec) -> None:
    base_spec.prompt.mode = "director"
    base_spec.prompt.sections.camera = "the camera pushes in on her hands, then holds"
    base_spec.prompt.camera = CameraDirection(move="push_in", intensity="moderate")
    out = compile_prompt(base_spec)
    assert out.lower().count("the camera") == 1, "the same direction should not be said twice"


def test_director_mode_with_nothing_filled_in_is_empty(base_spec: Spec) -> None:
    base_spec.prompt.mode = "director"
    base_spec.prompt.camera = CameraDirection(move="custom", custom="")
    assert compile_prompt(base_spec) == ""

"""Schema migrations and the Director 2.x import.

The v2 fixtures here are the shape ``js/ltx_director.js`` actually writes, and
one of them is read straight out of the shipped example workflow — so if the
importer drifts from reality, this fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from director import SPEC_SCHEMA
from director.core import Spec
from director.core.migrate import MigrationError, looks_like_v2_timeline, migrate
from director.migrations.v2_timeline import convert_with_report

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_WORKFLOW = REPO_ROOT / "example_workflows" / "LTX_Director_2_Workflow_Distilled.json"


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------

def test_missing_schema_is_treated_as_current() -> None:
    assert migrate({"project": {"fps": 30.0}})["schema"] == SPEC_SCHEMA


def test_v1_end_frame_flag_becomes_an_anchor() -> None:
    out = migrate({"schema": 1, "references": [
        {"id": "ref_1", "role": "keyframe", "is_end_frame": True},
        {"id": "ref_2", "role": "keyframe", "is_end_frame": False},
    ]})
    assert out["schema"] == SPEC_SCHEMA
    assert [r["anchor"] for r in out["references"]] == ["end", "index"]


def test_v2_bare_frame_becomes_a_time_point() -> None:
    out = migrate({
        "schema": 2,
        "ui": {"display_unit": "frames"},
        "references": [{"id": "ref_1", "role": "keyframe", "anchor": "index", "at": 48}],
    })
    assert out["references"][0]["at"] == {"frame": 48, "unit": "frames"}


def test_migrated_payload_decodes_into_a_spec() -> None:
    spec = Spec.from_dict({
        "schema": 1,
        "project": {"fps": 24.0, "frames": 121},
        "references": [{"id": "ref_1", "role": "keyframe", "is_end_frame": True, "at": 120}],
    })
    assert spec.references[0].anchor == "end"
    assert spec.references[0].at.frame == 120


def test_a_newer_schema_still_opens(caplog: pytest.LogCaptureFixture) -> None:
    # Losing unknown fields beats refusing to show the user their own prompts.
    payload = migrate({"schema": SPEC_SCHEMA + 5, "prompt": {"raw": "hello"}})
    assert payload["prompt"]["raw"] == "hello"


def test_an_unreadably_old_schema_says_so() -> None:
    with pytest.raises(MigrationError, match="older than"):
        migrate({"schema": 0})


def test_a_non_integer_schema_is_rejected() -> None:
    with pytest.raises(MigrationError, match="integer version"):
        migrate({"schema": "3.0"})


def test_migrate_does_not_mutate_its_input() -> None:
    original = {"schema": 1, "references": [{"id": "ref_1", "role": "keyframe"}]}
    snapshot = json.dumps(original, sort_keys=True)
    migrate(original)
    assert json.dumps(original, sort_keys=True) == snapshot


# --------------------------------------------------------------------------
# v2 detection
# --------------------------------------------------------------------------

def test_v2_blobs_are_recognised_by_their_markers() -> None:
    assert looks_like_v2_timeline({"motionSegments": [], "segments": []})
    assert looks_like_v2_timeline({"retakeMode": False})


def test_a_spec_is_never_mistaken_for_a_v2_blob() -> None:
    # A truncated or hand-edited Spec must not be routed through the importer.
    assert not looks_like_v2_timeline({"schema": 3})
    assert not looks_like_v2_timeline({"project": {}, "references": []})


# --------------------------------------------------------------------------
# v2 import
# --------------------------------------------------------------------------

def v2_blob(**overrides: Any) -> dict[str, Any]:
    blob: dict[str, Any] = {
        "global_prompt": "a harbour at night",
        "segments": [],
        "motionSegments": [],
        "audioSegments": [],
        "retakeMode": False,
        "mainTrackEnabled": True,
        "normalStartFrame": 0,
        "normalDurationFrames": 121,
        "inpaint_audio": True,
    }
    blob.update(overrides)
    return blob


def test_the_shipped_example_workflow_imports() -> None:
    data = json.loads(EXAMPLE_WORKFLOW.read_text(encoding="utf-8"))
    director = next(n for n in data["nodes"] if n["type"] == "LTXDirector")
    widgets = director["widgets_values"]
    timeline = json.loads(widgets[6])

    assert looks_like_v2_timeline(timeline), "the shipped example is not a v2 blob any more"

    result = convert_with_report(
        timeline,
        fps=float(widgets[14]),
        width=int(widgets[16]),
        height=int(widgets[17]),
        guide_strength=str(widgets[22]),
        img_compression=int(widgets[20]),
    )
    assert result.spec.schema == SPEC_SCHEMA
    assert result.spec.project.fps == 24.0
    # Round-trips like any other spec, i.e. the importer produces a real Spec.
    assert Spec.from_json(result.spec.to_json()).to_dict() == result.spec.to_dict()


def test_prompts_and_lengths_survive() -> None:
    blob = v2_blob(segments=[
        {"id": "1", "start": 0, "length": 40, "type": "text", "prompt": "the beam sweeps"},
        {"id": "2", "start": 40, "length": 81, "type": "text", "prompt": "rain intensifies"},
    ])
    spec = convert_with_report(blob).spec
    assert [s.text for s in spec.segments] == ["the beam sweeps", "rain intensifies"]
    assert [s.start for s in spec.segments] == [0, 40]
    assert [s.length for s in spec.segments] == [40, 81]
    assert spec.relay.enabled


def test_keyframes_become_references_with_anchors_and_strengths() -> None:
    blob = v2_blob(segments=[
        {"id": "1", "start": 0, "length": 1, "type": "image", "imageFile": "first.png"},
        {"id": "2", "start": 60, "length": 1, "type": "image", "imageFile": "mid.png"},
        {"id": "3", "start": 100, "length": 21, "type": "image",
         "imageFile": "last.png", "isEndFrame": True},
    ])
    spec = convert_with_report(blob, guide_strength="1.0, 0.5, 0.8").spec

    assert spec.project.mode == "fflf"
    anchors = [(r.anchor, round(r.strength, 3)) for r in spec.references]
    assert anchors == [("start", 1.0), ("index", 0.5), ("end", 0.8)]
    assert spec.references[1].at.frame == 60
    # The end-frame segment pins to its last frame, not its start.
    assert spec.references[2].at.frame == 120


def test_media_is_registered_once_per_file() -> None:
    blob = v2_blob(segments=[
        {"id": "1", "start": 0, "length": 1, "type": "image", "imageFile": "same.png"},
        {"id": "2", "start": 40, "length": 1, "type": "image", "imageFile": "same.png"},
    ])
    spec = convert_with_report(blob).spec
    assert len(spec.media) == 1
    assert spec.references[0].media == spec.references[1].media


def test_subfolders_are_split_and_separators_normalised() -> None:
    blob = v2_blob(segments=[
        {"id": "1", "start": 0, "length": 1, "type": "image",
         "imageFile": "whatdreamscost\\shot01.png"},
    ])
    entry = next(iter(convert_with_report(blob).spec.media.values()))
    assert entry.subfolder == "whatdreamscost"
    assert entry.filename == "shot01.png"
    assert "\\" not in entry.path


def test_base64_only_media_is_reported_rather_than_carried() -> None:
    blob = v2_blob(segments=[
        {"id": "1", "start": 0, "length": 1, "type": "image",
         "imageB64": "data:image/png;base64,AAAA"},
    ])
    result = convert_with_report(blob)
    assert "v2.base64_media" in {d.code for d in result.diagnostics}
    assert "base64" not in result.spec.to_json()


def test_motion_segments_become_iclora_references() -> None:
    blob = v2_blob(motionSegments=[
        {"id": "m1", "start": 0, "length": 121, "videoFile": "drive.mp4", "trimStart": 0},
    ])
    spec = convert_with_report(blob).spec
    assert spec.project.mode == "iclora"
    assert [r.role for r in spec.references] == ["motion"]


def test_audio_segments_become_clips_with_seconds_trim() -> None:
    blob = v2_blob(audioSegments=[
        {"id": "a1", "start": 12, "length": 96, "audioFile": "vo.wav", "trimStart": 24},
    ])
    spec = convert_with_report(blob, fps=24.0).spec
    assert len(spec.audio.clips) == 1
    clip = spec.audio.clips[0]
    assert clip.start == 12 and clip.length == 96
    # v2 stored trim in frames; the Spec stores seconds.
    assert clip.trim_start == pytest.approx(1.0)
    assert spec.audio.mode == "inpaint"


def test_the_old_render_window_is_rebased_to_zero() -> None:
    blob = v2_blob(
        normalStartFrame=24,
        normalDurationFrames=97,
        segments=[{"id": "1", "start": 24, "length": 1, "type": "image",
                   "imageFile": "a.png"}],
    )
    spec = convert_with_report(blob).spec
    assert spec.references[0].at.frame == 0
    assert spec.references[0].anchor == "start"
    assert spec.project.frames == 97


def test_content_before_the_render_window_is_dropped() -> None:
    blob = v2_blob(
        normalStartFrame=48,
        segments=[{"id": "1", "start": 0, "length": 10, "type": "image",
                   "imageFile": "gone.png"}],
    )
    assert convert_with_report(blob).spec.references == []


def test_non_conforming_duration_is_snapped_and_explained() -> None:
    result = convert_with_report(v2_blob(normalDurationFrames=120))
    assert result.spec.project.frames == 121
    assert "v2.frames_snapped" in {d.code for d in result.diagnostics}


def test_retake_is_imported_as_a_region_with_an_explanation() -> None:
    blob = v2_blob(retakeMode=True, retakeStart=24, retakeLength=48,
                   retakePrompt="he turns away")
    result = convert_with_report(blob)
    assert any(s.text == "he turns away" for s in result.spec.segments)
    d = next(d for d in result.diagnostics if d.code == "v2.retake_not_ported")
    assert "in/outpainting" in d.fix


def test_mode_inference_covers_the_common_shapes() -> None:
    def mode_for(**kw: Any) -> str:
        return convert_with_report(v2_blob(**kw)).spec.project.mode

    assert mode_for() == "t2v"
    assert mode_for(segments=[
        {"id": "1", "start": 0, "length": 1, "type": "image", "imageFile": "a.png"}]) == "i2v"
    assert mode_for(segments=[
        {"id": "1", "start": 0, "length": 1, "type": "image", "imageFile": "a.png"},
        {"id": "2", "start": 60, "length": 1, "type": "image", "imageFile": "b.png"}]) == "keyframes"


def test_import_goes_through_the_normal_migrate_entry_point() -> None:
    # Spec.from_dict must accept a raw v2 blob, so pasting one into the node works.
    spec = Spec.from_dict(v2_blob(segments=[
        {"id": "1", "start": 0, "length": 40, "type": "text", "prompt": "hello"},
    ]))
    assert spec.schema == SPEC_SCHEMA
    assert spec.prompt.raw == "a harbour at night"
    assert [s.text for s in spec.segments] == ["hello"]

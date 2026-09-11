"""Spec serialisation, identity and derived state."""

from __future__ import annotations

import json

import pytest

from director import SPEC_SCHEMA
from director.core import new_spec
from director.core.codec import DecodeError, from_jsonable
from director.core.ids import PREFIXES, digest_obj, is_id, new_id
from director.core.spec import Reference, Segment, Spec, Take, TimePoint


# --------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture",
    ["base_spec", "spec_i2v", "spec_fflf", "spec_keyframes", "spec_a2v",
     "spec_t2a", "spec_iclora", "spec_two_stage", "spec_relay"],
)
def test_round_trip_is_lossless(fixture: str, request: pytest.FixtureRequest) -> None:
    spec: Spec = request.getfixturevalue(fixture)
    assert Spec.from_json(spec.to_json()).to_dict() == spec.to_dict()


def test_round_trip_is_byte_stable(base_spec: Spec) -> None:
    # Two encodes of semantically identical specs must produce identical bytes,
    # otherwise the digest — and every cache keyed on it — is meaningless.
    once = base_spec.to_json()
    twice = Spec.from_json(once).to_json()
    assert once == twice


def test_empty_and_broken_input_yield_a_default_spec() -> None:
    # An empty node is a normal state, not an error.
    for value in (None, "", "   ", "not json", "[]", "42"):
        assert Spec.from_json(value).schema == SPEC_SCHEMA


def test_copy_is_deep(base_spec: Spec) -> None:
    clone = base_spec.copy()
    clone.prompt.raw = "changed"
    clone.project.frames = 201
    assert base_spec.prompt.raw != "changed"
    assert base_spec.project.frames != 201


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(PREFIXES))
def test_new_id_is_prefixed_and_recognised(kind: str) -> None:
    value = new_id(kind)
    assert value.startswith(PREFIXES[kind] + "_")
    assert is_id(value) and is_id(value, kind)


def test_unknown_id_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown id kind"):
        new_id("sandwich")


def test_ids_are_assigned_automatically_and_kept_on_reload() -> None:
    ref = Reference(role="keyframe")
    assert is_id(ref.id, "reference")
    reloaded = from_jsonable(Reference, {"id": ref.id, "role": "keyframe"})
    assert reloaded.id == ref.id


def test_reordering_does_not_change_ids(spec_keyframes: Spec) -> None:
    before = [r.id for r in spec_keyframes.references]
    spec_keyframes.references.reverse()
    spec_keyframes.normalise()
    assert sorted(r.id for r in spec_keyframes.references) == sorted(before)


# --------------------------------------------------------------------------
# digest
# --------------------------------------------------------------------------

def test_digest_ignores_presentation_and_history(base_spec: Spec) -> None:
    before = base_spec.digest()
    base_spec.ui.zoom = 3.5
    base_spec.ui.playhead = 77
    base_spec.ui.expanded = ["audio", "models"]
    base_spec.meta.name = "Renamed"
    base_spec.takes.append(Take(seed=1))
    assert base_spec.digest() == before, "cosmetic changes must not invalidate caches"


def test_digest_tracks_anything_that_changes_the_output(base_spec: Spec) -> None:
    before = base_spec.digest()
    base_spec.generation.seed += 1
    assert base_spec.digest() != before


def test_digest_is_order_independent_for_dicts() -> None:
    assert digest_obj({"a": 1, "b": 2}) == digest_obj({"b": 2, "a": 1})


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def test_normalise_snaps_geometry(base_spec: Spec) -> None:
    base_spec.project.frames = 120
    base_spec.project.width = 1281
    base_spec.project.height = 700
    base_spec.normalise()
    assert base_spec.project.frames == 121
    assert base_spec.project.width == 1280
    assert base_spec.project.height == 704


def test_normalise_clamps_references_into_the_shot(base_spec: Spec) -> None:
    ref = Reference(role="keyframe", anchor="index", strength=4.0)
    ref.at.frame = 9999
    base_spec.references.append(ref)
    base_spec.normalise()
    assert ref.at.frame == base_spec.project.frames - 1
    assert ref.strength == 1.0


def test_normalise_orders_segments(base_spec: Spec) -> None:
    base_spec.segments = [
        Segment(start=80, length=40, text="c"),
        Segment(start=0, length=40, text="a"),
        Segment(start=40, length=40, text="b"),
    ]
    base_spec.normalise()
    assert [s.text for s in base_spec.segments] == ["a", "b", "c"]


def test_normalise_is_idempotent(spec_keyframes: Spec) -> None:
    once = spec_keyframes.normalise().to_dict()
    twice = spec_keyframes.normalise().to_dict()
    assert once == twice


def test_zero_fps_is_repaired(base_spec: Spec) -> None:
    base_spec.project.fps = 0
    base_spec.normalise()
    assert base_spec.project.fps == 24.0


# --------------------------------------------------------------------------
# derived state
# --------------------------------------------------------------------------

def test_resolved_frame_maps_anchors_to_guide_indices(base_spec: Spec) -> None:
    total = base_spec.project.frames
    start = Reference(role="keyframe", anchor="start")
    end = Reference(role="keyframe", anchor="end")
    mid = Reference(role="keyframe", anchor="index", at=TimePoint(frame=48))
    assert start.resolved_frame(total) == 0
    assert end.resolved_frame(total) == -1, "last frame is -1, per LTXVAddGuide"
    assert mid.resolved_frame(total) == 48


def test_active_references_are_ordered_start_then_index_then_end(spec_keyframes: Spec) -> None:
    order = [r.anchor for r in spec_keyframes.active_references("keyframe")]
    assert order == ["start", "index", "end"]


def test_active_references_skip_disabled_and_dangling(spec_keyframes: Spec) -> None:
    spec_keyframes.references[0].enabled = False
    spec_keyframes.references[1].media = "med_does_not_exist"
    assert len(spec_keyframes.active_references("keyframe")) == 1


def test_relay_needs_more_than_one_populated_region(base_spec: Spec) -> None:
    assert not base_spec.relay_active
    base_spec.segments = [Segment(start=0, length=60, text="only one")]
    assert not base_spec.relay_active
    base_spec.segments.append(Segment(start=60, length=61, text="and another"))
    assert base_spec.relay_active


def test_relay_can_be_switched_off(spec_relay: Spec) -> None:
    assert spec_relay.relay_active
    spec_relay.relay.enabled = False
    assert not spec_relay.relay_active


def test_audio_only_mode_always_needs_audio(spec_t2a: Spec) -> None:
    spec_t2a.audio.enabled = False
    assert spec_t2a.needs_audio
    assert not spec_t2a.needs_video


def test_muting_disables_audio_for_video_modes(base_spec: Spec) -> None:
    base_spec.audio.mode = "mute"
    assert not base_spec.needs_audio


# --------------------------------------------------------------------------
# codec strictness
# --------------------------------------------------------------------------

def test_literal_fields_reject_unknown_values() -> None:
    with pytest.raises(DecodeError, match="expected one of"):
        from_jsonable(Reference, {"role": "sandwich"})


def test_decode_error_names_the_field() -> None:
    with pytest.raises(DecodeError) as excinfo:
        Spec.from_dict({"schema": 3, "project": {"mode": "nope"}})
    assert "project.mode" in str(excinfo.value)


def test_unknown_fields_are_ignored_not_fatal() -> None:
    # A project written by a newer Director must still open.
    spec = Spec.from_dict({"schema": SPEC_SCHEMA, "project": {"fps": 30.0}, "warp_drive": True})
    assert spec.project.fps == 30.0


def test_optional_fields_accept_null() -> None:
    spec = Spec.from_dict({
        "schema": SPEC_SCHEMA,
        "models": {"upscaler": None, "enhancer_clip": None},
        "audio": {"reference": None},
    })
    assert spec.models.upscaler is None
    assert spec.audio.reference is None


# --------------------------------------------------------------------------
# the no-blobs rule
# --------------------------------------------------------------------------

def test_spec_carries_no_media_payloads(spec_keyframes: Spec) -> None:
    # The whole point of the rebuild: project state stays small and portable.
    text = spec_keyframes.to_json()
    assert "base64" not in text and "data:image" not in text
    assert len(text) < 8192


def test_media_entries_store_references_not_contents(spec_keyframes: Spec) -> None:
    for entry in spec_keyframes.media.values():
        payload = json.dumps(entry.__dict__)
        assert len(payload) < 512

"""Conformance with what upstream actually builds.

Every other test says "the compiler does what we decided". This one says "what
we decided matches Lightricks and Comfy-Org". The fixture it reads is distilled
from the real LTX-2.5 reference graphs by ``tools/capture_upstream.py``, so it
keeps working without those repos present and its diff is reviewable when a new
LTX release lands.

If one of these fails after an upstream bump, the fixture is the spec and the
compiler is what changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from director.compiler import compile_spec
from director.compiler.signatures import NODE_SIGNATURES
from director.core.spec import DEFAULT_SIGMAS_STAGE1, DEFAULT_SIGMAS_STAGE2, Spec

from .graph_asserts import node_types, widget

FIXTURE = Path(__file__).parent / "fixtures" / "upstream_ltx25.json"
UPSTREAM: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
GRAPHS: dict[str, Any] = UPSTREAM["graphs"]
CONSTANTS: dict[str, Any] = UPSTREAM["constants"]


def upstream_types(*labels: str) -> set[str]:
    out: set[str] = set()
    for label in labels:
        out |= set(GRAPHS[label]["node_counts"])
    return out


# --------------------------------------------------------------------------
# signatures
# --------------------------------------------------------------------------

def test_the_fixture_actually_captured_something() -> None:
    # Guards against a silently empty capture making every check below vacuous.
    assert len(GRAPHS) >= 6
    assert CONSTANTS


@pytest.mark.parametrize("label", sorted(GRAPHS))
def test_our_slot_order_matches_the_serialised_graphs(label: str) -> None:
    """Slot order is referenced by index, so getting it wrong wires the graph wrong."""
    mismatches = []
    for node_type, observed in GRAPHS[label]["signatures"].items():
        signature = NODE_SIGNATURES.get(node_type)
        if signature is None:
            continue  # a node we never emit

        declared = [slot.name for slot in signature.inputs]
        seen = [name for name in observed["inputs"] if name in declared]
        if seen != [name for name in declared if name in seen]:
            mismatches.append(f"{node_type} inputs: upstream {observed['inputs']}, ours {declared}")

        if observed["outputs"] and observed["outputs"] != [s.name for s in signature.outputs]:
            mismatches.append(
                f"{node_type} outputs: upstream {observed['outputs']}, "
                f"ours {[s.name for s in signature.outputs]}"
            )

    assert not mismatches, "\n".join(mismatches)


@pytest.mark.parametrize("label", sorted(GRAPHS))
def test_our_widget_counts_match_the_serialised_graphs(label: str) -> None:
    mismatches = []
    for node_type, observed in GRAPHS[label]["signatures"].items():
        signature = NODE_SIGNATURES.get(node_type)
        if signature is None or not observed["widgets"]:
            continue
        # Dynamic-combo nodes serialise a variable number of widgets depending
        # on the selected variant, so they are compared as "at least".
        if any("." in slot.name for slot in signature.inputs):
            assert len(signature.widgets) >= observed["widgets"], node_type
            continue
        if len(signature.widgets) != observed["widgets"]:
            mismatches.append(
                f"{node_type}: upstream serialises {observed['widgets']} widget value(s), "
                f"we declare {len(signature.widgets)}"
            )
    assert not mismatches, "\n".join(mismatches)


@pytest.mark.parametrize("node_type", sorted(NODE_SIGNATURES))
def test_no_signature_declares_a_duplicate_slot(node_type: str) -> None:
    signature = NODE_SIGNATURES[node_type]
    names = [s.name for s in signature.inputs]
    assert len(names) == len(set(names)), f"{node_type} declares a slot twice"
    outputs = [s.name for s in signature.outputs]
    assert len(outputs) == len(set(outputs)), f"{node_type} declares an output twice"


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

def test_our_stage_one_schedule_is_the_upstream_one() -> None:
    upstream = [float(v) for v in CONSTANTS["sigmas_stage1"].split(",")]
    assert list(DEFAULT_SIGMAS_STAGE1) == upstream


def test_our_refine_schedule_is_the_upstream_one() -> None:
    upstream = [float(v) for v in CONSTANTS["sigmas_stage2"].split(",")]
    assert list(DEFAULT_SIGMAS_STAGE2) == pytest.approx(upstream, abs=1e-4)


def test_our_defaults_are_the_upstream_defaults(base_spec: Spec) -> None:
    generation = base_spec.generation
    assert generation.img_compression == CONSTANTS["img_compression"]
    assert generation.first_frame_strength == pytest.approx(CONSTANTS["first_frame_strength"])
    assert generation.guide_strength == pytest.approx(CONSTANTS["guide_strength"])
    for key, value in CONSTANTS["decode"].items():
        assert getattr(generation.decode, key) == value


def test_we_never_compute_latent_geometry_ourselves() -> None:
    """The 128/32/8 constants belong to EmptyLTXVLatentVideo, not to us.

    Hardcoding them is exactly what broke the previous Director when the VAE
    changed, so the compiler must not contain them.
    """
    import director.compiler.ltx25 as adapter

    source = Path(adapter.__file__).read_text(encoding="utf-8")
    assert "128" not in source, "latent channel count must come from the node, not from us"
    assert "// 32" not in source and "* 32" not in source, (
        "spatial scaling must come from the VAE, not from a literal"
    )


# --------------------------------------------------------------------------
# topology
# --------------------------------------------------------------------------

def test_t2v_uses_only_nodes_upstream_uses(base_spec: Spec) -> None:
    ours = set(node_types(compile_spec(base_spec).workflow))
    theirs = upstream_types(
        "t2v_i2v_single_stage", "t2v_i2v_two_stage", "template_t2v"
    )
    # Primitives feeding editable widgets are ours; everything doing work must
    # be a node upstream also uses.
    ours -= {"PrimitiveStringMultiline", "PreviewAny"}
    assert ours <= theirs, f"nodes upstream never uses: {sorted(ours - theirs)}"


def test_i2v_pins_the_way_upstream_pins(spec_i2v: Spec) -> None:
    ours = set(node_types(compile_spec(spec_i2v).workflow))
    assert "LTXVImgToVideoInplace" in ours
    assert "LTXVImgToVideoInplace" in GRAPHS["template_i2v"]["node_counts"]
    assert "LTXVPreprocess" in ours


def test_fflf_guides_the_way_upstream_guides(spec_fflf: Spec) -> None:
    counts = GRAPHS["template_flf2v"]["node_counts"]
    assert counts.get("LTXVAddGuide") == 2, "the reference FLF2V graph chains two guides"
    assert counts.get("LTXVCropGuides") == 1, "and crops once"

    ours = node_types(compile_spec(spec_fflf).workflow)
    assert ours.count("LTXVAddGuide") == 2
    assert ours.count("LTXVCropGuides") == 1


def test_two_stage_refines_the_way_upstream_refines(spec_two_stage: Spec) -> None:
    counts = GRAPHS["t2v_i2v_two_stage"]["node_counts"]
    assert counts.get("LTXVLatentUpsampler") == 1
    assert counts.get("SamplerCustomAdvanced") == 2

    ours = node_types(compile_spec(spec_two_stage).workflow)
    assert ours.count("LTXVLatentUpsampler") == 1
    assert ours.count("SamplerCustomAdvanced") == 2


def test_audio_is_concatenated_and_separated_like_upstream(base_spec: Spec) -> None:
    for label in ("t2v_i2v_single_stage", "template_t2v"):
        counts = GRAPHS[label]["node_counts"]
        assert counts.get("LTXVConcatAVLatent")
        assert counts.get("LTXVSeparateAVLatent")

    ours = node_types(compile_spec(base_spec).workflow)
    assert ours.count("LTXVConcatAVLatent") == ours.count("LTXVSeparateAVLatent") == 1


def test_a2v_freezes_supplied_audio_like_upstream(spec_a2v: Spec) -> None:
    # The 2.5 A2V example predates LTXVFreezeLatent and freezes with a
    # SolidMask + SetLatentNoiseMask pair. We use the node that now exists for
    # it, which is the same operation with one node instead of two.
    counts = GRAPHS["a2v_two_stage"]["node_counts"]
    assert counts.get("LTXVAudioVAEEncode") == 1
    assert counts.get("SetLatentNoiseMask"), "upstream freezes the encoded audio"

    ours = node_types(compile_spec(spec_a2v).workflow)
    assert ours.count("LTXVAudioVAEEncode") == 1
    assert ours.count("LTXVFreezeLatent") == 1


def test_t2a_uses_the_audio_only_pair(spec_t2a: Spec) -> None:
    counts = GRAPHS["t2a_single_stage"]["node_counts"]
    assert counts.get("LTXVAudioOnlyModel") == 1
    assert counts.get("LTXVAudioOnlyEmptyVideoLatent") == 1

    ours = node_types(compile_spec(spec_t2a).workflow)
    assert ours.count("LTXVAudioOnlyModel") == 1
    assert ours.count("LTXVAudioOnlyEmptyVideoLatent") == 1


def test_iclora_uses_the_loader_and_guide_upstream_uses(spec_iclora: Spec) -> None:
    counts = GRAPHS["iclora_motion_track"]["node_counts"]
    assert counts.get("LTXICLoRALoaderModelOnly") == 1
    assert counts.get("LTXAddVideoICLoRAGuide") == 1

    ours = node_types(compile_spec(spec_iclora).workflow)
    assert ours.count("LTXICLoRALoaderModelOnly") == 1
    assert ours.count("LTXAddVideoICLoRAGuide") == 1


def test_the_text_encoder_is_loaded_as_ltxv(base_spec: Spec) -> None:
    workflow = compile_spec(base_spec).workflow
    assert widget(workflow, "CLIPLoader", "type") == CONSTANTS["clip_type"]

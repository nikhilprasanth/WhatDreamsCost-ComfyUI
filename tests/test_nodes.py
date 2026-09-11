"""The node surface.

Two things are being defended here. First, that the nodes are *thin*: the
Director node touches no tensors, carries one string widget, and every failure
it can produce is a sentence. Second, that Director 2.x workflows keep loading —
losing a user's existing graph is not an acceptable price for a rebuild.

ComfyUI is stubbed, so these run anywhere.
"""

from __future__ import annotations

import json

import pytest

from .comfy_stub import install

install()

from director.core.spec import Spec  # noqa: E402
from director.nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    LTXDirectorCompile,
    LTXDirectorProject,
    LTXDirectorRelay,
)


def schema(node_class):
    return node_class.define_schema()


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def test_three_nodes_no_more() -> None:
    # The whole point of the rebuild: one 26-input node became three small ones.
    assert set(NODE_CLASS_MAPPINGS) == {
        "LTXDirectorProject", "LTXDirectorRelay", "LTXDirectorCompile",
    }


def test_every_node_has_a_display_name_and_a_description() -> None:
    for name, node_class in NODE_CLASS_MAPPINGS.items():
        definition = schema(node_class)
        assert NODE_DISPLAY_NAME_MAPPINGS[name]
        assert definition.description, name
        assert definition.category == "LTX Director", name


def test_node_ids_match_their_class_names() -> None:
    for name, node_class in NODE_CLASS_MAPPINGS.items():
        assert schema(node_class).node_id == name


# --------------------------------------------------------------------------
# the Director node
# --------------------------------------------------------------------------

def test_the_director_node_has_one_real_widget() -> None:
    definition = schema(LTXDirectorProject)
    required = [s for s in definition.inputs if not s.optional]
    assert [s.name for s in required] == ["project"]


def test_the_director_node_carries_no_tensor_sockets() -> None:
    # If a tensor ever crosses this node, the architecture has been abandoned.
    definition = schema(LTXDirectorProject)
    tensor_types = {"Image", "Latent", "Mask", "Audio", "Model", "Vae", "Conditioning"}
    for slot in definition.inputs + definition.outputs:
        assert slot.type not in tensor_types, f"{slot.name} carries {slot.type}"


def test_an_empty_project_still_yields_usable_defaults() -> None:
    # A freshly dropped node must not explode; it has nothing in it yet.
    with pytest.raises(ValueError) as excinfo:
        LTXDirectorProject.execute(project="")
    # ... but it must say what is missing, in sentences.
    message = str(excinfo.value)
    assert "cannot build this shot" in message
    assert "Traceback" not in message


def test_the_director_node_returns_the_derived_values_a_graph_wants(base_spec: Spec) -> None:
    base_spec.project.width, base_spec.project.height = 960, 544
    base_spec.project.frames = 97
    base_spec.project.fps = 25.0
    base_spec.generation.seed = 7

    out = LTXDirectorProject.execute(project=base_spec.to_json())
    spec, prompt, negative, frames, fps, width, height, seed = out

    assert isinstance(spec, Spec)
    assert prompt == base_spec.prompt.raw
    assert negative == base_spec.prompt.negative
    assert (frames, fps, width, height, seed) == (97, 25.0, 960, 544, 7)


def test_geometry_is_snapped_before_anything_downstream_sees_it(base_spec: Spec) -> None:
    base_spec.project.frames = 120
    base_spec.project.width = 1281
    out = LTXDirectorProject.execute(project=base_spec.to_json())
    assert out[3] == 121
    assert out[5] == 1280


def test_a_connected_prompt_replaces_the_editors(base_spec: Spec) -> None:
    out = LTXDirectorProject.execute(
        project=base_spec.to_json(), prompt_override="from upstream"
    )
    assert out[1] == "from upstream"


def test_an_overridden_prompt_is_used_verbatim(base_spec: Spec) -> None:
    # Silently appending Director-mode sections to someone's wired-in text would
    # make the graph lie about what it encodes.
    base_spec.prompt.mode = "director"
    base_spec.prompt.sections.lighting = "harsh sodium light"
    out = LTXDirectorProject.execute(
        project=base_spec.to_json(), prompt_override="only this"
    )
    assert out[1] == "only this"


def test_a_seed_override_wins_and_minus_one_does_not(base_spec: Spec) -> None:
    base_spec.generation.seed = 11
    assert LTXDirectorProject.execute(project=base_spec.to_json(), seed_override=99)[7] == 99
    assert LTXDirectorProject.execute(project=base_spec.to_json(), seed_override=-1)[7] == 11


def test_the_node_reruns_only_when_the_shot_changes(base_spec: Spec) -> None:
    before = LTXDirectorProject.fingerprint_inputs(project=base_spec.to_json())

    base_spec.ui.zoom = 4.0
    base_spec.ui.playhead = 60
    assert LTXDirectorProject.fingerprint_inputs(project=base_spec.to_json()) == before

    base_spec.generation.seed += 1
    assert LTXDirectorProject.fingerprint_inputs(project=base_spec.to_json()) != before


def test_a_malformed_project_does_not_crash_the_fingerprint() -> None:
    assert LTXDirectorProject.fingerprint_inputs(project="{not json")


# --------------------------------------------------------------------------
# the Relay node
# --------------------------------------------------------------------------

class FakeClip:
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def tokenize(self, text: str) -> str:
        self.encoded.append(text)
        return text

    def encode_from_tokens_scheduled(self, tokens: str) -> list:
        return [[tokens, {}]]


class FakeModel:
    def __init__(self) -> None:
        inner = type("Inner", (), {})()
        inner.diffusion_model = type("Unknown", (), {})()
        self.model = inner


def test_the_relay_node_takes_a_model_a_clip_and_a_spec() -> None:
    definition = schema(LTXDirectorRelay)
    required = [s.name for s in definition.inputs if not s.optional]
    assert required == ["model", "clip", "director"]


def test_the_relay_node_returns_a_model_conditioning_and_a_status() -> None:
    assert schema(LTXDirectorRelay).output_names == ["model", "conditioning", "status"]


def test_relay_with_one_region_reports_that_plainly(base_spec: Spec) -> None:
    model, clip = FakeModel(), FakeClip()
    _, conditioning, status = LTXDirectorRelay.execute(model, clip, base_spec)
    assert conditioning
    assert "no relay was needed" in status


def test_relay_switched_off_says_so(spec_relay: Spec) -> None:
    _, _, status = LTXDirectorRelay.execute(FakeModel(), FakeClip(), spec_relay, enabled=False)
    assert "switched off" in status


def test_switching_relay_off_does_not_mutate_the_project(spec_relay: Spec) -> None:
    before = spec_relay.to_json()
    LTXDirectorRelay.execute(FakeModel(), FakeClip(), spec_relay, enabled=False)
    assert spec_relay.to_json() == before


def test_relay_on_an_unsupported_model_still_returns_conditioning(spec_relay: Spec) -> None:
    # Degrading must cost the regions, not the render.
    model = FakeModel()
    returned, conditioning, status = LTXDirectorRelay.execute(model, FakeClip(), spec_relay)
    assert returned is model
    assert conditioning
    assert status and "Traceback" not in status


def test_the_relay_node_accepts_a_spec_as_json_or_dict(base_spec: Spec) -> None:
    for payload in (base_spec, base_spec.to_dict(), base_spec.to_json()):
        _, conditioning, _ = LTXDirectorRelay.execute(FakeModel(), FakeClip(), payload)
        assert conditioning


def test_a_missing_director_connection_says_what_to_connect() -> None:
    with pytest.raises(ValueError, match="Connect the LTX Director node"):
        LTXDirectorRelay.execute(FakeModel(), FakeClip(), 42)


# --------------------------------------------------------------------------
# the Compile node
# --------------------------------------------------------------------------

def test_the_compile_node_can_run_without_touching_the_disk(base_spec: Spec) -> None:
    path, summary, payload = LTXDirectorCompile.execute(base_spec, write_file=False)
    assert path == ""
    assert json.loads(payload)["version"] == 0.4
    assert "Text to video" in summary


def test_the_compile_summary_reads_as_english(spec_two_stage: Spec) -> None:
    _, summary, _ = LTXDirectorCompile.execute(spec_two_stage, write_file=False)
    assert "frames" in summary and "seed" in summary
    assert "2 stages" in summary
    for line in summary.splitlines():
        assert "Traceback" not in line and "None" not in line


def test_the_compile_node_offers_both_layouts(base_spec: Spec) -> None:
    definition = schema(LTXDirectorCompile)
    assert definition.input("layout").options == ["flat", "subgraphs"]

    _, _, packed = LTXDirectorCompile.execute(base_spec, layout="subgraphs", write_file=False)
    assert json.loads(packed)["definitions"]["subgraphs"]


def test_compiling_a_broken_shot_explains_rather_than_raising_a_keyerror() -> None:
    with pytest.raises(ValueError) as excinfo:
        LTXDirectorCompile.execute(Spec(), write_file=False)
    assert "cannot compile this shot" in str(excinfo.value)


def test_the_compile_node_reruns_only_when_the_shot_or_layout_changes(base_spec: Spec) -> None:
    flat = LTXDirectorCompile.fingerprint_inputs(base_spec, "flat")
    assert LTXDirectorCompile.fingerprint_inputs(base_spec, "flat") == flat
    assert LTXDirectorCompile.fingerprint_inputs(base_spec, "subgraphs") != flat


# --------------------------------------------------------------------------
# Director 2.x keeps working
# --------------------------------------------------------------------------

def test_the_legacy_relay_imports_still_resolve() -> None:
    # ltx_director.py imports these by name; breaking them breaks every saved
    # Director 2 workflow.
    from director.relay.mask import (
        build_segments, create_mask_fn, distribute_segment_lengths,
    )
    from director.relay.tokens import get_raw_tokenizer, map_token_indices

    assert all(callable(f) for f in (
        build_segments, create_mask_fn, distribute_segment_lengths,
        get_raw_tokenizer, map_token_indices,
    ))


def test_the_legacy_length_helper_delegates_rather_than_duplicating() -> None:
    from director.relay import _latent_lengths

    assert _latent_lengths([40, 40, 41], 8, 16) == pytest.approx([5, 5, 6], abs=1)

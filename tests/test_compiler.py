"""Compiler behaviour.

Three layers of assertion:

1. **Structural** — the workflow is a valid ComfyUI document: links resolve,
   slot indices match the signatures, nothing required is dangling, nothing is
   orphaned.
2. **Topological** — the graph matches what upstream builds for that mode. A
   guide chain implies a crop; a refine stage implies an upsampler and a second
   sampler; audio-only implies no video VAE.
3. **Purity** — the same spec compiles to byte-identical JSON, which is what
   makes the golden files meaningful.
"""

from __future__ import annotations

import json

import pytest

from director.compiler import available_families, compile_spec
from director.compiler.graph import SUBGRAPH_INPUT_ID, SUBGRAPH_OUTPUT_ID
from director.core.spec import DEFAULT_SIGMAS_STAGE1, DEFAULT_SIGMAS_STAGE2, Spec

from .graph_asserts import (
    assert_graph_integrity,
    assert_reachable,
    assert_required_inputs_connected,
    assert_slots_match_signatures,
    count,
    execution_order,
    node_types,
    only,
    widget,
)

TERMINALS = {"SaveVideo", "SaveAudioAdvanced", "PreviewAny"}

ALL_MODES = [
    "base_spec", "spec_i2v", "spec_fflf", "spec_keyframes",
    "spec_a2v", "spec_t2a", "spec_iclora", "spec_two_stage",
]


def compiled(spec: Spec):
    result = compile_spec(spec)
    assert result.ok, [str(d) for d in result.report.errors]
    return result


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ALL_MODES)
def test_every_mode_produces_a_structurally_valid_graph(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    workflow = compiled(request.getfixturevalue(fixture)).workflow
    assert_graph_integrity(workflow)
    assert_slots_match_signatures(workflow)
    assert_required_inputs_connected(workflow)
    assert_reachable(workflow, TERMINALS)


@pytest.mark.parametrize("fixture", ALL_MODES)
def test_every_mode_ends_at_a_save_node(fixture: str, request: pytest.FixtureRequest) -> None:
    workflow = compiled(request.getfixturevalue(fixture)).workflow
    assert TERMINALS & set(node_types(workflow))


def test_the_registry_knows_ltx25() -> None:
    assert "ltx2.5" in available_families()


# --------------------------------------------------------------------------
# purity
# --------------------------------------------------------------------------

def test_compiling_twice_gives_identical_bytes(spec_keyframes: Spec) -> None:
    first = json.dumps(compiled(spec_keyframes).workflow, sort_keys=True)
    second = json.dumps(compiled(spec_keyframes).workflow, sort_keys=True)
    assert first == second


def test_compiling_does_not_mutate_the_spec(spec_fflf: Spec) -> None:
    before = spec_fflf.to_json()
    compiled(spec_fflf)
    assert spec_fflf.to_json() == before


def test_a_changed_seed_changes_only_the_seed(base_spec: Spec) -> None:
    before = compiled(base_spec).workflow
    base_spec.generation.seed = 12345
    after = compiled(base_spec).workflow
    assert node_types(before) == node_types(after)
    assert widget(after, "RandomNoise", "noise_seed") == 12345


# --------------------------------------------------------------------------
# text to video
# --------------------------------------------------------------------------

def test_t2v_matches_the_upstream_spine(base_spec: Spec) -> None:
    workflow = compiled(base_spec).workflow
    types = set(node_types(workflow))
    assert {
        "UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode", "LTXVConditioning",
        "EmptyLTXVLatentVideo", "LTXVEmptyLatentAudio", "LTXVConcatAVLatent",
        "RandomNoise", "LTXVDualCFGGuider", "KSamplerSelect", "ManualSigmas",
        "SamplerCustomAdvanced", "LTXVSeparateAVLatent", "VAEDecodeTiled",
        "LTXVAudioVAEDecode", "CreateVideo", "SaveVideo",
    } <= types


def test_t2v_pins_nothing(base_spec: Spec) -> None:
    workflow = compiled(base_spec).workflow
    assert count(workflow, "LTXVAddGuide") == 0
    assert count(workflow, "LTXVImgToVideoInplace") == 0
    assert count(workflow, "LTXVCropGuides") == 0, "no guides means no crop"


def test_project_geometry_reaches_the_latent(base_spec: Spec) -> None:
    base_spec.project.width, base_spec.project.height = 960, 544
    base_spec.project.frames = 97
    workflow = compiled(base_spec).workflow
    assert widget(workflow, "EmptyLTXVLatentVideo", "width") == 960
    assert widget(workflow, "EmptyLTXVLatentVideo", "height") == 544
    assert widget(workflow, "EmptyLTXVLatentVideo", "length") == 97


def test_frame_rate_reaches_conditioning_audio_and_output(base_spec: Spec) -> None:
    base_spec.project.fps = 25.0
    workflow = compiled(base_spec).workflow
    assert widget(workflow, "LTXVConditioning", "frame_rate") == 25.0
    assert widget(workflow, "LTXVEmptyLatentAudio", "frame_rate") == 25.0
    assert widget(workflow, "CreateVideo", "fps") == 25.0


def test_the_distilled_schedule_is_emitted_verbatim(base_spec: Spec) -> None:
    workflow = compiled(base_spec).workflow
    sigmas = widget(workflow, "ManualSigmas", "sigmas")
    assert sigmas.startswith("1, 0.99375, 0.9875")
    assert sigmas.endswith("0")
    assert len(sigmas.split(",")) == len(DEFAULT_SIGMAS_STAGE1)


def test_the_compiled_prompt_is_what_lands_in_the_graph(base_spec: Spec) -> None:
    base_spec.prompt.mode = "director"
    base_spec.prompt.raw = ""
    base_spec.prompt.sections.scene = "a storm-lit harbour"
    workflow = compiled(base_spec).workflow
    values = [n["widgets_values"][0] for n in workflow["nodes"]
              if n["type"] == "PrimitiveStringMultiline"]
    assert "A storm-lit harbour." in values


# --------------------------------------------------------------------------
# image to video
# --------------------------------------------------------------------------

def test_i2v_pins_frame_zero_in_place(spec_i2v: Spec) -> None:
    workflow = compiled(spec_i2v).workflow
    assert count(workflow, "LTXVImgToVideoInplace") == 1
    # In-place pinning adds no keyframe tokens, so no crop is needed — that is
    # the whole reason to prefer it over a guide for a first frame.
    assert count(workflow, "LTXVCropGuides") == 0
    assert widget(workflow, "LTXVImgToVideoInplace", "strength") == pytest.approx(0.7)


def test_guide_images_are_compressed_the_way_ltx_expects(spec_i2v: Spec) -> None:
    workflow = compiled(spec_i2v).workflow
    assert widget(workflow, "LTXVPreprocess", "img_compression") == 18


def test_compression_can_be_switched_off(spec_i2v: Spec) -> None:
    spec_i2v.generation.img_compression = 0
    workflow = compiled(spec_i2v).workflow
    assert count(workflow, "LTXVPreprocess") == 0


def test_one_preprocess_per_file_not_per_reference(base_spec: Spec) -> None:
    from director.core.spec import MediaEntry, Reference

    base_spec.project.mode = "keyframes"
    entry = MediaEntry(filename="same.png", kind="image")
    base_spec.media[entry.id] = entry
    for frame, anchor in ((0, "start"), (48, "index")):
        ref = Reference(role="keyframe", media=entry.id, anchor=anchor)
        ref.at.frame = frame
        base_spec.references.append(ref)

    workflow = compiled(base_spec).workflow
    assert count(workflow, "LoadImage") == 1, "the same file must load once"
    assert count(workflow, "LTXVPreprocess") == 1, "and be compressed once"


# --------------------------------------------------------------------------
# guides
# --------------------------------------------------------------------------

def test_fflf_chains_two_guides_at_zero_and_minus_one(spec_fflf: Spec) -> None:
    workflow = compiled(spec_fflf).workflow
    guides = [n for n in workflow["nodes"] if n["type"] == "LTXVAddGuide"]
    assert len(guides) == 2
    indices = [g["widgets_values"][0] for g in guides]
    assert indices == [0, -1], "last frame is -1, so it survives a duration change"


def test_guides_always_bring_a_crop(spec_fflf: Spec, spec_keyframes: Spec) -> None:
    for spec in (spec_fflf, spec_keyframes):
        assert count(compiled(spec).workflow, "LTXVCropGuides") == 1


def test_the_crop_runs_before_the_upsampler(spec_fflf: Spec) -> None:
    # Keyframe tokens live on the latent's temporal axis. Upsampling them would
    # scale them as if they were picture.
    spec_fflf.generation.stages = 2
    spec_fflf.models.upscaler = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    order = execution_order(compiled(spec_fflf).workflow)
    assert order.index("LTXVCropGuides") < order.index("LTXVLatentUpsampler")


def test_the_crop_runs_after_sampling(spec_fflf: Spec) -> None:
    order = execution_order(compiled(spec_fflf).workflow)
    assert order.index("SamplerCustomAdvanced") < order.index("LTXVCropGuides")
    assert order.index("LTXVCropGuides") < order.index("VAEDecodeTiled")


def test_keyframes_are_ordered_start_then_middle_then_end(spec_keyframes: Spec) -> None:
    workflow = compiled(spec_keyframes).workflow
    guides = [n for n in workflow["nodes"] if n["type"] == "LTXVAddGuide"]
    assert [g["widgets_values"][0] for g in guides] == [0, 48, -1]


def test_guide_strength_comes_from_the_reference(spec_fflf: Spec) -> None:
    spec_fflf.references[0].strength = 0.4
    workflow = compiled(spec_fflf).workflow
    guides = [n for n in workflow["nodes"] if n["type"] == "LTXVAddGuide"]
    assert guides[0]["widgets_values"][1] == pytest.approx(0.4)


def test_a_disabled_reference_is_not_compiled(spec_keyframes: Spec) -> None:
    spec_keyframes.references[1].enabled = False
    workflow = compiled(spec_keyframes).workflow
    assert count(workflow, "LTXVAddGuide") == 2


# --------------------------------------------------------------------------
# two stage
# --------------------------------------------------------------------------

def test_refine_adds_an_upsampler_and_a_second_sampler(spec_two_stage: Spec) -> None:
    workflow = compiled(spec_two_stage).workflow
    assert count(workflow, "LTXVLatentUpsampler") == 1
    assert count(workflow, "SamplerCustomAdvanced") == 2
    assert count(workflow, "LatentUpscaleModelLoader") == 1


def test_stage_one_runs_at_half_size(spec_two_stage: Spec) -> None:
    spec_two_stage.project.width, spec_two_stage.project.height = 1280, 704
    workflow = compiled(spec_two_stage).workflow
    assert widget(workflow, "EmptyLTXVLatentVideo", "width") == 640
    assert widget(workflow, "EmptyLTXVLatentVideo", "height") == 352


def test_stage_one_size_stays_on_the_32_grid(spec_two_stage: Spec) -> None:
    spec_two_stage.project.width, spec_two_stage.project.height = 1504, 800
    workflow = compiled(spec_two_stage).workflow
    for axis in ("width", "height"):
        assert widget(workflow, "EmptyLTXVLatentVideo", axis) % 32 == 0


def test_single_stage_runs_at_full_size(base_spec: Spec) -> None:
    base_spec.project.width, base_spec.project.height = 1280, 704
    workflow = compiled(base_spec).workflow
    assert widget(workflow, "EmptyLTXVLatentVideo", "width") == 1280


def test_the_refine_schedule_is_the_three_step_one(spec_two_stage: Spec) -> None:
    workflow = compiled(spec_two_stage).workflow
    schedules = [n["widgets_values"][0] for n in workflow["nodes"]
                 if n["type"] == "ManualSigmas"]
    assert len(schedules) == 2
    assert len(schedules[1].split(",")) == len(DEFAULT_SIGMAS_STAGE2)
    assert schedules[1].startswith("0.85")


def test_the_first_frame_is_re_pinned_after_upsampling(spec_i2v: Spec) -> None:
    # LTXVLatentUpsampler drops noise_mask, so without this the opening frame
    # drifts in the refine pass.
    spec_i2v.generation.stages = 2
    spec_i2v.models.upscaler = "up.safetensors"
    workflow = compiled(spec_i2v).workflow
    pins = [n for n in workflow["nodes"] if n["type"] == "LTXVImgToVideoInplace"]
    assert len(pins) == 2
    assert pins[1]["widgets_values"][0] == pytest.approx(1.0), "re-pin is a hard pin"


def test_t2v_does_not_re_pin_anything(spec_two_stage: Spec) -> None:
    assert count(compiled(spec_two_stage).workflow, "LTXVImgToVideoInplace") == 0


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def test_video_modes_generate_audio_by_default(base_spec: Spec) -> None:
    workflow = compiled(base_spec).workflow
    assert count(workflow, "LTXVEmptyLatentAudio") == 1
    assert count(workflow, "LTXVAudioVAEDecode") == 1


def test_muting_drops_the_whole_audio_branch(base_spec: Spec) -> None:
    base_spec.audio.mode = "mute"
    workflow = compiled(base_spec).workflow
    assert count(workflow, "LTXVEmptyLatentAudio") == 0
    assert count(workflow, "LTXVConcatAVLatent") == 0
    assert count(workflow, "LTXVAudioVAEDecode") == 0


def test_a2v_encodes_freezes_and_muxes_the_supplied_track(spec_a2v: Spec) -> None:
    workflow = compiled(spec_a2v).workflow
    assert count(workflow, "LTXVAudioVAEEncode") == 1
    assert count(workflow, "LTXVFreezeLatent") == 1, "supplied audio must not be denoised"
    assert count(workflow, "TrimAudioDuration") == 1
    # The original waveform goes straight to the output; re-decoding it would
    # only degrade what the user supplied.
    assert count(workflow, "LTXVAudioVAEDecode") == 0


def test_a2v_trims_to_the_shot_length(spec_a2v: Spec) -> None:
    spec_a2v.project.frames = 97
    spec_a2v.project.fps = 24.0
    workflow = compiled(spec_a2v).workflow
    assert widget(workflow, "TrimAudioDuration", "duration") == pytest.approx(97 / 24.0)


def test_audio_inpainting_says_what_it_cannot_do(spec_a2v: Spec) -> None:
    spec_a2v.audio.mode = "inpaint"
    result = compile_spec(spec_a2v)
    assert result.ok
    codes = {d.code for d in result.report}
    assert "audio.inpaint_approximated" in codes


def test_t2a_drops_the_video_side_entirely(spec_t2a: Spec) -> None:
    workflow = compiled(spec_t2a).workflow
    assert count(workflow, "LTXVAudioOnlyModel") == 1
    assert count(workflow, "LTXVAudioOnlyEmptyVideoLatent") == 1
    assert count(workflow, "EmptyLTXVLatentVideo") == 0
    assert count(workflow, "VAEDecodeTiled") == 0
    assert count(workflow, "SaveAudioAdvanced") == 1
    assert count(workflow, "VAELoader") == 1, "audio VAE only"


def test_t2a_never_refines(spec_t2a: Spec) -> None:
    spec_t2a.generation.stages = 2
    spec_t2a.models.upscaler = "up.safetensors"
    workflow = compiled(spec_t2a).workflow
    assert count(workflow, "SamplerCustomAdvanced") == 1
    assert count(workflow, "LTXVLatentUpsampler") == 0


# --------------------------------------------------------------------------
# IC-LoRA
# --------------------------------------------------------------------------

def test_iclora_loads_the_lora_and_adds_a_video_guide(spec_iclora: Spec) -> None:
    workflow = compiled(spec_iclora).workflow
    assert count(workflow, "LTXICLoRALoaderModelOnly") == 1
    assert count(workflow, "LTXAddVideoICLoRAGuide") == 1
    assert count(workflow, "LTXVCropGuides") == 1


def test_the_lora_reports_its_own_downscale_factor(spec_iclora: Spec) -> None:
    # Reading it from the loader beats hardcoding a number per LoRA file.
    workflow = compiled(spec_iclora).workflow
    guide = only(workflow, "LTXAddVideoICLoRAGuide")
    linked = {i["name"] for i in guide["inputs"] if i.get("link") is not None}
    assert "latent_downscale_factor" in linked


def test_iclora_guides_sit_on_the_one_mod_eight_grid(spec_iclora: Spec) -> None:
    spec_iclora.references[0].anchor = "index"
    spec_iclora.references[0].at.frame = 12
    workflow = compiled(spec_iclora).workflow
    assert only(workflow, "LTXAddVideoICLoRAGuide")["widgets_values"][0] == 9


def test_plain_loras_load_before_ic_loras(spec_iclora: Spec) -> None:
    from director.core.spec import LoraRef

    spec_iclora.models.loras.insert(0, LoraRef(name="style.safetensors", kind="lora"))
    order = execution_order(compiled(spec_iclora).workflow)
    assert order.index("LoraLoaderModelOnly") < order.index("LTXICLoRALoaderModelOnly")


def test_a_disabled_lora_is_not_loaded(spec_iclora: Spec) -> None:
    from director.core.spec import LoraRef

    spec_iclora.models.loras.append(LoraRef(name="off.safetensors", kind="lora", enabled=False))
    assert count(compiled(spec_iclora).workflow, "LoraLoaderModelOnly") == 0


# --------------------------------------------------------------------------
# prompt enhancement
# --------------------------------------------------------------------------

def test_enhancement_is_off_by_default(base_spec: Spec) -> None:
    workflow = compiled(base_spec).workflow
    assert count(workflow, "TextGenerateLTX2Prompt") == 0
    assert count(workflow, "CLIPLoader") == 1


def test_enhancement_adds_an_encoder_and_a_visible_preview(base_spec: Spec) -> None:
    base_spec.prompt.enhance.enabled = True
    base_spec.models.enhancer_clip = "gemma4_e2b_it_bf16.safetensors"
    workflow = compiled(base_spec).workflow
    assert count(workflow, "TextGenerateLTX2Prompt") == 1
    assert count(workflow, "CLIPLoader") == 2
    # Prompt manipulation is never hidden.
    assert count(workflow, "PreviewAny") == 1


def test_enhancement_is_grounded_on_the_first_frame(spec_i2v: Spec) -> None:
    spec_i2v.prompt.enhance.enabled = True
    spec_i2v.models.enhancer_clip = "gemma4_e2b_it_bf16.safetensors"
    workflow = compiled(spec_i2v).workflow
    enhance = only(workflow, "TextGenerateLTX2Prompt")
    linked = {i["name"] for i in enhance["inputs"] if i.get("link") is not None}
    assert "image" in linked


def test_enhancement_can_skip_the_image(spec_i2v: Spec) -> None:
    spec_i2v.prompt.enhance.enabled = True
    spec_i2v.prompt.enhance.use_image = False
    spec_i2v.models.enhancer_clip = "gemma4_e2b_it_bf16.safetensors"
    workflow = compiled(spec_i2v).workflow
    enhance = only(workflow, "TextGenerateLTX2Prompt")
    linked = {i["name"] for i in enhance["inputs"] if i.get("link") is not None}
    assert "image" not in linked


# --------------------------------------------------------------------------
# generation settings
# --------------------------------------------------------------------------

def test_dual_cfg_carries_both_scales(base_spec: Spec) -> None:
    base_spec.generation.video_cfg = 2.5
    base_spec.generation.audio_cfg = 6.0
    workflow = compiled(base_spec).workflow
    guider = only(workflow, "LTXVDualCFGGuider")
    assert guider["widgets_values"] == [2.5, 6.0]


def test_seed_mode_maps_to_control_after_generate(base_spec: Spec) -> None:
    base_spec.generation.seed_mode = "random"
    workflow = compiled(base_spec).workflow
    assert widget(workflow, "RandomNoise", "control_after_generate") == "randomize"


def test_decode_tiling_is_carried_through(base_spec: Spec) -> None:
    base_spec.generation.decode.tile_size = 256
    base_spec.generation.decode.temporal_overlap = 16
    workflow = compiled(base_spec).workflow
    assert widget(workflow, "VAEDecodeTiled", "tile_size") == 256
    assert widget(workflow, "VAEDecodeTiled", "temporal_overlap") == 16


def test_the_save_prefix_is_respected(base_spec: Spec) -> None:
    base_spec.generation.save_prefix = "video/lighthouse"
    workflow = compiled(base_spec).workflow
    assert widget(workflow, "SaveVideo", "filename_prefix") == "video/lighthouse"


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def test_an_unknown_family_is_refused_by_name(base_spec: Spec) -> None:
    base_spec.models.family = "ltx9"
    result = compile_spec(base_spec)
    assert not result.ok
    codes = {d.code for d in result.report}
    assert "compile.no_compiler" in codes or "caps.family" in codes


def test_a_broken_spec_never_produces_a_graph() -> None:
    result = compile_spec(Spec())
    assert not result.ok
    assert result.workflow == {}, "a failed compile must not hand back half a graph"


def test_validation_errors_come_through_the_compile_report(base_spec: Spec) -> None:
    base_spec.project.frames = 120
    result = compile_spec(base_spec)
    assert not result.ok
    assert "frames.not_8k1" in {d.code for d in result.report}


# --------------------------------------------------------------------------
# subgraph layout
# --------------------------------------------------------------------------

def test_subgraph_layout_packs_sections_into_definitions(base_spec: Spec) -> None:
    result = compile_spec(base_spec, layout="subgraphs")
    assert result.ok
    workflow = result.workflow
    subgraphs = workflow["definitions"]["subgraphs"]
    assert len(subgraphs) >= 4
    assert {s["name"] for s in subgraphs} >= {"Models", "Prompt", "Canvas", "Generate", "Output"}
    # The parent canvas holds one node per section, not the whole pipeline.
    assert len(workflow["nodes"]) == len(subgraphs)


def test_subgraph_io_uses_the_reserved_proxy_ids(base_spec: Spec) -> None:
    workflow = compile_spec(base_spec, layout="subgraphs").workflow
    proxies = {
        link["origin_id"] for sub in workflow["definitions"]["subgraphs"]
        for link in sub["links"]
    } | {
        link["target_id"] for sub in workflow["definitions"]["subgraphs"]
        for link in sub["links"]
    }
    assert SUBGRAPH_INPUT_ID in proxies
    assert SUBGRAPH_OUTPUT_ID in proxies


def test_subgraph_boundaries_declare_matching_io_counts(base_spec: Spec) -> None:
    workflow = compile_spec(base_spec, layout="subgraphs").workflow
    by_id = {s["id"]: s for s in workflow["definitions"]["subgraphs"]}
    for node in workflow["nodes"]:
        definition = by_id[node["type"]]
        assert len(node["inputs"]) == len(definition["inputs"])
        assert len(node["outputs"]) == len(definition["outputs"])


def test_layouts_contain_the_same_work(base_spec: Spec) -> None:
    flat = compile_spec(base_spec, layout="flat").workflow
    packed = compile_spec(base_spec, layout="subgraphs").workflow
    inner = [n["type"] for sub in packed["definitions"]["subgraphs"] for n in sub["nodes"]]
    assert sorted(inner) == sorted(node_types(flat))


def test_subgraph_internals_are_structurally_sound(spec_keyframes: Spec) -> None:
    """Each definition's own links must resolve, exactly as the parent's do."""
    workflow = compile_spec(spec_keyframes, layout="subgraphs").workflow
    for sub in workflow["definitions"]["subgraphs"]:
        by_id = {n["id"]: n for n in sub["nodes"]}
        declared_inputs = len(sub["inputs"])
        declared_outputs = len(sub["outputs"])

        for link in sub["links"]:
            if link["origin_id"] == SUBGRAPH_INPUT_ID:
                assert link["origin_slot"] < declared_inputs
            else:
                origin = by_id[link["origin_id"]]
                assert link["origin_slot"] < len(origin["outputs"])
                assert link["id"] in (origin["outputs"][link["origin_slot"]]["links"] or [])

            if link["target_id"] == SUBGRAPH_OUTPUT_ID:
                assert link["target_slot"] < declared_outputs
            else:
                target = by_id[link["target_id"]]
                assert link["target_slot"] < len(target["inputs"])
                assert target["inputs"][link["target_slot"]]["link"] == link["id"]

        for slot in sub["inputs"] + sub["outputs"]:
            assert slot["linkIds"], f"{sub['name']} declares an unwired boundary slot"


def test_subgraph_link_ids_are_unique_across_the_document(base_spec: Spec) -> None:
    workflow = compile_spec(base_spec, layout="subgraphs").workflow
    ids = [link[0] for link in workflow["links"]]
    for sub in workflow["definitions"]["subgraphs"]:
        ids.extend(link["id"] for link in sub["links"])
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# API (execution) format
#
# What POST /prompt takes, and therefore what the Generate button queues.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ALL_MODES)
def test_the_api_form_covers_every_working_node(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    result = compiled(request.getfixturevalue(fixture))
    working = [t for t in node_types(result.workflow) if t not in ("MarkdownNote", "Note")]
    assert len(result.api) == len(working)


@pytest.mark.parametrize("fixture", ALL_MODES)
def test_api_links_resolve_to_real_nodes(fixture: str, request: pytest.FixtureRequest) -> None:
    api = compiled(request.getfixturevalue(fixture)).api
    for node_id, node in api.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                origin, slot = value
                assert origin in api, f"{node['class_type']}.{name} points at missing {origin}"
                assert isinstance(slot, int)


def test_api_node_ids_match_the_workflow(base_spec: Spec) -> None:
    result = compiled(base_spec)
    workflow_ids = {str(n["id"]) for n in result.workflow["nodes"]}
    assert set(result.api) <= workflow_ids


def test_frontend_only_widgets_are_not_sent_to_the_executor(base_spec: Spec) -> None:
    # control_after_generate is a litegraph affordance, not an input; the
    # executor rejects unknown inputs.
    api = compiled(base_spec).api
    for node in api.values():
        assert "control_after_generate" not in node["inputs"]


def test_api_carries_widget_values_and_links_together(spec_i2v: Spec) -> None:
    api = compiled(spec_i2v).api
    pin = next(n for n in api.values() if n["class_type"] == "LTXVImgToVideoInplace")
    assert isinstance(pin["inputs"]["vae"], list), "vae is wired"
    assert pin["inputs"]["strength"] == pytest.approx(0.7), "strength is a value"


def test_api_omits_unconnected_optional_inputs(base_spec: Spec) -> None:
    api = compiled(base_spec).api
    create = next(n for n in api.values() if n["class_type"] == "CreateVideo")
    base_spec.audio.mode = "mute"
    silent = next(
        n for n in compiled(base_spec).api.values() if n["class_type"] == "CreateVideo"
    )
    assert "audio" in create["inputs"]
    assert "audio" not in silent["inputs"], "a muted shot must not claim an audio input"


def test_the_api_form_is_deterministic(spec_keyframes: Spec) -> None:
    first = json.dumps(compiled(spec_keyframes).api, sort_keys=True)
    second = json.dumps(compiled(spec_keyframes).api, sort_keys=True)
    assert first == second


def test_every_api_node_names_a_class(spec_two_stage: Spec) -> None:
    for node in compiled(spec_two_stage).api.values():
        assert node["class_type"]
        assert isinstance(node["inputs"], dict)


# --------------------------------------------------------------------------
# Prompt Relay in a compiled graph
#
# The one place the compiled graph is not purely native nodes, because nothing
# upstream applies per-region prompts in a single sampling pass. A graph that
# quietly dropped them would render something other than what the editor showed.
# --------------------------------------------------------------------------

def test_a_shot_without_regions_compiles_to_native_nodes_only(base_spec: Spec) -> None:
    types = set(node_types(compiled(base_spec).workflow))
    assert not any(t.startswith("LTXDirector") for t in types)


def test_several_prompt_regions_bring_the_relay_node_into_the_graph(spec_relay: Spec) -> None:
    workflow = compiled(spec_relay).workflow
    assert count(workflow, "LTXDirectorProject") == 1
    assert count(workflow, "LTXDirectorRelay") == 1
    # The relay encodes the positive side, so only the negative needs an encoder.
    assert count(workflow, "CLIPTextEncode") == 1


def test_the_relay_is_announced_rather_than_slipped_in(spec_relay: Spec) -> None:
    result = compile_spec(spec_relay)
    diagnostic = next(d for d in result.report if d.code == "relay.in_graph")
    assert "stays in the graph" in diagnostic.message
    assert "Switch Prompt Relay off" in diagnostic.fix


def test_switching_relay_off_returns_to_a_purely_native_graph(spec_relay: Spec) -> None:
    spec_relay.relay.enabled = False
    types = set(node_types(compiled(spec_relay).workflow))
    assert not any(t.startswith("LTXDirector") for t in types)
    assert "CLIPTextEncode" in types


def test_the_relays_patched_model_is_what_gets_sampled(spec_relay: Spec) -> None:
    # Sampling the unpatched model would mean the regions had no effect at all.
    api = compiled(spec_relay).api
    relay_id = next(k for k, v in api.items() if v["class_type"] == "LTXDirectorRelay")
    guider = next(v for v in api.values() if v["class_type"] == "LTXVDualCFGGuider")
    assert guider["inputs"]["model"][0] == relay_id


def test_the_embedded_project_carries_only_what_the_relay_reads(spec_relay: Spec) -> None:
    from director.core.spec import MediaEntry

    # Give the shot a media registry and some takes; neither belongs in the graph.
    entry = MediaEntry(filename="unrelated.png", kind="image")
    spec_relay.media[entry.id] = entry
    workflow = compiled(spec_relay).workflow
    embedded = only(workflow, "LTXDirectorProject")["widgets_values"][0]

    assert "unrelated.png" not in embedded
    assert "the beam sweeps left" in embedded
    assert len(embedded) < 4096


def test_a_single_region_does_not_bring_the_relay_in(base_spec: Spec) -> None:
    from director.core.spec import Segment

    base_spec.segments = [Segment(start=0, length=121, text="only one")]
    types = set(node_types(compiled(base_spec).workflow))
    assert "LTXDirectorRelay" not in types

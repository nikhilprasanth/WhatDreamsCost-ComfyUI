"""Capability probing.

ComfyUI is faked by injecting ``nodes`` and ``folder_paths`` modules, which is
both how the probe is meant to be exercised and proof that it only ever asks
ComfyUI — never the filesystem.
"""

from __future__ import annotations

import sys
import types
from typing import Iterator

import pytest

from director.capabilities import capabilities, invalidate, probe
from director.capabilities.probe import family_of

# A plausible LTX-2.5 install: ComfyUI core LTX nodes plus the Lightricks pack.
CORE_NODES = {
    "EmptyLTXVLatentVideo", "LTXVConditioning", "SamplerCustomAdvanced",
    "LTXVEmptyLatentAudio", "LTXVAudioVAEDecode", "LTXVAudioVAEEncode",
    "LTXVConcatAVLatent", "LTXVSeparateAVLatent", "LTXVAddGuide", "LTXVCropGuides",
    "LTXVImgToVideoInplace", "LTXVDualCFGGuider", "LTXVModalityGuidance",
    "LTXVReferenceAudio", "LTXVFreezeLatent", "LTXVAddGeneratedKeyframes",
    "LTXVSeparateGeneratedKeyframes", "LTXVDurationPredictor", "LTXVLatentUpsampler",
    "LatentUpscaleModelLoader", "TextGenerateLTX2Prompt", "VAEDecodeTiled",
    "TrimAudioDuration",
}
PACK_NODES = {
    "LTXICLoRALoaderModelOnly", "LTXAddVideoICLoRAGuide",
    "LTXVAudioOnlyModel", "LTXVAudioOnlyEmptyVideoLatent",
    "LTXVSparseTrackEditor", "LTXVDrawTracks",
}

FULL_MODELS = {
    "diffusion_models": ["ltx-2.5-22b-distilled-transformer-bf16.safetensors"],
    "checkpoints": [],
    "vae": [
        "ltx-2.5-video-vae-bf16.safetensors",
        "ltx-2.5-audio-vae-bf16.safetensors",
    ],
    "text_encoders": [
        "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "gemma4_e2b_it_bf16.safetensors",
    ],
    "loras": [
        "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
        "squish-it.safetensors",
    ],
    "latent_upscale_models": ["ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"],
    "model_patches": ["ltx-2.4-duration-head.safetensors"],
}


@pytest.fixture
def fake_comfy(monkeypatch: pytest.MonkeyPatch):
    """Install fake ``nodes`` and ``folder_paths`` modules for the duration."""

    def install(node_names: set[str], models: dict[str, list[str]]) -> None:
        nodes_module = types.ModuleType("nodes")
        nodes_module.NODE_CLASS_MAPPINGS = {name: object for name in node_names}  # type: ignore[attr-defined]

        folder_module = types.ModuleType("folder_paths")

        def get_filename_list(folder: str) -> list[str]:
            if folder not in models:
                raise KeyError(folder)  # what ComfyUI does for an unknown folder
            return list(models[folder])

        folder_module.get_filename_list = get_filename_list  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "nodes", nodes_module)
        monkeypatch.setitem(sys.modules, "folder_paths", folder_module)
        invalidate()

    yield install
    invalidate()


@pytest.fixture
def full_install(fake_comfy) -> Iterator[None]:
    fake_comfy(CORE_NODES | PACK_NODES, FULL_MODELS)
    yield


# --------------------------------------------------------------------------
# absent ComfyUI
# --------------------------------------------------------------------------

def test_probing_without_comfyui_does_not_raise() -> None:
    invalidate()
    result = probe()
    assert result.available is False
    assert result.nodes == set()


def test_an_unknown_installation_reports_unknown_not_missing() -> None:
    invalidate()
    caps = capabilities()
    assert caps.reason("iclora").startswith("ComfyUI is not running")


def test_families_fall_back_to_what_can_be_compiled() -> None:
    # A user whose files are named oddly should still be able to work.
    invalidate()
    assert "ltx2.5" in capabilities().families


# --------------------------------------------------------------------------
# a complete install
# --------------------------------------------------------------------------

def test_a_full_install_enables_everything(full_install: None) -> None:
    caps = capabilities(refresh=True)
    assert caps.probe.available
    off = sorted(n for n, v in caps.flags.items() if not v)
    assert not off, f"unexpectedly unavailable: {off}"


def test_models_are_grouped_by_role(full_install: None) -> None:
    p = probe(refresh=True)
    assert p.video_vaes() == ["ltx-2.5-video-vae-bf16.safetensors"]
    assert p.audio_vaes() == ["ltx-2.5-audio-vae-bf16.safetensors"]
    assert p.encoders() == ["gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"]
    assert p.enhancers() == ["gemma4_e2b_it_bf16.safetensors"]
    assert p.ic_loras() == ["ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"]
    assert p.plain_loras() == ["squish-it.safetensors"]


def test_the_audio_vae_is_never_offered_as_the_video_vae(full_install: None) -> None:
    # Selecting the wrong one fails deep inside the sampler, so the grouping has
    # to be exact rather than approximate.
    p = probe(refresh=True)
    assert not set(p.video_vaes()) & set(p.audio_vaes())


def test_the_enhancer_is_never_offered_as_the_encoder(full_install: None) -> None:
    p = probe(refresh=True)
    assert not set(p.encoders()) & set(p.enhancers())


def test_the_family_comes_from_the_transformer(full_install: None) -> None:
    assert probe(refresh=True).families() == ("ltx2.5",)


# --------------------------------------------------------------------------
# partial installs
# --------------------------------------------------------------------------

def test_without_the_lightricks_pack_iclora_is_off_with_a_reason(fake_comfy) -> None:
    fake_comfy(CORE_NODES, FULL_MODELS)
    caps = capabilities(refresh=True)
    assert not caps.has("iclora")
    assert "LTXVideo custom nodes" in caps.reason("iclora")
    # Everything core still works.
    assert caps.has("mid_timeline_guides") and caps.has("audio") and caps.has("two_stage")


def test_without_an_upscaler_the_refine_pass_is_off(fake_comfy) -> None:
    models = {**FULL_MODELS, "latent_upscale_models": []}
    fake_comfy(CORE_NODES | PACK_NODES, models)
    caps = capabilities(refresh=True)
    assert not caps.has("two_stage")
    assert "latent_upscale_models" in caps.reason("two_stage")


def test_without_the_small_encoder_enhancement_is_off(fake_comfy) -> None:
    models = {
        **FULL_MODELS,
        "text_encoders": ["gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"],
    }
    fake_comfy(CORE_NODES | PACK_NODES, models)
    caps = capabilities(refresh=True)
    assert not caps.has("prompt_enhancer")
    assert "text_encoders" in caps.reason("prompt_enhancer")


def test_without_the_duration_head_suggesting_duration_is_off(fake_comfy) -> None:
    models = {**FULL_MODELS, "model_patches": []}
    fake_comfy(CORE_NODES | PACK_NODES, models)
    assert not capabilities(refresh=True).has("duration_predictor")


def test_an_old_comfyui_loses_audio_but_keeps_video(fake_comfy) -> None:
    old = CORE_NODES - {"LTXVEmptyLatentAudio", "LTXVAudioVAEDecode", "LTXVConcatAVLatent"}
    fake_comfy(old, FULL_MODELS)
    caps = capabilities(refresh=True)
    assert not caps.has("audio")
    assert caps.has("core_ltx") and caps.has("mid_timeline_guides")
    assert "silently" in caps.reason("audio")


def test_an_unregistered_model_folder_is_survivable(fake_comfy) -> None:
    # Older ComfyUI builds do not know latent_upscale_models at all.
    models = {k: v for k, v in FULL_MODELS.items() if k != "latent_upscale_models"}
    fake_comfy(CORE_NODES, models)
    caps = capabilities(refresh=True)
    assert caps.probe.available
    assert not caps.has("two_stage")


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------

def test_every_reason_is_a_sentence_a_filmmaker_can_act_on() -> None:
    from director.capabilities.features import FEATURES

    for feature in FEATURES:
        assert feature.reason.endswith("."), feature.name
        assert feature.reason[0].isupper(), feature.name
        assert "None" not in feature.reason and "Error" not in feature.reason, feature.name
        assert feature.label and feature.label[0].isupper(), feature.name


def test_asking_for_an_unknown_capability_is_a_mistake_not_a_default() -> None:
    invalidate()
    with pytest.raises(KeyError, match="Unknown capability"):
        capabilities().has("time_travel")


def test_the_probe_is_cached_until_invalidated(full_install: None) -> None:
    first = probe(refresh=True)
    assert probe() is first
    invalidate()
    assert probe() is not first


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("ltx-2.5-22b-distilled-transformer-bf16.safetensors", "ltx2.5"),
        ("ltx-2.3-22b-dev.safetensors", "ltx2.3"),
        ("LTX_2.0_something.safetensors", "ltx2.0"),
        ("my-custom-model.safetensors", None),
    ],
)
def test_family_detection_is_a_hint_not_a_rule(filename: str, expected: str | None) -> None:
    assert family_of(filename) == expected


def test_capabilities_serialise_for_the_frontend(full_install: None) -> None:
    payload = capabilities(refresh=True).to_dict()
    assert payload["available"] is True
    assert set(payload) == {"available", "flags", "labels", "missing", "families", "models"}
    assert payload["missing"] == {}
    assert payload["models"]["transformers"]

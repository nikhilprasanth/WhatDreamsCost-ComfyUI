"""Presets.

The rule being defended: a preset changes settings the user can see, name and
undo. Nothing hidden, nothing that only makes sense inside the preset.
"""

from __future__ import annotations

import pytest

from director.core.spec import Spec
from director.core.validate import validate
from director.presets import (
    QUALITY_PRESETS,
    VRAM_PRESETS,
    all_presets,
    apply_preset,
)

ALL = [p.name for p in QUALITY_PRESETS + VRAM_PRESETS]


@pytest.mark.parametrize("name", ALL)
def test_every_preset_changes_only_real_settings(name: str, base_spec: Spec) -> None:
    # A path that is not a setting would fail silently at apply time otherwise.
    apply_preset(base_spec, name)


@pytest.mark.parametrize("name", ALL)
def test_every_preset_leaves_a_valid_shot(name: str, base_spec: Spec) -> None:
    if name in ("quality", "final"):
        base_spec.models.upscaler = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    apply_preset(base_spec, name)
    report = validate(base_spec)
    assert report.ok, [str(d) for d in report.errors]


@pytest.mark.parametrize("name", ALL)
def test_applying_a_preset_reports_exactly_what_moved(name: str, base_spec: Spec) -> None:
    changed = apply_preset(base_spec, name)
    for path, (before, after) in changed.items():
        assert before != after, f"{path} was reported as changed but was not"
    # Applying the same preset again moves nothing: the report is honest.
    assert apply_preset(base_spec, name) == {}


@pytest.mark.parametrize("name", ALL)
def test_every_preset_is_described_for_a_person(name: str) -> None:
    preset = next(p for p in QUALITY_PRESETS + VRAM_PRESETS if p.name == name)
    assert preset.label and (preset.label[0].isupper() or preset.label[0].isdigit())
    assert preset.description.endswith(".")
    assert "sigma" not in preset.description.lower(), "descriptions are for filmmakers"


def test_an_unknown_preset_names_the_ones_that_exist(base_spec: Spec) -> None:
    with pytest.raises(ValueError) as excinfo:
        apply_preset(base_spec, "cinematic_magic")
    assert "balanced" in str(excinfo.value)


def test_quality_presets_differ_in_compute(base_spec: Spec) -> None:
    base_spec.models.upscaler = "up.safetensors"
    draft = base_spec.copy()
    apply_preset(draft, "draft")
    final = base_spec.copy()
    apply_preset(final, "final")

    assert draft.generation.stages < final.generation.stages
    assert draft.project.width * draft.project.height < final.project.width * final.project.height


def test_vram_presets_change_memory_not_look(base_spec: Spec) -> None:
    # A VRAM preset that also changed resolution would silently change the shot.
    before = (base_spec.project.width, base_spec.project.height, base_spec.generation.stages)
    for name in (p.name for p in VRAM_PRESETS):
        spec = base_spec.copy()
        changed = apply_preset(spec, name)
        assert (spec.project.width, spec.project.height, spec.generation.stages) == before
        assert all(
            path.startswith("generation.decode") or path == "generation.stage1_divisor"
            for path in changed
        ), changed


def test_smaller_vram_means_smaller_tiles(base_spec: Spec) -> None:
    sizes = []
    for name in ("vram_16", "vram_24", "vram_32", "vram_48"):
        spec = base_spec.copy()
        apply_preset(spec, name)
        sizes.append(spec.generation.decode.tile_size)
    assert sizes == sorted(sizes), "tile size must rise with available memory"


def test_quality_and_vram_presets_compose(base_spec: Spec) -> None:
    base_spec.models.upscaler = "up.safetensors"
    apply_preset(base_spec, "quality")
    apply_preset(base_spec, "vram_16")
    assert base_spec.generation.stages == 2, "the VRAM preset must not undo the quality one"
    assert base_spec.generation.decode.tile_size == 256


def test_the_quality_preset_is_recorded_on_the_shot(base_spec: Spec) -> None:
    apply_preset(base_spec, "draft")
    assert base_spec.generation.preset == "draft"


def test_a_vram_preset_does_not_claim_to_be_the_quality_preset(base_spec: Spec) -> None:
    apply_preset(base_spec, "draft")
    apply_preset(base_spec, "vram_48")
    assert base_spec.generation.preset == "draft"


def test_presets_serialise_for_the_editor() -> None:
    payload = all_presets()
    assert set(payload) == {"quality", "vram"}
    for group in payload.values():
        for preset in group:
            assert set(preset) == {"name", "label", "description", "changes"}
            assert preset["changes"], preset["name"]


def test_seed_hunt_is_actually_cheap(base_spec: Spec) -> None:
    apply_preset(base_spec, "seed_hunt")
    assert base_spec.generation.stages == 1
    assert base_spec.generation.seed_mode == "random"
    assert base_spec.project.width * base_spec.project.height < 1280 * 704

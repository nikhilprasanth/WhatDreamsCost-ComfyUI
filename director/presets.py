"""Presets, as explicit setting deltas.

A preset that changes hidden state is a preset you cannot reason about. Every
one here is a flat map of ``dotted.path -> value``, so the editor can show
exactly what changed, the user can undo any single part of it, and a preset can
be applied to a spec without the preset code knowing anything about the spec's
shape.

Two families:

``quality``
    How much compute the shot gets — from a seed-hunting draft to a final render.
``vram``
    How that compute is spent, for a given amount of video memory. Tiling and
    stage-1 resolution, not quality.

They compose: pick a quality preset, then a VRAM one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core.spec import DEFAULT_SIGMAS_STAGE1, DEFAULT_SIGMAS_STAGE2, Spec

__all__ = ["Preset", "QUALITY_PRESETS", "VRAM_PRESETS", "all_presets", "apply_preset"]


@dataclass(frozen=True)
class Preset:
    """One named set of changes."""

    name: str
    label: str
    description: str
    #: ``dotted.path -> value``, applied to a Spec in order.
    changes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "changes": self.changes,
        }


# --------------------------------------------------------------------------
# quality
# --------------------------------------------------------------------------

QUALITY_PRESETS: tuple[Preset, ...] = (
    Preset(
        "seed_hunt", "Seed hunt",
        "Small and fast, for finding a composition worth keeping. Generate several, "
        "then promote the one you like.",
        {
            "generation.stages": 1,
            "generation.sigmas_stage1": list(DEFAULT_SIGMAS_STAGE1),
            "generation.seed_mode": "random",
            "generation.stage1_divisor": 1,
            "project.width": 704,
            "project.height": 384,
            "generation.decode.tile_size": 512,
            "generation.decode.temporal_size": 32,
        },
    ),
    Preset(
        "draft", "Draft",
        "Half resolution, one pass. Quick enough to iterate on the prompt.",
        {
            "generation.stages": 1,
            "generation.sigmas_stage1": list(DEFAULT_SIGMAS_STAGE1),
            "generation.stage1_divisor": 1,
            "project.width": 960,
            "project.height": 544,
        },
    ),
    Preset(
        "balanced", "Balanced",
        "One pass at full size. The default.",
        {
            "generation.stages": 1,
            "generation.sigmas_stage1": list(DEFAULT_SIGMAS_STAGE1),
            "generation.stage1_divisor": 1,
            "project.width": 1280,
            "project.height": 704,
        },
    ),
    Preset(
        "quality", "Quality",
        "Generate at half size to lock composition and motion, then upscale and "
        "refine. Slower, and noticeably sharper.",
        {
            "generation.stages": 2,
            "generation.stage1_divisor": 2,
            "generation.sigmas_stage1": list(DEFAULT_SIGMAS_STAGE1),
            "generation.sigmas_stage2": list(DEFAULT_SIGMAS_STAGE2),
            "project.width": 1280,
            "project.height": 704,
        },
    ),
    Preset(
        "final", "Final render",
        "Two stages at full output size, fixed seed, with generous decode tiles.",
        {
            "generation.stages": 2,
            "generation.stage1_divisor": 2,
            "generation.sigmas_stage1": list(DEFAULT_SIGMAS_STAGE1),
            "generation.sigmas_stage2": list(DEFAULT_SIGMAS_STAGE2),
            "generation.seed_mode": "fixed",
            "project.width": 1920,
            "project.height": 1088,
            "generation.decode.tile_size": 768,
            "generation.decode.temporal_size": 128,
            "generation.decode.temporal_overlap": 32,
        },
    ),
)


# --------------------------------------------------------------------------
# memory
#
# Decode tiling dominates peak memory at decode time, and stage-1 resolution
# dominates it during sampling. These change those two things and nothing else,
# so a VRAM preset never silently changes how the shot looks.
# --------------------------------------------------------------------------

VRAM_PRESETS: tuple[Preset, ...] = (
    Preset(
        "vram_16", "16 GB",
        "Small decode tiles and a half-size first stage. Slowest to decode, but it fits.",
        {
            "generation.decode.tile_size": 256,
            "generation.decode.overlap": 32,
            "generation.decode.temporal_size": 32,
            "generation.decode.temporal_overlap": 8,
            "generation.stage1_divisor": 2,
        },
    ),
    Preset(
        "vram_24", "24 GB",
        "Moderate decode tiles.",
        {
            "generation.decode.tile_size": 384,
            "generation.decode.overlap": 48,
            "generation.decode.temporal_size": 48,
            "generation.decode.temporal_overlap": 8,
        },
    ),
    Preset(
        "vram_32", "32 GB",
        "The values the reference LTX-2.5 graphs ship with.",
        {
            "generation.decode.tile_size": 512,
            "generation.decode.overlap": 64,
            "generation.decode.temporal_size": 64,
            "generation.decode.temporal_overlap": 8,
        },
    ),
    Preset(
        "vram_48", "48 GB and above",
        "Large decode tiles. Fewer, bigger tiles decode faster.",
        {
            "generation.decode.tile_size": 1024,
            "generation.decode.overlap": 96,
            "generation.decode.temporal_size": 128,
            "generation.decode.temporal_overlap": 32,
        },
    ),
)


_BY_NAME = {p.name: p for p in QUALITY_PRESETS + VRAM_PRESETS}


def all_presets() -> dict[str, list[dict[str, Any]]]:
    return {
        "quality": [p.to_dict() for p in QUALITY_PRESETS],
        "vram": [p.to_dict() for p in VRAM_PRESETS],
    }


def apply_preset(spec: Spec, name: str) -> dict[str, tuple[Any, Any]]:
    """Apply a preset in place, returning ``path -> (before, after)``.

    The return value is what makes a preset inspectable: the editor shows it, so
    the user can see every setting that moved and put any of them back.
    """
    preset = _BY_NAME.get(name)
    if preset is None:
        raise ValueError(
            f"Unknown preset {name!r}. Known: {', '.join(sorted(_BY_NAME))}."
        )

    changed: dict[str, tuple[Any, Any]] = {}
    for path, value in preset.changes.items():
        before = _get(spec, path)
        if before == value:
            continue
        _set(spec, path, value)
        changed[path] = (before, value)

    spec.generation.preset = name if name in {p.name for p in QUALITY_PRESETS} else spec.generation.preset
    spec.normalise()
    return changed


def _resolve(spec: Spec, path: str) -> tuple[Any, str]:
    target: Any = spec
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    return target, parts[-1]


def _get(spec: Spec, path: str) -> Any:
    target, leaf = _resolve(spec, path)
    return getattr(target, leaf)


def _set(spec: Spec, path: str, value: Any) -> None:
    target, leaf = _resolve(spec, path)
    if not hasattr(target, leaf):
        raise ValueError(f"A preset refers to {path!r}, which is not a project setting.")
    setattr(target, leaf, value)

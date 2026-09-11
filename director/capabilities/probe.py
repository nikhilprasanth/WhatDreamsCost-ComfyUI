"""What this installation can actually do.

Asked of ComfyUI itself rather than of the filesystem: node availability comes
from the node registry, model availability from ``folder_paths``. Guessing from
directory listings is how you end up claiming a feature exists because a folder
is there.

Importable with no ComfyUI present — every import is guarded and an absent
ComfyUI yields an "unknown" probe rather than an exception. That is what lets
the compiler, the validator and the whole test suite run on a laptop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

log = logging.getLogger(__name__)

__all__ = ["Probe", "probe", "invalidate", "ModelSet"]


# --------------------------------------------------------------------------
# talking to ComfyUI, carefully
# --------------------------------------------------------------------------

def _registered_nodes() -> set[str] | None:
    """Node class names ComfyUI has registered, or None when it is not present."""
    try:
        import nodes  # type: ignore
    except Exception:
        return None
    mapping = getattr(nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mapping, dict):
        return None
    return set(mapping)


def _folder_listing(folder: str) -> list[str]:
    """Filenames ComfyUI knows about for a model folder.

    ``folder_paths`` owns the search path, including extra_model_paths.yaml, so
    it sees models a naive listdir of ``models/`` would miss entirely.
    """
    try:
        import folder_paths  # type: ignore
    except Exception:
        return []
    try:
        return list(folder_paths.get_filename_list(folder))
    except Exception:
        # An unregistered folder name is normal on older ComfyUI builds.
        return []


# --------------------------------------------------------------------------
# model discovery
# --------------------------------------------------------------------------

#: Folder → what we call it. ``diffusion_models`` and ``checkpoints`` both hold
#: transformers depending on how the user installed them.
MODEL_FOLDERS: dict[str, str] = {
    "diffusion_models": "unet",
    "checkpoints": "checkpoint",
    "vae": "vae",
    "text_encoders": "clip",
    "loras": "lora",
    "latent_upscale_models": "upscaler",
    "model_patches": "model_patch",
}


@dataclass
class ModelSet:
    """Everything installed, grouped the way the Director talks about it."""

    unet: list[str] = field(default_factory=list)
    checkpoint: list[str] = field(default_factory=list)
    vae: list[str] = field(default_factory=list)
    clip: list[str] = field(default_factory=list)
    lora: list[str] = field(default_factory=list)
    upscaler: list[str] = field(default_factory=list)
    model_patch: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "unet": self.unet,
            "checkpoint": self.checkpoint,
            "vae": self.vae,
            "clip": self.clip,
            "lora": self.lora,
            "upscaler": self.upscaler,
            "model_patch": self.model_patch,
        }


# Filename patterns. Used only to *suggest* defaults and to classify — never to
# decide whether a feature exists; that is the node registry's job.
_VIDEO_VAE = re.compile(r"video[-_]?vae|ltx.*vae(?!.*audio)", re.I)
_AUDIO_VAE = re.compile(r"audio[-_]?vae", re.I)
_ENHANCER = re.compile(r"e2b|enhanc", re.I)
_ENCODER = re.compile(r"with[-_]?proj|gemma", re.I)
_SPATIAL_UPSCALER = re.compile(r"spatial[-_]?upscaler", re.I)
_IC_LORA = re.compile(r"ic[-_]?lora", re.I)
_FAMILY = re.compile(r"ltx[-_ ]?(\d+\.\d+)", re.I)


def family_of(filename: str) -> str | None:
    """The LTX family a filename implies, e.g. ``ltx2.5``.

    A weak signal, and treated as one: it seeds a default selection, and the
    user can always override it.
    """
    match = _FAMILY.search(filename or "")
    return f"ltx{match.group(1)}" if match else None


def _matching(names: Iterable[str], pattern: re.Pattern[str]) -> list[str]:
    return [n for n in names if pattern.search(n)]


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------

@dataclass
class Probe:
    """A snapshot of what is installed.

    ``available`` is False when ComfyUI is not importable at all. In that state
    every flag is unknown rather than false, so nothing claims a feature is
    missing when it simply has not been asked.
    """

    available: bool = False
    nodes: set[str] = field(default_factory=set)
    models: ModelSet = field(default_factory=ModelSet)

    # -- nodes -------------------------------------------------------------

    def has_node(self, *names: str) -> bool:
        """True when every named node is registered."""
        if not self.available:
            return False
        return all(name in self.nodes for name in names)

    def missing_nodes(self, *names: str) -> list[str]:
        return [n for n in names if n not in self.nodes]

    # -- models ------------------------------------------------------------

    def video_vaes(self) -> list[str]:
        audio = set(_matching(self.models.vae, _AUDIO_VAE))
        preferred = [v for v in _matching(self.models.vae, _VIDEO_VAE) if v not in audio]
        return preferred or [v for v in self.models.vae if v not in audio]

    def audio_vaes(self) -> list[str]:
        return _matching(self.models.vae, _AUDIO_VAE)

    def encoders(self) -> list[str]:
        enhancers = set(self.enhancers())
        preferred = [c for c in _matching(self.models.clip, _ENCODER) if c not in enhancers]
        return preferred or [c for c in self.models.clip if c not in enhancers]

    def enhancers(self) -> list[str]:
        return _matching(self.models.clip, _ENHANCER)

    def upscalers(self) -> list[str]:
        return _matching(self.models.upscaler, _SPATIAL_UPSCALER) or self.models.upscaler

    def transformers(self) -> list[str]:
        return self.models.unet or self.models.checkpoint

    def ic_loras(self) -> list[str]:
        return _matching(self.models.lora, _IC_LORA)

    def plain_loras(self) -> list[str]:
        ic = set(self.ic_loras())
        return [n for n in self.models.lora if n not in ic]

    def families(self) -> tuple[str, ...]:
        """LTX families the installed transformers suggest, newest first."""
        found = {family_of(name) for name in self.transformers()}
        found.discard(None)
        return tuple(sorted((f for f in found if f), reverse=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "models": self.models.to_dict(),
            "grouped": {
                "transformers": self.transformers(),
                "video_vaes": self.video_vaes(),
                "audio_vaes": self.audio_vaes(),
                "encoders": self.encoders(),
                "enhancers": self.enhancers(),
                "upscalers": self.upscalers(),
                "ic_loras": self.ic_loras(),
                "loras": self.plain_loras(),
            },
            "families": list(self.families()),
        }


_CACHE: Probe | None = None


def probe(*, refresh: bool = False) -> Probe:
    """The current probe, cached for the process.

    Cached because ``get_filename_list`` walks the model folders and the
    frontend asks on every panel open. ``refresh=True`` after the user installs
    something; :func:`invalidate` does the same from elsewhere.
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE

    nodes = _registered_nodes()
    if nodes is None:
        log.debug("[Director] ComfyUI not importable; capabilities are unknown.")
        _CACHE = Probe(available=False)
        return _CACHE

    models = ModelSet()
    for folder, attribute in MODEL_FOLDERS.items():
        getattr(models, attribute).extend(_folder_listing(folder))

    _CACHE = Probe(available=True, nodes=nodes, models=models)
    return _CACHE


def invalidate() -> None:
    """Drop the cached probe. Call after models are added or nodes reloaded."""
    global _CACHE
    _CACHE = None

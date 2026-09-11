"""Capability and preset routes.

What the editor asks on open, so it can show only what this installation can
actually do — and, for anything it cannot, say why.
"""

from __future__ import annotations

from aiohttp import web

from .common import ok

__all__ = ["register"]


def register(routes) -> None:
    @routes.get("/ltxdirector/capabilities")
    async def get_capabilities(request: web.Request) -> web.Response:
        """Feature flags, installed models grouped by role, and suggested defaults.

        ``?refresh=1`` re-probes, for after the user installs something without
        restarting ComfyUI.
        """
        from ..capabilities import capabilities

        refresh = request.query.get("refresh") in ("1", "true", "yes")
        caps = capabilities(refresh=refresh)
        payload = caps.to_dict()
        payload["defaults"] = _suggested_defaults(caps)
        return ok(payload)

    @routes.get("/ltxdirector/presets")
    async def get_presets(request: web.Request) -> web.Response:
        """Quality and VRAM presets, as explicit setting deltas.

        Deltas rather than opaque modes, so the editor can show exactly what a
        preset changed and the user can undo any part of it.
        """
        from ..presets import all_presets

        return ok({"presets": all_presets()})

    @routes.get("/ltxdirector/vocabulary")
    async def get_vocabulary(request: web.Request) -> web.Response:
        """Camera moves, shot sizes, angles — the Director-mode dropdowns.

        Served rather than duplicated in JavaScript so the labels the editor
        shows and the prose the compiler emits cannot drift apart.
        """
        from ..core.prompt import (
            CAMERA_ANGLES,
            CAMERA_HEIGHTS,
            CAMERA_INTENSITY,
            CAMERA_MOVES,
            DEPTH_OF_FIELD,
            SHOT_SIZES,
        )

        return ok({
            "camera_moves": CAMERA_MOVES,
            "camera_intensity": CAMERA_INTENSITY,
            "shot_sizes": SHOT_SIZES,
            "camera_angles": CAMERA_ANGLES,
            "camera_heights": CAMERA_HEIGHTS,
            "depth_of_field": DEPTH_OF_FIELD,
        })


def _suggested_defaults(caps) -> dict[str, str | None]:
    """A first guess at model selection, so a new project is runnable.

    Only ever a suggestion: every value is a user-editable field, and a wrong
    guess costs one dropdown change rather than a failed render.
    """
    probe = caps.probe

    def first(values: list[str]) -> str | None:
        return values[0] if values else None

    return {
        "family": caps.families[0] if caps.families else None,
        "unet": first(probe.transformers()),
        "vae": first(probe.video_vaes()),
        "audio_vae": first(probe.audio_vaes()),
        "clip": first(probe.encoders()),
        "enhancer_clip": first(probe.enhancers()),
        "upscaler": first(probe.upscalers()),
    }

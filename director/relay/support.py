"""Whether a model's cross-attention can carry the Prompt Relay mask.

Separate from :mod:`director.relay.patch` because *asking* the question must not
require ComfyUI. The graceful-degradation path — "this model does not support
per-region prompts, so they were merged" — has to work in a plain Python process
and in the test suite, and it would not if the answer lived next to
``import comfy``.

Probed by attribute rather than by class name, so a subclassed or renamed model
still resolves and an unfamiliar one degrades instead of raising.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ModelSupport", "inspect_model", "detect_model_type"]


class ModelSupport:
    """Whether Prompt Relay can attach, and what geometry it needs if so.

    ``ok`` being False is a normal outcome, not an error: the caller encodes one
    prompt and carries on. ``reason`` is written for the user to read.
    """

    __slots__ = ("ok", "arch", "patch_size", "temporal_stride", "reason")

    def __init__(
        self,
        ok: bool,
        arch: str = "",
        patch_size: tuple[int, int, int] = (1, 1, 1),
        temporal_stride: int = 8,
        reason: str = "",
    ) -> None:
        self.ok = ok
        self.arch = arch
        self.patch_size = patch_size
        self.temporal_stride = temporal_stride
        self.reason = reason

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ModelSupport {self.arch or 'unsupported'} ok={self.ok}>"


def inspect_model(model: Any) -> ModelSupport:
    """Classify a model for Prompt Relay."""
    try:
        diff_model = model.model.diffusion_model
    except AttributeError:
        return ModelSupport(
            False,
            reason="The connected model does not expose a diffusion model, so per-region "
                   "prompts cannot be applied.",
        )

    if hasattr(diff_model, "patchifier"):
        if not getattr(diff_model, "transformer_blocks", None):
            return ModelSupport(
                False,
                reason="This LTX build exposes no transformer blocks for per-region prompts "
                       "to attach to.",
            )
        # The VAE's temporal ratio lives on the model. Reading it beats assuming
        # 8 — assuming is what broke the previous Director when the VAE changed.
        factors = getattr(diff_model, "vae_scale_factors", None)
        temporal = int(factors[0]) if factors else 8
        return ModelSupport(True, "ltx", (1, 1, 1), temporal)

    if hasattr(diff_model, "patch_size") and getattr(diff_model, "blocks", None):
        return ModelSupport(True, "wan", tuple(diff_model.patch_size), 4)

    return ModelSupport(
        False,
        reason=f"Per-region prompts are not supported for this model type "
               f"({type(diff_model).__name__}). LTX and Wan models are.",
    )


def detect_model_type(model: Any) -> tuple[str, tuple[int, int, int], int]:
    """``(arch, patch_size, temporal_stride)``, or raise.

    Kept because the Director 2.x node still calls it. New code should use
    :func:`inspect_model`, which reports rather than raises.
    """
    support = inspect_model(model)
    if not support.ok:
        raise ValueError(support.reason)
    return support.arch, support.patch_size, support.temporal_stride

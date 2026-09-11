"""The Prompt Relay node.

``MODEL + CLIP + DIRECTOR_SPEC → MODEL + CONDITIONING``. That is the entire
surface. In Director 2.x this was one behaviour of a node with twenty-six inputs
and eight outputs, which is why it was impossible to reason about.

Everything it can fail at, it fails softly: an unrecognised model, attention
another node has already claimed, a text encoder whose tokenizer it cannot read.
Each of those costs the user their per-region prompts and nothing else — the
regions are merged into one prompt, a warning is logged, and the shot renders.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

from ..relay import apply_relay
from .project import DirectorSpec
from .spec_input import as_spec

log = logging.getLogger(__name__)

__all__ = ["LTXDirectorRelay"]


class LTXDirectorRelay(io.ComfyNode):
    """Encode the shot's prompt, relaying per-region prompts over time."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXDirectorRelay",
            display_name="LTX Director Prompt Relay",
            category="LTX Director",
            description=(
                "Encodes the Director's prompt. When the timeline has more than one "
                "prompt region, it also biases cross-attention so each stretch of the "
                "video attends to its own text — in a single sampling pass, not one "
                "pass per region."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                DirectorSpec.Input("director"),
                io.Latent.Input(
                    "latent",
                    optional=True,
                    tooltip=(
                        "Optional. Connect the latent being sampled for exact token "
                        "geometry; without it the geometry is derived from the shot's "
                        "duration and resolution, which is correct for any graph the "
                        "Director compiled."
                    ),
                ),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Off encodes the regions as one prompt and leaves the model "
                        "untouched — useful for A/B comparison."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(
                    display_name="status",
                    tooltip="What the relay did, or why it could not.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        director,
        latent=None,
        enabled: bool = True,
    ) -> io.NodeOutput:
        spec = as_spec(director)

        if not enabled:
            spec = spec.copy()
            spec.relay.enabled = False

        result = apply_relay(model, clip, spec, latent=latent)

        for diagnostic in result.report:
            log.warning("[LTX Director] %s %s", diagnostic.message, diagnostic.fix)

        if result.applied:
            status = f"Relaying {result.segments} prompt regions."
        elif result.report.diagnostics:
            status = result.report.diagnostics[0].message
        elif not spec.relay.enabled:
            status = "Prompt Relay is switched off; one prompt was encoded."
        else:
            status = "One prompt region, so no relay was needed."

        return io.NodeOutput(result.model, result.conditioning, status)

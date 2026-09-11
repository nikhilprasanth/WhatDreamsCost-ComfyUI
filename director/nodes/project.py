"""The Director node.

One node, one string widget. Everything the editor produces is a Director Spec
serialised into ``project``; everything downstream reads that. The node itself
touches no tensors — that is the whole architecture in one sentence.

Its outputs are the handful of derived values a graph actually wants wired
(frames, fps, dimensions, seed, the compiled prompt) plus the spec itself, so
the Director can sit in a hand-built graph without anyone having to re-enter
numbers it already knows.
"""

from __future__ import annotations

import logging
from typing import Any

from comfy_api.latest import io

from ..capabilities import capabilities
from ..core.prompt import compile_prompt
from ..core.spec import Spec
from ..core.validate import validate

log = logging.getLogger(__name__)

__all__ = ["DirectorSpec", "LTXDirectorProject"]

#: The socket the Director's own nodes pass specs over. It carries a small JSON
#: document — never a tensor, never a blob.
DirectorSpec = io.Custom("DIRECTOR_SPEC")


class LTXDirectorProject(io.ComfyNode):
    """Describe a shot: timeline, references, prompt, audio, generation settings."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXDirectorProject",
            display_name="LTX Director",
            category="LTX Director",
            description=(
                "A filmmaking front end for LTX. Lay out a shot on a timeline, then "
                "either compile it to a native LTX graph or wire these outputs into "
                "one you have built yourself."
            ),
            inputs=[
                io.String.Input(
                    "project",
                    default="",
                    tooltip=(
                        "The Director project, as JSON. Managed by the editor — there is "
                        "no need to edit it by hand, but nothing stops you."
                    ),
                ),
                io.String.Input(
                    "prompt_override",
                    multiline=True,
                    default="",
                    optional=True,
                    force_input=True,
                    tooltip=(
                        "Optional. Connect a text node to drive the prompt from elsewhere "
                        "in the graph; it replaces whatever the editor holds."
                    ),
                ),
                io.Int.Input(
                    "seed_override",
                    default=-1,
                    min=-1,
                    max=0xFFFFFFFFFFFFFFFF,
                    optional=True,
                    tooltip="Optional. -1 uses the seed from the project.",
                ),
            ],
            outputs=[
                DirectorSpec.Output(display_name="director"),
                io.String.Output(display_name="prompt", tooltip="The compiled positive prompt."),
                io.String.Output(display_name="negative"),
                io.Int.Output(display_name="frames"),
                io.Float.Output(display_name="fps"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="seed"),
            ],
        )

    @classmethod
    def execute(
        cls,
        project: str = "",
        prompt_override: str = "",
        seed_override: int = -1,
    ) -> io.NodeOutput:
        spec = Spec.from_json(project)

        if prompt_override and prompt_override.strip():
            # An upstream text node wins, and switches to the mode where what is
            # typed is what is used — no sections silently appended.
            spec.prompt.mode = "expert"
            spec.prompt.raw = prompt_override
        if seed_override >= 0:
            spec.generation.seed = seed_override

        spec.normalise()

        report = validate(spec, _capabilities())
        for diagnostic in report:
            if diagnostic.level == "error":
                log.error("[LTX Director] %s %s", diagnostic.message, diagnostic.fix)
            elif diagnostic.level == "warning":
                log.warning("[LTX Director] %s %s", diagnostic.message, diagnostic.fix)
        report.raise_if_failed("build this shot")

        return io.NodeOutput(
            spec,
            compile_prompt(spec),
            spec.prompt.negative,
            spec.project.frames,
            float(spec.project.fps),
            spec.project.width,
            spec.project.height,
            int(spec.generation.seed),
        )

    @classmethod
    def fingerprint_inputs(cls, project: str = "", **_: Any) -> str:
        """Re-run only when the shot actually changes.

        Keyed on the settings digest rather than on the raw JSON, so moving the
        playhead or collapsing a panel does not invalidate the node — those are
        excluded from the digest by design.
        """
        try:
            return Spec.from_json(project).digest()
        except Exception:
            return project

    # ComfyUI has used both names for this hook; provide the older one too so
    # the node caches correctly on either.
    IS_CHANGED = fingerprint_inputs


def _capabilities() -> Any:
    """Capabilities, or None when they cannot be determined.

    None means the validator skips capability rules rather than assuming the
    worst — a probe that could not run must not block a render.
    """
    try:
        caps = capabilities()
    except Exception:  # pragma: no cover - defensive
        return None
    return caps if caps.probe.available else None

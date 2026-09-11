"""The Compile node.

Turns a shot into a native LTX workflow on disk. Running it produces a ``.json``
you can drag onto the canvas — the Director's "show me the real graph" escape
hatch, available from inside a graph rather than only from the editor's button.

Nothing here executes the compiled graph. Queueing is the editor's job (it posts
the API form to ``/prompt``), and keeping the two separate means this node can
be used purely to inspect what the Director would build.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from comfy_api.latest import io

from ..capabilities import capabilities
from ..compiler import compile_spec
from ..core.spec import Spec
from .project import DirectorSpec
from .spec_input import as_spec

log = logging.getLogger(__name__)

__all__ = ["LTXDirectorCompile"]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class LTXDirectorCompile(io.ComfyNode):
    """Compile the shot into a native ComfyUI workflow file."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXDirectorCompile",
            display_name="LTX Director Compile Workflow",
            category="LTX Director",
            description=(
                "Writes the shot out as a native LTX graph — loaders, conditioning, "
                "sampler, decode, save — with no Director node in it. Open the file to "
                "see and edit exactly what the Director would run."
            ),
            inputs=[
                DirectorSpec.Input("director"),
                io.Combo.Input(
                    "layout",
                    options=["flat", "subgraphs"],
                    default="flat",
                    tooltip=(
                        "Flat lays every node out with labelled group boxes. Subgraphs "
                        "packs each stage into one collapsible box, like the official "
                        "LTX templates."
                    ),
                ),
                io.String.Input(
                    "filename",
                    default="ltx_director_shot",
                    tooltip="Written to ComfyUI's output folder, under ltx_director/.",
                ),
                io.Boolean.Input(
                    "write_file",
                    default=True,
                    optional=True,
                    tooltip="Off compiles and reports without touching the disk.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="path"),
                io.String.Output(display_name="summary"),
                io.String.Output(display_name="workflow_json"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        director,
        layout: str = "flat",
        filename: str = "ltx_director_shot",
        write_file: bool = True,
    ) -> io.NodeOutput:
        spec = as_spec(director)
        result = compile_spec(spec, caps=_caps(), layout=layout)  # type: ignore[arg-type]

        for diagnostic in result.report:
            log.warning("[LTX Director] %s %s", diagnostic.message, diagnostic.fix)
        result.report.raise_if_failed("compile this shot")

        payload = json.dumps(result.workflow, indent=1, ensure_ascii=False)
        path = _write(payload, filename) if write_file else ""

        summary = _summarise(spec, result, path)
        log.info("[LTX Director] %s", summary.replace("\n", " "))
        return io.NodeOutput(path, summary, payload)

    @classmethod
    def fingerprint_inputs(cls, director=None, layout: str = "flat", **_: Any) -> str:
        try:
            return f"{as_spec(director).digest()}:{layout}"
        except Exception:
            return "unknown"

    IS_CHANGED = fingerprint_inputs


def _write(payload: str, filename: str) -> str:
    """Write the workflow under ``output/ltx_director/``, without clobbering.

    ComfyUI's output folder is the one place a custom node may write without
    surprising anyone, and a counter suffix means compiling twice leaves both
    versions to compare.
    """
    import folder_paths  # type: ignore

    directory = os.path.join(folder_paths.get_output_directory(), "ltx_director")
    os.makedirs(directory, exist_ok=True)

    stem = _SAFE_NAME.sub("_", filename).strip("._") or "ltx_director_shot"
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]

    path = os.path.join(directory, f"{stem}.json")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}_{counter:03d}.json")
        counter += 1

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return path


def _summarise(spec: Spec, result, path: str) -> str:
    """A few lines a person can read, not a dump of the graph."""
    nodes = len(result.workflow.get("nodes", []))
    inner = sum(
        len(sub.get("nodes", []))
        for sub in result.workflow.get("definitions", {}).get("subgraphs", [])
    )
    total = nodes + inner
    project = spec.project

    lines = [
        f"{_MODE_LABELS.get(project.mode, project.mode)} · "
        f"{project.width}×{project.height} · {project.frames} frames "
        f"({project.duration:.2f}s at {project.fps:g} fps)",
        f"{total} nodes, {spec.generation.stages} stage"
        f"{'s' if spec.generation.stages != 1 else ''}, seed {spec.generation.seed}",
    ]
    if path:
        lines.append(f"Written to {path}")
    warnings = [d for d in result.report if d.level == "warning"]
    for diagnostic in warnings:
        lines.append(f"Note: {diagnostic.message}")
    return "\n".join(lines)


_MODE_LABELS = {
    "t2v": "Text to video",
    "i2v": "Image to video",
    "fflf": "First and last frame",
    "keyframes": "Keyframes",
    "continue": "Continue shot",
    "a2v": "Audio to video",
    "t2a": "Text to audio",
    "iclora": "IC-LoRA control",
}


def _caps():
    try:
        caps = capabilities()
    except Exception:  # pragma: no cover - defensive
        return None
    return caps if caps.probe.available else None

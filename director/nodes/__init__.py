"""The Director's ComfyUI node surface.

Three nodes, deliberately:

``LTXDirectorProject``
    Describes the shot. Holds one string widget and touches no tensors.
``LTXDirectorRelay``
    Encodes the prompt, relaying per-region prompts when there is more than one.
``LTXDirectorCompile``
    Writes the shot out as a native LTX graph.

Director 2.x's ``LTXDirector``, ``LTXDirectorGuide`` and ``LTXDirectorCropGuides``
stay registered from the repository root so existing workflows keep loading;
they are not re-exported here.
"""

from __future__ import annotations

from .compile import LTXDirectorCompile
from .project import DirectorSpec, LTXDirectorProject
from .relay import LTXDirectorRelay

#: Everything the Director Next contributes, in menu order.
NODE_CLASSES = (
    LTXDirectorProject,
    LTXDirectorRelay,
    LTXDirectorCompile,
)

NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in NODE_CLASSES}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXDirectorProject": "LTX Director",
    "LTXDirectorRelay": "LTX Director Prompt Relay",
    "LTXDirectorCompile": "LTX Director Compile Workflow",
}

__all__ = [
    "NODE_CLASSES",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "DirectorSpec",
    "LTXDirectorCompile",
    "LTXDirectorProject",
    "LTXDirectorRelay",
]

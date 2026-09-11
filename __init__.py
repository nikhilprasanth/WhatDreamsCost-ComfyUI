"""WhatDreamsCost-ComfyUI — LTX Director and friends.

Two generations of nodes live here side by side, deliberately.

**Director Next** (``director/``) is the current one: a Director node that holds
a project description and nothing else, a Prompt Relay node, and a compiler that
turns a shot into a native LTX graph. See ``docs/ARCHITECTURE.md``.

**Director 2.x** (``ltx_director.py``, ``ltx_director_guide.py``) stays
registered so existing workflows keep loading. It is not the path new work
should take, and the editor offers to import its timelines.

Licensed GPL-3.0.
"""

from __future__ import annotations

import logging

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# -- Director Next ----------------------------------------------------------
from .director import __version__ as DIRECTOR_VERSION
from .director.nodes import (
    NODE_CLASS_MAPPINGS as DIRECTOR_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as DIRECTOR_NAMES,
    NODE_CLASSES as DIRECTOR_NODE_CLASSES,
)

# -- Director 2.x and the standalone utility nodes --------------------------
from .load_audio_ui import LoadAudioUI
from .load_video_ui import LoadVideoUI
from .ltx_director import LTXDirector
from .ltx_director_guide import LTXDirectorCropGuides, LTXDirectorGuide
from .ltx_keyframer import LTXKeyframer
from .ltx_sequencer import LTXSequencer
from .multi_image_loader import MultiImageLoader
from .speech_length_calculator import SpeechLengthCalculator

log = logging.getLogger(__name__)


class WhatDreamsCostExtension(ComfyExtension):
    """V3 node registration."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [*DIRECTOR_NODE_CLASSES, LTXDirector, LTXDirectorGuide]


async def comfy_entrypoint() -> WhatDreamsCostExtension:
    return WhatDreamsCostExtension()


#: The static map ComfyUI-Manager reads. Kept in step with the V3 registration
#: above by construction rather than by hand.
NODE_CLASS_MAPPINGS = {
    **DIRECTOR_NODES,
    "LTXKeyframer": LTXKeyframer,
    "MultiImageLoader": MultiImageLoader,
    "LTXSequencer": LTXSequencer,
    "SpeechLengthCalculator": SpeechLengthCalculator,
    "LoadAudioUI": LoadAudioUI,
    "LoadVideoUI": LoadVideoUI,
    "LTXDirector": LTXDirector,
    "LTXDirectorGuide": LTXDirectorGuide,
    "LTXDirectorCropGuides": LTXDirectorCropGuides,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **DIRECTOR_NAMES,
    "LTXKeyframer": "LTX Keyframer",
    "MultiImageLoader": "Multi Image Loader",
    "LTXSequencer": "LTX Sequencer",
    "SpeechLengthCalculator": "Speech Length Calculator",
    "LoadAudioUI": "Load Audio UI",
    "LoadVideoUI": "Load Video UI",
    "LTXDirector": "LTX Director 2 (legacy)",
    "LTXDirectorGuide": "LTX Director 2 Guide (legacy)",
    "LTXDirectorCropGuides": "LTX Director 2 Crop Guides (legacy)",
}

WEB_DIRECTORY = "./js"

# Routes are registered here, once, rather than as a side effect of importing a
# node module. A missing server is the normal state under test and is not an
# error.
try:
    from .director.api import register_routes

    register_routes()
except Exception as exc:  # pragma: no cover - never block node loading
    log.warning("[LTX Director] Routes could not be registered: %s", exc)

log.info("[LTX Director] Director Next %s loaded.", DIRECTOR_VERSION)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

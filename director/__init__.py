"""LTX Director Next — a filmmaking-oriented orchestration layer for LTX in ComfyUI.

The Director describes intent; native ComfyUI and LTX nodes perform the work.

Layout
------
``core``          project spec, time maths, validation, serialisation (no torch, no comfy)
``capabilities``  what is installed, expressed as feature flags with reasons
``compiler``      pure ``Spec -> ComfyUI workflow JSON`` translation, one adapter per LTX family
``relay``         Prompt Relay: per-segment conditioning in a single sampler pass
``nodes``         the thin ComfyUI node surface
``api``           aiohttp routes backing the frontend
``migrations``    Director 2.x import

See ``docs/ARCHITECTURE.md``.

This package is part of WhatDreamsCost-ComfyUI and is licensed GPL-3.0.
"""

__version__ = "3.0.0"

#: Director Spec schema version. Bump when the on-disk shape changes and add a
#: migration step in :mod:`director.core.migrate`.
SPEC_SCHEMA = 3

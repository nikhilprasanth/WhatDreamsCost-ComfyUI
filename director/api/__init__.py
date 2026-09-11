"""The Director's HTTP routes.

Registered from one place, on import of the package, rather than as a side
effect of importing a node module. That matters: routes appearing because
something else happened to be imported is how the previous Director ended up
installing a process-wide asyncio exception handler at import time.

Routes, all under ``/ltxdirector``:

===========================  ==========================================
``GET  /capabilities``       what is installed, as feature flags
``POST /validate``           a shot's diagnostics
``POST /compile``            a shot as a native graph (both forms)
``POST /queue``              compile and run, so Generate needs no canvas
``POST /media/check``        is this file already here (by hash)
``POST /media/upload``       chunked upload into the Director's workspace
``GET  /media/probe``        dimensions, duration, frame rate
``GET  /media/thumb``        cached thumbnail
``GET  /media/peaks``        cached waveform envelope
``GET  /media/list``         what is in the workspace
``POST /project/save``       write a ``.ltxdirector.json``
``GET  /project/load``       read one back
``GET  /project/list``       what has been saved
``POST /project/import``     convert a Director 2.x timeline
===========================  ==========================================
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

__all__ = ["register_routes"]

_REGISTERED = False


def register_routes() -> bool:
    """Attach every Director route. Safe to call more than once.

    Returns False when ComfyUI's server is not available — which is the normal
    state in a test run, and must not be an error.
    """
    global _REGISTERED
    if _REGISTERED:
        return True

    try:
        from server import PromptServer  # type: ignore
    except Exception:
        log.debug("[LTX Director] No ComfyUI server; routes not registered.")
        return False

    instance = getattr(PromptServer, "instance", None)
    if instance is None or not hasattr(instance, "routes"):
        log.debug("[LTX Director] ComfyUI server has no router yet.")
        return False

    from . import caps, media, project

    routes = instance.routes
    caps.register(routes)
    media.register(routes)
    project.register(routes)

    _REGISTERED = True
    log.info("[LTX Director] Routes registered under /ltxdirector.")
    return True

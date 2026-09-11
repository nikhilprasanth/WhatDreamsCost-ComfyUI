"""Shared helpers for the Director's HTTP routes.

Two things worth stating up front, because the rest of the API depends on them:

* **Every path the Director writes is confined to ComfyUI's own input and output
  folders.** :func:`safe_join` refuses anything that escapes, so a filename from
  the browser can never reach outside them.
* **Errors come back as sentences.** A route that fails returns the same
  ``{message, fix}`` shape the validator produces, so the editor has one way to
  display a problem regardless of where it came from.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from aiohttp import web

log = logging.getLogger(__name__)

__all__ = [
    "WORKSPACE",
    "ok",
    "fail",
    "input_directory",
    "output_directory",
    "workspace_directory",
    "safe_join",
    "read_json",
    "spec_from_body",
]

#: Subfolder of ComfyUI's input directory that the Director owns. Keeping the
#: Director's uploads out of the top level means a user's own input folder does
#: not fill up with timeline media.
WORKSPACE = "ltxdirector"


# --------------------------------------------------------------------------
# responses
# --------------------------------------------------------------------------

def ok(payload: Any = None, **extra: Any) -> web.Response:
    body: dict[str, Any] = {"ok": True}
    if isinstance(payload, dict):
        body.update(payload)
    elif payload is not None:
        body["result"] = payload
    body.update(extra)
    return web.json_response(body)


def fail(message: str, fix: str = "", *, status: int = 400, code: str = "") -> web.Response:
    """An error the editor can show verbatim.

    Same shape as a validator diagnostic, so the UI has one renderer.
    """
    log.warning("[LTX Director] %s %s", message, fix)
    return web.json_response(
        {"ok": False, "error": {"code": code, "message": message, "fix": fix}},
        status=status,
    )


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def input_directory() -> str:
    import folder_paths  # type: ignore

    return folder_paths.get_input_directory()


def output_directory() -> str:
    import folder_paths  # type: ignore

    return folder_paths.get_output_directory()


def workspace_directory(*parts: str, create: bool = True) -> str:
    """A directory under ``input/ltxdirector/``."""
    path = os.path.join(input_directory(), WORKSPACE, *parts)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def safe_join(root: str, *parts: str) -> str:
    """Join under ``root``, refusing anything that escapes it.

    The filename comes from the browser, so this is the boundary. Symlinks are
    resolved before the check, which is what stops a link inside the input
    folder being used to reach outside it.
    """
    candidate = os.path.realpath(os.path.join(root, *parts))
    base = os.path.realpath(root)
    if candidate != base and not candidate.startswith(base + os.sep):
        raise ValueError(
            f"Refusing a path outside {os.path.basename(base)}: {os.path.join(*parts)!r}."
        )
    return candidate


# --------------------------------------------------------------------------
# request bodies
# --------------------------------------------------------------------------

async def read_json(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        raise ValueError("The request body was not valid JSON.")
    if not isinstance(body, dict):
        raise ValueError("The request body must be a JSON object.")
    return body


def spec_from_body(body: dict[str, Any]):
    """Pull a Spec out of a request body.

    Accepts ``{"project": {...}}``, ``{"project": "<json>"}`` or the spec at the
    top level, because all three turn up: the editor sends one, a saved file is
    another, and someone testing with curl will send the third.
    """
    from ..core.spec import Spec

    payload = body.get("project", body)
    if isinstance(payload, str):
        return Spec.from_json(payload)
    if isinstance(payload, dict):
        return Spec.from_dict(payload)
    raise ValueError("No Director project was included in the request.")


def json_text(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, indent=indent, ensure_ascii=False)

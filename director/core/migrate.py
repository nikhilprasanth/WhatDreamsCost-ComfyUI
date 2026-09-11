"""Schema migrations for the Director Spec.

One ladder, one direction. Each step takes the payload at version *n* and
returns it at *n+1*; :func:`migrate` walks the ladder until the payload is
current. A payload that carries no ``schema`` key is either a Director 2.x
timeline blob — recognised by shape and handed to
:mod:`director.migrations.v2_timeline` — or an empty document.

Adding a schema version means bumping ``SPEC_SCHEMA`` in ``director/__init__``
and appending one function here. Nothing else in the codebase should branch on
a schema number.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

from .. import SPEC_SCHEMA

log = logging.getLogger(__name__)

__all__ = ["migrate", "looks_like_v2_timeline", "MigrationError"]


class MigrationError(ValueError):
    """Raised when a payload cannot be brought forward to the current schema."""


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

#: Keys that only ever appeared in a Director 2.x ``timeline_data`` blob.
_V2_MARKERS = ("motionSegments", "audioSegments", "retakeMode", "mainTrackEnabled")


def looks_like_v2_timeline(data: dict[str, Any]) -> bool:
    """True when ``data`` is a Director 2.x timeline rather than a Spec.

    Checked by marker keys rather than by absence of ``schema``, so a truncated
    or hand-edited Spec is not mistaken for a v2 blob.
    """
    if "schema" in data:
        return False
    return any(key in data for key in _V2_MARKERS)


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------

def _v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Schema 1 → 2: references gained an explicit anchor.

    Schema 1 encoded a last-frame anchor as ``is_end_frame``; schema 2 replaced
    that with the three-way ``anchor`` so a start anchor is expressible too.
    """
    for ref in data.get("references") or []:
        if "anchor" in ref:
            continue
        ref["anchor"] = "end" if ref.pop("is_end_frame", False) else "index"
    return data


def _v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
    """Schema 2 → 3: time positions became ``TimePoint`` objects.

    Schema 2 stored a bare ``at`` integer frame. Schema 3 stores
    ``{"frame": n, "unit": …}`` so the display preference travels with the
    position instead of living in a separate global setting.
    """
    default_unit = (data.get("ui") or {}).get("display_unit", "seconds")
    for ref in data.get("references") or []:
        at = ref.get("at")
        if isinstance(at, (int, float)):
            ref["at"] = {"frame": int(at), "unit": default_unit}
    return data


#: ``version -> upgrade`` for every step from the oldest supported schema.
_STEPS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    1: _v1_to_v2,
    2: _v2_to_v3,
}

#: Oldest schema we can still read. Below this, fail with a clear message
#: rather than guessing.
OLDEST_SUPPORTED = 1


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Bring ``data`` up to :data:`director.SPEC_SCHEMA`.

    Returns a new payload and never touches the caller's — the UI hands over a
    document it is still holding, and a step that rewrites nested reference
    dicts in place would corrupt it.

    Unknown *newer* schemas are passed through with a warning: a user who opened
    a project saved by a newer Director should see their prompts, not an error
    page — the codec will ignore fields it does not know.
    """
    if not isinstance(data, dict):
        raise MigrationError(
            f"Expected a Director project object, got {type(data).__name__}."
        )

    if looks_like_v2_timeline(data):
        from ..migrations.v2_timeline import convert  # local: avoids a cycle

        log.info("[Director] Importing a Director 2.x timeline.")
        return convert(data).to_dict()

    payload = copy.deepcopy(data)
    version = payload.get("schema")

    if version is None:
        # No markers, no schema: an empty or freshly hand-written document.
        payload["schema"] = SPEC_SCHEMA
        return payload

    if not isinstance(version, int):
        raise MigrationError(
            f"Project 'schema' must be an integer version, got {version!r}."
        )

    if version < OLDEST_SUPPORTED:
        raise MigrationError(
            f"This project was saved with Director schema {version}, which is older than "
            f"the oldest readable version ({OLDEST_SUPPORTED}). Open it in the Director "
            f"release that wrote it and re-save."
        )

    if version > SPEC_SCHEMA:
        log.warning(
            "[Director] Project uses schema %d but this build understands %d. "
            "Unknown fields will be dropped on save.",
            version, SPEC_SCHEMA,
        )
        return payload

    while version < SPEC_SCHEMA:
        step = _STEPS.get(version)
        if step is None:
            raise MigrationError(
                f"No migration registered from Director schema {version} to {version + 1}."
            )
        payload = step(payload)
        version += 1
        payload["schema"] = version

    return payload

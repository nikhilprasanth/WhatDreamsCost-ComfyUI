"""Compatibility shim — model patching moved to ``director.relay.patch``.

One behavioural note for anything still importing from here: ``apply_patches``
now **returns** a reason string instead of raising when it cannot attach, so a
model it does not recognise costs the caller its per-region prompts rather than
its render. Callers that ignored the return value are unaffected.

New code should import from ``director.relay.patch``.
"""

from __future__ import annotations

from .director.relay.patch import (
    ModelSupport,
    apply_patches,
    detect_model_type,
    get_current_node_id,
    inspect_model,
)

__all__ = [
    "ModelSupport",
    "apply_patches",
    "detect_model_type",
    "get_current_node_id",
    "inspect_model",
]

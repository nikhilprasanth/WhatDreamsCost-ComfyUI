"""Accepting a project on a node input.

The ``DIRECTOR_SPEC`` socket normally carries a :class:`~director.core.spec.Spec`,
but a workflow saved before a rename, a project pasted in by hand, or a graph
built by a script will hand over a dict or a JSON string instead. All three are
the same document, so all three are accepted — and anything else gets a message
saying which output to connect rather than a type error nobody can act on.
"""

from __future__ import annotations

from typing import Any

from ..core.spec import Spec

__all__ = ["as_spec"]


def as_spec(value: Any) -> Spec:
    """Coerce a node input into a Spec."""
    if isinstance(value, Spec):
        return value
    if isinstance(value, dict):
        return Spec.from_dict(value)
    if isinstance(value, str):
        return Spec.from_json(value)
    raise ValueError(
        "The Director input did not carry a project. Connect the LTX Director node's "
        "'director' output to this node."
    )

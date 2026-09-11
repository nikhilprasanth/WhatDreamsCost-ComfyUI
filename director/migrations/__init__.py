"""Importers for earlier Director formats.

Each module converts one legacy shape into a current :class:`director.core.spec.Spec`
and reports what it could not carry across. Conversions are lossy by design —
see the module docstrings for exactly what is dropped and why.
"""

from __future__ import annotations

from .v2_timeline import convert, convert_with_report

__all__ = ["convert", "convert_with_report"]

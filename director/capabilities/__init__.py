"""What this installation can do, expressed as feature flags with reasons.

Everything here is importable with no ComfyUI present: an absent ComfyUI yields
an "unknown" probe rather than an exception, which is what keeps the compiler
and the test suite runnable on a machine with no models installed.
"""

from __future__ import annotations

from .features import FEATURES, Capabilities, Feature, capabilities
from .probe import ModelSet, Probe, family_of, invalidate, probe

__all__ = [
    "FEATURES",
    "Capabilities",
    "Feature",
    "ModelSet",
    "Probe",
    "capabilities",
    "family_of",
    "invalidate",
    "probe",
]

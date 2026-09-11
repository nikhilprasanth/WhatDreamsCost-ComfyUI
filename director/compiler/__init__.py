"""Director → native ComfyUI workflow translation.

The compiler is a pure function. It imports no ``torch``, no ``comfy`` and
nothing from ComfyUI, so the whole of it is testable on a machine with no models
installed — which is what keeps the golden-file tests honest.

Supporting a new LTX release means adding one adapter module next to
:mod:`director.compiler.ltx25` and registering it. Nothing else changes.
"""

from __future__ import annotations

from .base import (
    CompileResult,
    Compiler,
    available_families,
    compile_spec,
    compiler_for,
    register,
)
from .graph import GraphBuilder, NodeHandle, Port, Section
from .ltx25 import LTX25Compiler
from .signatures import NODE_SIGNATURES, NodeSignature, signature_for

__all__ = [
    "NODE_SIGNATURES",
    "CompileResult",
    "Compiler",
    "GraphBuilder",
    "LTX25Compiler",
    "NodeHandle",
    "NodeSignature",
    "Port",
    "Section",
    "available_families",
    "compile_spec",
    "compiler_for",
    "register",
    "signature_for",
]

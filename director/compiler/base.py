"""The compiler contract and the family registry.

A compiler is a **pure function** ``Spec -> workflow JSON``. No filesystem, no
``torch``, no ComfyUI import. That is what makes it testable on a laptop with no
models installed, and what makes a golden-file diff mean something.

Supporting a new LTX release is a new module here plus one ``register`` call.
Nothing else in the codebase branches on a model family.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.validate import Diagnostic, Report
from .graph import GraphBuilder, Layout

if TYPE_CHECKING:  # pragma: no cover
    from ..core.spec import Spec

__all__ = [
    "CompileResult",
    "Compiler",
    "register",
    "compiler_for",
    "available_families",
    "compile_spec",
]


@dataclass
class CompileResult:
    """A compiled workflow, plus everything worth saying about it."""

    workflow: dict[str, Any]
    report: Report = field(default_factory=Report)
    #: Semantic name → node id, so callers can find the sampler or the save node
    #: without pattern-matching the graph.
    node_index: dict[str, int] = field(default_factory=dict)
    family: str = ""
    layout: Layout = "flat"

    @property
    def ok(self) -> bool:
        return self.report.ok

    @property
    def diagnostics(self) -> list[Diagnostic]:
        return self.report.diagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "family": self.family,
            "layout": self.layout,
            "node_index": self.node_index,
            **self.report.to_dict(),
        }


class Compiler(ABC):
    """Translates a Spec into a native ComfyUI graph for one LTX family."""

    #: The ``spec.models.family`` value this compiler answers to.
    family: str = ""

    #: Modes this compiler can build. Anything else is refused by name.
    modes: tuple[str, ...] = ()

    def supports(self, spec: "Spec") -> bool:
        return spec.models.family == self.family and spec.project.mode in self.modes

    @abstractmethod
    def compile(
        self,
        spec: "Spec",
        *,
        caps: Any = None,
        layout: Layout = "flat",
    ) -> CompileResult:
        """Build the workflow.

        Must be pure: the same spec, capabilities and layout produce
        byte-identical JSON.
        """

    # -- helpers for subclasses -------------------------------------------

    def _builder(self, spec: "Spec", layout: Layout) -> GraphBuilder:
        title = f"LTX Director — {spec.project.mode.upper()}"
        return GraphBuilder(layout=layout, title=title)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, Compiler] = {}


def register(compiler: Compiler) -> Compiler:
    """Register a compiler for its family. Later registrations replace earlier ones."""
    if not compiler.family:
        raise ValueError(f"{type(compiler).__name__} must declare a family.")
    _REGISTRY[compiler.family] = compiler
    return compiler


def available_families() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def compiler_for(family: str) -> Compiler | None:
    return _REGISTRY.get(family)


def compile_spec(
    spec: "Spec",
    *,
    caps: Any = None,
    layout: Layout = "flat",
) -> CompileResult:
    """Validate, then compile.

    Validation runs first and its diagnostics are carried through, so a caller
    gets one report covering both "this shot does not make sense" and "this shot
    could not be built". Errors stop compilation; warnings do not.
    """
    from ..core.validate import validate  # local: keeps import order simple

    report = validate(spec, caps)

    compiler = compiler_for(spec.models.family)
    if compiler is None:
        report.error(
            "compile.no_compiler",
            f"There is no compiler for model family {spec.models.family!r}.",
            f"Available: {', '.join(available_families()) or 'none'}.",
            "models.family",
        )
        return CompileResult(workflow={}, report=report, layout=layout)

    if spec.project.mode not in compiler.modes:
        report.error(
            "compile.mode_unsupported",
            f"The {spec.models.family} compiler cannot build "
            f"{spec.project.mode!r} shots yet.",
            f"Supported modes: {', '.join(compiler.modes)}.",
            "project.mode",
        )

    if not report.ok:
        return CompileResult(
            workflow={}, report=report, family=compiler.family, layout=layout
        )

    result = compiler.compile(spec, caps=caps, layout=layout)
    # Keep the validation diagnostics in front of anything the adapter adds.
    result.report.diagnostics[:0] = report.diagnostics
    result.family = compiler.family
    result.layout = layout
    return result

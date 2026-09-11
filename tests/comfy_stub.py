"""A minimal stand-in for the parts of ComfyUI the Director's nodes import.

Enough of ``comfy_api.latest`` to let node modules be imported and their schemas
built, so node wiring can be tested on a machine with no ComfyUI. Nothing here
executes a model — the nodes under test do not touch tensors, which is precisely
the property this makes checkable.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StubSlot:
    id: str = ""
    display_name: str = ""
    type: str = ""
    optional: bool = False
    default: Any = None
    tooltip: str = ""
    options: list | None = None
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.id or self.display_name


def _slot_factory(type_name: str):
    class _Slot:
        @staticmethod
        def Input(id: str = "", **kwargs: Any) -> StubSlot:
            return StubSlot(
                id=id,
                type=type_name,
                optional=bool(kwargs.pop("optional", False)),
                default=kwargs.pop("default", None),
                tooltip=str(kwargs.pop("tooltip", "")),
                options=kwargs.pop("options", None),
                display_name=str(kwargs.pop("display_name", "")),
                extra=kwargs,
            )

        @staticmethod
        def Output(display_name: str = "", **kwargs: Any) -> StubSlot:
            return StubSlot(
                id=display_name,
                display_name=display_name,
                type=type_name,
                tooltip=str(kwargs.pop("tooltip", "")),
                extra=kwargs,
            )

    _Slot.__name__ = type_name
    return _Slot


@dataclass
class StubSchema:
    node_id: str = ""
    display_name: str = ""
    category: str = ""
    description: str = ""
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    is_output_node: bool = False
    extra: dict = field(default_factory=dict)

    def input(self, name: str) -> StubSlot:
        for slot in self.inputs:
            if slot.name == name:
                return slot
        raise KeyError(name)

    @property
    def input_names(self) -> list[str]:
        return [s.name for s in self.inputs]

    @property
    def output_names(self) -> list[str]:
        return [s.display_name or s.type for s in self.outputs]


class StubNodeOutput:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter(self.args)

    def __getitem__(self, index: int) -> Any:
        return self.args[index]

    def __len__(self) -> int:
        return len(self.args)


class StubComfyNode:
    """Base class. Real ComfyUI gives this an executor; tests call execute directly."""


def install() -> None:
    """Install the stub modules, unless the real ComfyUI is already importable."""
    try:
        import comfy_api.latest  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    io_module = types.ModuleType("comfy_api.latest.io")
    for type_name in (
        "Model", "Clip", "Vae", "Latent", "Image", "Mask", "Audio", "Conditioning",
        "String", "Int", "Float", "Boolean", "Combo", "Custom", "MultiType",
        "DynamicCombo", "Video", "AnyType",
    ):
        setattr(io_module, type_name, _slot_factory(type_name))

    def custom(name: str):
        return _slot_factory(name)

    io_module.Custom = custom  # type: ignore[attr-defined]

    def schema(**kwargs: Any) -> StubSchema:
        known = {f for f in StubSchema.__dataclass_fields__ if f != "extra"}
        fields = {k: v for k, v in kwargs.items() if k in known}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        return StubSchema(**fields, extra=extra)

    io_module.Schema = schema  # type: ignore[attr-defined]
    io_module.NodeOutput = StubNodeOutput  # type: ignore[attr-defined]
    io_module.ComfyNode = StubComfyNode  # type: ignore[attr-defined]

    latest = types.ModuleType("comfy_api.latest")
    latest.io = io_module  # type: ignore[attr-defined]

    class StubExtension:
        async def get_node_list(self):  # pragma: no cover - never called in tests
            return []

    latest.ComfyExtension = StubExtension  # type: ignore[attr-defined]

    package = types.ModuleType("comfy_api")
    package.latest = latest  # type: ignore[attr-defined]

    sys.modules.setdefault("comfy_api", package)
    sys.modules["comfy_api.latest"] = latest
    sys.modules["comfy_api.latest.io"] = io_module

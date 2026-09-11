"""A minimal stand-in for the parts of ComfyUI the Director's nodes import.

:func:`install` provides enough of ``comfy_api.latest`` to import node modules
and build their schemas, so node wiring can be tested on a machine with no
ComfyUI. Nothing here executes a model — the nodes under test do not touch
tensors, which is precisely the property this makes checkable.

:func:`install_full` goes further and stubs the handful of ``comfy``,
``folder_paths``, ``server`` and ``node_helpers`` names the *legacy* Director 2.x
modules import, so the whole package can be loaded the way ComfyUI loads a custom
node. Only names that are actually imported are stubbed — ``torch`` stays real,
because a catch-all stub breaks its introspection.
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


# --------------------------------------------------------------------------
# a fuller stub, for loading the whole package
# --------------------------------------------------------------------------

class StubRoute:
    """One recorded registration, shaped like an aiohttp route definition."""

    __slots__ = ("method", "path")

    def __init__(self, method: str, path: str) -> None:
        self.method = method
        self.path = path


class StubRoutes:
    """Records route registrations instead of serving them.

    Iterable, like aiohttp's ``RouteTableDef``, so code that inspects the router
    to avoid registering twice behaves the same here as it does in ComfyUI.
    """

    def __init__(self) -> None:
        self.registered: list[tuple[str, str]] = []
        self._routes: list[StubRoute] = []

    def __iter__(self):
        return iter(self._routes)

    def __len__(self) -> int:
        return len(self._routes)

    def __getattr__(self, name: str):
        if name not in ("get", "post", "put", "delete", "patch"):
            raise AttributeError(name)

        def decorator(path: str):
            def wrap(fn):
                self.registered.append((name.upper(), path))
                self._routes.append(StubRoute(name.upper(), path))
                return fn
            return wrap

        return decorator


class StubPromptServer:
    instance: "StubPromptServer | None" = None

    def __init__(self) -> None:
        self.routes = StubRoutes()
        self.number = 0
        self.prompt_queue = types.SimpleNamespace(put=lambda item: None)


def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = sys.modules.get(name) or types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def install_full(*, input_dir: str | None = None, output_dir: str | None = None) -> StubPromptServer:
    """Stub everything the package imports at load time. Returns the fake server.

    A no-op for anything the real ComfyUI already provides.
    """
    import os

    install()
    noop = lambda *args, **kwargs: None  # noqa: E731

    root = input_dir or os.getcwd()
    _module(
        "folder_paths",
        get_filename_list=lambda folder: [],
        get_input_directory=lambda: root,
        get_output_directory=lambda: output_dir or root,
        get_full_path=lambda *a: None,
    )
    _module("node_helpers", conditioning_set_values=lambda cond, values: cond, open_image=noop)
    _module("nodes", NODE_CLASS_MAPPINGS={}, MAX_RESOLUTION=16384)

    comfy = _module("comfy")
    comfy.model_management = _module(
        "comfy.model_management",
        intermediate_device=lambda: "cpu",
        load_models_gpu=noop,
    )
    comfy.utils = _module("comfy.utils", common_upscale=noop, load_torch_file=noop)
    comfy.sd = _module("comfy.sd", load_lora_for_models=noop)
    _module("comfy.ldm")
    _module("comfy.ldm.modules")
    _module("comfy.ldm.modules.attention", attention_pytorch=noop, optimized_attention=noop)
    _module("comfy.samplers", CFGGuider=object)
    _module("comfy.ldm.lightricks")

    class _Patchifier:
        def __init__(self, *args, **kwargs) -> None:
            pass

    _module("comfy_extras")
    _module(
        "comfy_extras.nodes_lt",
        get_noise_mask=noop,
        LTXVAddGuide=object,
        SymmetricPatchifier=_Patchifier,
        conditioning_get_any_value=noop,
        get_keyframe_idxs=noop,
    )

    server = StubPromptServer()
    StubPromptServer.instance = server
    _module("server", PromptServer=StubPromptServer)
    return server

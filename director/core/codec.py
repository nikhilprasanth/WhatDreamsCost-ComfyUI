"""A small, strict dataclass ⇄ JSON codec.

The Director Spec is a deep tree of dataclasses that has to survive a round
trip through a node widget, a project file on disk and an HTTP body. Writing
``from_dict`` by hand for thirty classes is where drift creeps in, so the
conversion is derived from the type hints instead.

Deliberately narrow: it understands primitives, ``Optional``, ``list``,
``dict[str, …]``, ``Literal`` and nested dataclasses. Anything else raises at
decode time rather than producing a half-built object.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

__all__ = ["to_jsonable", "from_jsonable", "DecodeError"]

T = TypeVar("T")

_NoneType = type(None)

#: ``Optional[X]`` resolves to ``typing.Union`` while ``X | None`` resolves to
#: ``types.UnionType`` (PEP 604). Both appear in the spec, so both are matched.
_UNION_ORIGINS = (Union, types.UnionType)


class DecodeError(ValueError):
    """Raised when a payload cannot be decoded into the requested dataclass.

    Carries the dotted path to the offending value so the message points at a
    field rather than at the whole document.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        super().__init__(f"{path or '<root>'}: {message}")


# --------------------------------------------------------------------------
# encode
# --------------------------------------------------------------------------

def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, lists and dicts into JSON-safe primitives.

    Field order follows declaration order, which keeps ``json.dumps`` output
    stable and diffs small.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        # Collapse -0.0 and integral floats so round trips are byte-stable.
        return 0.0 if value == 0 else value
    return value


# --------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------

def from_jsonable(cls: type[T], data: Any, *, path: str = "") -> T:
    """Build ``cls`` from decoded JSON, filling defaults for missing fields."""
    return _decode(cls, data, path)


def _hints(cls: type) -> dict[str, Any]:
    # Resolved lazily and cached on the class; ``from __future__ import
    # annotations`` means the raw __annotations__ are strings.
    cached = cls.__dict__.get("__director_hints__")
    if cached is None:
        cached = get_type_hints(cls)
        setattr(cls, "__director_hints__", cached)
    return cached


def _decode(tp: Any, value: Any, path: str) -> Any:
    origin = get_origin(tp)

    # Optional[X] / Union[...] / X | None
    if origin in _UNION_ORIGINS:
        args = [a for a in get_args(tp) if a is not _NoneType]
        if value is None:
            return None
        if len(args) == 1:
            return _decode(args[0], value, path)
        for arg in args:  # first match wins; spec unions are disjoint by shape
            try:
                return _decode(arg, value, path)
            except (DecodeError, TypeError, ValueError):
                continue
        raise DecodeError(path, f"value {value!r} matches none of {args!r}")

    if origin is list:
        if value is None:
            return []
        if not isinstance(value, list):
            raise DecodeError(path, f"expected a list, got {type(value).__name__}")
        (item_tp,) = get_args(tp) or (Any,)
        return [_decode(item_tp, v, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin is dict:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise DecodeError(path, f"expected an object, got {type(value).__name__}")
        args = get_args(tp)
        val_tp = args[1] if len(args) == 2 else Any
        return {str(k): _decode(val_tp, v, f"{path}.{k}") for k, v in value.items()}

    if origin is typing.Literal:
        allowed = get_args(tp)
        if value in allowed:
            return value
        raise DecodeError(path, f"expected one of {list(allowed)}, got {value!r}")

    if dataclasses.is_dataclass(tp):
        if value is None:
            return tp()  # type: ignore[call-arg]
        if not isinstance(value, dict):
            raise DecodeError(path, f"expected an object, got {type(value).__name__}")
        hints = _hints(tp)
        kwargs: dict[str, Any] = {}
        for field in dataclasses.fields(tp):
            if field.name not in value:
                continue  # dataclass default applies
            kwargs[field.name] = _decode(
                hints[field.name], value[field.name], f"{path}.{field.name}" if path else field.name
            )
        try:
            return tp(**kwargs)  # type: ignore[call-arg]
        except TypeError as exc:  # a required field was genuinely absent
            raise DecodeError(path, str(exc)) from None

    if tp is Any:
        return value

    # primitives, with the coercions a JSON round trip actually needs
    if tp is bool:
        return bool(value)
    if tp is int:
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            raise DecodeError(path, f"expected an integer, got {value!r}") from None
    if tp is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise DecodeError(path, f"expected a number, got {value!r}") from None
    if tp is str:
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)

    raise DecodeError(path, f"unsupported field type {tp!r}")

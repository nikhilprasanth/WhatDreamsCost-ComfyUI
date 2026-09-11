"""Stable identifiers and content hashing.

Two rules make the rest of the system simple:

* Every object carries an id that never changes when it is moved, reordered or
  edited. Reordering a reference list must not invalidate a take's record of
  which references it used.
* Media is addressed by the hash of its bytes, not by its path. Dedup, cache
  keys and portable projects all fall out of that one choice.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from typing import Any, BinaryIO, Final, Iterable

__all__ = [
    "PREFIXES",
    "new_id",
    "is_id",
    "sha256_file",
    "sha256_bytes",
    "digest_obj",
    "short",
]

#: Kind prefixes. The prefix is part of the id so a stray id in a log or an
#: error message is self-describing.
PREFIXES: Final[dict[str, str]] = {
    "reference": "ref",
    "segment": "seg",
    "media": "med",
    "audio": "aud",
    "take": "tak",
    "shot": "sht",
    "project": "prj",
}

_ID_BYTES: Final[int] = 5  # 10 hex chars — ample for per-project uniqueness


def new_id(kind: str) -> str:
    """A fresh id for ``kind``, e.g. ``ref_9f2c1a04b7``.

    Raises for an unknown kind rather than inventing a prefix, so a typo shows
    up at the call site instead of producing ids nothing can route.
    """
    try:
        prefix = PREFIXES[kind]
    except KeyError:
        raise ValueError(
            f"Unknown id kind {kind!r}. Known kinds: {', '.join(sorted(PREFIXES))}."
        ) from None
    return f"{prefix}_{secrets.token_hex(_ID_BYTES)}"


def is_id(value: Any, kind: str | None = None) -> bool:
    """True when ``value`` looks like an id, optionally of a specific ``kind``."""
    if not isinstance(value, str) or "_" not in value:
        return False
    prefix, _, rest = value.partition("_")
    if not rest or any(c not in "0123456789abcdef" for c in rest):
        return False
    if kind is None:
        return prefix in PREFIXES.values()
    return prefix == PREFIXES.get(kind)


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------

_CHUNK: Final[int] = 1 << 20


def sha256_file(path: str | os.PathLike[str] | BinaryIO) -> str:
    """Hex sha256 of a file path or an already-open binary stream.

    Streams are read in 1 MiB chunks so a 4 GB video never lands in memory.
    An open stream is read from its current position and left where it ends;
    callers that need to reuse it should seek.
    """
    h = hashlib.sha256()
    if hasattr(path, "read"):
        stream: BinaryIO = path  # type: ignore[assignment]
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            h.update(chunk)
        return h.hexdigest()

    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_obj(obj: Any, *, exclude: Iterable[str] = ()) -> str:
    """A stable digest of a JSON-serialisable object.

    Used to key caches on settings and to record "which settings produced this
    take". Keys are sorted so dict ordering never changes the digest;
    ``exclude`` drops top-level keys that should not count as a settings change
    (UI collapse state, take history).
    """
    if isinstance(obj, dict) and exclude:
        skip = set(exclude)
        obj = {k: v for k, v in obj.items() if k not in skip}
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def short(digest: str, length: int = 8) -> str:
    """First ``length`` characters of a hash, for display only."""
    return digest[:length]

"""Media routes: upload, probe, thumbnails, waveform peaks.

The Director 2.x editor decoded media in the browser — seeking a ``<video>``
element frame by frame for thumbnails, decoding audio buffers to draw waveforms,
and keeping the results in the node's state, sometimes as base64 inside the
saved workflow. This module exists so none of that is necessary.

Three rules shape it:

* **Upload once.** Files are content-addressed. A file already on disk with the
  same hash is reused and no bytes are transferred, which is what makes dropping
  the same reference into three shots free.
* **Decode once.** Thumbnails and waveform peaks are generated server-side and
  cached on disk under the hash, so scrubbing a timeline re-reads a small PNG
  rather than re-decoding a video.
* **The browser never holds pixels.** Everything here returns a URL or a small
  array of numbers.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from aiohttp import web

from ..core.ids import sha256_file, short
from .common import fail, ok, safe_join, workspace_directory

log = logging.getLogger(__name__)

__all__ = ["register"]

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}
_VIDEO_EXT = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v", ".flv", ".wmv", ".mpg"}
_AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".wma"}

#: Upload chunk ceiling. ComfyUI's own /upload/image tops out around 100 MB, and
#: a timeline routinely carries files larger than that.
_CHUNK_LIMIT = 16 * 1024 * 1024

#: Thumbnails are for a timeline strip, so they are small on purpose.
_THUMB_WIDTH = 192

#: Waveform resolution. Enough to draw a recognisable envelope; far less than a
#: decoded buffer.
_PEAK_BUCKETS = 800


def kind_for(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in _VIDEO_EXT:
        return "video"
    if ext in _AUDIO_EXT:
        return "audio"
    if ext in _IMAGE_EXT:
        return "image"
    return "unknown"


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------

def probe_file(path: str) -> dict[str, Any]:
    """Dimensions, duration and frame rate, without decoding the whole file.

    Container metadata only — the point is that adding a 4 GB clip to the
    timeline costs a header read, not a decode.
    """
    filename = os.path.basename(path)
    info: dict[str, Any] = {
        "filename": filename,
        "kind": kind_for(filename),
        "size": os.path.getsize(path) if os.path.exists(path) else 0,
        "width": 0,
        "height": 0,
        "duration": 0.0,
        "fps": 0.0,
    }
    if not os.path.exists(path):
        return info

    if info["kind"] == "image":
        try:
            from PIL import Image

            with Image.open(path) as image:
                info["width"], info["height"] = image.size
        except Exception as exc:
            log.debug("[LTX Director] Could not read image header for %s: %s", filename, exc)
        return info

    try:
        import av
    except Exception:
        return info

    try:
        with av.open(path) as container:
            if container.duration:
                info["duration"] = float(container.duration) / av.time_base
            if container.streams.video:
                stream = container.streams.video[0]
                info["width"] = int(stream.width or 0)
                info["height"] = int(stream.height or 0)
                rate = stream.average_rate or stream.guessed_rate
                info["fps"] = float(rate) if rate else 0.0
                if not info["duration"] and stream.duration and stream.time_base:
                    info["duration"] = float(stream.duration * stream.time_base)
            elif container.streams.audio:
                stream = container.streams.audio[0]
                if not info["duration"] and stream.duration and stream.time_base:
                    info["duration"] = float(stream.duration * stream.time_base)
    except Exception as exc:
        log.debug("[LTX Director] Could not probe %s: %s", filename, exc)
    return info


# --------------------------------------------------------------------------
# caches
# --------------------------------------------------------------------------

def _cache_path(digest: str, suffix: str) -> str:
    """Cache files are keyed by content hash, so identical files share one entry."""
    return os.path.join(workspace_directory("cache"), f"{digest[:32]}{suffix}")


def _index_path() -> str:
    return os.path.join(workspace_directory(), "media_index.json")


def _load_index() -> dict[str, str]:
    """``sha256 -> relative path``. Best-effort: a corrupt index costs a re-upload."""
    try:
        with open(_index_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_index(index: dict[str, str]) -> None:
    try:
        with open(_index_path(), "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=1)
    except Exception as exc:  # pragma: no cover - disk problems are the user's
        log.warning("[LTX Director] Could not write the media index: %s", exc)


def _relative(path: str) -> str:
    """Path as ComfyUI's loaders expect it: relative to the input directory."""
    from .common import input_directory

    return os.path.relpath(path, input_directory()).replace(os.sep, "/")


# --------------------------------------------------------------------------
# thumbnails and peaks
# --------------------------------------------------------------------------

def make_thumbnail(source: str, destination: str, at_seconds: float = 0.0) -> bool:
    """One small JPEG, generated once and served from disk thereafter."""
    kind = kind_for(os.path.basename(source))
    try:
        from PIL import Image
    except Exception:
        return False

    try:
        if kind == "image":
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((_THUMB_WIDTH, _THUMB_WIDTH * 4))
                image.save(destination, "JPEG", quality=80)
            return True

        if kind != "video":
            return False

        import av

        with av.open(source) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if at_seconds > 0 and stream.time_base:
                # Seek to the nearest keyframe before the target; decoding from
                # the start of a long clip for one frame is the cost this whole
                # module exists to avoid.
                container.seek(
                    int(at_seconds / float(stream.time_base)), stream=stream, backward=True
                )
            for frame in container.decode(stream):
                image = frame.to_image()
                image.thumbnail((_THUMB_WIDTH, _THUMB_WIDTH * 4))
                image.save(destination, "JPEG", quality=80)
                return True
    except Exception as exc:
        log.debug("[LTX Director] Could not thumbnail %s: %s", source, exc)
    return False


def compute_peaks(path: str, buckets: int = _PEAK_BUCKETS) -> list[float]:
    """A waveform envelope: one absolute peak per bucket, 0..1.

    Returned as a few hundred floats rather than a decoded buffer, because the
    browser only ever needs enough to draw a shape.
    """
    try:
        import av
        import numpy as np
    except Exception:
        return []

    try:
        with av.open(path) as container:
            if not container.streams.audio:
                return []
            stream = container.streams.audio[0]
            stream.thread_type = "AUTO"
            chunks = []
            for frame in container.decode(stream):
                array = frame.to_ndarray()
                if array.ndim > 1:
                    array = array.mean(axis=0)
                chunks.append(array.astype("float32", copy=False))
            if not chunks:
                return []
            samples = np.concatenate(chunks)
    except Exception as exc:
        log.debug("[LTX Director] Could not read audio from %s: %s", path, exc)
        return []

    import numpy as np

    if samples.size == 0:
        return []
    peak = float(np.abs(samples).max()) or 1.0
    buckets = max(1, min(buckets, samples.size))
    edges = np.linspace(0, samples.size, buckets + 1, dtype=int)
    return [
        float(np.abs(samples[a:b]).max() / peak) if b > a else 0.0
        for a, b in zip(edges[:-1], edges[1:])
    ]


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

def register(routes) -> None:
    """Attach the media routes to ComfyUI's router."""

    @routes.post("/ltxdirector/media/check")
    async def check(request: web.Request) -> web.Response:
        """Is this file already here? Answered before any bytes are uploaded."""
        try:
            body = await request.json()
        except Exception:
            return fail("The upload check was not valid JSON.")
        digest = str(body.get("sha256") or "")
        if not digest:
            return fail("The upload check did not include a file hash.")

        existing = _load_index().get(digest)
        if existing:
            path = safe_join(workspace_directory(create=False), os.path.basename(existing))
            if os.path.exists(path):
                return ok({"exists": True, "path": existing, **probe_file(path)})
        return ok({"exists": False})

    @routes.post("/ltxdirector/media/upload")
    async def upload(request: web.Request) -> web.Response:
        """Upload one chunk of a file.

        Chunked because ComfyUI's own upload endpoint caps out well below the
        size of a usable source clip.
        """
        try:
            reader = await request.multipart()
        except Exception:
            return fail("The upload was not a valid multipart request.")

        fields: dict[str, Any] = {}
        directory = workspace_directory()
        written = 0
        target: str | None = None

        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "file":
                filename = os.path.basename(part.filename or "upload.bin")
                try:
                    target = safe_join(directory, filename)
                except ValueError as exc:
                    return fail(str(exc), "Rename the file and try again.")
                mode = "ab" if str(fields.get("append", "")).lower() == "true" else "wb"
                with open(target, mode) as handle:
                    while True:
                        chunk = await part.read_chunk(1 << 20)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > _CHUNK_LIMIT:
                            return fail(
                                "That upload chunk was too large.",
                                "The editor splits large files automatically; reload the "
                                "page if this persists.",
                                status=413,
                            )
                        handle.write(chunk)
            else:
                fields[part.name or ""] = (await part.read()).decode("utf-8", "replace")

        if target is None:
            return fail("The upload did not include a file.")

        if str(fields.get("final", "true")).lower() != "true":
            return ok({"path": _relative(target), "complete": False})

        digest = sha256_file(target)
        index = _load_index()
        index[digest] = _relative(target)
        _save_index(index)

        info = probe_file(target)
        return ok({"path": _relative(target), "complete": True, "sha256": digest, **info})

    @routes.get("/ltxdirector/media/probe")
    async def probe(request: web.Request) -> web.Response:
        """Dimensions, duration and frame rate for a file already on disk."""
        from .common import input_directory

        name = request.query.get("path", "")
        if not name:
            return fail("No file was named.")
        try:
            path = safe_join(input_directory(), name)
        except ValueError as exc:
            return fail(str(exc))
        if not os.path.exists(path):
            return fail(
                f"{os.path.basename(name)} is no longer in ComfyUI's input folder.",
                "Re-add the file to the timeline.",
                status=404,
            )
        return ok({"path": name, "sha256": sha256_file(path), **probe_file(path)})

    @routes.get("/ltxdirector/media/thumb")
    async def thumb(request: web.Request) -> web.Response:
        """A cached thumbnail. Generated on first request, served from disk after."""
        from .common import input_directory

        name = request.query.get("path", "")
        try:
            at = float(request.query.get("t", "0"))
        except ValueError:
            at = 0.0
        if not name:
            return fail("No file was named.")
        try:
            source = safe_join(input_directory(), name)
        except ValueError as exc:
            return fail(str(exc))
        if not os.path.exists(source):
            return fail("That file is no longer on disk.", status=404)

        digest = sha256_file(source)
        cached = _cache_path(f"{digest}{short(str(round(at, 2)), 8)}", ".jpg")
        if not os.path.exists(cached):
            if not make_thumbnail(source, cached, at):
                return fail(
                    f"A preview could not be made for {os.path.basename(name)}.",
                    "The file may be in a format ComfyUI cannot read.",
                    status=415,
                )
        return web.FileResponse(cached, headers={"Cache-Control": "public, max-age=86400"})

    @routes.get("/ltxdirector/media/peaks")
    async def peaks(request: web.Request) -> web.Response:
        """Waveform envelope, cached by content hash rather than by path."""
        from .common import input_directory

        name = request.query.get("path", "")
        if not name:
            return fail("No file was named.")
        try:
            source = safe_join(input_directory(), name)
        except ValueError as exc:
            return fail(str(exc))
        if not os.path.exists(source):
            return fail("That file is no longer on disk.", status=404)

        digest = sha256_file(source)
        cached = _cache_path(digest, ".peaks.json")
        if os.path.exists(cached):
            try:
                with open(cached, encoding="utf-8") as handle:
                    return ok({"peaks": json.load(handle)})
            except Exception:
                pass

        values = compute_peaks(source)
        if not values:
            return ok({"peaks": [], "note": "This file has no readable audio track."})
        try:
            with open(cached, "w", encoding="utf-8") as handle:
                json.dump(values, handle)
        except Exception:  # pragma: no cover
            pass
        return ok({"peaks": values})

    @routes.get("/ltxdirector/media/list")
    async def listing(request: web.Request) -> web.Response:
        """Everything in the Director's workspace, for the "add existing" picker."""
        directory = workspace_directory()
        entries = []
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            kind = kind_for(name)
            if kind == "unknown":
                continue
            entries.append({"path": _relative(path), **probe_file(path)})
        return ok({"media": entries})

"""Project routes: validate, compile, queue, save, load, import.

These are the operations the editor cannot do for itself, and the reason it does
not need to: validation and compilation are the same code the nodes run, so what
the editor shows and what ComfyUI executes cannot disagree.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from aiohttp import web

from .common import fail, ok, read_json, safe_join, spec_from_body, workspace_directory

log = logging.getLogger(__name__)

__all__ = ["register"]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")
_EXTENSION = ".ltxdirector.json"


def _projects_directory() -> str:
    return workspace_directory("projects")


def _caps():
    """Capabilities, or None when ComfyUI could not be probed.

    None makes the validator skip capability rules rather than assume the worst.
    """
    from ..capabilities import capabilities

    try:
        caps = capabilities()
    except Exception:  # pragma: no cover - defensive
        return None
    return caps if caps.probe.available else None


def register(routes) -> None:
    """Attach the project routes to ComfyUI's router."""

    @routes.post("/ltxdirector/validate")
    async def validate_project(request: web.Request) -> web.Response:
        """Check a shot and return diagnostics.

        Called as the user edits, so it runs the real rules rather than a
        browser-side approximation that could drift from them.
        """
        from ..core.validate import validate

        try:
            body = await read_json(request)
            spec = spec_from_body(body)
        except ValueError as exc:
            return fail(str(exc))

        spec.normalise()
        report = validate(spec, _caps())
        return ok({
            **report.to_dict(),
            "project": spec.to_dict(),
            "digest": spec.digest(),
        })

    @routes.post("/ltxdirector/compile")
    async def compile_project(request: web.Request) -> web.Response:
        """Compile a shot into a native graph.

        Returns both forms: the workflow the canvas opens and the API form the
        queue endpoint executes.
        """
        from ..compiler import compile_spec

        try:
            body = await read_json(request)
            spec = spec_from_body(body)
        except ValueError as exc:
            return fail(str(exc))

        layout = body.get("layout", "flat")
        if layout not in ("flat", "subgraphs"):
            return fail(
                f"{layout!r} is not a graph layout the Director knows.",
                "Use 'flat' or 'subgraphs'.",
            )

        spec.normalise()
        result = compile_spec(spec, caps=_caps(), layout=layout)
        include_api = bool(body.get("include_api", True))
        include_workflow = bool(body.get("include_workflow", True))

        payload = result.to_dict(include_api=include_api)
        if not include_workflow:
            payload.pop("workflow", None)
        return web.json_response({"ok": result.ok, **payload})

    @routes.post("/ltxdirector/queue")
    async def queue_project(request: web.Request) -> web.Response:
        """Compile and queue a shot, so Generate needs no canvas.

        The compiled graph is handed to ComfyUI's own prompt queue exactly as if
        the user had pressed Run on it — which is the point: nothing about
        execution is special-cased for the Director.
        """
        from ..compiler import compile_spec

        try:
            body = await read_json(request)
            spec = spec_from_body(body)
        except ValueError as exc:
            return fail(str(exc))

        spec.normalise()
        result = compile_spec(spec, caps=_caps(), layout="flat")
        if not result.ok:
            return web.json_response(
                {"ok": False, **result.to_dict(include_api=False)}, status=422
            )

        client_id = body.get("client_id")
        try:
            prompt_id = _enqueue(result.api, result.workflow, client_id)
        except Exception as exc:
            log.exception("[LTX Director] Could not queue the shot.")
            return fail(
                "The shot compiled but ComfyUI would not accept it.",
                str(exc),
                status=502,
            )

        return ok({
            "prompt_id": prompt_id,
            "node_index": result.node_index,
            **result.report.to_dict(),
        })

    # -- project files -----------------------------------------------------

    @routes.post("/ltxdirector/project/save")
    async def save_project(request: web.Request) -> web.Response:
        try:
            body = await read_json(request)
            spec = spec_from_body(body)
        except ValueError as exc:
            return fail(str(exc))

        name = _SAFE_NAME.sub("", str(body.get("name") or spec.meta.name or "shot")).strip()
        if not name:
            return fail("That project name has no usable characters.", "Try a simpler name.")

        spec.meta.name = name
        spec.meta.modified = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not spec.meta.created:
            spec.meta.created = spec.meta.modified

        path = safe_join(_projects_directory(), f"{name}{_EXTENSION}")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(spec.to_json(indent=1))
        except OSError as exc:
            return fail(f"The project could not be saved: {exc.strerror}.", status=500)

        return ok({"name": name, "path": os.path.basename(path), "bytes": os.path.getsize(path)})

    @routes.get("/ltxdirector/project/load")
    async def load_project(request: web.Request) -> web.Response:
        from ..core.spec import Spec

        name = request.query.get("name", "")
        if not name:
            return fail("No project was named.")
        try:
            path = safe_join(_projects_directory(), f"{_SAFE_NAME.sub('', name)}{_EXTENSION}")
        except ValueError as exc:
            return fail(str(exc))
        if not os.path.exists(path):
            return fail(f"There is no saved project called {name!r}.", status=404)

        try:
            spec = Spec.from_json(open(path, encoding="utf-8").read())
        except Exception as exc:
            return fail(
                f"{name} could not be opened: the file is not a Director project.",
                str(exc),
                status=422,
            )
        return ok({"project": spec.to_dict()})

    @routes.get("/ltxdirector/project/list")
    async def list_projects(request: web.Request) -> web.Response:
        directory = _projects_directory()
        entries = []
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(_EXTENSION):
                continue
            path = os.path.join(directory, filename)
            entries.append({
                "name": filename[: -len(_EXTENSION)],
                "modified": os.path.getmtime(path),
                "bytes": os.path.getsize(path),
            })
        entries.sort(key=lambda e: e["modified"], reverse=True)
        return ok({"projects": entries})

    @routes.post("/ltxdirector/project/import")
    async def import_project(request: web.Request) -> web.Response:
        """Convert a Director 2.x timeline into a project.

        Returns the diagnostics alongside it, because the conversion is lossy in
        ways the user needs to know about — base64 media has to be re-added, and
        retake's masking has no equivalent.
        """
        from ..migrations.v2_timeline import convert_with_report

        try:
            body = await read_json(request)
        except ValueError as exc:
            return fail(str(exc))

        timeline = body.get("timeline")
        if isinstance(timeline, str):
            try:
                timeline = json.loads(timeline)
            except json.JSONDecodeError:
                return fail("The timeline data was not valid JSON.")
        if not isinstance(timeline, dict):
            return fail(
                "No Director 2 timeline was included.",
                "Copy the old node's timeline_data value and try again.",
            )

        result = convert_with_report(
            timeline,
            fps=float(body.get("fps", 24.0) or 24.0),
            width=int(body.get("width", 0) or 0),
            height=int(body.get("height", 0) or 0),
            guide_strength=str(body.get("guide_strength", "") or ""),
            img_compression=int(body.get("img_compression", 18) or 18),
        )
        return ok({
            "project": result.spec.to_dict(),
            "diagnostics": [d.to_dict() for d in result.diagnostics],
        })


# --------------------------------------------------------------------------
# queueing
# --------------------------------------------------------------------------

def _enqueue(api_prompt: dict[str, Any], workflow: dict[str, Any], client_id: Any) -> str:
    """Hand a compiled graph to ComfyUI's prompt queue.

    Validated through ComfyUI's own validator first, so a graph it would reject
    is reported as a readable failure here rather than as a silent no-op in the
    queue.
    """
    import uuid

    import execution  # type: ignore
    from server import PromptServer  # type: ignore

    server = PromptServer.instance
    prompt_id = str(uuid.uuid4())

    valid = execution.validate_prompt(prompt_id, api_prompt, None)
    if not valid[0]:
        raise ValueError(_describe_rejection(valid))

    extra_data: dict[str, Any] = {"extra_pnginfo": {"workflow": workflow}}
    if client_id is not None:
        extra_data["client_id"] = client_id

    number = server.number
    server.number += 1
    server.prompt_queue.put((number, prompt_id, api_prompt, extra_data, valid[2]))
    return prompt_id


def _describe_rejection(valid: tuple) -> str:
    """Turn ComfyUI's validation tuple into one sentence."""
    error = valid[1] if len(valid) > 1 else None
    if isinstance(error, dict):
        message = error.get("message") or "the graph was rejected"
        details = error.get("details")
        return f"{message}{f' ({details})' if details else ''}"
    return str(error or "the graph was rejected")

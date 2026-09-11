"""Does this actually install?

Loads the repository the way ComfyUI loads a custom node — by path, as a package
with relative imports — and checks that every node registers, every route
attaches, and the Director 2.x nodes still resolve.

Everything else in the suite tests a module. This tests the thing a user
installs.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

from .comfy_stub import install_full

REPO_ROOT = Path(__file__).resolve().parent.parent

# The legacy modules need torch and PyAV at import time; skip rather than fail
# where they are absent, since none of the Director Next code needs either.
pytest.importorskip("torch")
pytest.importorskip("av")
pytest.importorskip("PIL")


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory):
    """The loaded package, plus the fake server that recorded its routes."""
    workspace = tmp_path_factory.mktemp("comfy")
    server = install_full(input_dir=str(workspace), output_dir=str(workspace))

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    spec = importlib.util.spec_from_file_location(
        "WhatDreamsCost_ComfyUI_under_test",
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, server


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def test_the_package_loads(package) -> None:
    module, _ = package
    assert module.WEB_DIRECTORY == "./js"


def test_every_node_registers(package) -> None:
    module, _ = package
    assert set(module.NODE_CLASS_MAPPINGS) == {
        # Director Next
        "LTXDirectorProject", "LTXDirectorRelay", "LTXDirectorCompile",
        # Director 2.x, still working
        "LTXDirector", "LTXDirectorGuide", "LTXDirectorCropGuides",
        # the standalone utility nodes
        "LTXKeyframer", "LTXSequencer", "MultiImageLoader",
        "SpeechLengthCalculator", "LoadAudioUI", "LoadVideoUI",
    }


def test_every_registered_node_has_a_display_name(package) -> None:
    module, _ = package
    assert set(module.NODE_DISPLAY_NAME_MAPPINGS) == set(module.NODE_CLASS_MAPPINGS)
    for name, label in module.NODE_DISPLAY_NAME_MAPPINGS.items():
        assert label and label[0].isupper(), name


def test_the_legacy_nodes_are_labelled_as_such(package) -> None:
    # They keep working; they should not look like the recommended path.
    module, _ = package
    for name in ("LTXDirector", "LTXDirectorGuide", "LTXDirectorCropGuides"):
        assert "legacy" in module.NODE_DISPLAY_NAME_MAPPINGS[name].lower(), name


def test_the_v3_entry_point_agrees_with_the_static_map(package) -> None:
    """ComfyUI-Manager reads the static map; ComfyUI reads the entry point.

    They are built from one source, and this is what proves it.
    """
    module, _ = package
    extension = asyncio.run(module.comfy_entrypoint())
    registered = {cls.__name__ for cls in asyncio.run(extension.get_node_list())}
    assert registered <= set(module.NODE_CLASS_MAPPINGS)
    assert {"LTXDirectorProject", "LTXDirectorRelay", "LTXDirectorCompile"} <= registered


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

def test_every_director_route_attaches(package) -> None:
    _, server = package
    paths = {path for _, path in server.routes.registered}
    assert {
        "/ltxdirector/capabilities",
        "/ltxdirector/presets",
        "/ltxdirector/vocabulary",
        "/ltxdirector/validate",
        "/ltxdirector/compile",
        "/ltxdirector/queue",
        "/ltxdirector/media/check",
        "/ltxdirector/media/upload",
        "/ltxdirector/media/probe",
        "/ltxdirector/media/thumb",
        "/ltxdirector/media/peaks",
        "/ltxdirector/media/list",
        "/ltxdirector/project/save",
        "/ltxdirector/project/load",
        "/ltxdirector/project/list",
        "/ltxdirector/project/import",
    } <= paths


def test_routes_register_once(package) -> None:
    from director.api import register_routes

    _, server = package
    before = len(server.routes.registered)
    register_routes()
    assert len(server.routes.registered) == before


def test_the_legacy_routes_are_untouched(package) -> None:
    # Breaking these would break the Director 2.x editor in saved workflows.
    _, server = package
    paths = {path for _, path in server.routes.registered}
    assert {"/ltx_director_check_file", "/ltx_director_upload_chunk"} <= paths


def test_director_routes_are_namespaced(package) -> None:
    _, server = package
    new = [p for _, p in server.routes.registered if "ltxdirector" in p]
    assert len(new) >= 16
    for path in new:
        assert path.startswith("/ltxdirector/"), path


# --------------------------------------------------------------------------
# the frontend the package serves
# --------------------------------------------------------------------------

def test_the_editor_is_served_from_the_web_directory() -> None:
    assert (REPO_ROOT / "js" / "ltx_director_next.js").exists()
    assert (REPO_ROOT / "js" / "director" / "styles.css").exists()


def test_only_one_module_registers_the_extension() -> None:
    """Every .js under the web directory is loaded by ComfyUI as an extension.

    Modules that merely export are harmless, but two files both calling
    registerExtension for the same node would be a real conflict.
    """
    registering = []
    for path in (REPO_ROOT / "js").rglob("*.js"):
        source = path.read_text(encoding="utf-8")
        if "LTXDirectorProject" in source and "registerExtension" in source:
            registering.append(path.name)
    assert registering == ["ltx_director_next.js"]


def test_the_editor_is_smaller_than_the_one_it_replaces() -> None:
    # Not vanity: every .js under the web directory is parsed on every ComfyUI
    # page load, used or not.
    new = sum(p.stat().st_size for p in (REPO_ROOT / "js" / "director").rglob("*.js"))
    new += (REPO_ROOT / "js" / "ltx_director_next.js").stat().st_size
    old = (REPO_ROOT / "js" / "ltx_director.js").stat().st_size
    assert new < old, f"{new} bytes vs the old editor's {old}"

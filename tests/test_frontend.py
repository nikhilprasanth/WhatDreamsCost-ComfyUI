"""Run the frontend test suite, and check the browser mirror against Python.

Two jobs:

1. Shell out to Node's test runner, so ``pytest`` is the one command that runs
   everything. Skipped, not failed, where Node is absent — a contributor without
   Node should still be able to work on the Python side.
2. Check the duplicated constants directly. ``js/director/state/time.js`` mirrors
   ``director/core/time.py`` so a marker drag can snap at pointer speed; the
   duplicate is only safe while the two agree, and this is what makes a drift
   fail here rather than in someone's render.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from director import SPEC_SCHEMA
from director.core import time as ptime

REPO_ROOT = Path(__file__).resolve().parent.parent
JS_DIR = REPO_ROOT / "js" / "director"
TIME_JS = JS_DIR / "state" / "time.js"
STORE_JS = JS_DIR / "state" / "store.js"


# --------------------------------------------------------------------------
# the Node suite
# --------------------------------------------------------------------------

node = pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")


@node
def test_the_frontend_suite_passes() -> None:
    result = subprocess.run(
        ["node", "--test", "tests/js/*.test.mjs"],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@node
@pytest.mark.parametrize(
    "path",
    sorted(str(p.relative_to(REPO_ROOT)) for p in JS_DIR.rglob("*.js")),
)
def test_every_frontend_module_parses(path: str) -> None:
    source = (REPO_ROOT / path).read_text(encoding="utf-8")
    result = subprocess.run(
        ["node", "--input-type=module", "--check"],
        input=source, cwd=REPO_ROOT, capture_output=True, text=True,
        encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stderr


@node
def test_the_extension_entry_point_parses() -> None:
    source = (REPO_ROOT / "js" / "ltx_director_next.js").read_text(encoding="utf-8")
    result = subprocess.run(
        ["node", "--input-type=module", "--check"],
        input=source, cwd=REPO_ROOT, capture_output=True, text=True,
        encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------
# the mirror
# --------------------------------------------------------------------------

def _js_const(source: str, name: str) -> str:
    match = re.search(rf"export const {name}\s*=\s*([^;]+);", source)
    assert match, f"{name} is not exported from time.js any more"
    return match.group(1).strip()


def test_the_stride_constants_match_python() -> None:
    source = TIME_JS.read_text(encoding="utf-8")
    assert int(_js_const(source, "TEMPORAL_STRIDE")) == ptime.TEMPORAL_STRIDE
    assert int(_js_const(source, "SPATIAL_STRIDE")) == ptime.SPATIAL_STRIDE


def test_the_schema_version_matches_python() -> None:
    source = STORE_JS.read_text(encoding="utf-8")
    match = re.search(r"export const SCHEMA\s*=\s*(\d+);", source)
    assert match, "SCHEMA is not exported from store.js any more"
    assert int(match.group(1)) == SPEC_SCHEMA


def test_the_digest_exclusions_match_python() -> None:
    from director.core.spec import DIGEST_EXCLUDE

    source = STORE_JS.read_text(encoding="utf-8")
    match = re.search(r"const COSMETIC = new Set\(\[([^\]]*)\]\)", source)
    assert match, "COSMETIC is not defined in store.js any more"
    js_keys = set(re.findall(r'"([^"]+)"', match.group(1)))
    assert js_keys == set(DIGEST_EXCLUDE)


@node
def test_the_mirror_agrees_with_python_on_every_snap() -> None:
    """Compare the two implementations directly, over the whole useful range."""
    script = """
    import { snapFrames, snapDim, latentFrames } from "./js/director/state/time.js";
    const rows = [];
    for (let n = 0; n < 500; n += 1) rows.push([n, snapFrames(n), snapDim(n), latentFrames(n)]);
    process.stdout.write(JSON.stringify(rows));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stderr

    import json

    for n, js_frames, js_dim, js_latent in json.loads(result.stdout):
        assert js_frames == ptime.snap_frames(n), f"snapFrames({n})"
        assert js_dim == ptime.snap_dim(n), f"snapDim({n})"
        assert js_latent == ptime.latent_frames_for(n), f"latentFrames({n})"


# --------------------------------------------------------------------------
# the rules the rebuild exists to keep
# --------------------------------------------------------------------------

def test_nothing_reads_the_graph_on_every_repaint() -> None:
    """The v2 editor traversed links inside onDrawForeground, sixty times a second."""
    source = (REPO_ROOT / "js" / "ltx_director_next.js").read_text(encoding="utf-8")
    # Look for the hook being installed, not for the word: the module docstring
    # explains why it is absent, and that explanation should not fail the test.
    assert not re.search(r"prototype\.onDrawForeground\s*=", source)
    assert "onDrawForeground?" not in source


def test_the_editor_never_stores_decoded_media() -> None:
    for path in list(JS_DIR.rglob("*.js")) + [REPO_ROOT / "js" / "ltx_director_next.js"]:
        source = path.read_text(encoding="utf-8")
        assert "toDataURL" not in source, f"{path.name} encodes canvas data"
        assert "readAsDataURL" not in source, f"{path.name} reads files as data URLs"


def test_the_project_widget_is_the_only_serialised_state() -> None:
    source = (REPO_ROOT / "js" / "ltx_director_next.js").read_text(encoding="utf-8")
    # The DOM widget must not write a second copy of the project into the workflow.
    assert "serialize: false" in source

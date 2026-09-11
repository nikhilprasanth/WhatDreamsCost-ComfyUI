#!/usr/bin/env python3
"""Re-capture ``tests/fixtures/upstream_ltx25.json`` from upstream reference graphs.

The Director's compiler is checked against what Lightricks and Comfy-Org
actually build, not against what their docs say. That check has to keep working
on a machine that does not have those repos, so the relevant facts — node-type
counts and slot orders per reference graph — are distilled into a committed
fixture.

Run this after pulling a new LTX release, then look at the diff. A change here
is a change in what the Director should emit.

Usage::

    python tools/capture_upstream.py --ltxvideo ../ComfyUI-LTXVideo [--templates DIR]

``--templates`` points at a checkout of ``Comfy-Org/workflow_templates`` (or any
directory holding ``video_ltx2_5_*.json``). Both arguments are optional; whatever
is found is captured and the rest is skipped with a note.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "upstream_ltx25.json"

#: label -> path relative to the LTX-2.5 example_workflows directory
LTXVIDEO_GRAPHS = {
    "t2v_i2v_two_stage": "LTX-2.5_T2V_I2V_Two_Stage_Distilled.json",
    "t2v_i2v_single_stage": "LTX-2.5_T2V_I2V_Single_Stage_Distilled.json",
    "a2v_two_stage": "LTX-2.5_A2V_Two_Stage_Distilled.json",
    "t2a_single_stage": "LTX-2.5_T2A_Single_Stage_Distilled.json",
    "iclora_ingredients": "LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json",
    "iclora_motion_track": "LTX-2.5_ICLoRA_Motion_Track_Distilled.json",
}

TEMPLATE_GRAPHS = {
    "template_t2v": "video_ltx2_5_t2v.json",
    "template_i2v": "video_ltx2_5_i2v.json",
    "template_flf2v": "video_ltx2_5_flf2v.json",
}

#: Annotation and plumbing nodes carry no topology, so they are dropped to keep
#: the fixture about the pipeline.
IGNORED_TYPES = {"MarkdownNote", "Note", "Reroute"}


def is_subgraph_instance(node_type: str) -> bool:
    """Subgraph instances use the definition's UUID as their type."""
    return len(node_type) == 36 and node_type.count("-") == 4


def all_nodes(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every real node, flattened across subgraph definitions."""
    nodes = list(doc.get("nodes", []))
    for sub in doc.get("definitions", {}).get("subgraphs", []) or []:
        nodes.extend(sub.get("nodes", []))
    return [
        n for n in nodes
        if n.get("type") and not is_subgraph_instance(n["type"])
        and n["type"] not in IGNORED_TYPES
    ]


def capture(doc: dict[str, Any]) -> dict[str, Any]:
    nodes = all_nodes(doc)
    counts = collections.Counter(n["type"] for n in nodes)

    signatures: dict[str, dict[str, Any]] = {}
    for node in nodes:
        entry = signatures.setdefault(
            node["type"], {"inputs": [], "outputs": [], "widgets": 0}
        )
        # Unlinked widget inputs are omitted from the serialised form, so the
        # longest observed list is the closest thing to the full signature.
        inputs = [i.get("name") for i in node.get("inputs") or []]
        if len(inputs) > len(entry["inputs"]):
            entry["inputs"] = inputs
        outputs = [o.get("name") for o in node.get("outputs") or []]
        if len(outputs) > len(entry["outputs"]):
            entry["outputs"] = outputs
        widgets = node.get("widgets_values")
        if isinstance(widgets, list):
            entry["widgets"] = max(entry["widgets"], len(widgets))

    return {"node_counts": dict(sorted(counts.items())), "signatures": signatures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltxvideo", type=Path, default=Path("../ComfyUI-LTXVideo"))
    parser.add_argument("--templates", type=Path, default=None)
    parser.add_argument("--commit", default="", help="Upstream commit, recorded for provenance.")
    args = parser.parse_args(argv)

    existing = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.exists() else {}
    graphs: dict[str, Any] = {}
    sources: dict[str, str] = {}

    examples = args.ltxvideo / "example_workflows" / "2.5"
    for label, filename in LTXVIDEO_GRAPHS.items():
        path = examples / filename
        if not path.exists():
            print(f"skip {label}: {path} not found", file=sys.stderr)
            continue
        graphs[label] = capture(json.loads(path.read_text(encoding="utf-8")))
        sources[label] = filename

    if args.templates:
        for label, filename in TEMPLATE_GRAPHS.items():
            path = args.templates / filename
            if not path.exists():
                print(f"skip {label}: {path} not found", file=sys.stderr)
                continue
            graphs[label] = capture(json.loads(path.read_text(encoding="utf-8")))
            sources[label] = filename

    if not graphs:
        print("nothing captured; check --ltxvideo / --templates", file=sys.stderr)
        return 1

    # Keep anything we could not re-capture this run rather than losing it.
    merged = dict(existing.get("graphs", {}))
    merged.update(graphs)

    payload = {
        "_provenance": {
            "captured": _today(),
            "ltxvideo_commit": args.commit or existing.get("_provenance", {}).get(
                "ltxvideo_commit", "unknown"
            ),
            "sources": {**existing.get("_provenance", {}).get("sources", {}), **sources},
            "note": (
                "Node-type counts extracted from the upstream LTX-2.5 reference graphs, "
                "flattened across subgraph definitions. Committed so the Director's "
                "compiler can be checked against what upstream actually builds without "
                "needing those repos present. Regenerate with tools/capture_upstream.py."
            ),
        },
        "graphs": merged,
        "constants": existing.get("constants", {}),
    }
    FIXTURE.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"captured {len(graphs)} graph(s) into {FIXTURE.relative_to(REPO_ROOT)}")
    return 0


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


if __name__ == "__main__":
    raise SystemExit(main())

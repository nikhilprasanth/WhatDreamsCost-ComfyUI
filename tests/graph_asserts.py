"""Structural checks for a compiled workflow.

A graph that loads but is wired to the wrong socket is the worst failure mode
available to a compiler: it looks fine. These assertions exist to catch that
class of mistake without a running ComfyUI.
"""

from __future__ import annotations

from typing import Any

from director.compiler.signatures import signature_for


def _links(workflow: dict[str, Any]) -> list[tuple[int, int, int, int, int, str]]:
    """Normalise the two serialised link shapes into one.

    Top-level links are arrays; subgraph links are objects. Both are valid and
    both appear in upstream graphs.
    """
    out = []
    for link in workflow.get("links", []):
        if isinstance(link, dict):
            out.append((
                link["id"], link["origin_id"], link["origin_slot"],
                link["target_id"], link["target_slot"], link["type"],
            ))
        else:
            out.append(tuple(link))  # type: ignore[arg-type]
    return out


def assert_graph_integrity(workflow: dict[str, Any]) -> None:
    """Every structural invariant a ComfyUI workflow must hold."""
    nodes = workflow["nodes"]
    by_id = {n["id"]: n for n in nodes}

    assert len(by_id) == len(nodes), "duplicate node ids"
    assert workflow["version"] == 0.4, "workflow schema version must be 0.4"
    assert workflow["last_node_id"] >= max(by_id, default=0)

    seen_link_ids: set[int] = set()
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in _links(workflow):
        assert link_id not in seen_link_ids, f"duplicate link id {link_id}"
        seen_link_ids.add(link_id)

        assert origin_id in by_id, f"link {link_id} comes from missing node {origin_id}"
        assert target_id in by_id, f"link {link_id} goes to missing node {target_id}"

        origin = by_id[origin_id]
        outputs = origin.get("outputs") or []
        assert origin_slot < len(outputs), (
            f"link {link_id}: {origin['type']} has {len(outputs)} outputs, "
            f"slot {origin_slot} requested"
        )
        assert link_id in (outputs[origin_slot].get("links") or []), (
            f"link {link_id} is not recorded on {origin['type']} output {origin_slot}"
        )

        target = by_id[target_id]
        inputs = target.get("inputs") or []
        assert target_slot < len(inputs), (
            f"link {link_id}: {target['type']} has {len(inputs)} inputs, "
            f"slot {target_slot} requested"
        )
        assert inputs[target_slot].get("link") == link_id, (
            f"link {link_id} is not recorded on {target['type']} input {target_slot} "
            f"({inputs[target_slot].get('name')})"
        )

        assert _types_compatible(outputs[origin_slot]["type"], inputs[target_slot]["type"]), (
            f"link {link_id}: {origin['type']}.{outputs[origin_slot]['name']} is "
            f"{outputs[origin_slot]['type']} but {target['type']}."
            f"{inputs[target_slot]['name']} wants {inputs[target_slot]['type']}"
        )

    assert workflow["last_link_id"] >= max(seen_link_ids, default=0)


def assert_slots_match_signatures(workflow: dict[str, Any]) -> None:
    """Serialised slot names and order must match the registered signature.

    The serialised form omits unlinked widget inputs, so this checks that what
    *is* present appears in signature order with nothing invented.
    """
    for node in workflow["nodes"]:
        signature = signature_for(node["type"])

        declared = [s.name for s in signature.inputs]
        present = [i["name"] for i in node.get("inputs") or []]
        assert set(present) <= set(declared), (
            f"{node['type']} serialises unknown inputs "
            f"{sorted(set(present) - set(declared))}"
        )
        assert present == [n for n in declared if n in present], (
            f"{node['type']} serialises inputs out of declaration order: {present}"
        )

        for slot in node.get("inputs") or []:
            if signature.input(slot["name"]).widget:
                assert slot.get("link") is not None, (
                    f"{node['type']}.{slot['name']} is a widget input serialised with no "
                    f"link; unlinked widget inputs must be omitted"
                )

        outputs = [o["name"] for o in node.get("outputs") or []]
        assert outputs == [s.name for s in signature.outputs], (
            f"{node['type']} outputs {outputs}, signature says "
            f"{[s.name for s in signature.outputs]}"
        )

        if signature.widgets:
            assert len(node["widgets_values"]) == len(signature.widgets), (
                f"{node['type']} has {len(node['widgets_values'])} widget values, "
                f"signature declares {len(signature.widgets)}"
            )


def assert_required_inputs_connected(workflow: dict[str, Any]) -> None:
    """Nothing that must be wired is left dangling.

    A missing VAE or model link is a run-time explosion deep inside the sampler,
    which is exactly the kind of error the Director exists to prevent.
    """
    for node in workflow["nodes"]:
        signature = signature_for(node["type"])
        linked = {i["name"] for i in node.get("inputs") or [] if i.get("link") is not None}
        for slot in signature.inputs:
            if slot.optional or slot.widget:
                continue
            assert slot.name in linked, (
                f"{node['type']}#{node['id']}.{slot.name} is required but not connected"
            )


def assert_reachable(workflow: dict[str, Any], terminal_types: set[str]) -> None:
    """Every node feeds something that eventually reaches an output node.

    Catches orphans: nodes that were built and then never wired in, which waste
    execution time and confuse anyone reading the graph.
    """
    nodes = {n["id"]: n for n in workflow["nodes"]}
    consumers: dict[int, list[int]] = {nid: [] for nid in nodes}
    for _, origin_id, _, target_id, _, _ in _links(workflow):
        consumers[origin_id].append(target_id)

    reaching: dict[int, bool] = {}

    def reaches(node_id: int, seen: frozenset[int] = frozenset()) -> bool:
        if node_id in reaching:
            return reaching[node_id]
        if node_id in seen:
            return False
        if nodes[node_id]["type"] in terminal_types:
            reaching[node_id] = True
            return True
        result = any(reaches(c, seen | {node_id}) for c in consumers[node_id])
        reaching[node_id] = result
        return result

    orphans = [
        f"{nodes[nid]['type']}#{nid}"
        for nid in nodes
        if not reaches(nid) and nodes[nid]["type"] not in ("MarkdownNote", "Note", "PreviewAny")
    ]
    assert not orphans, f"nodes that reach no output: {orphans}"


def node_types(workflow: dict[str, Any]) -> list[str]:
    return [n["type"] for n in workflow["nodes"]]


def count(workflow: dict[str, Any], node_type: str) -> int:
    return node_types(workflow).count(node_type)


def only(workflow: dict[str, Any], node_type: str) -> dict[str, Any]:
    matches = [n for n in workflow["nodes"] if n["type"] == node_type]
    assert len(matches) == 1, f"expected exactly one {node_type}, found {len(matches)}"
    return matches[0]


def execution_order(workflow: dict[str, Any]) -> list[str]:
    return [n["type"] for n in sorted(workflow["nodes"], key=lambda n: n["order"])]


def widget(workflow: dict[str, Any], node_type: str, name: str) -> Any:
    node = only(workflow, node_type)
    signature = signature_for(node_type)
    return node["widgets_values"][signature.widget_names.index(name)]


def _types_compatible(source: str, target: str) -> bool:
    if "*" in (source, target):
        return True
    source_set = {t.strip() for t in source.split(",")}
    target_set = {t.strip() for t in target.split(",")}
    return bool(source_set & target_set)

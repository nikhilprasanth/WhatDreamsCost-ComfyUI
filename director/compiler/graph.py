"""A builder for ComfyUI workflow JSON (schema 0.4).

The point of this module is that the LTX adapter reads like the pipeline it
describes::

    with g.section("Canvas") as s:
        latent = s.node("EmptyLTXVLatentVideo", width=1280, height=704, length=121)
        pre    = s.node("LTXVPreprocess", image=image, img_compression=18)
        pinned = s.node("LTXVImgToVideoInplace",
                        vae=vae, image=pre.out(), latent=latent.out(), strength=0.7)

One call per node. A keyword whose value is a :class:`Port` becomes a link; any
other value becomes a widget value. Slot indices, link records, widget ordering,
positions, groups and execution order are this module's problem, not the
adapter's.

Two output layouts:

``flat`` (default)
    One graph, with :data:`groups` boxes around each section. This is the format
    with the longest-standing support in litegraph, so it is what the Director
    emits unless asked otherwise. Opening it gives the user the native LTX
    graph with nothing of the Director left in it.

``subgraphs``
    Sections become entries in ``definitions.subgraphs``, matching the shape of
    the official LTX-2.5 templates: a handful of tidy boxes you expand to edit.
    Structurally validated here; see ``docs/ARCHITECTURE.md`` for the caveat.

Building is deterministic — same input, byte-identical output — because the
compiler contract says so and because a golden-file test depends on it.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from .signatures import NodeSignature, signature_for

__all__ = ["Port", "NodeHandle", "Section", "GraphBuilder", "Layout"]

Layout = Literal["flat", "subgraphs"]

#: Node ids litegraph reserves for a subgraph's input and output proxies.
SUBGRAPH_INPUT_ID = -10
SUBGRAPH_OUTPUT_ID = -20

#: Layout grid. Only affects how the opened graph looks.
_COL_WIDTH = 440
_ROW_HEIGHT = 260
_SECTION_GAP = 120
_PAD = 40


@dataclass(frozen=True)
class Port:
    """One output of one node — the thing you pass to wire nodes together."""

    node: "NodeHandle"
    index: int
    type: str

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.node.type}#{self.node.id}:{self.index} {self.type}>"


@dataclass
class NodeHandle:
    id: int
    type: str
    signature: NodeSignature
    section: "Section"
    title: str = ""
    widget_values: list[Any] = field(default_factory=list)
    #: ``input name -> Port``
    links_in: dict[str, Port] = field(default_factory=dict)
    pos: tuple[int, int] = (0, 0)

    def out(self, which: int | str = 0) -> Port:
        """A port for output ``which``, by index or by name."""
        index = which if isinstance(which, int) else self.signature.output_index(which)
        try:
            slot = self.signature.outputs[index]
        except IndexError:
            raise KeyError(
                f"{self.type} has {len(self.signature.outputs)} output(s); asked for {which!r}."
            ) from None
        return Port(self, index, slot.type)

    def set(self, **values: Any) -> "NodeHandle":
        """Set further widget values or links after creation.

        Needed for the occasional cycle-shaped wiring where a node has to exist
        before one of its inputs does.
        """
        self.section._apply(self, values)
        return self


@dataclass
class Section:
    """A named region of the graph — a group box, or one subgraph."""

    name: str
    builder: "GraphBuilder"
    color: str = "#3f789e"
    nodes: list[NodeHandle] = field(default_factory=list)
    #: Ports this section exposes. Only meaningful in ``subgraphs`` layout.
    exports: dict[str, Port] = field(default_factory=dict)

    def node(self, node_type: str, *, title: str = "", **values: Any) -> NodeHandle:
        """Add a node. Port values become links; everything else is a widget value."""
        signature = signature_for(node_type)
        handle = NodeHandle(
            id=self.builder._next_node_id(),
            type=node_type,
            signature=signature,
            section=self,
            title=title,
            widget_values=[default for _, default in signature.widgets],
        )
        self.nodes.append(handle)
        self.builder._nodes.append(handle)
        self._apply(handle, values)
        return handle

    def note(self, text: str, *, title: str = "") -> NodeHandle:
        """A MarkdownNote, for explaining a choice inside the graph itself."""
        return self.node("MarkdownNote", title=title, text=text)

    def _apply(self, handle: NodeHandle, values: dict[str, Any]) -> None:
        signature = handle.signature
        widget_names = signature.widget_names
        for name, value in values.items():
            if isinstance(value, Port):
                try:
                    signature.input_index(name)
                except KeyError:
                    raise KeyError(
                        f"{handle.type} has no input {name!r}. "
                        f"Inputs: {[s.name for s in signature.inputs]}."
                    ) from None
                handle.links_in[name] = value
                continue

            if name in widget_names:
                handle.widget_values[widget_names.index(name)] = value
                continue

            # A widget-backed input whose widget is recorded under a short name
            # (TextGenerateLTX2Prompt's "sampling_mode.seed" is "seed" in
            # widgets_values, for instance).
            short = name.rsplit(".", 1)[-1]
            if short in widget_names:
                handle.widget_values[widget_names.index(short)] = value
                continue

            raise KeyError(
                f"{handle.type} has no widget or input {name!r}. "
                f"Widgets: {list(widget_names)}; inputs: {[s.name for s in signature.inputs]}."
            )

    def export(self, name: str, port: Port) -> None:
        """Expose ``port`` as a named output of this section."""
        self.exports[name] = port


class GraphBuilder:
    """Accumulates sections and emits a ComfyUI workflow."""

    def __init__(
        self,
        *,
        layout: Layout = "flat",
        graph_id: str | None = None,
        title: str = "LTX Director",
    ) -> None:
        self.layout: Layout = layout
        self.title = title
        # A fixed namespace keeps compilation deterministic: a random uuid4 here
        # would break the "same input, identical output" contract.
        self.graph_id = graph_id or str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"ltxdirector/{title}")
        )
        self._sections: list[Section] = []
        self._nodes: list[NodeHandle] = []
        self._node_id = 0
        self._link_id = 0

    # -- construction ------------------------------------------------------

    def _next_node_id(self) -> int:
        self._node_id += 1
        return self._node_id

    def _next_link_id(self) -> int:
        self._link_id += 1
        return self._link_id

    @contextmanager
    def section(self, name: str, *, color: str = "#3f789e") -> Iterator[Section]:
        """Open a named region. Nodes created inside it belong to it."""
        sec = Section(name=name, builder=self, color=color)
        self._sections.append(sec)
        yield sec

    def find(self, node_type: str) -> list[NodeHandle]:
        """Every node of a type, in creation order. Used by tests and by the node index."""
        return [n for n in self._nodes if n.type == node_type]

    # -- emission ----------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """The workflow document."""
        self._assign_positions()
        if self.layout == "subgraphs":
            return self._build_subgraphs()
        return self._build_flat()

    # -- layout ------------------------------------------------------------

    def _assign_positions(self) -> None:
        """Lay sections out as columns, nodes as rows. Deterministic by construction."""
        x = _PAD
        for sec in self._sections:
            # Wrap long sections into extra columns so nothing runs off the canvas.
            rows = max(1, min(len(sec.nodes), 6))
            for index, node in enumerate(sec.nodes):
                col, row = divmod(index, rows)
                node.pos = (x + col * _COL_WIDTH, _PAD + row * _ROW_HEIGHT)
            columns = (len(sec.nodes) + rows - 1) // rows if sec.nodes else 1
            x += max(1, columns) * _COL_WIDTH + _SECTION_GAP

    def _section_bounds(self, sec: Section) -> list[int]:
        if not sec.nodes:
            return [0, 0, _COL_WIDTH, _ROW_HEIGHT]
        xs = [n.pos[0] for n in sec.nodes]
        ys = [n.pos[1] for n in sec.nodes]
        x2 = max(n.pos[0] + n.signature.size[0] for n in sec.nodes)
        y2 = max(n.pos[1] + n.signature.size[1] for n in sec.nodes)
        left, top = min(xs) - 30, min(ys) - 70
        return [left, top, x2 - left + 30, y2 - top + 30]

    # -- node serialisation ------------------------------------------------

    def _serialise_node(
        self, node: NodeHandle, link_ids_in: dict[str, int], link_ids_out: dict[int, list[int]]
    ) -> dict[str, Any]:
        """One node record.

        ``inputs`` lists non-widget inputs always, and widget-backed inputs only
        when they are linked — which is what the serialised format does, and what
        makes the slot indices line up.
        """
        inputs: list[dict[str, Any]] = []
        for slot in node.signature.inputs:
            linked = slot.name in link_ids_in
            if slot.widget and not linked:
                continue
            record: dict[str, Any] = {"name": slot.name, "type": slot.type}
            if slot.widget:
                record["widget"] = {"name": slot.name}
            record["link"] = link_ids_in.get(slot.name)
            inputs.append(record)

        outputs = [
            {
                "name": slot.name,
                "type": slot.type,
                "links": link_ids_out.get(index) or None,
            }
            for index, slot in enumerate(node.signature.outputs)
        ]

        record = {
            "id": node.id,
            "type": node.type,
            "pos": list(node.pos),
            "size": list(node.signature.size),
            "flags": {},
            "order": self._nodes.index(node),
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "properties": {"Node name for S&R": node.type},
            "widgets_values": list(node.widget_values),
        }
        if node.title:
            record["title"] = node.title
        if not node.signature.inputs:
            record.pop("inputs")
        if not node.signature.outputs:
            record.pop("outputs")
        if not node.signature.widgets:
            record["widgets_values"] = []
        return record

    def _resolve_links(
        self, nodes: list[NodeHandle]
    ) -> tuple[list[tuple[int, int, int, int, int, str]], dict[int, dict[str, int]], dict[int, dict[int, list[int]]]]:
        """Allocate link ids for every wire between ``nodes``.

        Returns the link tuples plus per-node lookups so the node records can
        reference them. Links whose origin is outside ``nodes`` are skipped;
        the subgraph packer handles those separately.
        """
        included = {n.id for n in nodes}
        links: list[tuple[int, int, int, int, int, str]] = []
        inbound: dict[int, dict[str, int]] = {n.id: {} for n in nodes}
        outbound: dict[int, dict[int, list[int]]] = {n.id: {} for n in nodes}

        for node in nodes:
            for input_name, port in node.links_in.items():
                if port.node.id not in included:
                    continue
                link_id = self._next_link_id()
                target_slot = self._serialised_input_index(node, input_name)
                links.append(
                    (link_id, port.node.id, port.index, node.id, target_slot, port.type)
                )
                inbound[node.id][input_name] = link_id
                outbound[port.node.id].setdefault(port.index, []).append(link_id)
        return links, inbound, outbound

    def _serialised_input_index(self, node: NodeHandle, input_name: str) -> int:
        """Index of ``input_name`` in the node's *serialised* inputs list.

        Unlinked widget inputs are omitted from that list, so this is not the
        same as the signature index.
        """
        index = 0
        for slot in node.signature.inputs:
            linked = slot.name in node.links_in
            if slot.widget and not linked:
                continue
            if slot.name == input_name:
                return index
            index += 1
        raise KeyError(f"{node.type} has no serialised input {input_name!r}.")

    # -- flat layout -------------------------------------------------------

    def _build_flat(self) -> dict[str, Any]:
        links, inbound, outbound = self._resolve_links(self._nodes)
        nodes = [
            self._serialise_node(n, inbound[n.id], outbound[n.id]) for n in self._nodes
        ]
        groups = [
            {
                "id": index + 1,
                "title": sec.name,
                "bounding": self._section_bounds(sec),
                "color": sec.color,
                "font_size": 24,
                "flags": {},
            }
            for index, sec in enumerate(self._sections)
            if sec.nodes
        ]
        return self._envelope(nodes, [list(link) for link in links], groups, [])

    # -- subgraph layout ---------------------------------------------------

    def _build_subgraphs(self) -> dict[str, Any]:
        """Pack each section into a subgraph definition.

        A wire that crosses a section boundary becomes three things: a link to
        the producing subgraph's output proxy, a link between the two subgraph
        instances on the parent canvas, and a link from the consuming
        subgraph's input proxy. Sections with no cross-boundary wires stay
        entirely internal.
        """
        definitions: list[dict[str, Any]] = []
        #: (section index) -> parent-canvas node id
        instance_ids: dict[int, int] = {}
        #: (producing node id, output index) -> (section index, output slot)
        produced: dict[tuple[int, int], tuple[int, int]] = {}
        section_of = {n.id: self._sections.index(n.section) for n in self._nodes}

        # Pass 1: work out which ports each section must export.
        exports: list[list[Port]] = [[] for _ in self._sections]
        for node in self._nodes:
            for port in node.links_in.values():
                src_section = section_of[port.node.id]
                if src_section == section_of[node.id]:
                    continue
                key = (port.node.id, port.index)
                if key not in produced:
                    produced[key] = (src_section, len(exports[src_section]))
                    exports[src_section].append(port)

        # Pass 2: work out which ports each section must import, in order.
        imports: list[list[tuple[int, str, Port]]] = [[] for _ in self._sections]
        for node in self._nodes:
            for input_name, port in node.links_in.items():
                if section_of[port.node.id] == section_of[node.id]:
                    continue
                imports[section_of[node.id]].append((node.id, input_name, port))

        # Pass 3: emit one definition per section.
        #: section index -> its definition, so pass 4 never has to index by position.
        definition_of: dict[int, dict[str, Any]] = {}
        for index, sec in enumerate(self._sections):
            if not sec.nodes:
                continue
            instance_ids[index] = self._next_node_id()
            definition = self._subgraph_definition(index, sec, exports, imports, produced)
            definition_of[index] = definition
            definitions.append(definition)

        # Pass 4: the parent canvas — one node per section, wired between them.
        parent_nodes: list[dict[str, Any]] = []
        parent_links: list[list[Any]] = []
        parent_outbound: dict[int, dict[int, list[int]]] = {i: {} for i in instance_ids.values()}
        parent_inbound: dict[int, list[int | None]] = {
            i: [None] * len(imports[si]) for si, i in instance_ids.items()
        }

        for section_index, instance_id in instance_ids.items():
            for slot, (_, _, port) in enumerate(imports[section_index]):
                src_section, src_slot = produced[(port.node.id, port.index)]
                src_instance = instance_ids[src_section]
                link_id = self._next_link_id()
                parent_links.append(
                    [link_id, src_instance, src_slot, instance_id, slot, port.type]
                )
                parent_outbound[src_instance].setdefault(src_slot, []).append(link_id)
                parent_inbound[instance_id][slot] = link_id

        for position, (section_index, instance_id) in enumerate(instance_ids.items()):
            sec = self._sections[section_index]
            parent_nodes.append({
                "id": instance_id,
                "type": definition_of[section_index]["id"],
                "pos": [_PAD + position * _COL_WIDTH, _PAD],
                "size": [360, 120],
                "flags": {},
                "order": position,
                "mode": 0,
                "title": sec.name,
                "inputs": [
                    {
                        "name": _import_name(imports[section_index], slot),
                        "type": port.type,
                        "link": parent_inbound[instance_id][slot],
                    }
                    for slot, (_, _, port) in enumerate(imports[section_index])
                ],
                "outputs": [
                    {
                        "name": _export_name(port, slot),
                        "type": port.type,
                        "links": parent_outbound[instance_id].get(slot) or None,
                    }
                    for slot, port in enumerate(exports[section_index])
                ],
                "properties": {"Node name for S&R": sec.name},
                "widgets_values": [],
            })

        return self._envelope(parent_nodes, parent_links, [], definitions)

    def _subgraph_definition(
        self,
        index: int,
        sec: Section,
        exports: list[list[Port]],
        imports: list[list[tuple[int, str, Port]]],
        produced: dict[tuple[int, int], tuple[int, int]],
    ) -> dict[str, Any]:
        links, inbound, outbound = self._resolve_links(sec.nodes)
        link_records: list[dict[str, Any]] = [
            {
                "id": lid,
                "origin_id": oid,
                "origin_slot": oslot,
                "target_id": tid,
                "target_slot": tslot,
                "type": ltype,
            }
            for lid, oid, oslot, tid, tslot, ltype in links
        ]

        # Inputs: one per imported wire, fed from the -10 proxy.
        input_records: list[dict[str, Any]] = []
        for slot, (target_id, input_name, port) in enumerate(imports[index]):
            link_id = self._next_link_id()
            target = next(n for n in sec.nodes if n.id == target_id)
            link_records.append({
                "id": link_id,
                "origin_id": SUBGRAPH_INPUT_ID,
                "origin_slot": slot,
                "target_id": target_id,
                "target_slot": self._serialised_input_index(target, input_name),
                "type": port.type,
            })
            inbound[target_id][input_name] = link_id
            input_records.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.graph_id}/{sec.name}/in/{slot}")),
                "name": _import_name(imports[index], slot),
                "type": port.type,
                "linkIds": [link_id],
            })

        # Outputs: one per exported port, feeding the -20 proxy.
        output_records: list[dict[str, Any]] = []
        for slot, port in enumerate(exports[index]):
            link_id = self._next_link_id()
            link_records.append({
                "id": link_id,
                "origin_id": port.node.id,
                "origin_slot": port.index,
                "target_id": SUBGRAPH_OUTPUT_ID,
                "target_slot": slot,
                "type": port.type,
            })
            outbound[port.node.id].setdefault(port.index, []).append(link_id)
            output_records.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.graph_id}/{sec.name}/out/{slot}")),
                "name": _export_name(port, slot),
                "type": port.type,
                "linkIds": [link_id],
            })

        node_records = [
            self._serialise_node(n, inbound[n.id], outbound[n.id]) for n in sec.nodes
        ]
        bounds = self._section_bounds(sec)
        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.graph_id}/{sec.name}")),
            "version": 1,
            "state": {
                "lastGroupId": 0,
                "lastNodeId": self._node_id,
                "lastLinkId": self._link_id,
                "lastRerouteId": 0,
            },
            "revision": 0,
            "config": {},
            "name": sec.name,
            "inputNode": {
                "id": SUBGRAPH_INPUT_ID,
                "bounding": [bounds[0] - 220, bounds[1], 180, max(60, 24 * len(input_records))],
            },
            "outputNode": {
                "id": SUBGRAPH_OUTPUT_ID,
                "bounding": [bounds[0] + bounds[2] + 60, bounds[1], 160,
                             max(60, 24 * len(output_records))],
            },
            "inputs": input_records,
            "outputs": output_records,
            "widgets": [],
            "nodes": node_records,
            "groups": [],
            "links": link_records,
            "extra": {},
        }

    # -- envelope ----------------------------------------------------------

    def _envelope(
        self,
        nodes: list[dict[str, Any]],
        links: list[Any],
        groups: list[dict[str, Any]],
        subgraphs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "id": self.graph_id,
            "revision": 0,
            "last_node_id": self._node_id,
            "last_link_id": self._link_id,
            "nodes": nodes,
            "links": links,
            "groups": groups,
            "definitions": {"subgraphs": subgraphs},
            "config": {},
            "extra": {"ltxdirector": {"title": self.title}},
            "version": 0.4,
        }


def _export_name(port: Port, slot: int) -> str:
    name = port.node.signature.outputs[port.index].name
    return name or f"out_{slot}"


def _import_name(entries: list[tuple[int, str, Port]], slot: int) -> str:
    """A readable, unique name for a subgraph input.

    Two different nodes can want an input called ``vae``; suffixing keeps the
    names distinct without making them cryptic.
    """
    _, input_name, _ = entries[slot]
    seen = sum(1 for i in range(slot) if entries[i][1] == input_name)
    return input_name if seen == 0 else f"{input_name}_{seen}"

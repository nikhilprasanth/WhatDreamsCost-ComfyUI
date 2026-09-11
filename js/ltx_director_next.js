/**
 * LTX Director Next — ComfyUI extension entry point.
 *
 * This file does one thing: attach the editor to the `LTXDirectorProject` node.
 * Everything else lives in `js/director/`, imported as modules.
 *
 * Two things it deliberately does *not* do, both of which the previous Director
 * did and paid for:
 *
 * - It does not read the graph on every canvas repaint. `onDrawForeground` used
 *   to traverse links and write to the DOM sixty times a second; here the only
 *   per-frame work in the whole editor is the timeline's canvas, and that only
 *   redraws when something changed.
 * - It does not keep state in three places. The project lives in one widget,
 *   the store owns it, and the views are derived.
 */

import { app } from "../../scripts/app.js";
import { Shell } from "./director/ui/shell.js";
import { Store } from "./director/state/store.js";

const NODE = "LTXDirectorProject";
const STYLE_ID = "ltxd-styles";
const MIN_SIZE = [980, 720];

/** Load the stylesheet once, from beside this file. */
function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const href = new URL("./director/styles.css", import.meta.url).href;
  document.head.append(
    Object.assign(document.createElement("link"), { id: STYLE_ID, rel: "stylesheet", href })
  );
}

/** The hidden widget holding the serialised project. */
function projectWidget(node) {
  return node.widgets?.find((w) => w.name === "project") ?? null;
}

/**
 * Hide a widget without removing it.
 *
 * It has to stay in `node.widgets` so ComfyUI serialises it — the project is
 * the node's entire state. Both the litegraph and Vue node renderers are
 * catered for, because a user may be on either.
 */
function hideWidget(widget) {
  if (!widget || widget._ltxdHidden) return;
  widget._ltxdHidden = true;
  widget.hidden = true;
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
  if (widget.element) widget.element.style.display = "none";
}

app.registerExtension({
  name: "WhatDreamsCost.LTXDirectorNext",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      ensureStyles();
      attach(this);
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this._ltxd?.shell.destroy();
      this._ltxd = null;
      onRemoved?.apply(this, arguments);
    };

    // Loading a saved workflow rewrites the widget after creation, so the store
    // has to be told. Without this the editor shows a blank project over a
    // populated one.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      onConfigure?.apply(this, arguments);
      const widget = projectWidget(this);
      if (this._ltxd && widget) {
        this._ltxd.store.loadJSON(widget.value, { label: "load", record: false });
        this._ltxd.shell.renderBars();
        this._ltxd.shell.renderAdvanced();
        this._ltxd.shell.prompt.render();
      }
    };
  },
});

function attach(node) {
  const widget = projectWidget(node);
  hideWidget(widget);

  const store = new Store({
    persist: (json) => {
      if (widget) widget.value = json;
      // Mark the graph dirty so ComfyUI knows there is something to save.
      node.graph?.setDirtyCanvas?.(true, false);
    },
  });

  const shell = new Shell(store, { node });

  const container = document.createElement("div");
  container.className = "ltxd-mount";
  container.append(shell.root);

  const domWidget = node.addDOMWidget(NODE, "ltxdirector", container, {
    // The project is the `project` widget's value; this one must not serialise
    // a second copy of it into the workflow.
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => 640,
  });

  node._ltxd = { store, shell, container, domWidget };

  if (node.size[0] < MIN_SIZE[0]) node.size[0] = MIN_SIZE[0];
  if (node.size[1] < MIN_SIZE[1]) node.size[1] = MIN_SIZE[1];

  store.loadJSON(widget?.value ?? "", { label: "load", record: false });

  // The editor writes on a debounce; make sure nothing is in flight when the
  // page goes away or the workflow is serialised.
  const flush = () => store.flush();
  window.addEventListener("beforeunload", flush);
  const onSerialize = node.onSerialize;
  node.onSerialize = function (info) {
    flush();
    onSerialize?.apply(this, arguments);
  };
}

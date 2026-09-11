/**
 * Tiny DOM helpers.
 *
 * Deliberately small. The editor builds a few hundred elements, not a few
 * thousand, and the timeline — the one part that could be thousands — is a
 * single canvas rather than DOM. So there is no need for a framework here, and
 * the cost of one would be paid on every ComfyUI page load.
 */

/**
 * Build an element.
 *
 * `el("div.row", { title: "…" }, child, "text")` — the tag string carries
 * classes so the common case is one argument.
 */
export function el(spec, props = null, ...children) {
  const [tag, ...classes] = String(spec).split(".");
  const node = document.createElement(tag || "div");
  if (classes.length) node.className = classes.join(" ");

  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = [node.className, value].filter(Boolean).join(" ");
      else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
      else if (key === "dataset") Object.assign(node.dataset, value);
      else if (key.startsWith("on") && typeof value === "function") {
        node.addEventListener(key.slice(2).toLowerCase(), value);
      } else if (key === "html") node.innerHTML = value;
      else if (key in node) node[key] = value;
      else node.setAttribute(key, value);
    }
  }

  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Replace an element's children in one go. */
export function fill(node, ...children) {
  node.replaceChildren(
    ...children.flat().filter((c) => c !== null && c !== undefined && c !== false)
      .map((c) => (c instanceof Node ? c : document.createTextNode(String(c))))
  );
  return node;
}

/** A labelled row, the editor's basic unit of layout. */
export function field(label, control, hint = "") {
  return el("label.ltxd-field", { title: hint },
    el("span.ltxd-field-label", null, label),
    control
  );
}

/** A `<select>` bound to a value, from `[value, label]` pairs or a plain array. */
export function select(options, value, onChange, { title = "" } = {}) {
  const node = el("select.ltxd-select", { title, onchange: (e) => onChange(e.target.value) });
  for (const option of options) {
    const [optionValue, label] = Array.isArray(option) ? option : [option, option];
    node.append(el("option", { value: optionValue, selected: optionValue === value }, label));
  }
  return node;
}

/** A number input that reports parsed values, and never NaN. */
export function number(value, onChange, { min, max, step = 1, title = "" } = {}) {
  return el("input.ltxd-number", {
    type: "number", value, min, max, step, title,
    onchange: (e) => {
      const parsed = step % 1 === 0 ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
      if (!Number.isNaN(parsed)) onChange(parsed);
      else e.target.value = value;
    },
  });
}

/** A checkbox with a label, returned as one element. */
export function toggle(label, checked, onChange, { title = "" } = {}) {
  return el("label.ltxd-toggle", { title },
    el("input", { type: "checkbox", checked, onchange: (e) => onChange(e.target.checked) }),
    el("span", null, label)
  );
}

/** A button. `variant` is "primary", "ghost" or omitted. */
export function button(label, onClick, { variant = "", title = "", disabled = false } = {}) {
  return el(`button.ltxd-button${variant ? "." + variant : ""}`, {
    type: "button", title, disabled, onclick: onClick,
  }, label);
}

/**
 * A collapsible section.
 *
 * Progressive disclosure is the editor's main defence against the wall of
 * widgets the previous Director became, so this is used for everything beyond
 * the four things a beginner needs.
 */
export function section(title, { open = false, onToggle = null, actions = null } = {}) {
  const body = el("div.ltxd-section-body");
  const chevron = el("span.ltxd-chevron", null, "▾");
  const header = el("div.ltxd-section-header", {
    onclick: (event) => {
      if (event.target.closest(".ltxd-section-actions")) return;
      root.classList.toggle("open");
      onToggle?.(root.classList.contains("open"));
    },
  }, chevron, el("span.ltxd-section-title", null, title));

  if (actions) header.append(el("span.ltxd-section-actions", null, actions));

  const root = el(`div.ltxd-section${open ? ".open" : ""}`, null, header, body);
  root.body = body;
  return root;
}

/** Debounce, with a `flush` for "save before the page unloads". */
export function debounce(fn, ms = 200) {
  let timer = null;
  let pending = null;
  const wrapped = (...args) => {
    pending = args;
    clearTimeout(timer);
    timer = setTimeout(() => { timer = null; const a = pending; pending = null; fn(...a); }, ms);
  };
  wrapped.flush = () => {
    if (timer) { clearTimeout(timer); timer = null; }
    if (pending) { const a = pending; pending = null; fn(...a); }
  };
  wrapped.cancel = () => { clearTimeout(timer); timer = null; pending = null; };
  return wrapped;
}

/** Drag handling that cleans up after itself even when the pointer leaves. */
export function drag(target, { onStart, onMove, onEnd }) {
  target.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    target.setPointerCapture?.(event.pointerId);
    const context = onStart?.(event) ?? {};
    if (context === false) return;

    const move = (e) => onMove?.(e, context);
    const up = (e) => {
      target.releasePointerCapture?.(event.pointerId);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      onEnd?.(e, context);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
  });
}

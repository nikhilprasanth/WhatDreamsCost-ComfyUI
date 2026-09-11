/**
 * The prompt editor.
 *
 * Three modes, and the important property is that switching between them never
 * changes what gets encoded without the user seeing it:
 *
 *   Simple    one box. What you type is what is encoded.
 *   Director  structured sections plus camera and lens controls, compiled into
 *             prose — because LTX-2.5's encoder was trained on prose, not tags.
 *   Expert    one box again, identical in behaviour to Simple. The difference
 *             is what the panel shows, not what the compiler does.
 *
 * The compiled prompt is always visible, and always comes from the server — the
 * same function the graph will run. Prompt manipulation is never hidden.
 */

import { validate, vocabulary } from "../net/api.js";
import { button, el, field, section, select, toggle } from "../util/dom.js";

const MODES = [
  ["simple", "Simple"],
  ["director", "Director"],
  ["expert", "Expert"],
];

const SECTIONS = [
  ["subject", "Subject", "Who or what is in the shot, described once."],
  ["scene", "Scene", "Where it is, and what the place looks like."],
  ["action", "Action", "What happens, in the order it happens."],
  ["camera", "Camera", "Your own camera note. Overrides the camera controls below."],
  ["acting", "Performance", "Expression, gesture, how a line is delivered."],
  ["lighting", "Lighting", "Sources, quality, colour."],
  ["sound", "Sound", "Ambience, music, effects."],
  ["dialogue", "Dialogue", "Quote it exactly, in the language it is spoken."],
  ["technical", "Technical", "Film stock, grain, aspect — anything about the image itself."],
];

export class PromptPanel {
  constructor(store, { onChange } = {}) {
    this.store = store;
    this.onChange = onChange ?? (() => {});
    this.vocabulary = null;

    this.compiled = el("pre.ltxd-compiled", { title: "Exactly what the text encoder receives." });
    this.body = el("div.ltxd-prompt-body");
    this.modeBar = el("div.ltxd-segmented");

    this.root = el("div.ltxd-prompt", null,
      el("div.ltxd-prompt-head", null,
        el("span.ltxd-label", null, "Prompt"),
        this.modeBar,
      ),
      this.body,
      this._preview(),
    );

    this._loadVocabulary();
    store.subscribe((_, reason) => {
      if (reason === "undo" || reason === "redo" || reason === "load") this.render();
      this.refreshPreview();
    });
  }

  async _loadVocabulary() {
    try {
      this.vocabulary = await vocabulary();
    } catch {
      // Director mode degrades to free text rather than failing: the sections
      // still work, only the dropdowns are missing.
      this.vocabulary = null;
    }
    this.render();
  }

  _preview() {
    const copy = button("Copy", async () => {
      try {
        await navigator.clipboard.writeText(this.compiled.textContent ?? "");
        copy.textContent = "Copied";
        setTimeout(() => { copy.textContent = "Copy"; }, 1200);
      } catch {
        copy.textContent = "Copy failed";
        setTimeout(() => { copy.textContent = "Copy"; }, 1600);
      }
    }, { variant: "ghost", title: "Copy the compiled prompt." });

    this.previewSection = section("Compiled prompt", {
      open: false,
      actions: copy,
      onToggle: (open) => this._rememberSection("prompt-preview", open),
    });
    this.previewSection.body.append(this.compiled);
    return this.previewSection;
  }

  _rememberSection(name, open) {
    const expanded = new Set(this.store.ui.expanded);
    if (open) expanded.add(name); else expanded.delete(name);
    this.store.setUi({ expanded: [...expanded] });
  }

  render() {
    this._renderModeBar();
    const mode = this.store.prompt.mode;
    this.body.replaceChildren(
      mode === "director" ? this._directorEditor() : this._plainEditor(mode),
      this._negativeEditor(),
    );
    this.previewSection.classList.toggle(
      "open", this.store.ui.expanded.includes("prompt-preview")
    );
    this.refreshPreview();
  }

  _renderModeBar() {
    this.modeBar.replaceChildren(...MODES.map(([value, label]) => {
      const active = this.store.prompt.mode === value;
      return el(`button.ltxd-segment${active ? ".active" : ""}`, {
        type: "button",
        title: value === "expert"
          ? "The same encoding as Simple; the panel just gets out of the way."
          : value === "director"
            ? "Structured sections, compiled into prose."
            : "One box. What you type is what is encoded.",
        onclick: () => {
          this.store.update("prompt mode", (state) => { state.prompt.mode = value; });
          this.render();
        },
      }, label);
    }));
  }

  _plainEditor(mode) {
    const textarea = el("textarea.ltxd-textarea", {
      rows: mode === "expert" ? 10 : 5,
      placeholder: "Describe the shot: who, where, what happens, how it is filmed.",
      value: this.store.prompt.raw,
      oninput: (event) => {
        this.store.update("prompt", (state) => { state.prompt.raw = event.target.value; },
          { coalesce: true });
      },
    });
    return el("div.ltxd-prompt-plain", null, textarea);
  }

  _negativeEditor() {
    return field("Negative", el("textarea.ltxd-textarea.small", {
      rows: 2,
      placeholder: "What to avoid — blurry, low contrast, washed out.",
      value: this.store.prompt.negative,
      oninput: (event) => {
        this.store.update("negative prompt", (state) => {
          state.prompt.negative = event.target.value;
        }, { coalesce: true });
      },
    }), "Encoded separately as the negative conditioning.");
  }

  _directorEditor() {
    const wrap = el("div.ltxd-prompt-director");

    for (const [key, label, hint] of SECTIONS) {
      const value = this.store.prompt.sections[key] ?? "";
      wrap.append(field(label, el("textarea.ltxd-textarea.small", {
        rows: value.length > 90 ? 3 : 2,
        placeholder: hint,
        value,
        title: hint,
        oninput: (event) => {
          this.store.update(`prompt ${key}`, (state) => {
            state.prompt.sections[key] = event.target.value;
          }, { coalesce: true });
        },
      }), hint));
    }

    wrap.append(this._cameraControls(), this._lensControls());
    return wrap;
  }

  _cameraControls() {
    const camera = this.store.prompt.camera;
    const moves = this.vocabulary?.camera_moves ?? {};
    const intensities = this.vocabulary?.camera_intensity ?? {};

    const options = [["", "— not specified —"], ...Object.keys(moves)
      .filter((key) => key && key !== "custom")
      .map((key) => [key, humanise(key)]), ["custom", "Custom…"]];

    const set = (changes) => {
      this.store.update("camera", (state) => { Object.assign(state.prompt.camera, changes); });
      this.render();
    };

    const rows = [
      field("Camera move", select(options, camera.move, (value) => set({ move: value }),
        { title: "Compiled into a sentence, not forced on the model." })),
    ];

    if (camera.move && camera.move !== "custom") {
      rows.push(field("Intensity", select(
        Object.keys(intensities).filter((k) => k !== "custom").map((k) => [k, humanise(k)]),
        camera.intensity, (value) => set({ intensity: value })
      )));
    }

    if (camera.move === "custom" || camera.custom) {
      rows.push(field("Camera note", el("input.ltxd-input", {
        type: "text",
        placeholder: "the camera spirals through a doorway",
        value: camera.custom,
        oninput: (event) => {
          this.store.update("camera note", (state) => {
            state.prompt.camera.custom = event.target.value;
          }, { coalesce: true });
        },
      })));
    }

    const note = this.store.prompt.sections.camera.trim()
      ? el("p.ltxd-note", null,
          "The Camera section above is filled in, so it is used and these controls are ignored.")
      : null;

    const panel = section("Camera", {
      open: this.store.ui.expanded.includes("camera"),
      onToggle: (open) => this._rememberSection("camera", open),
    });
    panel.body.append(...rows, note);
    return panel;
  }

  _lensControls() {
    const lens = this.store.prompt.lens;
    const vocab = this.vocabulary ?? {};
    const set = (changes) => {
      this.store.update("lens", (state) => { Object.assign(state.prompt.lens, changes); });
    };

    const dropdown = (key, table, label) => field(label, select(
      [["", "— not specified —"], ...Object.keys(table ?? {}).map((k) => [k, humanise(k)])],
      lens[key], (value) => set({ [key]: value })
    ));

    const panel = section("Framing", {
      open: this.store.ui.expanded.includes("lens"),
      onToggle: (open) => this._rememberSection("lens", open),
    });
    panel.body.append(
      dropdown("shot_size", vocab.shot_sizes, "Shot size"),
      dropdown("angle", vocab.camera_angles, "Angle"),
      dropdown("height", vocab.camera_heights, "Camera height"),
      dropdown("dof", vocab.depth_of_field, "Depth of field"),
      field("Focal length", el("input.ltxd-number", {
        type: "number", min: 0, max: 400, step: 1, value: lens.focal_mm,
        title: "0 leaves it unstated.",
        onchange: (event) => set({ focal_mm: parseInt(event.target.value, 10) || 0 }),
      })),
    );
    return panel;
  }

  /**
   * Refresh the compiled preview.
   *
   * Server-side, and debounced by the caller: this is the same function the
   * graph runs, so the preview cannot drift from the render.
   */
  async refreshPreview() {
    if (!this.previewSection.classList.contains("open")) return;
    this._previewAbort?.abort();
    const controller = new AbortController();
    this._previewAbort = controller;
    try {
      const result = await validate(this.store.state, { signal: controller.signal });
      this.compiled.textContent = result.prompt || "(no prompt yet)";
    } catch (error) {
      if (error?.name !== "AbortError") {
        this.compiled.textContent = "(the compiled prompt could not be fetched)";
      }
    }
  }
}

function humanise(key) {
  return key.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}

export { toggle };

/**
 * The Director's chrome.
 *
 * Default view: mode, duration, timeline, references, prompt, Generate. Five
 * things. Everything else — models, quality, audio, relay, takes — is a
 * collapsed section, and its open state lives in the project so it survives a
 * reload.
 *
 * The previous Director showed every control at once, which is how a node ends
 * up 1375 pixels wide with a sidebar. Progressive disclosure is the main defence
 * against that happening again.
 */

import * as api from "../net/api.js";
import { filesFrom, isSupported, pathFor, toMediaEntry, upload } from "../net/media.js";
import { latentFrames, snapFrames, timecode } from "../state/time.js";
import { button, debounce, el, field, number, section, select, toggle } from "../util/dom.js";
import { Inspector } from "./inspector.js";
import { PromptPanel } from "./prompt.js";
import { ReferencePanel } from "./references.js";
import { Timeline } from "./timeline.js";

const MODES = [
  ["t2v", "Text", "Generate from the prompt alone."],
  ["i2v", "Image", "Start from a still and move from there."],
  ["fflf", "First / Last", "Pin the opening and closing frames."],
  ["keyframes", "Keyframes", "Pin any number of frames along the timeline."],
  ["continue", "Continue", "Carry on from the last frame of a previous shot."],
  ["a2v", "Audio", "Follow a soundtrack you supply."],
  ["t2a", "Audio only", "Generate sound with no picture."],
  ["iclora", "Control", "Follow a clip's motion, depth, edges or pose."],
];

/** Features each mode needs, so unavailable modes can be disabled with a reason. */
const MODE_FEATURE = {
  fflf: "mid_timeline_guides",
  keyframes: "mid_timeline_guides",
  i2v: "first_frame",
  continue: "first_frame",
  a2v: "audio_import",
  t2a: "audio_only",
  iclora: "iclora",
};

export class Shell {
  constructor(store, { node } = {}) {
    this.store = store;
    this.node = node;
    this.caps = null;
    this.presets = null;

    this.status = el("div.ltxd-status");
    this.diagnostics = el("div.ltxd-diagnostics");
    this.modeBar = el("div.ltxd-modes");
    this.projectBar = el("div.ltxd-projectbar");
    this.advanced = el("div.ltxd-advanced");

    this.timeline = new Timeline(store, { onSelect: () => this.inspector.render() });
    this.references = new ReferencePanel(store, { onNotify: (n) => this.notify(n) });
    this.prompt = new PromptPanel(store);
    this.inspector = new Inspector(store);

    this.root = el("div.ltxd-root", null,
      el("div.ltxd-topbar", null, this.modeBar, this.projectBar),
      this.timeline.root,
      el("div.ltxd-columns", null,
        el("div.ltxd-col.main", null, this.prompt.root),
        el("div.ltxd-col.side", null,
          this._sectioned("References", "references", this.references.root, true),
          this._sectioned("Selection", "inspector", this.inspector.root, true),
        ),
      ),
      this.advanced,
      this.diagnostics,
      this._actionBar(),
      this.status,
    );

    this._revalidate = debounce(() => this.revalidate(), 400);
    this._bindGlobalDrop();
    this._bindKeys();

    store.subscribe((_, reason) => {
      this.renderBars();
      if (reason !== "ui") this._revalidate();
    });

    this._load();
  }

  destroy() {
    this.timeline.destroy();
    this._revalidate.cancel();
  }

  async _load() {
    try {
      this.caps = await api.capabilities();
    } catch (error) {
      this.notify({ message: error.message, fix: error.fix ?? "" });
    }
    try {
      this.presets = await api.presets();
    } catch {
      this.presets = null;
    }
    this._applySuggestedModels();
    this.renderBars();
    this.renderAdvanced();
    this.revalidate();
  }

  /**
   * Fill in model selections the user has not made.
   *
   * A suggestion, never an override: a new project should be runnable without
   * visiting the Models section, but a chosen model is never replaced.
   */
  _applySuggestedModels() {
    const defaults = this.caps?.defaults;
    if (!defaults) return;
    const models = this.store.models;
    const missing = ["unet", "vae", "audio_vae", "clip"].some((key) => !models[key]);
    if (!missing) return;

    this.store.update("models", (state) => {
      for (const key of ["family", "unet", "vae", "audio_vae", "clip", "enhancer_clip", "upscaler"]) {
        if (!state.models[key] && defaults[key]) state.models[key] = defaults[key];
      }
    }, { cosmetic: true });
  }

  // -- top bars ----------------------------------------------------------

  renderBars() {
    this._renderModes();
    this._renderProjectBar();
  }

  _renderModes() {
    const current = this.store.project.mode;
    this.modeBar.replaceChildren(...MODES.map(([value, label, hint]) => {
      const feature = MODE_FEATURE[value];
      const available = !feature || !this.caps || this.caps.flags?.[feature] !== false;
      const reason = available ? hint : this.caps?.missing?.[feature] ?? hint;

      return el(`button.ltxd-mode${current === value ? ".active" : ""}${available ? "" : ".unavailable"}`, {
        type: "button",
        title: reason,
        disabled: !available,
        onclick: () => this.store.setMode(value),
      }, label);
    }));
  }

  _renderProjectBar() {
    const project = this.store.project;
    const seconds = project.frames / project.fps;

    this.projectBar.replaceChildren(
      field("Duration", el("div.ltxd-time-entry", null,
        number(Number(seconds.toFixed(2)), (value) => {
          this.store.setDuration(snapFrames(Math.round(value * project.fps)));
        }, { min: 0.1, max: 120, step: 0.1 }),
        el("span.ltxd-unit", null, "s"),
        el("span.ltxd-readout", null, `${project.frames} frames`),
      ), "LTX needs 1 plus a multiple of 8 frames, so this snaps upward."),

      field("FPS", number(project.fps, (fps) => {
        this.store.update("fps", (state) => { state.project.fps = fps; });
      }, { min: 1, max: 60, step: 1 })),

      field("Size", el("div.ltxd-size", null,
        number(project.width, (width) => {
          this.store.update("width", (state) => { state.project.width = width; this.store.normalise(); });
        }, { min: 64, max: 3840, step: 32 }),
        el("span.ltxd-x", null, "×"),
        number(project.height, (height) => {
          this.store.update("height", (state) => { state.project.height = height; this.store.normalise(); });
        }, { min: 64, max: 2176, step: 32 }),
      ), "Both must be multiples of 32."),

      field("Seed", el("div.ltxd-seed", null,
        number(this.store.generation.seed, (seed) => {
          this.store.update("seed", (state) => { state.generation.seed = seed; });
        }, { min: 0, max: Number.MAX_SAFE_INTEGER, step: 1 }),
        button("⟳", () => {
          this.store.update("seed", (state) => {
            state.generation.seed = Math.floor(Math.random() * 0xffffffff);
          });
        }, { variant: "ghost", title: "New random seed" }),
      )),
    );
  }

  _sectioned(title, key, body, openByDefault = false) {
    const open = this.store.ui.expanded.includes(key)
      || (openByDefault && !this.store.ui.expanded.length);
    const panel = section(title, {
      open,
      onToggle: (isOpen) => {
        const expanded = new Set(this.store.ui.expanded);
        if (isOpen) expanded.add(key); else expanded.delete(key);
        this.store.setUi({ expanded: [...expanded] });
      },
    });
    panel.body.append(body);
    return panel;
  }

  // -- advanced sections -------------------------------------------------

  renderAdvanced() {
    this.advanced.replaceChildren(
      this._sectioned("Quality", "quality", this._qualityPanel()),
      this._sectioned("Models", "models", this._modelPanel()),
      this._sectioned("Audio", "audio", this._audioPanel()),
      this._sectioned("Prompt Relay", "relay", this._relayPanel()),
      this._sectioned("Takes", "takes", this._takesPanel()),
    );
  }

  _qualityPanel() {
    const generation = this.store.generation;
    const quality = this.presets?.presets?.quality ?? [];
    const vram = this.presets?.presets?.vram ?? [];

    const applyPreset = (name, group) => {
      const preset = group.find((p) => p.name === name);
      if (!preset) return;
      this.store.transaction(`preset ${name}`, (state) => {
        for (const [path, value] of Object.entries(preset.changes)) {
          const parts = path.split(".");
          let target = state;
          for (const part of parts.slice(0, -1)) target = target[part];
          target[parts[parts.length - 1]] = structuredClone(value);
        }
      });
      this.store.normalise();
      // Presets are inspectable by construction: say what moved.
      this.notify({
        level: "info",
        message: `${preset.label}: ${Object.keys(preset.changes).length} settings changed.`,
        fix: preset.description,
      });
      this.renderAdvanced();
      this.renderBars();
    };

    return el("div.ltxd-panel", null,
      quality.length
        ? field("Preset", select(
            [["", "— custom —"], ...quality.map((p) => [p.name, p.label])],
            generation.preset, (name) => name && applyPreset(name, quality),
            { title: quality.map((p) => `${p.label}: ${p.description}`).join("\n") },
          ))
        : null,
      vram.length
        ? field("Memory", select(
            [["", "— choose —"], ...vram.map((p) => [p.name, p.label])],
            "", (name) => name && applyPreset(name, vram),
            { title: "Changes decode tiling only. It never changes how the shot looks." },
          ))
        : null,
      field("Stages", select([[1, "One pass"], [2, "Generate then refine"]].map(
        ([v, l]) => [String(v), l]), String(generation.stages), (value) => {
          this.store.update("stages", (state) => { state.generation.stages = Number(value); });
          this.renderAdvanced();
        }), "Two stages generate small, then upscale and refine."),
      field("Guidance", el("div.ltxd-pair", null,
        number(generation.video_cfg, (v) => this.store.update("cfg", (s) => {
          s.generation.video_cfg = v;
        }), { step: 0.1, min: 0, max: 20, title: "Video" }),
        number(generation.audio_cfg, (v) => this.store.update("cfg", (s) => {
          s.generation.audio_cfg = v;
        }), { step: 0.1, min: 0, max: 20, title: "Audio" }),
      ), "Video and audio guidance. Distilled models want 1.0 for both."),
      field("Guide compression", number(generation.img_compression, (v) => {
        this.store.update("compression", (state) => { state.generation.img_compression = v; });
      }, { min: 0, max: 100, step: 1 }),
        "LTX expects conditioning frames to carry compression artefacts. 18 is the reference value; 0 disables it."),
      field("Save as", el("input.ltxd-input", {
        type: "text", value: generation.save_prefix,
        oninput: (event) => this.store.update("save prefix", (state) => {
          state.generation.save_prefix = event.target.value;
        }, { coalesce: true }),
      })),
    );
  }

  _modelPanel() {
    const models = this.store.models;
    const available = this.caps?.models ?? {};
    const pick = (key, options, label, hint = "") => field(label, select(
      [["", "— none —"], ...(options ?? []).map((n) => [n, n])],
      models[key] ?? "", (value) => {
        this.store.update("models", (state) => { state.models[key] = value || null; });
      }, { title: hint },
    ), hint);

    return el("div.ltxd-panel", null,
      pick("unet", available.transformers, "Transformer"),
      pick("clip", available.encoders, "Text encoder"),
      pick("vae", available.video_vaes, "Video VAE"),
      pick("audio_vae", available.audio_vaes, "Audio VAE"),
      pick("upscaler", available.upscalers, "Spatial upscaler", "Needed for the refine stage."),
      pick("enhancer_clip", available.enhancers, "Prompt enhancer",
        "A small Gemma-4 encoder that expands a short prompt."),
      field("Enhance prompt", toggle("Expand the prompt before encoding",
        this.store.prompt.enhance.enabled, (enabled) => {
          this.store.update("enhance", (state) => { state.prompt.enhance.enabled = enabled; });
        }), "Adds a model load. The expanded text is shown, never hidden."),
      this.caps && !this.caps.available
        ? el("p.ltxd-note", null, "Models could not be listed; ComfyUI did not answer.")
        : null,
    );
  }

  _audioPanel() {
    const audio = this.store.audio;
    return el("div.ltxd-panel", null,
      field("Audio", select([
        ["generate", "Generate with the video"],
        ["import", "Use a file I supply"],
        ["inpaint", "Fill gaps around my clips"],
        ["mute", "No audio"],
      ], audio.mode, (mode) => {
        this.store.update("audio mode", (state) => {
          state.audio.mode = mode;
          state.audio.enabled = mode !== "mute";
        });
        this.renderAdvanced();
      })),
      audio.mode !== "mute" && audio.mode !== "generate"
        ? button("Add audio file…", () => this._pickAudio(), { variant: "ghost" })
        : null,
      ...audio.clips.map((clip) => {
        const entry = this.store.mediaFor(clip.media);
        return el("div.ltxd-audio-row", null,
          el("span", null, entry?.filename ?? "missing file"),
          button("✕", () => this.store.removeAudioClip(clip.id), { variant: "ghost" }),
        );
      }),
      audio.mode === "inpaint"
        ? el("p.ltxd-note", null,
            "Gap filling is approximated: the compiled graph generates across the whole "
            + "track. Use “Use a file I supply” to keep your audio exactly as it is.")
        : null,
    );
  }

  _relayPanel() {
    const relay = this.store.relay;
    const populated = this.store.segments.filter((s) => s.text.trim()).length;

    return el("div.ltxd-panel", null,
      field("Prompt Relay", toggle("Let each region drive its own stretch",
        relay.enabled, (enabled) => {
          this.store.update("relay", (state) => { state.relay.enabled = enabled; });
        }), "One sampling pass, with cross-attention biased toward each region's words."),
      field("Boundary sharpness", el("div.ltxd-slider-row", null,
        el("input.ltxd-slider", {
          type: "range", min: 0.001, max: 0.9, step: 0.001, value: relay.epsilon,
          oninput: (event) => this.store.update("relay epsilon", (state) => {
            state.relay.epsilon = parseFloat(event.target.value);
          }, { coalesce: true }),
        }),
        el("span.ltxd-readout", null, relay.epsilon.toFixed(3)),
      ), "Low values cut sharply between regions; high values blend them."),
      el("p.ltxd-note", null, populated > 1
        ? `${populated} regions over ${latentFrames(this.store.project.frames)} latent frames.`
        : "Double-click the prompt track on the timeline to add a region."),
    );
  }

  _takesPanel() {
    const takes = this.store.takes ?? [];
    if (!takes.length) {
      return el("p.ltxd-note", null, "Takes appear here as you generate.");
    }
    return el("div.ltxd-takes", null, ...takes.map((take) => el("div.ltxd-take", null,
      button(take.starred ? "★" : "☆", () => {
        this.store.update("take", (state) => {
          const found = state.takes.find((t) => t.id === take.id);
          if (found) found.starred = !found.starred;
        }, { cosmetic: true });
        this.renderAdvanced();
      }, { variant: "ghost", title: take.starred ? "Unstar" : "Star this take" }),
      el("span.ltxd-take-seed", null, `seed ${take.seed}`),
      el("span.ltxd-take-time", null, take.created ?? ""),
      button("Reuse seed", () => {
        this.store.update("seed", (state) => {
          state.generation.seed = take.seed;
          state.generation.seed_mode = "fixed";
        });
        this.renderBars();
      }, { variant: "ghost", title: "Put this take's seed back in the seed field." }),
    )));
  }

  async _pickAudio() {
    const input = el("input", { type: "file", accept: "audio/*,video/*", style: { display: "none" } });
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        const entry = toMediaEntry(await upload(file));
        const mediaId = this.store.addMedia(entry);
        this.store.addAudioClip({ media: mediaId, start: 0, length: this.store.project.frames });
        this.renderAdvanced();
        this._loadPeaks(mediaId, entry);
      } catch (error) {
        this.notify({ message: error.message, fix: error.fix ?? "" });
      }
    };
    input.click();
  }

  async _loadPeaks(mediaId, entry) {
    try {
      const result = await api.audioPeaks(pathFor(entry));
      if (result.peaks?.length) this.timeline.setPeaks(mediaId, result.peaks);
    } catch {
      // A waveform is a nicety; its absence is not worth telling anyone about.
    }
  }

  // -- actions -----------------------------------------------------------

  _actionBar() {
    this.generateButton = button("Generate", () => this.generate(), { variant: "primary" });
    return el("div.ltxd-actions", null,
      this.generateButton,
      button("Compile workflow", () => this.compile(), {
        variant: "ghost",
        title: "Write the native LTX graph to the output folder, with no Director node in it.",
      }),
      button("Save project", () => this.save(), { variant: "ghost" }),
      button("Undo", () => { this.store.undo(); this.renderAdvanced(); }, { variant: "ghost" }),
      button("Redo", () => { this.store.redo(); this.renderAdvanced(); }, { variant: "ghost" }),
    );
  }

  async generate() {
    this.store.flush();
    this.generateButton.disabled = true;
    this.generateButton.textContent = "Queueing…";
    try {
      const result = await api.queue(this.store.state);
      this.store.recordTake({
        seed: this.store.generation.seed,
        created: new Date().toLocaleTimeString(),
        settings_digest: result.digest ?? "",
      });
      this.notify({ level: "info", message: "Queued. Watch ComfyUI's queue for progress." });
      this._showDiagnostics(result.diagnostics ?? []);
      this.renderAdvanced();
    } catch (error) {
      this.notify({ message: error.message, fix: error.fix ?? "" });
    } finally {
      this.generateButton.disabled = false;
      this.generateButton.textContent = "Generate";
    }
  }

  async compile() {
    this.store.flush();
    try {
      const result = await api.compile(this.store.state, { layout: "flat" });
      const nodes = result.workflow?.nodes?.length ?? 0;
      this.notify({
        level: "info",
        message: `Compiled to ${nodes} native nodes.`,
        fix: "Opening it on the canvas…",
      });
      this._openOnCanvas(result.workflow);
      this._showDiagnostics(result.diagnostics ?? []);
    } catch (error) {
      this.notify({ message: error.message, fix: error.fix ?? "" });
    }
  }

  /** Load a compiled workflow onto the ComfyUI canvas. */
  async _openOnCanvas(workflow) {
    if (!workflow) return;
    try {
      const { app } = await import("../../../../scripts/app.js");
      await app.loadGraphData(structuredClone(workflow), true, true, "LTX Director shot");
    } catch (error) {
      this.notify({
        message: "The graph compiled but could not be opened on the canvas.",
        fix: "It was still written to the output folder; open it from there.",
      });
      console.error("[LTX Director]", error);
    }
  }

  async save() {
    const name = prompt("Save this shot as:", this.store.state.meta.name ?? "shot");
    if (!name) return;
    try {
      await api.saveProject(this.store.state, name);
      this.store.update("meta", (state) => { state.meta.name = name; }, { cosmetic: true });
      this.notify({ level: "info", message: `Saved as ${name}.` });
    } catch (error) {
      this.notify({ message: error.message, fix: error.fix ?? "" });
    }
  }

  // -- validation and messages ------------------------------------------

  async revalidate() {
    try {
      const result = await api.validate(this.store.state);
      this._showDiagnostics(result.diagnostics ?? []);
      this.generateButton.disabled = !result.ok;
      this.generateButton.title = result.ok
        ? "Compile this shot and queue it."
        : "Fix the problems above first.";
    } catch {
      // A failed validate must not block editing; the server may just be busy.
      this.diagnostics.replaceChildren();
    }
  }

  _showDiagnostics(diagnostics) {
    const shown = diagnostics.filter((d) => d.level !== "info" || d.fix);
    this.diagnostics.replaceChildren(...shown.map((d) => el(`div.ltxd-diag.${d.level}`, null,
      el("span.ltxd-diag-message", null, d.message),
      d.fix ? el("span.ltxd-diag-fix", null, d.fix) : null,
    )));
  }

  /** A transient message in the status bar. */
  notify({ message, fix = "", level = "error" }) {
    this.status.className = `ltxd-status ${level}`;
    this.status.replaceChildren(
      el("span", null, message),
      fix ? el("span.ltxd-status-fix", null, fix) : null,
    );
    clearTimeout(this._statusTimer);
    this._statusTimer = setTimeout(() => {
      this.status.replaceChildren();
      this.status.className = "ltxd-status";
    }, level === "error" ? 9000 : 4500);
  }

  // -- input -------------------------------------------------------------

  _bindGlobalDrop() {
    // Dropping anywhere on the panel works, not only on the reference zone.
    this.root.addEventListener("dragover", (event) => {
      if (event.dataTransfer?.types?.includes("Files")) event.preventDefault();
    });
    this.root.addEventListener("drop", (event) => {
      const files = filesFrom(event);
      if (!files.length) return;
      event.preventDefault();
      event.stopPropagation();
      this.references._ingest(files);
    });
  }

  _bindKeys() {
    this.root.addEventListener("keydown", (event) => {
      const editing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        event.shiftKey ? this.store.redo() : this.store.undo();
        this.renderAdvanced();
        return;
      }
      if (editing) return;

      if (event.key === "Delete" || event.key === "Backspace") {
        const id = this.store.ui.selection;
        if (!id) return;
        event.preventDefault();
        this.store.removeReference(id);
        this.store.removeSegment(id);
        this.store.removeAudioClip(id);
      }
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        const step = event.shiftKey ? 8 : 1;
        const delta = event.key === "ArrowLeft" ? -step : step;
        this.store.setUi({
          playhead: Math.max(0, Math.min(
            this.store.ui.playhead + delta, this.store.project.frames - 1
          )),
        });
      }
    });
    this.root.tabIndex = 0;
  }
}

export { timecode };

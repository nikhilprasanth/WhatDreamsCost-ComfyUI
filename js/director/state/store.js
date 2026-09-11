/**
 * The project store.
 *
 * One mutable document, typed mutations, an undo stack, and a change event.
 * Views subscribe; views never mutate each other. Serialisation to the node
 * widget happens in exactly one place, debounced.
 *
 * The previous Director kept state in three places at once — `node.properties`,
 * hidden widgets, and editor instance fields — kept in sync by hand. Every
 * state bug it had came from that. Here there is one object, and everything
 * else is derived.
 */

import { debounce } from "../util/dom.js";
import { snapDim, snapFrames } from "./time.js";

/** Matches `director/__init__.py:SPEC_SCHEMA`. */
export const SCHEMA = 3;

/** Top-level keys excluded from the digest, mirroring `spec.DIGEST_EXCLUDE`. */
const COSMETIC = new Set(["ui", "takes", "meta"]);

let counter = 0;
function newId(prefix) {
  counter += 1;
  const random = Math.random().toString(16).slice(2, 8);
  return `${prefix}_${Date.now().toString(16).slice(-4)}${random}${counter.toString(16)}`;
}

export const ids = {
  reference: () => newId("ref"),
  segment: () => newId("seg"),
  media: () => newId("med"),
  audio: () => newId("aud"),
  take: () => newId("tak"),
};

/** A blank project. Kept in step with `Spec`'s dataclass defaults. */
export function emptyProject() {
  return {
    schema: SCHEMA,
    meta: { name: "Untitled shot", created: "", modified: "", director_version: "3.0.0", note: "" },
    project: { fps: 24.0, width: 1280, height: 704, frames: 121, mode: "t2v" },
    prompt: {
      mode: "simple", raw: "", negative: "",
      sections: {
        subject: "", scene: "", action: "", camera: "", acting: "",
        lighting: "", sound: "", dialogue: "", technical: "",
      },
      camera: { move: "", intensity: "moderate", custom: "" },
      lens: { shot_size: "", angle: "", height: "", focal_mm: 0, dof: "", lighting: "", composition: "" },
      enhance: { enabled: false, seed: 0, max_length: 600, use_image: true },
    },
    references: [],
    segments: [],
    audio: { enabled: true, mode: "generate", clips: [], reference: null, identity_guidance: 3.0 },
    media: {},
    generation: {
      preset: "balanced", stages: 1, seed: 42, seed_mode: "fixed", seed_list: [],
      video_cfg: 1.0, audio_cfg: 1.0,
      sampler: "euler_ancestral", refine_sampler: "euler",
      sigmas_stage1: [], sigmas_stage2: [],
      img_compression: 18, first_frame_strength: 0.7, guide_strength: 0.7,
      decode: { tile_size: 512, overlap: 64, temporal_size: 64, temporal_overlap: 8, bit_depth: 8 },
      stage1_divisor: 2, save_prefix: "video/LTXDirector",
    },
    models: {
      family: "ltx2.5", unet: "", vae: "", audio_vae: "", clip: "",
      enhancer_clip: null, upscaler: null, loras: [],
    },
    relay: {
      enabled: true, epsilon: 0.001,
      video_strength: 1.0, audio_strength: 1.0,
      video_window_scale: 1.0, audio_window_scale: 1.0,
    },
    takes: [],
    ui: { display_unit: "seconds", expanded: [], zoom: 1.0, playhead: 0, selection: "" },
  };
}

/** Deep clone. `structuredClone` where available; JSON otherwise. */
function clone(value) {
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

/** Fill in anything a loaded project is missing, without overwriting it. */
function withDefaults(loaded) {
  const merge = (base, incoming) => {
    if (incoming === null || incoming === undefined) return base;
    if (Array.isArray(base) || typeof base !== "object") return incoming;
    if (typeof incoming !== "object" || Array.isArray(incoming)) return incoming;
    const out = { ...base };
    for (const [key, value] of Object.entries(incoming)) {
      out[key] = key in base ? merge(base[key], value) : value;
    }
    return out;
  };
  return merge(emptyProject(), loaded);
}

export class Store {
  /**
   * @param {object} options
   * @param {(json: string) => void} options.persist  writes to the node widget
   * @param {number} options.historyLimit             undo depth
   */
  constructor({ persist, historyLimit = 100 } = {}) {
    this.state = emptyProject();
    this.persist = persist ?? (() => {});
    this.historyLimit = historyLimit;

    this._listeners = new Set();
    this._undo = [];
    this._redo = [];
    this._batch = null;

    // One place writes to the node, and it waits until the user stops typing.
    this._save = debounce(() => this.persist(this.toJSON()), 250);
  }

  // -- subscription ------------------------------------------------------

  /** Subscribe to changes. Returns an unsubscribe function. */
  subscribe(listener) {
    this._listeners.add(listener);
    return () => this._listeners.delete(listener);
  }

  _emit(reason) {
    for (const listener of this._listeners) {
      try {
        listener(this.state, reason);
      } catch (error) {
        console.error("[LTX Director] A view threw while handling a change:", error);
      }
    }
  }

  // -- reading -----------------------------------------------------------

  get project() { return this.state.project; }
  get prompt() { return this.state.prompt; }
  get references() { return this.state.references; }
  get segments() { return this.state.segments; }
  get audio() { return this.state.audio; }
  get media() { return this.state.media; }
  get generation() { return this.state.generation; }
  get models() { return this.state.models; }
  get ui() { return this.state.ui; }

  mediaFor(id) { return id ? this.state.media[id] ?? null : null; }

  toJSON() { return JSON.stringify(this.state); }

  /** Everything that affects the output, for change detection. */
  digestSource() {
    const copy = {};
    for (const [key, value] of Object.entries(this.state)) {
      if (!COSMETIC.has(key)) copy[key] = value;
    }
    return JSON.stringify(copy);
  }

  // -- writing -----------------------------------------------------------

  /**
   * Apply a mutation.
   *
   * `label` groups consecutive edits for undo: dragging a marker produces one
   * undo entry, not sixty.
   */
  update(label, mutate, { cosmetic = false, coalesce = false } = {}) {
    if (this._batch) {
      mutate(this.state);
      this._batch.labels.add(label);
      return;
    }

    const before = clone(this.state);
    mutate(this.state);

    if (!cosmetic) {
      const last = this._undo[this._undo.length - 1];
      if (!(coalesce && last && last.label === label)) {
        this._undo.push({ label, state: before });
        if (this._undo.length > this.historyLimit) this._undo.shift();
      }
      this._redo.length = 0;
    }

    this._save();
    this._emit(label);
  }

  /** Group several mutations into one undo entry. */
  transaction(label, body) {
    if (this._batch) { body(this.state); return; }
    const before = clone(this.state);
    this._batch = { labels: new Set([label]) };
    try {
      body(this.state);
    } finally {
      this._batch = null;
    }
    this._undo.push({ label, state: before });
    if (this._undo.length > this.historyLimit) this._undo.shift();
    this._redo.length = 0;
    this._save();
    this._emit(label);
  }

  /** Replace the whole project — loading a file, importing, or undo. */
  load(project, { label = "load", record = true } = {}) {
    const next = withDefaults(project ?? {});
    if (record) {
      this._undo.push({ label, state: clone(this.state) });
      this._redo.length = 0;
    }
    this.state = next;
    this.normalise();
    this._save();
    this._emit(label);
  }

  loadJSON(json, options) {
    if (!json || !json.trim()) { this.load(emptyProject(), { ...options, record: false }); return; }
    try {
      this.load(JSON.parse(json), options);
    } catch {
      // A malformed widget value is a normal state for a freshly dropped node.
      this.load(emptyProject(), { ...options, record: false });
    }
  }

  // -- history -----------------------------------------------------------

  get canUndo() { return this._undo.length > 0; }
  get canRedo() { return this._redo.length > 0; }

  undo() {
    const entry = this._undo.pop();
    if (!entry) return false;
    this._redo.push({ label: entry.label, state: clone(this.state) });
    this.state = entry.state;
    this._save();
    this._emit("undo");
    return true;
  }

  redo() {
    const entry = this._redo.pop();
    if (!entry) return false;
    this._undo.push({ label: entry.label, state: clone(this.state) });
    this.state = entry.state;
    this._save();
    this._emit("redo");
    return true;
  }

  /** Write immediately. For beforeunload and for queueing a render. */
  flush() { this._save.flush(); }

  // -- invariants --------------------------------------------------------

  /** Apply the rules the compiler assumes. Cheap, and safe to call often. */
  normalise() {
    const project = this.state.project;
    project.width = snapDim(project.width);
    project.height = snapDim(project.height);
    project.frames = snapFrames(project.frames);
    if (!(project.fps > 0)) project.fps = 24;

    const last = Math.max(0, project.frames - 1);
    for (const reference of this.state.references) {
      reference.at.frame = Math.min(Math.max(0, reference.at.frame), last);
      reference.strength = Math.min(Math.max(0, reference.strength), 1);
    }
    for (const clip of this.state.audio.clips) {
      clip.start = Math.max(0, clip.start);
      clip.length = Math.max(1, clip.length);
    }
    this.state.segments.sort((a, b) => a.start - b.start);
    this.state.ui.playhead = Math.min(Math.max(0, this.state.ui.playhead), last);
  }

  // -- convenience mutations --------------------------------------------

  setDuration(frames) {
    this.update("duration", (state) => {
      state.project.frames = snapFrames(Math.max(1, frames));
      this.normalise();
    }, { coalesce: true });
  }

  setMode(mode) {
    this.update("mode", (state) => { state.project.mode = mode; });
  }

  addMedia(entry) {
    const id = entry.id ?? ids.media();
    this.update("add media", (state) => { state.media[id] = { ...entry, id }; });
    return id;
  }

  addReference(reference) {
    const full = {
      id: ids.reference(), role: "keyframe", media: null, label: "", enabled: true,
      at: { frame: 0, unit: this.state.ui.display_unit }, anchor: "index",
      strength: 1.0, fit: "cover", crop: "center",
      ...reference,
    };
    this.update("add reference", (state) => { state.references.push(full); });
    return full;
  }

  updateReference(id, changes, options) {
    this.update("edit reference", (state) => {
      const reference = state.references.find((r) => r.id === id);
      if (reference) Object.assign(reference, changes);
    }, options);
  }

  removeReference(id) {
    this.update("remove reference", (state) => {
      state.references = state.references.filter((r) => r.id !== id);
    });
  }

  addSegment(segment) {
    const full = { id: ids.segment(), start: 0, length: 24, text: "", ...segment };
    this.update("add prompt region", (state) => { state.segments.push(full); });
    return full;
  }

  updateSegment(id, changes, options) {
    this.update("edit prompt region", (state) => {
      const segment = state.segments.find((s) => s.id === id);
      if (segment) Object.assign(segment, changes);
    }, options);
  }

  removeSegment(id) {
    this.update("remove prompt region", (state) => {
      state.segments = state.segments.filter((s) => s.id !== id);
    });
  }

  addAudioClip(clip) {
    const full = {
      id: ids.audio(), media: null, start: 0, length: this.state.project.frames,
      trim_start: 0, gain: 1, enabled: true, ...clip,
    };
    this.update("add audio", (state) => { state.audio.clips.push(full); });
    return full;
  }

  removeAudioClip(id) {
    this.update("remove audio", (state) => {
      state.audio.clips = state.audio.clips.filter((c) => c.id !== id);
    });
  }

  /** Cosmetic: never recorded for undo, never invalidates the node. */
  setUi(changes) {
    this.update("ui", (state) => { Object.assign(state.ui, changes); }, { cosmetic: true });
  }

  recordTake(take) {
    this.update("take", (state) => {
      state.takes.unshift({ id: ids.take(), starred: false, note: "", ...take });
      // Capped: this is a shot log, not an archive.
      if (state.takes.length > 50) state.takes.length = 50;
    }, { cosmetic: true });
  }
}

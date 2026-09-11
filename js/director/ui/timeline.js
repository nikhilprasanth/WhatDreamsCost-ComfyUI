/**
 * The timeline.
 *
 * One `<canvas>`, redrawn on `requestAnimationFrame` and only when something
 * changed. The previous Director drew markers as DOM and re-read the graph on
 * every canvas repaint; this draws a few hundred rectangles in about a
 * millisecond and does nothing at all while idle.
 *
 * Three tracks, top to bottom:
 *
 *   ruler       time, ticks and the playhead
 *   references  keyframes and IC-LoRA guides, as draggable markers
 *   prompts     Prompt Relay regions, as draggable bars
 *   audio       clips, with a waveform where one is available
 *
 * Thumbnails are `<img>` elements loaded from a URL and drawn into the canvas.
 * The browser's cache does the caching; decoded pixels never enter project
 * state.
 */

import { thumbnailUrl } from "../net/api.js";
import { framesToSeconds, label as unitLabel, timecode } from "../state/time.js";
import { el } from "../util/dom.js";

const RULER_H = 26;
const REF_H = 54;
const SEG_H = 34;
const AUDIO_H = 40;
const GUTTER = 10;

const COLOURS = {
  background: "#1b1b1f",
  band: "#212127",
  grid: "#2e2e36",
  text: "#b9b9c4",
  faint: "#6f6f7d",
  playhead: "#e7b34a",
  reference: "#4a8fd4",
  referenceEnd: "#c77ad4",
  referenceOff: "#4a4a55",
  segment: "#3f9e6b",
  segmentEmpty: "#3a4a42",
  audio: "#a8794a",
  selection: "#ffffff",
};

/** One row of the timeline, so hit-testing and drawing agree on geometry. */
class Band {
  constructor(kind, top, height) {
    this.kind = kind;
    this.top = top;
    this.height = height;
  }
  get bottom() { return this.top + this.height; }
  contains(y) { return y >= this.top && y < this.bottom; }
}

export class Timeline {
  constructor(store, { onSelect } = {}) {
    this.store = store;
    this.onSelect = onSelect ?? (() => {});

    this.canvas = el("canvas.ltxd-timeline-canvas");
    this.root = el("div.ltxd-timeline", null, this.canvas);
    this.ctx = this.canvas.getContext("2d");

    this._dirty = true;
    this._frame = null;
    this._thumbs = new Map();   // path -> HTMLImageElement
    this._peaks = new Map();    // media id -> number[]
    this._bands = [];
    this._drag = null;
    this._hover = null;

    this._observer = new ResizeObserver(() => this.invalidate());
    this._observer.observe(this.root);

    this._bindPointer();
    store.subscribe(() => this.invalidate());
  }

  destroy() {
    this._observer.disconnect();
    if (this._frame) cancelAnimationFrame(this._frame);
    this._frame = null;
  }

  /** Ask for a repaint. Cheap enough to call from anywhere. */
  invalidate() {
    this._dirty = true;
    if (this._frame !== null) return;
    this._frame = requestAnimationFrame(() => {
      this._frame = null;
      if (this._dirty) this.draw();
    });
  }

  // -- geometry ----------------------------------------------------------

  get totalFrames() {
    return Math.max(1, this.store.project.frames);
  }

  /** Pixels per frame, given the current width. */
  get scale() {
    return (this.canvas.width / (window.devicePixelRatio || 1) - GUTTER * 2) / this.totalFrames;
  }

  frameToX(frame) {
    return GUTTER + frame * this.scale;
  }

  xToFrame(x) {
    const frame = Math.round((x - GUTTER) / this.scale);
    return Math.min(Math.max(0, frame), this.totalFrames - 1);
  }

  _layout() {
    const bands = [new Band("ruler", 0, RULER_H)];
    let top = RULER_H;
    bands.push(new Band("references", top, REF_H)); top += REF_H;
    bands.push(new Band("segments", top, SEG_H)); top += SEG_H;
    if (this.store.audio.enabled) { bands.push(new Band("audio", top, AUDIO_H)); top += AUDIO_H; }
    this._bands = bands;
    return top;
  }

  bandFor(kind) {
    return this._bands.find((b) => b.kind === kind);
  }

  // -- drawing -----------------------------------------------------------

  draw() {
    this._dirty = false;
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(320, this.root.clientWidth);
    const height = this._layout();

    if (this.canvas.width !== Math.round(width * dpr) ||
        this.canvas.height !== Math.round(height * dpr)) {
      this.canvas.width = Math.round(width * dpr);
      this.canvas.height = Math.round(height * dpr);
      this.canvas.style.width = `${width}px`;
      this.canvas.style.height = `${height}px`;
    }
    this.root.style.height = `${height}px`;

    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLOURS.background;
    ctx.fillRect(0, 0, width, height);

    this._drawBands(ctx, width);
    this._drawRuler(ctx, width);
    this._drawSegments(ctx);
    this._drawReferences(ctx);
    this._drawAudio(ctx);
    this._drawPlayhead(ctx, height);
  }

  _drawBands(ctx, width) {
    for (const band of this._bands) {
      if (band.kind === "ruler") continue;
      ctx.fillStyle = COLOURS.band;
      ctx.fillRect(0, band.top + 1, width, band.height - 2);
    }
  }

  _drawRuler(ctx, width) {
    const { fps } = this.store.project;
    const unit = this.store.ui.display_unit;
    const total = this.totalFrames;

    // Choose a tick spacing that leaves at least ~70px between labels, so the
    // ruler stays readable at any zoom without overlapping text.
    const minSpacing = 70;
    const candidates = unit === "frames"
      ? [1, 2, 4, 8, 16, 24, 48, 96, 192, 384]
      : [fps / 4, fps / 2, fps, fps * 2, fps * 5, fps * 10, fps * 30, fps * 60];
    const step = candidates.find((c) => c * this.scale >= minSpacing)
      ?? candidates[candidates.length - 1];

    ctx.font = "10px system-ui, sans-serif";
    ctx.textBaseline = "middle";

    for (let frame = 0; frame <= total; frame += step) {
      const x = this.frameToX(Math.round(frame));
      ctx.strokeStyle = COLOURS.grid;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, RULER_H - 7);
      ctx.lineTo(Math.round(x) + 0.5, RULER_H);
      ctx.stroke();

      ctx.fillStyle = COLOURS.faint;
      ctx.fillText(unitLabel(Math.round(frame), unit, fps, total), Math.round(x) + 3, 9);
    }

    ctx.strokeStyle = COLOURS.grid;
    ctx.beginPath();
    ctx.moveTo(0, RULER_H - 0.5);
    ctx.lineTo(width, RULER_H - 0.5);
    ctx.stroke();
  }

  _drawSegments(ctx) {
    const band = this.bandFor("segments");
    if (!band) return;
    const selection = this.store.ui.selection;

    for (const segment of this.store.segments) {
      const x1 = this.frameToX(segment.start);
      const x2 = this.frameToX(segment.start + segment.length);
      const w = Math.max(3, x2 - x1);
      const filled = segment.text.trim().length > 0;

      ctx.fillStyle = filled ? COLOURS.segment : COLOURS.segmentEmpty;
      roundRect(ctx, x1, band.top + 5, w, band.height - 12, 3);
      ctx.fill();

      if (selection === segment.id) {
        ctx.strokeStyle = COLOURS.selection;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      if (w > 40) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(x1 + 4, band.top, w - 8, band.height);
        ctx.clip();
        ctx.fillStyle = "#eef3ef";
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillText(segment.text.trim() || "empty region", x1 + 7, band.top + band.height / 2);
        ctx.restore();
      }
    }
  }

  _drawReferences(ctx) {
    const band = this.bandFor("references");
    if (!band) return;
    const total = this.totalFrames;
    const selection = this.store.ui.selection;

    for (const reference of this.store.references) {
      const frame = resolveFrame(reference, total);
      const x = this.frameToX(frame);
      const enabled = reference.enabled && reference.media;
      const colour = !enabled ? COLOURS.referenceOff
        : reference.anchor === "end" ? COLOURS.referenceEnd
        : COLOURS.reference;

      const thumbW = 40;
      const thumbH = band.height - 14;
      const left = Math.min(Math.max(GUTTER, x - thumbW / 2),
        this.canvas.width / (window.devicePixelRatio || 1) - thumbW - GUTTER);
      const top = band.top + 7;

      ctx.fillStyle = colour;
      roundRect(ctx, left, top, thumbW, thumbH, 3);
      ctx.fill();

      const image = this._thumbnail(reference);
      if (image?.complete && image.naturalWidth) {
        ctx.save();
        roundRect(ctx, left + 1, top + 1, thumbW - 2, thumbH - 2, 2);
        ctx.clip();
        drawCover(ctx, image, left + 1, top + 1, thumbW - 2, thumbH - 2);
        ctx.restore();
      }

      if (selection === reference.id) {
        ctx.strokeStyle = COLOURS.selection;
        ctx.lineWidth = 1.5;
        roundRect(ctx, left, top, thumbW, thumbH, 3);
        ctx.stroke();
      }

      // The stem shows the true frame even when the thumbnail was nudged
      // inward to stay on screen.
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, band.top);
      ctx.lineTo(Math.round(x) + 0.5, band.bottom);
      ctx.stroke();

      reference._hit = { left, top, width: thumbW, height: thumbH, frame };
    }
  }

  _drawAudio(ctx) {
    const band = this.bandFor("audio");
    if (!band) return;
    const selection = this.store.ui.selection;

    for (const clip of this.store.audio.clips) {
      const x1 = this.frameToX(clip.start);
      const x2 = this.frameToX(clip.start + clip.length);
      const w = Math.max(3, x2 - x1);
      const top = band.top + 5;
      const height = band.height - 12;

      ctx.fillStyle = clip.enabled ? COLOURS.audio : COLOURS.referenceOff;
      roundRect(ctx, x1, top, w, height, 3);
      ctx.fill();

      const peaks = this._peaks.get(clip.media);
      if (peaks?.length && w > 12) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(x1, top, w, height);
        ctx.clip();
        ctx.strokeStyle = "rgba(255,255,255,0.55)";
        ctx.beginPath();
        const middle = top + height / 2;
        for (let i = 0; i < w; i += 1) {
          const peak = peaks[Math.floor((i / w) * peaks.length)] ?? 0;
          const amplitude = (peak * (height - 6)) / 2;
          ctx.moveTo(x1 + i + 0.5, middle - amplitude);
          ctx.lineTo(x1 + i + 0.5, middle + amplitude);
        }
        ctx.stroke();
        ctx.restore();
      }

      if (selection === clip.id) {
        ctx.strokeStyle = COLOURS.selection;
        ctx.lineWidth = 1.5;
        roundRect(ctx, x1, top, w, height, 3);
        ctx.stroke();
      }
    }
  }

  _drawPlayhead(ctx, height) {
    const x = Math.round(this.frameToX(this.store.ui.playhead)) + 0.5;
    ctx.strokeStyle = COLOURS.playhead;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();

    ctx.fillStyle = COLOURS.playhead;
    ctx.beginPath();
    ctx.moveTo(x - 4, 0);
    ctx.lineTo(x + 4, 0);
    ctx.lineTo(x, 7);
    ctx.closePath();
    ctx.fill();
  }

  // -- thumbnails and waveforms -----------------------------------------

  _thumbnail(reference) {
    const entry = this.store.mediaFor(reference.media);
    if (!entry) return null;
    const path = entry.subfolder ? `${entry.subfolder}/${entry.filename}` : entry.filename;
    const at = entry.kind === "video"
      ? framesToSeconds(reference.at.frame, this.store.project.fps)
      : 0;
    const key = `${path}@${at.toFixed(2)}`;

    let image = this._thumbs.get(key);
    if (!image) {
      image = new Image();
      image.decoding = "async";
      // Repaint when it lands; until then the marker draws as a flat colour.
      image.onload = () => this.invalidate();
      image.onerror = () => {};
      image.src = thumbnailUrl(path, at);
      this._thumbs.set(key, image);
      // Bounded: a long session should not accumulate every frame ever hovered.
      if (this._thumbs.size > 200) {
        this._thumbs.delete(this._thumbs.keys().next().value);
      }
    }
    return image;
  }

  /** Cache a waveform for a media id. Called by the audio view once loaded. */
  setPeaks(mediaId, peaks) {
    this._peaks.set(mediaId, peaks);
    this.invalidate();
  }

  // -- interaction -------------------------------------------------------

  _bindPointer() {
    const canvas = this.canvas;

    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      const { x, y } = this._local(event);
      const hit = this._hitTest(x, y);

      canvas.setPointerCapture(event.pointerId);
      this.store.setUi({ selection: hit?.id ?? "" });
      this.onSelect(hit);

      if (!hit) {
        this._drag = { kind: "playhead" };
        this._scrub(x);
        return;
      }

      const frame = this.xToFrame(x);
      this._drag = { kind: hit.kind, id: hit.id, grabOffset: frame - hit.frame, edge: hit.edge };
    });

    canvas.addEventListener("pointermove", (event) => {
      const { x, y } = this._local(event);
      if (!this._drag) {
        const hit = this._hitTest(x, y);
        const cursor = hit?.edge ? "ew-resize" : hit ? "grab" : "default";
        if (canvas.style.cursor !== cursor) canvas.style.cursor = cursor;
        return;
      }
      this._applyDrag(x, event.shiftKey);
    });

    const end = (event) => {
      if (!this._drag) return;
      canvas.releasePointerCapture?.(event.pointerId);
      this._drag = null;
      this.store.normalise();
      this.invalidate();
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);

    canvas.addEventListener("dblclick", (event) => {
      const { x, y } = this._local(event);
      const band = this._bands.find((b) => b.contains(y));
      if (band?.kind === "segments" && !this._hitTest(x, y)) {
        // Double-clicking empty prompt track adds a region there — the fastest
        // way to start using Prompt Relay.
        const start = this.xToFrame(x);
        const length = Math.max(8, Math.round(this.totalFrames / 4));
        this.store.addSegment({ start, length: Math.min(length, this.totalFrames - start) });
      }
    });
  }

  _local(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  _hitTest(x, y) {
    const band = this._bands.find((b) => b.contains(y));
    if (!band) return null;

    if (band.kind === "references") {
      // Reverse so the topmost drawn marker wins, as it does visually.
      for (const reference of [...this.store.references].reverse()) {
        const hit = reference._hit;
        if (!hit) continue;
        if (x >= hit.left && x <= hit.left + hit.width &&
            y >= hit.top && y <= hit.top + hit.height) {
          return { kind: "reference", id: reference.id, frame: hit.frame, object: reference };
        }
      }
      return null;
    }

    if (band.kind === "segments") {
      for (const segment of this.store.segments) {
        const x1 = this.frameToX(segment.start);
        const x2 = this.frameToX(segment.start + segment.length);
        if (x < x1 - 4 || x > x2 + 4) continue;
        const edge = x <= x1 + 5 ? "start" : x >= x2 - 5 ? "end" : null;
        return { kind: "segment", id: segment.id, frame: segment.start, edge, object: segment };
      }
      return null;
    }

    if (band.kind === "audio") {
      for (const clip of this.store.audio.clips) {
        const x1 = this.frameToX(clip.start);
        const x2 = this.frameToX(clip.start + clip.length);
        if (x < x1 - 4 || x > x2 + 4) continue;
        const edge = x <= x1 + 5 ? "start" : x >= x2 - 5 ? "end" : null;
        return { kind: "audio", id: clip.id, frame: clip.start, edge, object: clip };
      }
      return null;
    }

    return null;
  }

  _scrub(x) {
    this.store.setUi({ playhead: this.xToFrame(x) });
  }

  _applyDrag(x, snapToStride) {
    const drag = this._drag;
    const total = this.totalFrames;
    const raw = this.xToFrame(x);

    if (drag.kind === "playhead") { this._scrub(x); return; }

    // Shift snaps to the 8-frame grid guides actually land on, so a user who
    // cares about exact placement can get it without arithmetic.
    const snap = (frame) => (snapToStride ? Math.round(frame / 8) * 8 : frame);

    if (drag.kind === "reference") {
      const frame = Math.min(Math.max(0, snap(raw - drag.grabOffset)), total - 1);
      this.store.updateReference(drag.id, { anchor: "index", at: { frame, unit: this.store.ui.display_unit } },
        { coalesce: true });
      return;
    }

    if (drag.kind === "segment" || drag.kind === "audio") {
      const list = drag.kind === "segment" ? this.store.segments : this.store.audio.clips;
      const item = list.find((i) => i.id === drag.id);
      if (!item) return;
      const update = drag.kind === "segment"
        ? (changes) => this.store.updateSegment(drag.id, changes, { coalesce: true })
        : (changes) => this.store.update("edit audio", (state) => {
            const clip = state.audio.clips.find((c) => c.id === drag.id);
            if (clip) Object.assign(clip, changes);
          }, { coalesce: true });

      if (drag.edge === "start") {
        const end = item.start + item.length;
        const start = Math.min(Math.max(0, snap(raw)), end - 1);
        update({ start, length: end - start });
      } else if (drag.edge === "end") {
        const end = Math.min(Math.max(item.start + 1, snap(raw)), total);
        update({ length: end - item.start });
      } else {
        const start = Math.min(Math.max(0, snap(raw - drag.grabOffset)), total - item.length);
        update({ start });
      }
    }
  }

  /** A readout for the status bar. */
  playheadLabel() {
    const { fps } = this.store.project;
    return `${timecode(this.store.ui.playhead, fps)} · frame ${this.store.ui.playhead}`;
  }
}

// --------------------------------------------------------------------------
// drawing helpers
// --------------------------------------------------------------------------

function roundRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

/** Draw an image filling a box, cropping rather than distorting. */
function drawCover(ctx, image, x, y, width, height) {
  const scale = Math.max(width / image.naturalWidth, height / image.naturalHeight);
  const w = image.naturalWidth * scale;
  const h = image.naturalHeight * scale;
  ctx.drawImage(image, x + (width - w) / 2, y + (height - h) / 2, w, h);
}

/** Where a reference sits, resolving the start and end anchors. */
export function resolveFrame(reference, totalFrames) {
  if (reference.anchor === "start") return 0;
  if (reference.anchor === "end") return Math.max(0, totalFrames - 1);
  return Math.min(Math.max(0, reference.at.frame), Math.max(0, totalFrames - 1));
}

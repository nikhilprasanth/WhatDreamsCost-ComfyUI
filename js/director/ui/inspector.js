/**
 * The inspector.
 *
 * Properties for whatever is selected on the timeline. Kept in one place rather
 * than scattered across per-element popups, so there is exactly one answer to
 * "where do I change this".
 *
 * Time is entered in whatever unit the user is reading in — seconds, frames or
 * percent — and converted to frames on the way in. The stored value is always
 * frames, so changing the project fps re-labels a position rather than moving
 * it.
 */

import { fromFrames, timecode, toFrames } from "../state/time.js";
import { button, el, field, number, select, toggle } from "../util/dom.js";

const UNITS = [
  ["seconds", "Seconds"],
  ["frames", "Frames"],
  ["percent", "Percent"],
];

export class Inspector {
  constructor(store) {
    this.store = store;
    this.body = el("div.ltxd-inspector-body");
    this.title = el("span.ltxd-label", null, "Nothing selected");
    this.root = el("div.ltxd-inspector", null,
      el("div.ltxd-inspector-head", null, this.title, this._unitPicker()),
      this.body,
    );
    store.subscribe(() => this.render());
    this.render();
  }

  _unitPicker() {
    this.unitSelect = select(UNITS, this.store.ui.display_unit, (unit) => {
      // A display preference, so it must not dirty the shot.
      this.store.setUi({ display_unit: unit });
      this.render();
    }, { title: "How times are shown. Positions are always stored as frames." });
    return this.unitSelect;
  }

  get selected() {
    const id = this.store.ui.selection;
    if (!id) return null;
    const reference = this.store.references.find((r) => r.id === id);
    if (reference) return { kind: "reference", object: reference };
    const segment = this.store.segments.find((s) => s.id === id);
    if (segment) return { kind: "segment", object: segment };
    const clip = this.store.audio.clips.find((c) => c.id === id);
    if (clip) return { kind: "audio", object: clip };
    return null;
  }

  render() {
    this.unitSelect.value = this.store.ui.display_unit;
    const selection = this.selected;

    if (!selection) {
      this.title.textContent = "Nothing selected";
      this.body.replaceChildren(el("p.ltxd-note", null,
        "Select a keyframe, prompt region or audio clip on the timeline to edit it. "
        + "Double-click the prompt track to add a region."));
      return;
    }

    if (selection.kind === "reference") this._reference(selection.object);
    else if (selection.kind === "segment") this._segment(selection.object);
    else this._audio(selection.object);
  }

  // -- time entry --------------------------------------------------------

  /** A number input in the user's chosen unit, storing frames. */
  _timeField(label, frame, onFrame, hint = "") {
    const unit = this.store.ui.display_unit;
    const { fps, frames: total } = this.store.project;
    const value = fromFrames(frame, unit, fps, total);
    const step = unit === "frames" ? 1 : unit === "percent" ? 1 : 0.01;

    return field(label, el("div.ltxd-time-entry", null,
      number(Number(value.toFixed(unit === "seconds" ? 2 : 0)), (entered) => {
        onFrame(Math.max(0, Math.min(toFrames(entered, unit, fps, total), total - 1)));
      }, { step, min: 0 }),
      el("span.ltxd-unit", null, unit === "seconds" ? "s" : unit === "percent" ? "%" : "f"),
      el("span.ltxd-readout", null, timecode(frame, fps)),
    ), hint);
  }

  // -- per-kind panels ---------------------------------------------------

  _reference(reference) {
    const entry = this.store.mediaFor(reference.media);
    this.title.textContent = reference.label || entry?.filename || "Reference";

    const rows = [
      field("Enabled", toggle("Use this reference", reference.enabled, (enabled) => {
        this.store.updateReference(reference.id, { enabled });
      })),
      field("Anchor", select(
        [["start", "First frame"], ["index", "At a time"], ["end", "Last frame"]],
        reference.anchor,
        (anchor) => this.store.updateReference(reference.id, { anchor }),
      ), "First and last frame anchors follow the shot if you change its length."),
    ];

    if (reference.anchor === "index") {
      rows.push(this._timeField("Position", reference.at.frame, (frame) => {
        this.store.updateReference(reference.id, {
          at: { frame, unit: this.store.ui.display_unit },
        });
      }, "Multi-frame guides are moved to the nearest 8-frame boundary."));
    }

    rows.push(
      field("Strength", el("div.ltxd-slider-row", null,
        el("input.ltxd-slider", {
          type: "range", min: 0, max: 1, step: 0.01, value: reference.strength,
          oninput: (event) => this.store.updateReference(
            reference.id, { strength: parseFloat(event.target.value) }, { coalesce: true }
          ),
        }),
        el("span.ltxd-readout", null, `${Math.round(reference.strength * 100)}%`),
      ), "How hard the frame is held. Below 1 lets the model reinterpret it."),
      field("Fit", select(
        [["cover", "Crop to fill"], ["contain", "Fit inside"], ["stretch", "Stretch"]],
        reference.fit,
        (fit) => this.store.updateReference(reference.id, { fit }),
      )),
      field("Label", el("input.ltxd-input", {
        type: "text", value: reference.label, placeholder: entry?.filename ?? "",
        oninput: (event) => this.store.updateReference(
          reference.id, { label: event.target.value }, { coalesce: true }
        ),
      })),
    );

    if (entry) {
      rows.push(el("p.ltxd-note", null,
        `${entry.filename} — ${entry.width}×${entry.height}`
        + (entry.duration ? `, ${entry.duration.toFixed(2)}s` : "")));
    }

    rows.push(button("Remove reference", () => this.store.removeReference(reference.id),
      { variant: "ghost" }));

    this.body.replaceChildren(...rows);
  }

  _segment(segment) {
    this.title.textContent = "Prompt region";
    const total = this.store.project.frames;

    this.body.replaceChildren(
      field("Text", el("textarea.ltxd-textarea.small", {
        rows: 3,
        placeholder: "What happens during this stretch of the shot.",
        value: segment.text,
        oninput: (event) => this.store.updateSegment(
          segment.id, { text: event.target.value }, { coalesce: true }
        ),
      }), "Applied most strongly across this region, and fades outside it."),

      this._timeField("Starts", segment.start, (frame) => {
        const end = segment.start + segment.length;
        this.store.updateSegment(segment.id, {
          start: Math.min(frame, end - 1), length: Math.max(1, end - frame),
        });
      }),
      this._timeField("Ends", Math.min(segment.start + segment.length, total - 1), (frame) => {
        this.store.updateSegment(segment.id, {
          length: Math.max(1, frame - segment.start),
        });
      }),

      el("p.ltxd-note", null,
        this.store.segments.filter((s) => s.text.trim()).length > 1
          ? "Two or more regions with text switch Prompt Relay on: one sampling pass, "
            + "with each region's words applying over its own stretch."
          : "Add a second region with text to switch Prompt Relay on."),

      button("Remove region", () => this.store.removeSegment(segment.id), { variant: "ghost" }),
    );
  }

  _audio(clip) {
    const entry = this.store.mediaFor(clip.media);
    this.title.textContent = entry?.filename ?? "Audio clip";

    this.body.replaceChildren(
      field("Enabled", toggle("Use this clip", clip.enabled, (enabled) => {
        this.store.update("edit audio", (state) => {
          const found = state.audio.clips.find((c) => c.id === clip.id);
          if (found) found.enabled = enabled;
        });
      })),
      this._timeField("Starts", clip.start, (frame) => {
        this.store.update("edit audio", (state) => {
          const found = state.audio.clips.find((c) => c.id === clip.id);
          if (found) found.start = frame;
        });
      }),
      field("Trim in", number(Number(clip.trim_start.toFixed(2)), (value) => {
        this.store.update("edit audio", (state) => {
          const found = state.audio.clips.find((c) => c.id === clip.id);
          if (found) found.trim_start = Math.max(0, value);
        });
      }, { step: 0.01, min: 0 }), "Seconds to skip from the start of the file."),
      entry?.duration
        ? el("p.ltxd-note", null, `${entry.filename} — ${entry.duration.toFixed(2)}s long`)
        : null,
      button("Remove clip", () => this.store.removeAudioClip(clip.id), { variant: "ghost" }),
    );
  }
}

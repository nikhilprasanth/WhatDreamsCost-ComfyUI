/**
 * Reference slots.
 *
 * Drag an image in, or paste one, or pick one already in the workspace. Each
 * slot shows a thumbnail, where on the timeline it lands, and how strongly it
 * holds — and the whole thing is a URL and a few numbers, never pixel data in
 * project state.
 *
 * Roles matter to the compiler: a `keyframe` becomes a pinned frame, a `motion`
 * or `control` reference becomes an IC-LoRA guide, and the identity roles
 * (character, environment, object, style) are for reference-sheet workflows.
 */

import { thumbnailUrl } from "../net/api.js";
import { toMediaEntry, upload, pathFor, filesFrom, isSupported } from "../net/media.js";
import { framesToSeconds, timecode } from "../state/time.js";
import { button, el, select } from "../util/dom.js";
import { resolveFrame } from "./timeline.js";

const ROLES = [
  ["keyframe", "Keyframe", "Pinned at a point on the timeline."],
  ["character", "Character", "Identity reference for a reference-sheet workflow."],
  ["environment", "Environment", "A place the shot should match."],
  ["object", "Object", "A prop the shot should match."],
  ["style", "Style", "A look to follow."],
  ["motion", "Motion", "A clip whose movement the shot follows, through an IC-LoRA."],
  ["control", "Control", "A depth, edge or pose annotation, through an IC-LoRA."],
];

const ANCHORS = [
  ["start", "First frame"],
  ["index", "At a time"],
  ["end", "Last frame"],
];

export class ReferencePanel {
  constructor(store, { onNotify } = {}) {
    this.store = store;
    this.onNotify = onNotify ?? (() => {});

    this.list = el("div.ltxd-refs");
    this.dropZone = el("div.ltxd-dropzone", {
      tabIndex: 0,
      title: "Drop images or clips here, or paste from the clipboard.",
    },
      el("span", null, "Drop images or clips here"),
      button("Browse…", () => this.picker.click(), { variant: "ghost" }),
    );

    this.picker = el("input", {
      type: "file",
      multiple: true,
      accept: "image/*,video/*",
      style: { display: "none" },
      onchange: (event) => {
        this._ingest([...event.target.files]);
        event.target.value = "";
      },
    });

    this.root = el("div.ltxd-ref-panel", null, this.list, this.dropZone, this.picker);
    this._bindDrop();
    store.subscribe(() => this.render());
    this.render();
  }

  _bindDrop() {
    const zone = this.dropZone;
    const stop = (event) => { event.preventDefault(); event.stopPropagation(); };

    for (const name of ["dragenter", "dragover"]) {
      zone.addEventListener(name, (event) => { stop(event); zone.classList.add("over"); });
    }
    for (const name of ["dragleave", "dragend"]) {
      zone.addEventListener(name, (event) => { stop(event); zone.classList.remove("over"); });
    }
    zone.addEventListener("drop", (event) => {
      stop(event);
      zone.classList.remove("over");
      const files = filesFrom(event);
      if (files.length) this._ingest(files);
      else this.onNotify({
        message: "That file is not an image, video or audio file.",
        fix: "Drop a PNG, JPEG, MP4, MOV, WAV or MP3.",
      });
    });

    // Paste works when the panel has focus, which is what a user expects after
    // copying a frame from somewhere else.
    zone.addEventListener("paste", (event) => {
      const files = filesFrom(event);
      if (files.length) { stop(event); this._ingest(files); }
    });
  }

  /** Upload files and add a reference for each. */
  async _ingest(files, { role = "keyframe", anchor = null } = {}) {
    const usable = files.filter((file) => isSupported(file.name));
    if (!usable.length) return;

    this.dropZone.classList.add("busy");
    for (const file of usable) {
      try {
        const result = await upload(file);
        const entry = toMediaEntry(result);
        const mediaId = this.store.addMedia(entry);
        const existing = this.store.references.filter((r) => r.role === "keyframe");
        this.store.addReference({
          role,
          media: mediaId,
          label: entry.filename,
          anchor: anchor ?? (existing.length === 0 ? "start" : "index"),
          at: { frame: this.store.ui.playhead, unit: this.store.ui.display_unit },
        });
        if (result.reused) {
          this.onNotify({
            message: `${entry.filename} was already on the server, so nothing was uploaded.`,
            level: "info",
          });
        }
      } catch (error) {
        this.onNotify({ message: error.message, fix: error.fix ?? "" });
      }
    }
    this.dropZone.classList.remove("busy");
  }

  render() {
    const references = this.store.references;
    this.list.replaceChildren(
      ...references.map((reference) => this._card(reference))
    );
    this.dropZone.classList.toggle("compact", references.length > 0);
  }

  _card(reference) {
    const entry = this.store.mediaFor(reference.media);
    const total = this.store.project.frames;
    const frame = resolveFrame(reference, total);
    const selected = this.store.ui.selection === reference.id;

    const thumb = el("div.ltxd-ref-thumb");
    if (entry) {
      const at = entry.kind === "video"
        ? framesToSeconds(reference.at.frame, this.store.project.fps)
        : 0;
      thumb.append(el("img", {
        src: thumbnailUrl(pathFor(entry), at),
        alt: entry.filename,
        loading: "lazy",
        onerror: (event) => {
          // The file was deleted off disk. Degrade to a label rather than a
          // broken image, and say so.
          event.target.replaceWith(el("span.ltxd-missing", null, "missing"));
        },
      }));
    } else {
      thumb.append(el("span.ltxd-missing", null, "empty"));
    }

    const where = reference.anchor === "start" ? "First frame"
      : reference.anchor === "end" ? "Last frame"
      : timecode(frame, this.store.project.fps);

    const card = el(`div.ltxd-ref${selected ? ".selected" : ""}${reference.enabled ? "" : ".off"}`, {
      onclick: () => this.store.setUi({ selection: reference.id }),
      title: entry ? `${entry.filename} — ${where}` : "No image yet",
    },
      thumb,
      el("div.ltxd-ref-body", null,
        select(ROLES.map(([v, l]) => [v, l]), reference.role, (role) => {
          this.store.updateReference(reference.id, { role });
        }, { title: ROLES.find(([v]) => v === reference.role)?.[2] ?? "" }),
        select(ANCHORS, reference.anchor, (anchor) => {
          this.store.updateReference(reference.id, { anchor });
        }, { title: "Where on the timeline this lands." }),
        el("div.ltxd-ref-meta", null,
          el("span", null, where),
          el("span.ltxd-dot", null, "·"),
          el("span", null, `${Math.round(reference.strength * 100)}%`),
        ),
      ),
      el("div.ltxd-ref-actions", null,
        button(reference.enabled ? "◉" : "○", (event) => {
          event.stopPropagation();
          this.store.updateReference(reference.id, { enabled: !reference.enabled });
        }, { variant: "ghost", title: reference.enabled ? "Disable" : "Enable" }),
        button("✕", (event) => {
          event.stopPropagation();
          this.store.removeReference(reference.id);
        }, { variant: "ghost", title: "Remove" }),
      ),
    );
    return card;
  }
}

export { ROLES, ANCHORS };

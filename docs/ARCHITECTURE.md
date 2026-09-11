# LTX Director Next — Architecture

Companion to [`RESEARCH.md`](./RESEARCH.md). Read that first; this document assumes its findings.

---

## 1. The thesis

> The Director describes **intent**. Native ComfyUI and LTX nodes perform the **work**.

Concretely: a Director project is a small JSON document. A pure function turns that document into
a native ComfyUI workflow. The only Python that touches a tensor at runtime is the one capability
no upstream node provides (Prompt Relay), and it is one focused node.

Everything else follows from that.

---

## 2. What was wrong, restated as forces

| Force | Consequence for the design |
| --- | --- |
| Hardcoded `128` channels, `//32`, `*32` broke on VAE changes | Geometry is never computed by us — it comes from `EmptyLTXVLatentVideo` and `vae.downscale_index_formula` |
| Four re-implementations of core nodes | Emit the core node instead; the compiler's job is wiring, not maths |
| Media re-decoded every queue | The spec references media by **id + content hash**; decode happens once, in a node that ComfyUI can cache |
| Tensors in custom sockets | Sockets carry specs (small, hashable, serialisable); tensors stay on LATENT/IMAGE/CONDITIONING |
| Monkey-patched `.forward` on every block | One `set_model_attn2_patch`-style hook via `transformer_options`, applied to a cloned model, chaining any prior wrapper |
| 11.7k-line JS file | ES modules, one concern each, no business logic in the browser |
| No validation | A `validate()` pass over the spec runs before compile and before execute, producing human sentences |
| No tests | The compiler is a pure function over JSON — testable with no GPU, no models, no ComfyUI import |

---

## 3. Module map

```
director/
  __init__.py             package marker, version
  core/
    ids.py                stable id + content-hash helpers
    time.py               frame ⇄ second ⇄ percent, 8k+1 snapping, /32 snapping
    spec.py               the Director Spec dataclasses + (de)serialisation
    validate.py           Spec → [Diagnostic]; human sentences, not KeyErrors
    migrate.py            schema migrations, incl. Director 2.x timeline import
  capabilities/
    probe.py              what is installed: nodes, models, loras, LTX family
    features.py           capability → feature flags the UI consumes
  compiler/
    graph.py              tiny builder for ComfyUI workflow schema 0.4 + subgraphs
    base.py               Compiler ABC + registry + CompileResult/Diagnostic
    ltx25.py              the LTX-2.5 adapter (T2V/I2V/FFLF/multi-key/A2V/T2A/IC-LoRA)
    presets.py            quality/VRAM presets as explicit setting deltas
  relay/
    mask.py               Gaussian temporal penalty (moved from prompt_relay.py)
    tokens.py             prompt assembly + token-range mapping
    patch.py              model patching (moved from patches.py, de-risked)
  nodes/
    project.py            LTXDirectorProject  — the UI node; outputs a SPEC
    relay.py              LTXDirectorRelay    — SPEC + model + clip → model + conditioning
    compile.py            LTXDirectorCompile  — SPEC → workflow JSON on disk / to canvas
    legacy.py             thin shims so v2 graphs keep loading
  api/
    media.py              /ltxdirector/media/* — upload, dedup, thumbnail, probe
    project.py            /ltxdirector/project/* — save/load/list
    caps.py               /ltxdirector/capabilities
  presets/
    quality.json          Draft / Balanced / Quality / Seed-hunt / Final
    vram.json             16 / 24 / 32 / 48 GB profiles
    camera.json           camera move + lens vocabulary (prompt-side)
  migrations/
    v2_timeline.py        Director 2.x timeline_data → Spec

js/director/
  index.js                extension registration, nothing else
  state/store.js          the single mutable project store + undo/redo
  state/schema.js         mirror of the Spec (kept in sync by a generated file)
  ui/shell.js             chrome: mode bar, sections, collapse state
  ui/timeline.js          ruler, tracks, markers, scrub, drag
  ui/references.js        reference slots, drag/drop, paste, thumbnails
  ui/prompt.js            Simple / Director / Expert prompt editor
  ui/audio.js             audio track + waveform
  ui/takes.js             take manager
  ui/inspector.js         selected-element property panel
  net/api.js              fetch wrappers for /ltxdirector/*
  net/media.js            upload with dedup + chunking
  util/*.js               dom, geometry, format, events
```

Deliberately **not** in the plan: a database, a job queue, a bundler/build step. The frontend is
plain ES modules served by ComfyUI's static handler, so `git clone` remains the install.

---

## 4. The Director Spec

One versioned JSON document. It is the contract between the UI, the compiler, the validator, the
runtime nodes, and the `.ltxdirector.json` project file.

```jsonc
{
  "schema": 3,                       // integer, migrations keyed off this
  "meta":  { "name": "…", "created": "…", "director_version": "3.0.0" },

  "project": {
    "fps": 24.0,
    "width": 1280, "height": 704,    // always /32
    "frames": 121,                   // always 8k+1
    "mode": "t2v|i2v|fflf|keyframes|continue|a2v|t2a|iclora"
  },

  "prompt": {
    "mode": "simple|director|expert",
    "raw": "…",                      // Simple mode / Expert mode text
    "negative": "…",
    "sections": {                    // Director mode, all optional
      "subject": "…", "scene": "…", "action": "…", "camera": "…",
      "acting": "…", "lighting": "…", "sound": "…", "dialogue": "…",
      "technical": "…"
    },
    "camera":  { "move": "dolly_in", "intensity": "moderate", "custom": "" },
    "lens":    { "shot_size": "MS", "angle": "eye", "focal_mm": 35, "dof": "shallow" },
    "enhance": { "enabled": false, "seed": 0 },
    "compiled_cache": null           // last compiled prompt, for the preview panel
  },

  "references": [
    { "id": "ref_a1b2", "role": "keyframe|character|environment|object|style|motion|control",
      "media": "med_…", "enabled": true,
      "at": { "unit": "seconds|frames|percent", "value": 2.4 },
      "anchor": "start|end|index",   // "end" ⇒ frame_idx -1
      "strength": 1.0, "fit": "cover|contain|stretch", "crop": "center|disabled" }
  ],

  "segments": [                      // Prompt Relay regions; empty ⇒ relay off
    { "id": "seg_…", "start": 0, "length": 48, "text": "…" }
  ],

  "audio": {
    "enabled": true, "mode": "generate|import|inpaint|mute",
    "clips": [ { "id": "aud_…", "media": "med_…", "start": 0, "length": 96,
                 "trim_start": 0.0, "gain": 1.0 } ],
    "reference": null                // media id for LTXVReferenceAudio (ID-LoRA)
  },

  "media": {                         // content-addressed registry, no blobs
    "med_…": { "filename": "…", "kind": "image|video|audio",
               "sha256": "…", "width": 0, "height": 0,
               "duration": 0.0, "fps": 0.0, "subfolder": "" }
  },

  "generation": {
    "preset": "balanced",
    "stages": 2,
    "seed": 42, "seed_mode": "fixed|random|increment|decrement|list",
    "seed_list": [],
    "video_cfg": 1.0, "audio_cfg": 1.0,
    "sampler": "euler_ancestral",
    "sigmas_stage1": [...], "sigmas_stage2": [...],
    "img_compression": 18,
    "decode": { "tile_size": 512, "overlap": 64,
                "temporal_size": 64, "temporal_overlap": 8 }
  },

  "models": {
    "family": "ltx2.5",
    "unet": "…", "vae": "…", "audio_vae": "…",
    "clip": "…", "enhancer_clip": null,
    "upscaler": null,
    "loras": [ { "name": "…", "strength": 1.0, "kind": "lora|iclora" } ]
  },

  "relay": { "enabled": true, "epsilon": 0.001,
             "video_strength": 1.0, "audio_strength": 1.0,
             "video_window_scale": 1.0, "audio_window_scale": 1.0 },

  "takes": [                         // lightweight history, capped
    { "id": "tak_…", "seed": 42, "starred": false, "at": "…",
      "output": "…", "settings_digest": "…" }
  ]
}
```

Rules that make this work:

* **No tensors, no base64, no waveforms.** Media is a registry entry: filename + sha256 +
  probed metadata. The browser never holds decoded pixels in project state.
* **Time is stored in frames**, always. `at.unit` is a *display* preference; the stored `value` is
  converted on entry and on fps change. One source of truth, no drift.
* **Every id is stable** and prefixed by kind. Reordering never changes an id.
* **`schema` is an integer.** `migrate.py` owns the ladder; there is no "guess the version".

---

## 5. State flow

```
        ┌──────────── browser ────────────┐        ┌──────────── python ────────────┐
        │                                  │        │                                │
  user  │  ui/*  ──mutate──▶  store.js      │        │   nodes/project.py             │
  input │                       │           │        │        │                       │
        │                       │ serialise │        │        │ deserialise           │
        │                       ▼           │        │        ▼                       │
        │              node widget          │        │    core/spec.py  ──▶ Spec      │
        │              "project" (STRING) ──┼────────┼──▶       │                     │
        │                       ▲           │        │          ├──▶ core/validate    │
        │                       │           │        │          │      └─ diagnostics │
        │   /ltxdirector/* ◀────┘           │        │          │                     │
        │   (media, probe, caps)            │        │          ├──▶ DIRECTOR_SPEC out │
        └──────────────────────────────────┘        │          │                     │
                                                     │          ▼                     │
                                    ┌────────────────┴──  compiler/ltx25.py           │
                                    │                          │                      │
                       nodes/compile.py                 workflow JSON (schema 0.4)    │
                                    │                          │                      │
                       nodes/relay.py ──▶ MODEL + CONDITIONING │                      │
                                                                ▼                      │
                                                    native LTX + core nodes ──▶ VIDEO  │
                                                     └──────────────────────────────────┘
```

Two execution paths, deliberately:

**Path A — Compile (recommended).** The Director node emits a spec; `LTXDirectorCompile` writes a
native workflow and the frontend loads it onto the canvas (or into a new tab). From then on the
user has a plain LTX graph with no Director in it. *This is the future-proof path.*

**Path B — Inline.** The Director node sits in a prebuilt graph and feeds `LTXDirectorRelay` plus
ordinary core nodes. Used for iteration, seed hunting, and by the shipped example workflow. No
Director node ever holds a tensor it did not receive on a standard socket.

---

## 6. The compiler

### 6.1 Contract

```python
class Compiler(ABC):
    family: str                       # "ltx2.5"
    def supports(self, spec: Spec) -> bool: ...
    def compile(self, spec: Spec, caps: Capabilities) -> CompileResult: ...

@dataclass
class CompileResult:
    workflow: dict                    # ComfyUI workflow JSON, schema 0.4
    diagnostics: list[Diagnostic]
    node_index: dict[str, int]        # semantic name → node id, for round-tripping
```

`compile` is **pure**: same spec + same capabilities ⇒ byte-identical workflow. No filesystem, no
torch, no ComfyUI import. That is what makes it testable and what makes `git diff` on a compiled
workflow meaningful.

### 6.2 Graph builder

`compiler/graph.py` is a ~250-line builder over the 0.4 schema:

```python
g = GraphBuilder()
with g.subgraph("Load Models") as sub:
    unet = sub.node("UNETLoader", unet_name=spec.models.unet, weight_dtype="default")
    sub.output("MODEL", unet.out(0))
...
g.link(sampler.out("output"), separate.inp("av_latent"))
```

It owns id allocation, link records, `widgets_values` ordering, subgraph definitions and
promoted-input plumbing. Node *input order* per type lives in one table (`NODE_SIGNATURES`)
derived from the audit — so a schema change is a one-line edit, not a hunt.

### 6.3 The LTX-2.5 adapter

`compiler/ltx25.py` builds the six subgraphs from §2.1 of the research doc:

| Subgraph | Built from |
| --- | --- |
| `Load Models` | `UNETLoader`, `CLIPLoader`×1–2, `VAELoader`×2, `LatentUpscaleModelLoader`?, `LTXICLoRALoaderModelOnly`? |
| `Prompt` | `TextGenerateLTX2Prompt`?, `CLIPTextEncode`×2, `LTXVConditioning` |
| `Canvas` | `EmptyLTXVLatentVideo`, `LTXVEmptyLatentAudio`, mode-specific conditioning, `LTXVConcatAVLatent` |
| `Sample` | `RandomNoise`, `LTXVDualCFGGuider`, `KSamplerSelect`, `ManualSigmas`, `SamplerCustomAdvanced`, `LTXVSeparateAVLatent`, `LTXVCropGuides`? |
| `Refine` | `LTXVLatentUpsampler`, re-pin, concat, second `SamplerCustomAdvanced` |
| `Decode` | `VAEDecodeTiled`, `LTXVAudioVAEDecode`, `CreateVideo`, `SaveVideo` |

Mode handling is a dispatch table, not a chain of `if`s:

```python
CANVAS_BUILDERS = {
    Mode.T2V:       _canvas_t2v,
    Mode.I2V:       _canvas_i2v,       # LTXVImgToVideoInplace
    Mode.FFLF:      _canvas_guides,    # LTXVAddGuide(0) + LTXVAddGuide(-1)
    Mode.KEYFRAMES: _canvas_guides,    # LTXVAddGuide(n) …
    Mode.CONTINUE:  _canvas_i2v,       # I2V pinned to the previous clip's last frame
    Mode.A2V:       _canvas_a2v,       # LTXVAudioVAEEncode + LTXVFreezeLatent
    Mode.T2A:       _canvas_t2a,       # LTXVAudioOnlyModel + dummy video latent
    Mode.ICLORA:    _canvas_iclora,    # LTXAddVideoICLoRAGuide
}
```

`FFLF`, `KEYFRAMES` and `ICLORA` set `needs_crop_guides = True`, which is the only thing that
decides whether `LTXVCropGuides` is emitted. Research §7.5 made that a rule; the code makes it a
single flag.

### 6.4 Adding LTX 2.6

Write `compiler/ltx26.py`, register it, add a probe signature. Nothing else changes — not the
spec, not the UI, not the validator, not the tests for 2.5.

---

## 7. Capabilities

`capabilities/probe.py` answers, cheaply and cached-per-process:

* which node classes are registered (`nodes.NODE_CLASS_MAPPINGS`) — never "is the folder there";
* which checkpoints / VAEs / text encoders / loras / upscalers exist
  (`folder_paths.get_filename_list`) — never `os.listdir`;
* which LTX family the selected files imply, from **filename patterns as a last resort only**,
  preferring node availability (`LTXVDualCFGGuider` ⇒ 2.5-era core;
  `LTXVAddGeneratedKeyframes` ⇒ DFR available; `LTXICLoRALoaderModelOnly` ⇒ ComfyUI-LTXVideo
  installed).

`capabilities/features.py` turns that into flags the UI reads over `/ltxdirector/capabilities`:

```json
{ "audio": true, "iclora": true, "dfr_keyframes": true, "dual_cfg": true,
  "prompt_enhancer": false, "duration_predictor": false, "two_stage": true,
  "families": ["ltx2.5"], "missing": { "prompt_enhancer": "gemma4_e2b_* not found in text_encoders/" } }
```

A missing feature is **hidden with a reason**, never a crash. The reason string is what the UI
shows on hover.

---

## 8. Validation and errors

`core/validate.py` returns `Diagnostic(level, code, message, fix, path)`:

```
ERROR  frames.not_8k1     "Duration 5.0 s at 24 fps is 120 frames; LTX needs 1 + a multiple of 8."
                          fix: "Use 121 frames (5.04 s) or 113 frames (4.71 s)."   path: project.frames

ERROR  ref.model_mode     "Reference frame at 4.2 s can't be compiled: the selected model family
                           (ltx2.5) has no node for mid-timeline guides without IC-LoRA."
                          fix: "Install ComfyUI-LTXVideo, or move the reference to the start or end."

WARN   ref.duplicate      "References A and C both land on frame 96."
                          fix: "Move one, or disable it."
```

Rules enforced, from research §7: 8k+1 frames, /32 dimensions, guide index legality (multiple of 8
for multi-frame guides, ≡1 mod 8 for IC-LoRA), reference ordering and collision, audio clip bounds
within the timeline, required model components present for the chosen mode, relay segment lengths
summing within the latent, and mode/capability compatibility.

Validation runs in three places with the same code: as the user edits (frontend calls
`/ltxdirector/validate`), at compile, and at execute. One rule set, three surfaces.

---

## 9. Prompt Relay, de-risked

Kept, because nothing upstream does single-pass per-segment conditioning. Changed in three ways:

1. **Moved out of the mega-node.** `LTXDirectorRelay` takes `MODEL + CLIP + DIRECTOR_SPEC` and
   returns `MODEL + CONDITIONING`. That is its whole surface.
2. **Geometry comes from the latent, not from constants.** The optional `latent` input supplies
   `samples.shape`; with no latent, latent frames are derived from `spec.project.frames` via the
   VAE ratio reported by the model, not a literal `8`.
3. **Patching chains instead of asserting.** `relay/patch.py` keeps the existing approach of
   wrapping `get_model_object(key)` (which returns any prior patch), so stacking with NAG and
   friends works — but it stops raising on the Wan path and stops silently disabling the user's
   attention backend when no mask is actually produced.

The maths in `relay/mask.py` and `relay/tokens.py` is carried over unchanged; it is correct and
already version-agnostic.

---

## 10. Frontend architecture

**One store, many views.** `state/store.js` holds the project, applies typed mutations, keeps an
undo stack of inverse patches, and emits change events. Views subscribe; views never mutate each
other. Serialisation to the node widget is debounced and happens in exactly one place.

**No business logic in the browser.** Frame/time conversion, 8k+1 snapping and validation are
called over HTTP against the same Python that the compiler uses, with a small optimistic local
mirror for the hot path (dragging a marker). The mirror is generated from `core/time.py` constants,
so it cannot drift.

**Rendering budget.** The timeline is one `<canvas>`, redrawn on `requestAnimationFrame` only when
dirty. Markers are drawn, not DOM. `onDrawForeground` does nothing but request a repaint if dirty —
the per-frame graph traversal in v2 is deleted and replaced by an `onConnectionsChange` handler.

**Thumbnails.** Requested from `/ltxdirector/media/thumb?id=…&t=…`, generated server-side with
PyAV, cached on disk under the workspace folder, and referenced by URL. The browser never seeks a
`<video>` element in a loop, and decoded frames never enter project state.

**Progressive disclosure.** Default view: mode bar, timeline, prompt, generate. Everything else is
a collapsed section whose open/closed state lives in the project (so it survives reload) but is
excluded from the settings digest (so it doesn't dirty takes).

---

## 11. Performance strategy

| Cost in v2 | Strategy |
| --- | --- |
| Re-decode every image/video each queue | Media is content-addressed; `LTXDirectorMedia` decode nodes are ordinary ComfyUI nodes, so the executor caches them on unchanged inputs |
| In-memory H.264 round-trip per image | Emit `LTXVPreprocess` instead — same effect, upstream-maintained, and cached by the executor |
| Manual latent allocation | Emit `EmptyLTXVLatentVideo` / `LTXVEmptyLatentAudio` |
| Audio decoded then re-encoded to latent each run | `LTXVAudioVAEEncode`, cached by the executor; waveform peaks cached on disk keyed by sha256 |
| Model cloned+patched per run | Keep the per-node clone cache, but key it on `(model, spec_digest)` rather than a stack-walked node id |
| Full-res tensors crossing sockets | Spec sockets only; tensors ride LATENT/IMAGE/CONDITIONING |
| Base64 in the workflow JSON | Removed from the spec entirely; upload-or-reject |
| 470 KB JS parsed on every page load | Modules, lazily imported per section |

Budget targets, measured in Phase 8: Director node creation < 50 ms; timeline redraw < 4 ms at 200
markers; spec serialisation < 2 ms; compile < 30 ms; project JSON < 64 KB for a 20-reference shot.

---

## 12. Compatibility and migration

* `LTXDirector` and `LTXDirectorGuide` **stay registered** and keep working. v2 graphs load.
* `migrations/v2_timeline.py` converts a v2 `timeline_data` blob to a Spec: prompts, segment
  lengths, keyframe positions and end-frame flags, guide strengths, audio clips, fps, dimensions,
  resize method, compression. Motion/IC-LoRA segments map to `references[role="motion"]`.
* The frontend offers **"Import Director 2 timeline"** on an empty project, and the Python side
  exposes the same migration so a saved `.json` timeline can be converted headlessly.
* What is deliberately *not* carried over: base64 media (re-upload required, with a clear message),
  the `GUIDE_DATA` / `MOTION_GUIDE_DATA` socket protocol, and retake's hand-built masking (replaced
  by the inpaint IC-LoRA path).

---

## 13. Decisions and their reasons

| Decision | Alternatives considered | Why |
| --- | --- | --- |
| Compile to a native graph | Keep executing inside one node | Survives LTX releases; users can inspect and edit; no hidden pipeline |
| Emit schema 0.4 subgraphs | Flat graph | Matches official templates; expand/collapse is free; a 40-node graph stays legible |
| Spec as a JSON string widget | Many typed widgets | One source of truth; trivially serialisable; versioned; diffable |
| Time stored in frames | Store seconds | Frames are what the model consumes; seconds drift on fps change |
| Validation in Python, mirrored in JS | Validate in JS | One rule set; the browser mirror is a latency optimisation, not a second implementation |
| Keep Prompt Relay | Use `LTXVLoopingSampler` | Single-pass vs chunked sampling — different trade-off, and the single pass has no seams |
| Content-addressed media | Path-keyed | Dedup, cache keys, and portable projects all fall out of the hash |
| No build step | Bundle with esbuild/vite | `git clone` must remain the install story for a ComfyUI custom node |

---

## 14. Risks

| Risk | Mitigation |
| --- | --- |
| ComfyUI subgraph schema shifts | The builder is one module; `NODE_SIGNATURES` is one table; golden-file tests catch drift immediately |
| Prompt Relay breaks on an LTX internals change | Feature-probe before patching; if the expected attributes are absent, disable relay with a diagnostic rather than crash |
| ComfyUI-LTXVideo not installed | Everything except IC-LoRA and audio-only works on core alone; those features hide with a reason |
| Users depend on v2 behaviour we changed | v2 nodes stay registered; migration is explicit and lossy-with-a-message, never silent |
| Scope | Phases are independently shippable; core + compiler + T2V/I2V/FFLF is a useful product on its own |

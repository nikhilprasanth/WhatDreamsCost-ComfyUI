# LTX Director Next — Implementation Plan

Milestones are independently testable and independently shippable. Each lists acceptance criteria
that can be checked without a GPU unless stated otherwise.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

**Status at 2026-09-11:** M1–M7, M9 and M10 complete; M8 partly done (see below).
589 Python tests and 40 frontend tests, none needing a GPU, model weights or a
ComfyUI install. Measured numbers are in [`PERFORMANCE.md`](./PERFORMANCE.md).

---

## `[x]` M1 — Core: time, ids, spec, validation

`director/core/{ids,time,spec,validate}.py`

Scope
* Frame ⇄ second ⇄ percent conversion; `snap_frames` (8k+1), `snap_dim` (/32); duration helpers.
* `Spec` dataclasses mirroring `ARCHITECTURE.md` §4, with `from_dict` / `to_dict` round-trip.
* `validate(spec, caps) -> list[Diagnostic]` covering the rules in `RESEARCH.md` §7.
* Stable id generation and content hashing.

Acceptance
* `to_dict(from_dict(x)) == x` for a corpus of specs including every mode.
* `snap_frames(n)` is idempotent and always `≡ 1 (mod 8)`; `snap_dim` always `≡ 0 (mod 32)`.
* Round-tripping a spec through JSON is byte-stable (sorted keys, no floats that drift).
* Every diagnostic has a non-empty `message` and `fix`; no diagnostic text contains a Python type
  name or exception class.
* No import of `torch`, `comfy`, `folder_paths`, or `server` anywhere under `core/`.

---

## `[x]` M2 — Compiler: graph builder + LTX-2.5 adapter

`director/compiler/{graph,base,ltx25,presets}.py`

Scope
* `GraphBuilder`: node/link/subgraph emission for workflow schema `0.4`.
* `NODE_SIGNATURES` table: per node type, the ordered input names and widget names, taken from the
  audit.
* `LTX25Compiler` with the six subgraphs and the mode dispatch table.
* Presets as explicit setting deltas (`presets/quality.json`, `presets/vram.json`).

Acceptance
* Compiling each of T2V, I2V, FFLF, KEYFRAMES, A2V, T2A, ICLORA produces a workflow whose node
  type multiset matches the corresponding upstream reference graph.
* `LTXVCropGuides` is present **iff** the mode adds guides.
* Two-stage compiles emit exactly two `SamplerCustomAdvanced`, one `LTXVLatentUpsampler`, and
  stage-2 sigmas `0.85, 0.7250, 0.4219, 0.0`.
* Every link references an existing node and a valid slot; no dangling inputs on required slots.
* `compile()` is pure: two calls with the same inputs produce identical JSON.
* Golden-file tests: committed expected JSON for each mode; a diff is a deliberate act.

---

## `[x]` M3 — Capabilities probe

`director/capabilities/{probe,features}.py`

Scope
* Node-registry and `folder_paths` interrogation, cached per process with an explicit invalidate.
* Feature flags with human-readable reasons for anything unavailable.

Acceptance
* Probe runs with ComfyUI absent (returns an "unknown" capability set, no exception) — so the
  compiler tests stay importable.
* No `os.listdir` / `glob` over model folders.
* Each false flag carries a `missing[...]` sentence naming the file or node that would enable it.

---

## `[x]` M4 — Runtime nodes

`director/nodes/{project,relay,compile,legacy}.py`

Scope
* `LTXDirectorProject` — V3 schema, one `project` STRING widget plus a handful of promoted
  convenience inputs; outputs `DIRECTOR_SPEC` and derived scalars (frames, fps, width, height, seed).
* `LTXDirectorRelay` — `MODEL + CLIP + DIRECTOR_SPEC (+ optional LATENT)` → `MODEL + CONDITIONING`.
* `LTXDirectorCompile` — `DIRECTOR_SPEC` → writes a workflow JSON, returns its path.
* `legacy.py` — keeps `LTXDirector`, `LTXDirectorGuide`, `LTXDirectorCropGuides` registered.

Acceptance
* Loading an existing v2 workflow still resolves every node.
* `LTXDirectorRelay` with `relay.enabled = false` returns the model unmodified and a plain encode.
* `LTXDirectorRelay` on a model lacking the expected transformer attributes emits a diagnostic and
  degrades to a plain encode instead of raising.
* No node output carries a tensor inside a custom socket type.

---

## `[x]` M5 — Relay extraction

`director/relay/{mask,tokens,patch}.py`

Scope
* Move `prompt_relay.py` → `relay/mask.py` + `relay/tokens.py`; move `patches.py` → `relay/patch.py`.
* Add provenance headers. Add a feature probe before patching. Stop disabling the user's attention
  backend when no mask is produced.
* Original modules become re-export shims so v2 imports keep working.

Acceptance
* Mask maths is unchanged: existing penalty-matrix outputs reproduce bit-for-bit for a fixed input.
* `apply_patches` on an unrecognised architecture returns cleanly with a diagnostic.
* `from .prompt_relay import build_segments` still works.

---

## `[x]` M6 — API layer

`director/api/{media,project,caps}.py`

Scope
* `/ltxdirector/capabilities`
* `/ltxdirector/media/upload` (chunked, sha256-dedup), `/media/probe`, `/media/thumb`, `/media/peaks`
* `/ltxdirector/project/{save,load,list}`
* `/ltxdirector/validate` and `/ltxdirector/compile`
* Routes registered from one module, not at import of the node file.

Acceptance
* Uploading the same file twice performs one write and returns the same media id.
* Thumbnails are generated once and served from disk thereafter.
* Audio peaks are cached keyed by sha256, not by path.
* No route handler decodes media that a cache already covers.

---

## `[x]` M7 — Frontend

`js/director/**`

Scope, in order: store + undo → shell → timeline → references → prompt → audio → inspector → takes.

Acceptance
* Creating the node adds no per-frame graph traversal (`onDrawForeground` does not read links).
  — **met**, and asserted by `test_frontend.py`.
* 200 markers on the timeline redraw within one frame budget.
  — the **state** work per drag frame is measured at 0.38 ms. The canvas raster
  itself needs a real browser and remains a manual check; see `PERFORMANCE.md`.
* Undo/redo restores exact spec equality.
* Reloading the page restores the project from the widget alone.
* Deleting a media file off disk degrades to a placeholder with a message; it does not throw.
* No decoded image data is stored in the project.

---

## `[~]` M8 — Advanced workflows

Done
* **Continue** mode compiles (it is image-to-video pinned to a supplied frame).
* **Audio modes** — generate, import, gap-fill (approximated, with a warning), mute.
* **Two-stage refinement**, including the first-frame re-pin after upsampling and
  the crop-before-upsample ordering that a guided two-stage shot needs.
* **IC-LoRA control**, with the LoRA's own downscale factor wired through.
* **Ingredients / reference sheet** via the `character`, `environment`, `object`
  and `style` reference roles.
* **Seed-hunt and quality presets**, applied as visible setting deltas.
* **Takes** — a capped log of seed, time and settings digest, with *Reuse seed*.

Not done, and honestly so
* **"Use last frame as next shot" is not one click.** Continue mode works, but
  extracting the previous clip's final frame is manual: save the frame, drop it
  in. Automating it needs the Director to reach into ComfyUI's output history,
  which is a larger piece of plumbing than it first appears.
* **"Promote take" is not one click.** Takes offers *Reuse seed*; switching the
  preset from Seed hunt to Quality afterwards is a second action.
* **Seed hunter does not queue N runs.** The preset sets one stage, a small size
  and a random seed; pressing Generate several times is the loop.
* **Multi-shot sequencing** is not built. LTX-2.5 has native multishot within one
  generation, which the prompt already reaches — a sequence of *separate* shots
  with shared state is a project-level feature the Spec does not yet model.

Acceptance, restated for what was built
* Continue mode compiles to a valid I2V graph — covered by the compiler tests.
* The seed-hunt preset produces a single-stage shot with no upscaler and a random
  seed — covered by `test_presets.py`.
* A take records the seed it ran with, and *Reuse seed* puts it back.

---

## `[x]` M9 — Migration

`director/migrations/v2_timeline.py`

Acceptance
* A real v2 `timeline_data` blob from the shipped example workflow converts without loss of
  prompts, segment lengths, keyframe positions, end-frame flags, guide strengths, audio clips, fps
  or dimensions.
* Base64-only media produces an explicit diagnostic naming what must be re-added.
* Migration is covered by a test using the committed example workflow as the fixture.

---

## `[x]` M10 — Optimisation and polish

Profile node creation, timeline redraw, spec serialisation, compile time, project size. Then
documentation (`README`, `QUICKSTART`, `WORKFLOWS`, `ADVANCED`, `MIGRATION`), example workflows
regenerated by the compiler, tooltips and error text review.

Acceptance
* Every budget in `ARCHITECTURE.md` §11 measured and recorded in `docs/PERFORMANCE.md`.
* Shipped example workflows are compiler output, not hand-edited.
* `pytest` green; no test requires a GPU or model weights.

---

## Test strategy

| Layer | How |
| --- | --- |
| `core/` | Pure unit tests; property tests for the snapping invariants |
| `compiler/` | Golden-file comparison per mode + structural assertions (link integrity, node multiset) |
| `capabilities/` | Injected fake registry and fake `folder_paths` |
| `migrations/` | Fixture = the committed v2 example workflow |
| `nodes/` | Import-level tests with ComfyUI stubbed; no execution |
| frontend | Headless DOM harness for store/undo/serialisation; manual checklist for drag behaviour |

Tests live in `tests/` and must run with `pytest` on a machine with no ComfyUI, no torch and no
models installed. Anything that cannot is out of scope for CI and is documented as a manual check.


---

## What was found along the way

Worth recording, because each of these was a real defect the tests caught rather
than a design choice:

* **Two signature errors** in the node table — `ResizeImageMaskNode`'s outputs
  and `GetVideoComponents`' output count — found by
  `test_upstream_conformance.py` comparing against the real upstream graphs.
  Either would have produced a graph that loaded and was wired to the wrong
  socket, which is the worst available failure mode.
* **Python and JavaScript disagreed on `snap_dim(80)`.** Python's `round()` uses
  banker's rounding and gave 64; `Math.round` gave 96. Found by
  `test_frontend.py` comparing the two implementations over 500 values. Python
  now rounds halves up to match.
* **Guides were cropped after the upsampler, not before.** Keyframe tokens live
  on the latent's temporal axis, so a two-stage shot with guides would have
  upsampled them as if they were picture.
* **Prompt Relay was silently dropped when compiling.** A shot with three prompt
  regions compiled to the same graph as a shot with none. The compiler now emits
  the relay pair and says so.
* **`migrate()` mutated its caller's document** through a shallow copy, so
  importing a project corrupted the one the editor was still holding.

## Deliberately not done

* **Retake mode** is not carried forward. See `MIGRATION.md` — the honest
  replacement is the in/outpainting IC-LoRA.
* **Audio gap-filling** is approximated, with a warning, because no upstream node
  builds the per-frame audio mask it needs.
* **Low-VRAM loaders** (`LowVRAMCheckpointLoader` and friends) are not emitted.
  They are a sequencing concern the compiler has no opinion about yet; add them
  by hand to a compiled graph.
* **Subgraph layout** is structurally validated and golden-tested, but has not
  been opened in a running ComfyUI. `flat` is the default for that reason.

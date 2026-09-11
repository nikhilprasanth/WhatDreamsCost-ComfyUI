# LTX Director Next — Implementation Plan

Milestones are independently testable and independently shippable. Each lists acceptance criteria
that can be checked without a GPU unless stated otherwise.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## M1 — Core: time, ids, spec, validation

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

## M2 — Compiler: graph builder + LTX-2.5 adapter

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

## M3 — Capabilities probe

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

## M4 — Runtime nodes

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

## M5 — Relay extraction

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

## M6 — API layer

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

## M7 — Frontend

`js/director/**`

Scope, in order: store + undo → shell → timeline → references → prompt → audio → inspector → takes.

Acceptance
* Creating the node adds no per-frame graph traversal (`onDrawForeground` does not read links).
* 200 markers on the timeline redraw within one frame budget.
* Undo/redo restores exact spec equality.
* Reloading the page restores the project from the widget alone.
* Deleting a media file off disk degrades to a placeholder with a message; it does not throw.
* No decoded image data is stored in the project.

---

## M8 — Advanced workflows

Continuation (last frame → next first frame), multi-shot, ingredients/reference sheet, audio
modes, seed hunter + promote-take, two-stage refinement, IC-LoRA control modes.

Acceptance
* "Use last frame as next shot" produces a valid I2V spec referencing the previous take's output.
* Seed hunter compiles N single-stage graphs at draft settings without an upscaler.
* "Promote take" compiles a two-stage graph pinned to the promoted take's seed and references.

---

## M9 — Migration

`director/migrations/v2_timeline.py`

Acceptance
* A real v2 `timeline_data` blob from the shipped example workflow converts without loss of
  prompts, segment lengths, keyframe positions, end-frame flags, guide strengths, audio clips, fps
  or dimensions.
* Base64-only media produces an explicit diagnostic naming what must be re-added.
* Migration is covered by a test using the committed example workflow as the fixture.

---

## M10 — Optimisation and polish

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

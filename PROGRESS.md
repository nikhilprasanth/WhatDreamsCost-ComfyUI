# LTX Director Next — progress

Branch `director-next`, 11 commits on top of `main`. Working tree clean, tests
green. Last updated 2026-09-11.

This is the working record: what exists, what does not, what is known to be
weak, and what to pick up next. The reasoning behind the design is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the audit it came from is in
[`docs/RESEARCH.md`](docs/RESEARCH.md).

---

## Status

| | |
| --- | --- |
| Phases 0–9 | complete, except the parts of M8 listed under [Not done](#not-done) |
| Python | 8,376 lines across 35 modules under `director/` |
| Editor | 117 KB across 13 ES modules, replacing a 470 KB single file |
| Tests | **602 Python** + **40 frontend**, none needing a GPU, models or ComfyUI |
| Lint | ruff clean on `F` and `E9` |
| Verified against a live ComfyUI | **no** — see [Untested](#untested) |

Run everything:

```bash
python -m pytest              # 602 tests, ~3.5 s (includes the frontend suite via Node)
python tools/profile_director.py
node tools/profile_editor.mjs
```

---

## What was found

The research phase changed the plan in three ways, and these are the load-bearing
findings:

1. **Upstream is LTX-2.5, not 2.3.** `ComfyUI-LTXVideo`'s README still documents
   2.3 and never mentions 2.5. The truth is in its `example_workflows/2.5/`
   directory and in ComfyUI core (`comfy_extras/nodes_lt*.py`). Everything the
   compiler emits was extracted from those graphs, not from documentation.
2. **The old Director's problem was ownership, not size.** It hand-allocated
   latents as `torch.zeros([1, 128, …, h // 32, w // 32])` and read guide
   dimensions as `latent_width * 32`. Correct for the 2.3 VAE; silently wrong the
   moment it changed. It also re-implemented four nodes that already existed
   upstream (`LTXVPreprocess`, `EmptyLTXVLatentVideo`, `LTXVEmptyLatentAudio`,
   guide insertion).
3. **Prompt Relay is the genuine differentiator.** Upstream's only multi-prompt
   answer (`LTXVLoopingSampler` + `MultiPromptProvider`) samples temporal tiles
   sequentially — N times the cost, with seams. Single-pass attention biasing has
   no equivalent anywhere, so it stays.

Full detail, including the exact node topology per mode and the constraints
derived from node source rather than folklore, is in `docs/RESEARCH.md`.

---

## What exists

### `director/core/` — the Spec

One versioned JSON document that the UI, validator, compiler, runtime nodes and
project file all agree on. No tensors, no base64, no waveforms; media is a
registry entry keyed by content hash.

* `time.py` — the `1 + 8k` frame rule and the `/32` dimension rule, guide-index
  snapping for both `LTXVAddGuide` (`0 mod 8`) and `LTXAddVideoICLoRAGuide`
  (`1 mod 8`), and frame/second/percent conversion. Frames are the storage unit,
  so changing fps re-labels positions instead of moving them.
* `spec.py` — the dataclass tree, stable ids, and a settings digest that excludes
  presentation and take history.
* `codec.py` — a strict hint-driven dataclass/JSON codec. Unknown fields are
  ignored, so a project from a newer build still opens.
* `prompt.py` — camera and lens choices compile to *prose*, because that is what
  the LTX-2.5 encoder was trained on. User wording is never paraphrased.
* `validate.py` — every rule traces to a node source, and every error carries a
  sentence plus a fix.

Hard rule, enforced by test: nothing here imports `torch`, `comfy`,
`folder_paths` or `server`.

### `director/compiler/` — Spec → native graph

A pure function. Same input, byte-identical output.

* `signatures.py` — slot and widget order for all 53 node types we emit,
  extracted from the real upstream graphs. A workflow references inputs by
  *index*, so this table is the difference between a correct graph and one that
  looks correct.
* `graph.py` — one call makes one node; a `Port` argument becomes a link. Emits
  workflow schema 0.4 in two layouts (flat with group boxes, or packed into
  subgraph definitions), plus the API form that `POST /prompt` executes.
* `ltx25.py` — eight modes via a dispatch table, one or two stages.

### `director/relay/` — Prompt Relay

Extracted from the mega-node into a package with one entry point. `support.py`
deliberately imports no `comfy`, so "this model cannot do per-region prompts" is
answerable in a plain Python process — the degradation path must not itself need
ComfyUI.

### `director/nodes/` — three nodes

`LTXDirectorProject` (holds the shot, **no tensor sockets at all**),
`LTXDirectorRelay`, `LTXDirectorCompile`. Director 2.x's three nodes stay
registered, labelled *(legacy)*.

### `director/api/` — 16 routes under `/ltxdirector`

Media is content-addressed: uploading a file already on disk transfers no bytes,
and thumbnails and waveform peaks are generated server-side and cached by hash.

### `js/director/` — the editor

One canvas timeline redrawn only when something changed, one store with undo,
and serialisation to the node widget in exactly one place.

---

## Defects the tests caught

Each of these would have shipped. Recorded because the class of bug matters more
than the instance:

| Defect | How it was caught |
| --- | --- |
| Two wrong node signatures (`ResizeImageMaskNode` outputs, `GetVideoComponents` output count) | `test_upstream_conformance.py`, comparing against the real upstream graphs. Either would have produced a graph that **loads and is wired to the wrong socket** — the worst available failure mode. |
| Python and JavaScript disagreed on `snap_dim(80)`: 64 vs 96 | `test_frontend.py`, comparing the two implementations over 500 values. Python's `round()` uses banker's rounding; `Math.round` does not. Python now rounds halves up. |
| Guides cropped *after* the upsampler | Noticed while wiring the two-stage path; keyframe tokens would have been upsampled as if they were picture. |
| Prompt Relay silently dropped when compiling | Building the examples: a shot with three prompt regions compiled to the same graph as one with none. |
| `migrate()` mutated its caller's document | `test_migrate.py`, via a shallow copy that let a step rewrite nested reference dicts in place. |

---

## Not done

M8 is marked `[~]` in the plan, honestly:

* **"Use last frame as next shot" is not one click.** Continue mode compiles
  correctly, but extracting the previous clip's final frame is manual. Automating
  it needs the Director to read ComfyUI's output history — more plumbing than it
  first appears.
* **"Promote take" is not one click.** Takes offers *Reuse seed*; switching the
  preset from Seed hunt to Quality is a second action.
* **Seed hunter does not queue N runs.** The preset sets one stage, a small size
  and a random seed; pressing Generate several times is the loop.
* **Multi-shot sequencing is not modelled.** LTX-2.5 has native multishot within
  one generation, which the prompt already reaches. A sequence of *separate*
  shots with shared state is a project-level feature the Spec does not describe.

Deliberately excluded, with reasons written down:

* **Retake mode** is not carried forward. Director 2.x's own README said it "is
  not potent enough"; the reason is that it built a temporal mask by hand rather
  than using the path upstream supports. Region edits belong to the
  in/outpainting IC-LoRA. See `docs/MIGRATION.md`.
* **Audio gap-filling is approximated**, with a warning. Doing it properly needs
  a per-audio-frame noise mask and no upstream node builds one.
* **Low-VRAM loaders** are not emitted. A sequencing concern the compiler has no
  opinion about yet; add them by hand to a compiled graph.

---

## Untested

**No live ComfyUI run.** There is no ComfyUI install on this machine, so nothing
here has been executed against a real server or a real model. What *is* verified:

* `tests/test_package.py` loads the repository exactly the way ComfyUI loads a
  custom node — by path, as a package with relative imports — with only the
  ComfyUI names the modules actually import stubbed. All 12 nodes register, all
  16 routes attach, and the V3 entry point agrees with the static map
  ComfyUI-Manager reads.
* Every compiled graph is structurally checked: links resolve, slot indices match
  the signatures, nothing required dangles, nothing is orphaned.
* Node types and slot orders are checked against `tests/fixtures/upstream_ltx25.json`,
  distilled from the real upstream graphs.

**Subgraph layout has never been opened in ComfyUI.** It is golden-tested and its
internal link structure is validated, but `flat` is the default for that reason.

**Canvas raster time is not measured.** The state work per drag frame is 0.38 ms
at 200 markers; the drawing itself needs a browser to measure.

### First live launch — what to check

1. The node appears as **LTX Director** and the editor renders.
2. Capabilities populate: model dropdowns fill, unavailable modes grey out with
   a reason on hover.
3. Drop an image → it uploads once, a thumbnail appears on the timeline.
4. **Generate** queues and produces a video.
5. **Compile workflow** opens a graph on the canvas that runs unmodified.
6. A saved Director 2.x workflow still loads.

---

## Measured

Everything well inside the budgets in `docs/ARCHITECTURE.md` §11. Full table in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

| | Median | Budget |
| --- | ---: | ---: |
| Compile text-to-video | 0.27 ms | 30 ms |
| Compile 20 keyframes, two stages | 1.59 ms | 30 ms |
| Validate a 200-reference shot | 0.12 ms | 50 ms |
| One drag frame, 200 markers | 0.38 ms | 4 ms |
| 20-reference project JSON | 9.3 KB | 64 KB |

The Director 2.x example workflow is 190–205 KB by comparison, and could grow
without limit because media could be inlined as base64.

---

## Where things live

```
director/
  core/          spec, time, prompt, validation, migration   (no torch, no comfy)
  compiler/      signatures, graph builder, LTX-2.5 adapter  (pure function)
  capabilities/  what is installed, as flags with reasons
  relay/         Prompt Relay: mask, tokens, support, patch
  nodes/         the three ComfyUI nodes
  api/           16 routes under /ltxdirector
  migrations/    Director 2.x timeline import
  presets.py     quality and memory presets, as visible deltas

js/
  ltx_director_next.js   extension entry point — registration only
  director/              store, timeline, prompt, references, inspector, shell

tools/
  capture_upstream.py    re-capture the upstream conformance fixture
  build_examples.py      regenerate the shipped example workflows
  profile_director.py    Python timings against the budgets
  profile_editor.mjs     editor timings

docs/                    RESEARCH, ARCHITECTURE, IMPLEMENTATION_PLAN,
                         QUICKSTART, WORKFLOWS, ADVANCED, MIGRATION, PERFORMANCE
```

Unchanged and still working: `ltx_director.py`, `ltx_director_guide.py`,
`ltx_sequencer.py`, `ltx_keyframer.py`, `multi_image_loader.py`,
`load_{audio,video}_ui.py`, `speech_length_calculator.py`.
`prompt_relay.py` and `patches.py` are now documented re-export shims.

---

## Next

In rough order of value:

1. **Run it.** Everything above is inference from source; the first live launch
   is the real test.
2. **Close the M8 gaps** — last-frame-to-next-shot, promote-take, batch seed
   hunting. All three are editor work over the existing compiler.
3. **Verify the subgraph layout** in a real ComfyUI, then consider making it the
   default; it is what makes a compiled graph look like the official templates.
4. **Model multi-shot sequences** in the Spec — a list of shots sharing
   references and models, compiled to a queue of graphs.
5. **Emit the low-VRAM loaders** when the capability probe sees them.

### When a new LTX version lands

1. `git pull` in `ComfyUI-LTXVideo`.
2. `python tools/capture_upstream.py --ltxvideo ../ComfyUI-LTXVideo --commit <sha>`
   and **read the diff** — it is the difference between what upstream builds and
   what the compiler assumes.
3. If the pipeline changed, copy `director/compiler/ltx25.py` to a new adapter
   and register it. Nothing else branches on a model family.
4. `DIRECTOR_UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py`, and read
   that diff too.
5. `python tools/build_examples.py`.

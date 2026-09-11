# Performance

Measured on 2026-09-11, Python 3.11 / Node 26 on a Windows laptop, with no GPU
involved. Reproduce with:

```bash
python tools/profile_director.py
node tools/profile_editor.mjs
```

Budgets are from [`ARCHITECTURE.md`](./ARCHITECTURE.md) §11. Everything below is
the work the Director does *around* generation — sampling time is the model's,
and nothing here touches it.

---

## Python

| Measurement | Median | p95 | Budget |
| --- | ---: | ---: | ---: |
| Serialise a simple shot | 0.07 ms | 0.08 ms | 2 ms |
| Serialise a 20-reference shot | 0.38 ms | 0.41 ms | 2 ms |
| Parse a simple shot | 0.17 ms | 0.19 ms | 2 ms |
| Parse a 20-reference shot | 0.86 ms | 0.91 ms | 2 ms |
| Settings digest | 0.08 ms | 0.11 ms | 2 ms |
| Validate a simple shot | <0.01 ms | 0.01 ms | 5 ms |
| Validate a 20-reference shot | 0.02 ms | 0.04 ms | 5 ms |
| Validate a 200-reference shot | 0.12 ms | 0.21 ms | 50 ms |
| Compile text-to-video | 0.27 ms | 0.43 ms | 30 ms |
| Compile 20 keyframes, two stages | 1.59 ms | 2.03 ms | 30 ms |
| Compile as subgraphs | 2.45 ms | 2.59 ms | 30 ms |
| Compile 200 keyframes | 34.87 ms | 35.73 ms | 200 ms |

## Editor

| Measurement | Median | p95 | Budget |
| --- | ---: | ---: | ---: |
| Serialise 20 references | 0.03 ms | 0.03 ms | 2 ms |
| Serialise 200 references | 0.20 ms | 0.23 ms | 8 ms |
| Digest 200 references | 0.20 ms | 0.21 ms | 8 ms |
| Normalise 200 references | <0.01 ms | 0.02 ms | 4 ms |
| One drag frame, 200 references | 0.38 ms | 0.48 ms | 4 ms |
| Undo with 200 references | 0.80 ms | 0.86 ms | 16 ms |

## Sizes

| Thing | Size | Budget |
| --- | ---: | ---: |
| Simple shot | 1.8 KB | |
| 20-reference shot | 9.3 KB | 64 KB |
| 200-reference shot | 73.4 KB | |
| Compiled workflow, text to video | 9.3 KB | |
| Compiled workflow, 20 keyframes two stages | 47.7 KB | |

For comparison, the Director 2.x example workflow is **190–205 KB**, and could
grow without limit because media could be stored inline as base64.

---

## What is not measured here

**Canvas raster time.** The timeline's drawing budget — under one frame at 200
markers — needs a real browser. The numbers above cover the state work that runs
per drag frame; the drawing itself is a few hundred `fillRect` and `drawImage`
calls on a canvas a few hundred pixels tall, redrawn only when something
changed. Check it in DevTools if you suspect otherwise.

**Media decode.** Probing, thumbnailing and waveform extraction are server-side,
once per file, cached by content hash. First touch of a 4 GB clip reads a header;
a thumbnail seeks to the nearest keyframe rather than decoding from the start.
Every touch after that is a disk read of a small JPEG or a JSON array.

**Sampling, VAE encode and decode.** The model's, and unchanged by the Director.

---

## Where the old Director spent time, and what replaced it

| Director 2.x | Cost | Director Next |
| --- | --- | --- |
| Re-decoded every image and video segment per queue | seconds of CPU before sampling started | `LoadImage` / `LoadVideo` nodes, cached by ComfyUI's executor on unchanged inputs |
| Re-encoded each guide through an in-memory H.264 round trip | proportional to guide count and resolution | `LTXVPreprocess`, same effect, cached by the executor |
| Re-decoded every audio file into a 44.1 kHz buffer per queue | proportional to track length | `LTXVAudioVAEEncode`, cached by the executor |
| Allocated latents by hand from literal constants | correct until the VAE changed | `EmptyLTXVLatentVideo` |
| Passed full-resolution tensors plus the timeline JSON between two nodes | held in memory for the whole run | small JSON specs; tensors ride LATENT/IMAGE/CONDITIONING |
| `onDrawForeground` traversed graph links and wrote to the DOM | every canvas repaint, ~60/s, idle or not | a dirty flag and one `requestAnimationFrame` |
| Extracted thumbnails by seeking a `<video>` element frame by frame | per marker, in the browser | one server-side JPEG per position, cached by hash |
| Base64 media could land inside the saved workflow | megabytes per workflow | file references only; there is a test asserting no `readAsDataURL` |
| 470 KB of JavaScript in one file, parsed every page load | every ComfyUI page, whether the node is used or not | 148 KB across modules, loaded when the node is |

---

## Keeping it this way

Three tests defend the properties above, and will fail rather than let them
regress:

* `test_frontend.py::test_nothing_reads_the_graph_on_every_repaint` — no
  `onDrawForeground` hook on the node.
* `test_frontend.py::test_the_editor_never_stores_decoded_media` — no
  `toDataURL` or `readAsDataURL` anywhere in the editor.
* `test_nodes.py::test_the_director_node_carries_no_tensor_sockets` — the
  Director node cannot hold a tensor, so it cannot retain one.

And two that keep the numbers honest:

* `test_upstream_conformance.py::test_we_never_compute_latent_geometry_ourselves`
  — the compiler contains no `128`, no `// 32`, no `* 32`.
* `test_golden.py` — any change to what the compiler emits shows up as a diff.

# Quick start

Your first generated shot, in about five minutes plus however long the models
take to download.

---

## 1. Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI
```

Restart ComfyUI. Nothing else to build — the editor is plain ES modules.

Recommended, not required: [`ComfyUI-LTXVideo`][ltxvideo] from Lightricks. Without
it you still get text, image, first/last-frame and keyframe shots; with it you also
get IC-LoRA control and audio-only generation. The Director tells you which is
which rather than guessing — anything unavailable is greyed out with the reason
on hover.

[ltxvideo]: https://github.com/Lightricks/ComfyUI-LTXVideo

## 2. Get the models

LTX-2.5, from the [Hugging Face repository][hf]:

| File | Goes in |
| --- | --- |
| `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | `models/diffusion_models/` |
| `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | `models/text_encoders/` |
| `ltx-2.5-video-vae-bf16.safetensors` | `models/vae/` |
| `ltx-2.5-audio-vae-bf16.safetensors` | `models/vae/` |

Two more, only if you want them:

| File | Goes in | For |
| --- | --- | --- |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `models/latent_upscale_models/` | the refine pass |
| `gemma4_e2b_it_bf16.safetensors` | `models/text_encoders/` | prompt enhancement |

[hf]: https://huggingface.co/Lightricks/LTX-2.5

## 3. Add the node

Double-click the canvas, search **LTX Director**, and drop it in. It arrives
wide on purpose — it is an editor, not a widget.

If the models above are installed, they are already selected. The Director
suggests them once, on a project that has not chosen any; it never overrides a
choice you made.

## 4. Describe the shot

Type into the prompt box:

> A lighthouse beam sweeps across a storm-lit harbour, rain streaking through the
> light as waves break over the sea wall.

Set the duration. The field is in seconds; the readout beside it shows the frame
count, which LTX requires to be **1 plus a multiple of 8**. Ask for 5 seconds at
24 fps and you get 121 frames, not 120 — it rounds up, so a shot is never quieter
than you asked for.

## 5. Generate

Press **Generate**. The Director compiles the shot into a native LTX graph and
hands it to ComfyUI's ordinary prompt queue. Watch the queue as usual; the
finished video lands in your output folder.

If Generate is greyed out, the panel above it says why, and what to do about it.

---

## Adding an image

Drag a still onto the panel, or press **Browse…**. It uploads once — drop the
same file into another shot and nothing is transferred at all.

The mode bar switches to **Image** by itself when you add the first still. The
image lands as a keyframe anchored to the first frame; drag its marker along the
timeline to move it, or set the anchor to **Last frame** in the Selection panel
to make it the closing frame instead.

Two images, one anchored to the start and one to the end, is a first/last-frame
shot. Pick **First / Last** in the mode bar and the Director pins both.

## Changing the prompt over time

Double-click the green prompt track on the timeline. That adds a **prompt
region** — a stretch of the shot with its own text.

Add a second, give each one a line, and Prompt Relay switches on: one sampling
pass where each region's words apply most strongly over its own stretch of the
video. It is not several generations stitched together, so there are no seams.

The Selection panel shows the region you clicked, and the boundary-sharpness
slider under **Prompt Relay** controls how hard the changeover is.

## Seeing the real graph

Press **Compile workflow**. The Director writes the native LTX graph — loaders,
conditioning, sampler, decode, save — and opens it on the canvas. There is no
Director node in it. Every value is an ordinary widget you can change.

That is the same graph Generate runs. Nothing is hidden, and nothing happens
inside the Director that you cannot see written out.

---

## If something goes wrong

The panel above the buttons shows problems as sentences with fixes:

> **121 frames is not a length LTX can generate — it needs 1 plus a multiple of 8.**
> Use 121 frames (5.04 s) or 113 frames (4.71 s).

> **Image to Video needs a first-frame image, and none is anchored to the start.**
> Add a keyframe reference and set its anchor to Start.

Errors stop Generate; warnings do not. If a feature is missing rather than
misconfigured, the message names the file or component that would enable it.

## Where to go next

* [`WORKFLOWS.md`](./WORKFLOWS.md) — each mode, and when to reach for it.
* [`ADVANCED.md`](./ADVANCED.md) — guides, conditioning, memory, Prompt Relay.
* [`MIGRATION.md`](./MIGRATION.md) — bringing a Director 2.x timeline across.
* [`ARCHITECTURE.md`](./ARCHITECTURE.md) — how it is put together.

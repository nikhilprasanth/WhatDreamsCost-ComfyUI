# Advanced guide

The parts worth understanding once you are past a first render: how conditioning
actually reaches the model, what Prompt Relay does, where memory goes, and what
the Director deliberately does not do.

---

## The constraints, and why they are not negotiable

Two rules come from the model, not from us. Both are checked before anything is
compiled.

**Frame count must be `1 + 8k`.** The video VAE compresses eight pixel frames
into one latent frame, so a latent has `(length - 1) // 8 + 1` frames.
`EmptyLTXVLatentVideo` declares `length` with step 8 for exactly this reason.
The Director rounds *up* — five seconds at 24 fps is 121 frames, not 120 — so a
shot is never shorter than you asked for.

**Width and height must be multiples of 32.** The latent is `height // 32` by
`width // 32`. Anything else truncates.

A third rule comes from the guide nodes: a **multi-frame** guide's start index
must be a multiple of 8 (or `1 (mod 8)` for IC-LoRA guides, which use a
start-end patchifier). Single stills can sit anywhere. The Director warns with
the frame the model will actually use rather than moving your marker silently.

---

## How conditioning reaches the model

Three different mechanisms, and picking the right one matters.

### In-place pinning — `LTXVImgToVideoInplace`

Encodes an image and writes it into the latent at index 0, then sets the noise
mask there to `1 - strength`.

Adds **no tokens**. Cheapest possible first frame, and the reason an
image-to-video shot needs no crop afterwards. Only works at index 0.

### Guides — `LTXVAddGuide`

Encodes an image or clip and **appends** it to the latent's temporal axis as
keyframe tokens, recording where in time each one belongs via `keyframe_idxs` on
the conditioning.

This is what lets a frame be pinned anywhere, including `-1` for the last frame.
The cost is that those tokens are still on the latent when sampling ends, so
`LTXVCropGuides` has to remove them before decode — otherwise they come out as
extra frames.

The Director sets one flag when any guide is added and emits the crop from it.
There is no path where a guide is added and the crop is forgotten.

### IC-LoRA guides — `LTXAddVideoICLoRAGuide`

Same idea, but the reference is processed at the scale the LoRA was trained at,
which is why `LTXICLoRALoaderModelOnly` reports a `latent_downscale_factor` and
why the Director wires it through rather than letting you type a number.

---

## Prompt Relay

### What it does

Builds one prompt out of the shot's text plus every region's text, finds where
each region's tokens landed, and adds a **Gaussian penalty** to cross-attention
so video tokens at a given moment attend mostly to the words for that moment.

The penalty, per region:

```
cost = strength * relu(|frame - midpoint| - window)² / (2σ²)
```

`window` is roughly half the region, so inside it the penalty is zero and the
text applies fully. Outside, it grows quadratically. `σ = 1 / ln(1/epsilon)`,
which is what the boundary-sharpness slider sets: below about 0.1 the boundary
is hard, 0.5 and above blends.

### Why one pass matters

`LTXVLoopingSampler` with `MultiPromptProvider` is upstream's multi-prompt
answer: sample temporal tile 1, then tile 2, then tile 3. Different trade-off —
N times the sampling cost, and joins where the tiles meet.

Prompt Relay is one pass over the whole latent. No seams, no multiple. The cost
is a `[queries × keys]` penalty matrix, built once per shape and cached.

### Where it attaches

The model is cloned and each transformer block's `attn2` (and `audio_attn2`,
where present) forward is **wrapped**, not replaced. Wrapping composes: if
another node has already patched the same attention — KJNodes' NAG, for
instance — Prompt Relay's mask is added to theirs and both apply.

Two things it does carefully:

* The attention backend is overridden to the PyTorch path **only when the relay
  actually contributed a mask**. Sage and similar backends silently drop
  arbitrary masks, so our own mask needs it; imposing it on someone else's mask
  would change their node's behaviour behind its back.
* Everything that can fail, fails softly. An unrecognised architecture,
  attention another node has claimed, a text encoder whose tokenizer cannot be
  read — each of those merges the regions into one prompt, logs why, and renders.
  Losing per-region prompts should cost you per-region prompts, not the shot.

### In a compiled graph

Prompt Relay is the one thing the Director does that has no upstream
equivalent, so a compiled graph that uses it keeps two Director nodes in it. The
compiler says so when it happens. Switch relay off and the graph is purely
native nodes again.

Provenance: the method is Prompt Relay, described at
<https://gordonchen19.github.io/Prompt-Relay/> and first implemented for ComfyUI
in Kijai's `ComfyUI-PromptRelay`. This implementation is GPL-3.0 with the rest
of the repository.

---

## Prompting LTX-2.5

The encoder was trained on **flowing prose**, not tags. ComfyUI's own
`TextGenerateLTX2Prompt` system prompt is explicit: *"Express these as flowing
prose: a medium shot frames…, captured from a front-facing angle as the camera
slowly pans…. Never as medium shot, static camera —"*.

That is why Director mode's camera and lens controls emit sentences rather than
keywords. "Dolly in" at moderate intensity becomes *"the camera slowly dollies in
toward the subject"*, appended as a clause.

Three properties the prompt compiler holds to:

* **Your words are never paraphrased.** Director mode orders sections and adds
  terminal punctuation and a capital letter. It does not rewrite.
* **Simple and Expert compile identically.** They differ in what the panel shows,
  never in what is encoded, so switching between them cannot change a render.
* **An untouched camera control adds nothing.** "Not specified" and "static" are
  different values — the second is a direction worth stating, the first is
  silence.

Press **Compiled prompt** to see exactly what the encoder will receive. It comes
from the server, from the same function the graph runs.

### Prompt enhancement

`TextGenerateLTX2Prompt` runs a small Gemma-4 model over your prompt and expands
it. Off by default, because it costs a model load. When on, the expanded text is
shown in a `PreviewAny` node in the compiled graph — the Director does not
manipulate a prompt without showing you the result.

On an image-to-video shot the enhancer is grounded on the opening frame, which
is what the I2V system prompt is written for.

---

## Sampling

LTX-2.5's distilled model ships a **fixed sigma schedule**, not a step count:

```
stage 1:  1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0
stage 2:  0.85, 0.7250, 0.4219, 0.0
```

Eight steps, then three. These are emitted verbatim into a `ManualSigmas` node.
They are editable in the compiled graph, but they are not a hint to be tuned —
they are what the model was distilled against.

**Guidance** uses `LTXVDualCFGGuider`, which applies separate scales to the video
and audio halves of the packed latent. Distilled models want 1.0 for both.
Raising them on a distilled model mostly produces artefacts.

**Sampler** is `euler_ancestral` for the first stage and `euler` for the refine,
matching the reference graphs.

---

## Memory

Where it actually goes, in order:

1. **The transformer**, 22B parameters. Not something the Director can change —
   pick a quantised checkpoint if it does not fit.
2. **Decode.** `VAEDecodeTiled` holds `tile_size × tile_size × temporal_size`
   worth of latent at a time. This is usually what runs out, and it is what the
   memory presets change. Fewer, larger tiles are faster and need more.
3. **Sampling.** Driven by latent size, so by resolution and duration. The
   two-stage path exists partly for this: the expensive eight steps run at
   quarter the area.

Things the Director deliberately does **not** do:

* Load or unload models. ComfyUI's memory manager is authoritative, and a second
  opinion about when to evict would only fight it.
* Hold tensors. The Director node has no tensor sockets at all — there is a test
  asserting that — so it cannot retain anything between runs.
* Cache decoded media in browser state. Thumbnails are URLs; waveforms are a few
  hundred floats.

For the lowest-memory path, `ComfyUI-LTXVideo` ships `LowVRAMCheckpointLoader`
and friends that sequence model loading to fit 32 GB. The Director does not emit
them yet; add them by hand to a compiled graph.

---

## Making the graph do something else

The compiled workflow is ordinary. Everything you might want to add is a node
you can wire in:

| To add | Use |
| --- | --- |
| Spatio-temporal guidance | `LTXVApplySTG` before the guider |
| Stronger lip-sync | `LTXVModalityGuidance` on the model |
| Speaker identity from a clip | `LTXVReferenceAudio` on model + conditioning |
| Detail keyframes (DFR) | `LTXVAddGeneratedKeyframes` before the concat |
| Video extension | `LTXVExtendSampler` in place of the sampler |
| Tiled sampling for long shots | `LTXVTiledSampler` |
| A non-distilled checkpoint | Replace `ManualSigmas` with `BasicScheduler` |

If one of these turns out to be something most shots want, it belongs in the
compiler rather than in everyone's hand-edits — that is what
`director/compiler/ltx25.py` is for, and the mode dispatch table is one row per
mode.

---

## Adding a new LTX release

The compiler is one adapter per model family:

1. Copy `director/compiler/ltx25.py` to `ltx26.py`, set `family`, change what
   differs.
2. `register(LTX26Compiler())`.
3. Update `director/compiler/signatures.py` for any node whose slots moved.
4. Re-run `python tools/capture_upstream.py --ltxvideo ../ComfyUI-LTXVideo` and
   read the diff. That fixture is what `tests/test_upstream_conformance.py`
   checks the compiler against, and it has already caught two signature errors
   that would have produced silently miswired graphs.
5. `DIRECTOR_UPDATE_GOLDEN=1 pytest tests/test_golden.py`, and read that diff too.

Nothing else branches on a model family. Not the spec, not the UI, not the
validator.

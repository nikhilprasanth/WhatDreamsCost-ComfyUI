# Workflow guide

Each mode, what it compiles to, and when to reach for it.

Every mode produces a graph you can open and read — press **Compile workflow**
and the native version appears on the canvas. The node names below are the ones
you will find in it.

---

## Choosing a mode

```
Only sound, no picture?                         → Audio only
Have a soundtrack the video should follow?      → Audio
Have a clip whose motion you want restaged?     → Control
Have a still to open on?
  └─ and one to close on?                       → First / Last
  └─ and more along the way?                    → Keyframes
  └─ just the one?                              → Image
Carrying on from a shot you already made?       → Continue
Nothing but words?                              → Text
```

---

## Text

Nothing pinned. The prompt and the duration are the whole input.

Compiles to: `EmptyLTXVLatentVideo` → `LTXVEmptyLatentAudio` →
`LTXVConcatAVLatent` → `SamplerCustomAdvanced` → `LTXVSeparateAVLatent` →
`VAEDecodeTiled` + `LTXVAudioVAEDecode` → `CreateVideo` → `SaveVideo`.

Picture and sound are generated together by one model, in one pass. That is why
there is no separate audio step: LTX-2.5 is a joint audio/video transformer, and
the two latents travel concatenated through the sampler.

## Image

One still, anchored to the first frame.

Compiles to the Text graph plus `LTXVPreprocess` → `LTXVImgToVideoInplace`.

`LTXVImgToVideoInplace` writes the encoded image into the latent at index 0. It
adds no extra tokens, which is why an image-to-video shot needs no
`LTXVCropGuides` afterwards — and why it is cheaper than pinning the same frame
as a guide.

**Strength** defaults to 0.7, the value the reference graphs use. Lower lets the
model reinterpret the opening frame; 1.0 holds it exactly and can make the first
moment of motion stiff.

### Why the image is compressed first

`LTXVPreprocess` applies H.264 artefacts to conditioning frames, at CRF 18 by
default. That is not damage — LTX was trained on compressed footage, and a
pristine guide sits outside what it expects, which tends to make motion stick.
Set **Guide compression** to 0 under Quality if you want to see the difference.

## First / Last

Two stills: one anchored to the start, one to the end.

Compiles to `LTXVAddGuide(frame_idx=0)` → `LTXVAddGuide(frame_idx=-1)` →
sampler → `LTXVCropGuides` → decode.

The closing frame uses index `-1` rather than a computed frame number, so it
stays the last frame if you change the shot's length later.

`LTXVCropGuides` is not optional. Guides append keyframe tokens to the latent's
temporal axis; without the crop they decode as extra frames on the end.

## Keyframes

Any number of pinned frames. The Director chains one `LTXVAddGuide` per
reference, in order: start anchors first, then indexed ones by time, then end
anchors.

**Placement.** A single still can sit anywhere. A *video* used as a guide has to
start on a multiple of 8, and the Director warns you with the frame it will
actually use rather than moving it silently. Hold **Shift** while dragging a
marker to snap to that grid.

**Strength** is per reference, in the Selection panel. Mixing strengths is
useful: pin the opening frame hard and let a mid-shot reference sit at 0.5 as a
suggestion.

## Continue

Image-to-video, seeded from the last frame of a shot you already made.

There is no separate machinery for this and there does not need to be: a
continuation *is* an image-to-video shot whose opening frame came from
somewhere specific. Extract the final frame of the previous clip, drop it in,
adjust the prompt, generate.

For consistency across a longer sequence, add the same character or environment
references to every shot and use the Ingredients IC-LoRA — identity holds far
better through a shared reference sheet than through last-frame chaining alone.

## Audio

A soundtrack you supply drives the picture.

Compiles to `LoadAudio` → `TrimAudioDuration` → `LTXVAudioVAEEncode` →
`LTXVFreezeLatent` → `LTXVConcatAVLatent`.

Freezing matters. The encoded audio goes through the sampler with its noise mask
at zero, so it is never denoised — it only provides cross-attention for the
video to follow. The original waveform is then muxed straight into the output
rather than decoded back, because re-decoding audio the user supplied can only
degrade it.

**Audio modes**, under the Audio section:

| Mode | What happens |
| --- | --- |
| Generate with the video | LTX writes the soundtrack. The default. |
| Use a file I supply | Your audio, held exactly, video follows it. |
| Fill gaps around my clips | *Approximated* — see below. |
| No audio | No audio branch at all; a slightly smaller, faster graph. |

Gap filling is the one thing here that does not do what its name says. Doing it
properly needs a per-audio-frame noise mask, and no upstream node builds one, so
the compiled graph generates across the whole track. The Director tells you this
in a warning rather than shipping something that quietly differs from the
timeline.

## Audio only

Sound with no picture.

Compiles to `LTXVAudioOnlyModel` on the model and
`LTXVAudioOnlyEmptyVideoLatent` as a placeholder, with no video VAE, no decode
and no `CreateVideo`.

The placeholder is not a waste. LTX splits its input positionally into
`[video, audio]`, so the sampler needs *something* at index 0; with
`LTXVAudioOnlyModel` active the cross-attention between the two streams is
severed and the placeholder is never attended to.

Needs `ComfyUI-LTXVideo`.

## Control (IC-LoRA)

A reference clip steers generation through an IC-LoRA.

Compiles to `LTXICLoRALoaderModelOnly` on the model and
`LTXAddVideoICLoRAGuide` per reference, then `LTXVCropGuides`.

The LoRA's own `latent_downscale_factor` output is wired into the guide rather
than typed in — different IC-LoRAs were trained at different reference scales,
and reading it from the loader means you cannot get it wrong.

IC-LoRA guides sit at `1 (mod 8)`, not `0 (mod 8)`, because they use a
start-end patchifier. The Director snaps to that and shows you where.

Needs `ComfyUI-LTXVideo`.

---

## Prompt Relay

Several prompts over one shot, in **one** sampling pass.

Double-click the prompt track to add a region; give two or more regions text and
the relay switches on. Each region's words apply most strongly over its own
stretch of the video, fading outside it.

This is not the same as generating several clips and joining them, and it is not
the same as `LTXVLoopingSampler`, which samples temporal tiles in sequence.
Prompt Relay biases cross-attention inside a single pass — so no seams, and no
multiple of the sampling cost.

**Boundary sharpness** (the `epsilon` slider) sets how hard the changeover is.
Low values cut; 0.5 and above blend.

A compiled graph that uses Prompt Relay keeps two Director nodes in it — the
project and the relay — because nothing upstream does this. That is called out
in a diagnostic when you compile, with the note that switching relay off gives
you a purely native graph.

---

## One stage or two

Under **Quality**.

**One pass** generates at the output size. Faster, and everything else being
equal, softer.

**Generate then refine** generates at half size, upscales the latent with
`LTXVLatentUpsampler`, and runs a short three-step refine over it. Composition
and motion are decided cheaply at the small size; detail arrives in the second
pass. This is what the upstream two-stage graphs do and it is the better default
when you care about the picture.

Two things the Director handles for you here:

* An image-to-video shot is **re-pinned** after upsampling, at full strength.
  `LTXVLatentUpsampler` drops the noise mask, so without it the opening frame
  drifts during the refine.
* Guides are **cropped before** the upsampler, not after. Keyframe tokens live
  on the latent's temporal axis and upsampling would scale them as if they were
  picture.

## Seed hunting

Under **Quality**, choose the **Seed hunt** preset: one stage, small, random
seed each run. Queue it several times, watch what comes back, and note the seed
of the one you like.

Then open **Takes**, press **Reuse seed** on it, switch the preset to **Quality**
or **Final render**, and generate again. Same seed, same composition, far more
compute spent on it.

Takes are a log, not an archive: seed, time, and the settings digest, capped at
fifty. The videos themselves are in your output folder where ComfyUI put them.

## Memory

Under **Quality**, the **Memory** dropdown. It changes decode tiling and, on the
16 GB profile, the first-stage divisor — and nothing else. A memory preset never
changes resolution, duration or sampling, so it cannot quietly change how the
shot looks.

Fewer, larger decode tiles are faster and need more VRAM. If decoding is where
you run out of memory, step down one profile before reducing the shot itself.

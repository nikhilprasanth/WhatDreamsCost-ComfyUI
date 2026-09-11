# Migrating from Director 2.x

**Your existing workflows keep working.** `LTXDirector`, `LTXDirectorGuide` and
`LTXDirectorCropGuides` are still registered — they are labelled *(legacy)* in
the node menu, and nothing about them changed except where two helper modules
live. Open a saved Director 2 workflow and it loads.

This document is about moving a shot across when you want to.

---

## The short version

1. Open the Director 2 node and copy its `timeline_data` widget value.
2. Add an **LTX Director** node.
3. Paste into the import dialog, or `POST` it to `/ltxdirector/project/import`.
4. Read what the importer tells you it could not carry.

---

## What comes across

| Director 2 | Director Next |
| --- | --- |
| `global_prompt` | the shot prompt |
| text segments' `prompt` | prompt regions (Prompt Relay) |
| segment `start` / `length` | region timing, rebased to zero |
| image segments | keyframe references |
| `isEndFrame` | anchor set to **Last frame** |
| first segment at frame 0 | anchor set to **First frame** |
| `guide_strength` (positional) | per-reference strength |
| `motionSegments` | motion references, IC-LoRA mode |
| `audioSegments` | audio clips, with trim converted to seconds |
| `inpaint_audio` | audio mode: fill gaps / use file |
| `normalStartFrame` / `normalDurationFrames` | shot length, rebased and snapped |
| frame rate, dimensions, compression | the same settings |

The importer also **infers the mode**: a motion segment means Control, a first
and a last frame mean First / Last, several keyframes mean Keyframes, one means
Image, none means Text.

## What does not, and why

### Inline base64 media

Director 2 could store an image inside the timeline as base64. Director Next
stores file references only — that is most of why a project is now a few
kilobytes rather than a few megabytes, and why a saved workflow no longer
carries pixel data.

A segment that only ever existed as base64 produces:

> **A keyframe was stored inline as base64 rather than as a file, so it could not
> be imported.**
> Drag the image back onto the timeline; Director Next uploads it once and
> references it by name.

Everything else in that timeline still imports. Re-add the image and you are
done.

### Retake mode

The region and its prompt come across as a prompt region, with:

> **This timeline used Retake mode. Its region and prompt were imported, but the
> re-generation itself was not — v2 built the mask by hand, which is why it was
> never potent enough.**
> Use IC-LoRA mode with the in/outpainting LoRA for region edits.

That is the honest position. Director 2's own README said retake "is not potent
enough"; the reason is that it assembled a temporal mask itself rather than
using the path upstream actually supports. Regenerating part of a clip is a job
for the in/outpainting IC-LoRA, which has dedicated upstream example graphs.

### Override audio

Taking the soundtrack from the IC-LoRA video is now done by loading that audio
onto the audio track, so it is visible and trimmable like any other clip. The
importer notes it; add the clip if you want it kept.

### Editor chrome

Track visibility, panel heights, filename display, snapping preferences. None of
it described the shot.

---

## Duration changes on import

Director 2 let the timeline be any length. LTX needs `1 + 8k` frames, so a
120-frame timeline becomes 121 and the importer says so:

> **The v2 timeline was 120 frames long; LTX needs 1 plus a multiple of 8, so
> the shot is now 121 frames.**

If the old timeline had a render window that did not start at frame 0, everything
is rebased: a keyframe at frame 24 in a window starting at 24 becomes a first
frame. Anything entirely before the window is dropped — it was not being
rendered.

---

## Importing from Python

```python
import json
from director.migrations.v2_timeline import convert_with_report

timeline = json.loads(open("old_workflow.json").read())
node = next(n for n in timeline["nodes"] if n["type"] == "LTXDirector")
widgets = node["widgets_values"]

result = convert_with_report(
    json.loads(widgets[6]),          # timeline_data
    fps=float(widgets[14]),
    width=int(widgets[16]),
    height=int(widgets[17]),
    guide_strength=str(widgets[22]),
    img_compression=int(widgets[20]),
)

print(result.spec.to_json(indent=1))
for diagnostic in result.diagnostics:
    print(diagnostic.level, diagnostic.message, diagnostic.fix)
```

`Spec.from_dict` also accepts a raw v2 timeline directly — it is recognised by
shape — so pasting one into the `project` widget works too.

---

## What changed underneath

Worth knowing if you built anything on top of Director 2.

**Node layout.** One node with 26 inputs and 8 outputs became three: the
Director (project in, derived values out), the Prompt Relay
(model + clip + spec → model + conditioning), and the Compile node.

**The `GUIDE_DATA` and `MOTION_GUIDE_DATA` sockets are gone.** They carried
full-resolution tensors *and* a copy of the timeline JSON between two nodes, with
no schema. Guides are now inserted by `LTXVAddGuide` and
`LTXAddVideoICLoRAGuide` — the upstream nodes that do exactly this and track the
model.

**Moved modules.** `prompt_relay.py` → `director/relay/{mask,tokens}.py`;
`patches.py` → `director/relay/{patch,support}.py`. Both old paths remain as
documented re-export shims, so existing imports resolve.

One behaviour change in those: `apply_patches` now **returns** a reason string
instead of raising when it cannot attach. A model it does not recognise costs
the caller its per-region prompts rather than its render. Callers that ignored
the return value are unaffected.

**Latent geometry.** Director 2 built latents by hand with literal constants —
`torch.zeros([1, 128, ..., h // 32, w // 32])` — and read guide dimensions as
`latent_width * 32`. Those numbers were right for the LTX-2.3 VAE and would
silently corrupt geometry if it changed. Director Next never computes them:
`EmptyLTXVLatentVideo` and the VAE's own `downscale_index_formula` do. There is
a test asserting the compiler contains no such literal.

---

## Keeping both

There is no need to choose. The two sets of nodes coexist, the legacy example
workflows still open, and the Director 2 editor is untouched. Move a shot across
when you want what the new one does — and if something does not come across that
you needed, that is worth reporting rather than working around.

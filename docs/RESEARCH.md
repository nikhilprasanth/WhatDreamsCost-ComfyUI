# LTX Director — Phase 0 Research

Audit date: **2026-09-11**
Repo state: `WhatDreamsCost-ComfyUI` @ `a3c809c` (v2.0.5), branch `director-next` cut from `main`.
Upstream reference clone: `Lightricks/ComfyUI-LTXVideo` @ `15d09ab` (2026-08-20).

Everything below was verified against source in this workspace or fetched from upstream
`master` during the audit. Where the original task brief and reality disagree, reality wins and
the discrepancy is called out under [Corrections to the brief](#corrections-to-the-brief).

---

## 1. What LTX actually is today

### 1.1 Version reality

| Thing | Status |
| --- | --- |
| LTX-2.5 | **Current.** Example graphs live in `ComfyUI-LTXVideo/example_workflows/2.5/` (10 graphs + README). |
| LTX-2.3 | Previous generation, still shipped in `example_workflows/2.3/`. |
| LTX-2.0 | Legacy, `example_workflows/2.0/`. |
| `ComfyUI-LTXVideo/README.md` | **Stale** — still documents 2.3 as current and never mentions 2.5. Do not trust it; trust the graphs and the Python. |

2.3 and 2.5 checkpoints/components are **not interchangeable** — different transformer, different
VAEs, different text encoder. Anything the Director emits must pick one family and stay internally
consistent.

### 1.2 LTX-2.5 model set

| Role | File | ComfyUI folder | Loader node |
| --- | --- | --- | --- |
| Transformer (distilled) | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` / `…-comfy-int8-convrot.safetensors` | `diffusion_models/` | `UNETLoader` |
| Video VAE | `ltx-2.5-video-vae-bf16.safetensors` | `vae/` | `VAELoader` |
| Audio VAE | `ltx-2.5-audio-vae-bf16.safetensors` | `vae/` | `VAELoader` |
| Text encoder | `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` / `…-comfy-int8-convrot.safetensors` | `text_encoders/` | `CLIPLoader(type="ltxv")` |
| Prompt enhancer | `gemma4_e2b_it_bf16.safetensors` / `gemma4_e2b_it_int8_convrot.safetensors` | `text_encoders/` | `CLIPLoader(type="ltxv")` |
| Spatial upscaler | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `latent_upscale_models/` | `LatentUpscaleModelLoader` |

Two notable changes from 2.3:

* The **text projection is baked into the encoder file**. 2.3 needed `DualCLIPLoader` with a
  separate `ltx-2.3_text_projection` file; 2.5 uses a single `CLIPLoader`.
* There is a **second, small text encoder** (`gemma4_e2b`) used only by the prompt-enhancer node.
  It is optional.

IC-LoRAs are still 2.3-named (`ltx-2.3-22b-ic-lora-*`) even in the 2.5 graphs — the 2.5 example
workflows load `ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors` against the 2.5 transformer. So
IC-LoRA file naming is **not** a reliable version signal.

### 1.3 What 2.5 adds that a Director should care about

* **Native multishot** — one generation can yield several connected shots holding character,
  environment, lighting and voice across cuts. This changes the answer to "multi-shot: one
  generation or several?" (see §5.4).
* **Auto duration** — `LTXVDurationPredictor` (ComfyUI core) runs a duration head and returns
  `num_frames` snapped to the 8k+1 grid, before diffusion.
* **Generated keyframes / Diffusion Fidelity Rendering** — `comfy_extras/nodes_lt_keyframes.py`
  adds keyframe tokens that are *denoised alongside* the video but not decoded, then can be pinned
  as frozen guides on a later (upscaled) canvas.
* **Dual CFG** — `LTXVDualCFGGuider` gives independent `video_cfg` / `audio_cfg` on the packed AV
  latent. The official 2.5 templates use this instead of plain `CFGGuider`.
* **Modality guidance** — `LTXVModalityGuidance` strengthens A/V coupling (lip-sync) with one
  extra forward pass per step.
* **Reference audio / ID-LoRA** — `LTXVReferenceAudio` transfers speaker identity from a ~5 s clip.
* **`LTXVFreezeLatent`** — first-class "don't denoise this" instead of the
  `SolidMask` + `SetLatentNoiseMask` hack the 2.5 A2V example still uses.

---

## 2. The canonical LTX-2.5 graph

This is the single most valuable artifact of the audit: the exact node topology the Director's
compiler must reproduce. Extracted from `ComfyUI-LTXVideo/example_workflows/2.5/*.json` and the
official ComfyUI templates `video_ltx2_5_{t2v,i2v,flf2v}.json`.

### 2.1 Shared spine

```
UNETLoader ─────────────────────────────────┐
CLIPLoader (gemma4-12b-with-proj, "ltxv") ──┤
VAELoader (video vae) ──────────────────────┤
VAELoader (audio vae) ──────────────────────┤
LatentUpscaleModelLoader (two-stage only) ──┘

prompt ─▶ [TextGenerateLTX2Prompt]? ─▶ CLIPTextEncode ─┐
negative ───────────────────────────▶ CLIPTextEncode ──┴─▶ LTXVConditioning(frame_rate)

EmptyLTXVLatentVideo(w, h, length, 1) ──▶ [mode-specific conditioning] ──┐
LTXVEmptyLatentAudio(audio_vae, frames, fps) ───────────────────────────┴─▶ LTXVConcatAVLatent

RandomNoise ─┐
LTXVDualCFGGuider(model, pos, neg, video_cfg, audio_cfg) ─┤
KSamplerSelect("euler_ancestral") ────────────────────────┼─▶ SamplerCustomAdvanced ─▶ LTXVSeparateAVLatent
ManualSigmas("1.0, 0.99375, …, 0.0") ─────────────────────┘                              │        │
                                                                                    video│        │audio
VAEDecodeTiled(512, 64, 64, 16) ◀───────────────────────────────────────────────────────┘        │
LTXVAudioVAEDecode ◀─────────────────────────────────────────────────────────────────────────────┘
        └─▶ CreateVideo(images, audio, fps, bit_depth) ─▶ SaveVideo
```

### 2.2 Mode deltas

| Mode | Insert between `EmptyLTXVLatentVideo` and `LTXVConcatAVLatent` |
| --- | --- |
| **T2V** | nothing |
| **I2V** | `LTXVPreprocess(img_compression=18)` → `LTXVImgToVideoInplace(vae, image, latent, strength=0.7, bypass)` |
| **FFLF** | `LTXVPreprocess` ×2 → `LTXVAddGuide(frame_idx=0, strength=0.7)` → `LTXVAddGuide(frame_idx=-1, strength=0.7)`; **`LTXVCropGuides` after sampling, before decode** |
| **Multi-keyframe** | chained `LTXVAddGuide(frame_idx=N)` + `LTXVCropGuides` |
| **A2V** | `LTXVAudioVAEEncode(TrimAudioDuration(audio))` → freeze audio (`LTXVFreezeLatent`, or the older `SolidMask`+`SetLatentNoiseMask`) and mux the original waveform into `CreateVideo` instead of decoding |
| **T2A** | `LTXVAudioOnlyModel` on the model; `LTXVAudioOnlyEmptyVideoLatent` as the dummy video latent; no video VAE/decode |
| **IC-LoRA** | `LTXICLoRALoaderModelOnly` on the model + `LTXAddVideoICLoRAGuide(positive, negative, vae, latent, image, frame_idx, strength, latent_downscale_factor, crop, …)`; `LTXVCropGuides` after sampling |

### 2.3 Two-stage (recommended default)

Stage 1 runs at **half resolution** (`ComfyMathExpression "a/2"` on width and height), 8 steps.
Then:

```
stage1 video latent ─▶ LTXVLatentUpsampler(upscale_model, vae)
                    ─▶ [LTXVImgToVideoInplace again, strength=1.0]   (I2V only, re-pins frame 0)
                    ─▶ LTXVConcatAVLatent(stage1 audio latent)
                    ─▶ SamplerCustomAdvanced with ManualSigmas("0.85, 0.7250, 0.4219, 0.0")
```

### 2.4 Hard numbers worth pinning

| Quantity | Value | Source |
| --- | --- | --- |
| Distilled stage-1 sigmas (8 steps) | `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0` | every 2.5 graph |
| Refine stage-2 sigmas (3 steps) | `0.85, 0.7250, 0.4219, 0.0` | two-stage graphs |
| Sampler | `euler_ancestral` (`euler` for refine in some graphs) | 2.5 graphs |
| CFG | `1.0` video / `1.0` audio for distilled | `LTXVDualCFGGuider` widget values |
| Frame count | `1 + floor(fps * seconds / 8) * 8` (LTXVideo graphs) or `fps * seconds + 1` (ComfyUI templates) | `ComfyMathExpression` |
| Latent geometry | `[B, 128, (len-1)//8 + 1, H//32, W//32]` | `EmptyLTXVLatentVideo.execute` |
| `LTXVPreprocess` compression | `18` | every graph |
| Decode tiling | `tile_size 512, overlap 64, temporal 64–128, temporal_overlap 8–32` | 2.5 graphs |
| Resize before encode | `ResizeImageMaskNode("scale longer dimension", 1536, "lanczos")` | I2V graphs |

Note the two frame-count formulas disagree. `1 + floor(fps*s/8)*8` is 8k+1 by construction;
`fps*s+1` only lands on 8k+1 when `fps*s` is a multiple of 8. **The Director must use the 8k+1
form** — `LTXVAddGuide` and the patchifier assume it.

### 2.5 Modern ComfyUI workflow format

The 2.5 graphs are workflow schema **`version: 0.4`** with a `definitions.subgraphs` array. Each
subgraph is `{id (uuid), name, inputs[], outputs[], nodes[], links[]}` and is instantiated on the
parent canvas as a node whose `type` is the subgraph UUID. Widget promotion works by the subgraph
node carrying a `widgets_values` array matching its promoted inputs.

This matters enormously: **the Director can emit a subgraph-structured graph**, so "Compile
Workflow" produces something that looks like the official templates — five tidy boxes — rather
than 40 loose nodes. Expanding a subgraph is a built-in frontend affordance, which gives us
"Compact Mode / Expanded Mode" for free.

### 2.6 Core LTX node inventory (ComfyUI master, verified)

`comfy_extras/nodes_lt.py`
: `EmptyLTXVLatentVideo`, `LTXVImgToVideo`, `LTXVImgToVideoInplace`, `ModelSamplingLTXV`,
  `LTXVConditioning`, `LTXVScheduler`, `GetICLoRAParameters`, `LTXVAddGuide`, `LTXVAddLatentGuide`,
  `LTXVPreprocess`, `LTXVCropGuides`, `LTXVConcatAVLatent`, `LTXVSeparateAVLatent`,
  `LTXVReferenceAudio`, `LTXVDualCFGGuider`, `LTXVModalityGuidance`, `LTXVSpatioTemporalGuidance`,
  `LTXVDurationPredictor`

`comfy_extras/nodes_lt_audio.py`
: `LTXVAudioVAELoader`, `LTXVAudioVAEEncode`, `LTXVAudioVAEDecode`, `LTXVEmptyLatentAudio`,
  `LTXAVTextEncoderLoader`

`comfy_extras/nodes_lt_keyframes.py`
: `LTXVAddGeneratedKeyframes`, `LTXVSeparateGeneratedKeyframes`,
  `LTXVGeneratedKeyframesToGuides`, `LTXVFreezeLatent`

`comfy_extras/nodes_lt_upsampler.py`
: `LTXVLatentUpsampler`

`comfy_extras/nodes_textgen.py`
: `TextGenerate`, `TextGenerateLTX2Prompt`

`ComfyUI-LTXVideo` (custom nodes) adds ~60 more: `LTXVBaseSampler`, `LTXVExtendSampler`,
`LTXVInContextSampler`, `LTXVLoopingSampler` + `MultiPromptProvider`, `LTXVTiledSampler`,
`LTXVAddGuideAdvanced`, `LTXAddVideoICLoRAGuide(+Advanced)`, `LTXICLoRALoaderModelOnly`,
`LTXVSetAudioRefTokens`, `LTXVAudioOnlyModel`, `LTXVAudioOnlyEmptyVideoLatent`,
`LowVRAM{Checkpoint,AudioVAE,LatentUpscaleModel}Loader`, `LTXVSaveConditioning` /
`LTXVLoadConditioning`, `LTXVSparseTrackEditor` / `LTXVDrawTracks`, `LTXVPromptEnhancer`,
`LTXVGemmaEnhancePrompt`, `GemmaAPITextEncode`, `DynamicConditioning`, STG nodes, Q8 nodes,
HDR nodes, inpaint/vanish nodes, latent-norm nodes.

**Licence note:** `ComfyUI-LTXVideo` is under the **LTX-2 Community License Agreement**, not
Apache/MIT. This repo is **GPL-3.0**. Those are not obviously compatible, so the rule for Director
Next is: *reference their node IDs and input names as interface facts; never vendor their code.*

---

## 3. Audit of the existing Director (v2.0.5)

### 3.1 Shape of the thing

| File | Lines / size | Role |
| --- | --- | --- |
| `js/ltx_director.js` | **11,769 lines / 470 KB** | One `TimelineEditor` class + one `app.registerExtension` |
| `ltx_director.py` | 1,380 lines / 62 KB | `LTXDirector` node + 4 aiohttp routes + media/audio helpers |
| `ltx_director_guide.py` | 700 lines / 35 KB | `LTXDirectorGuide`, `LTXDirectorCropGuides` |
| `prompt_relay.py` | 240 lines | Attention-mask maths (clean, reusable) |
| `patches.py` | 210 lines | Cross-attention monkey-patching for Wan + LTX |
| `load_video_ui.py` / `.js` | 23 KB / 81 KB | Standalone video loader node |
| `load_audio_ui.py` / `.js` | 6 KB / 28 KB | Standalone audio loader node |
| `ltx_keyframer.py`, `ltx_sequencer.py`, `multi_image_loader.py` | small | Predecessors, self-described as superseded |

### 3.2 What it actually does

`LTXDirector.execute` is a single node with **26 inputs and 8 outputs** that internally:

1. Parses `timeline_data` (a JSON blob serialised from the browser).
2. For each image/video segment: decodes from disk (PIL / PyAV), resizes, then **re-encodes and
   re-decodes through an in-memory H.264 mp4** to simulate compression artefacts.
3. Allocates the video latent by hand: `torch.zeros([1, 128, (len-1)//8+1, h//32, w//32])`.
4. Builds combined audio by decoding every audio segment with PyAV into a 44.1 kHz buffer.
5. Encodes audio to a latent and hand-builds a 3-D `noise_mask` for audio inpainting.
6. Tokenises the global + per-segment prompts, computes token ranges, builds a Gaussian temporal
   penalty matrix, and **monkey-patches every transformer block's `attn2` / `audio_attn2`**.
7. Emits two bespoke socket types (`GUIDE_DATA`, `MOTION_GUIDE_DATA`) carrying **raw tensors plus
   the entire timeline JSON string** to `LTXDirectorGuide`, which does the VAE encoding and guide
   insertion.

### 3.3 Things worth keeping

These are real, load-bearing, and not available upstream:

* **Prompt Relay** (`prompt_relay.py` + `patches.py`). Per-segment prompts in a *single* sampler
  pass via an additive Gaussian cross-attention penalty. Upstream's only multi-prompt answer is
  `LTXVLoopingSampler` + `MultiPromptProvider`, which is *chunked sequential sampling* — a
  different (slower, seam-prone) trade-off. Prompt Relay is the Director's genuine moat.
* **The timeline metaphor itself.** Draggable prompt/keyframe/audio/IC-LoRA segments on a
  frame-accurate ruler with thumbnails, snapping, in/out points, split-at-playhead.
* **Timestamp → frame-index arithmetic** with a seconds/frames display toggle.
* **Timeline save/load** to JSON.
* **End-frame flag** (turn any keyframe into a last-frame anchor) — maps cleanly onto
  `LTXVAddGuide(frame_idx=-1)`.
* **Audio inpainting** — generate audio only in the gaps between imported clips. Genuinely
  clever; the mechanism (a noise mask over audio latent frames) survives into 2.5.
* **Chunked upload endpoint** (`/ltx_director_upload_chunk`) with a dedup pre-check
  (`/ltx_director_check_file`) — large media never hits the 100 MB `/upload/image` ceiling.
* **Audio peak extraction** for waveform display, cached server-side.

### 3.4 Problems

Ordered by how much they constrain the rebuild.

**A. Hardcoded LTX-2.3 geometry.**
`ltx_director.py:1185` builds the latent as `[1, 128, …, h//32, w//32]` with literal constants, and
`ltx_director_guide.py:401` computes `target_width = latent_width * 32`. Both ignore
`vae.downscale_index_formula`, which is the value the core nodes actually use. Any VAE change
silently corrupts geometry. `EmptyLTXVLatentVideo` already does this correctly.

**B. Re-implements core nodes.**
`_compress_image` is `LTXVPreprocess`. Manual latent allocation is `EmptyLTXVLatentVideo`. Manual
audio-latent allocation is `LTXVEmptyLatentAudio`. Guide insertion in `LTXDirectorGuide` is
`LTXVAddGuide` / `LTXAddVideoICLoRAGuide`. Four re-implementations that must be re-verified on
every upstream change.

**C. Work is redone on every queue.**
Every run re-decodes every image and video segment from disk, re-resizes, re-encodes/decodes H.264,
re-decodes every audio file, and re-encodes audio to latents. None of it is cached, and none of it
is keyed by content hash. For a 10-segment timeline this is seconds of CPU before sampling starts.

**D. Tensors travel through the graph as custom sockets.**
`GUIDE_DATA` carries a list of full-resolution `[N,H,W,3]` float32 tensors *plus a copy of the
entire timeline JSON string*, and `LTXDirectorGuide` re-parses that JSON to rediscover retake
state. Two nodes, one hidden protocol, no schema.

**E. Monkey-patching the model.**
`patches.py` replaces `.forward` on every transformer block through `add_object_patch`, and
explicitly raises if another node got there first ("Stacking is not supported"). It also bypasses
the user's attention backend (SageAttention etc.) whenever a mask is present. The mask *concept* is
sound; the delivery mechanism is brittle.

**F. Frontend cost.**
- 11,769 lines in one file, one class, no modules, no tests.
- `onDrawForeground` calls `_syncGlobalPromptFromLink()` — a graph link traversal plus DOM writes
  — **on every canvas repaint**.
- Thumbnails are extracted browser-side by seeking a `<video>` element frame by frame.
- Base64 image fallbacks (`imageB64`) can land inside the saved workflow JSON.
- Node state is spread across `this.properties`, hidden widgets, and `TimelineEditor` instance
  fields, kept in sync by hand.

**G. Global side effects at import.**
`ltx_director.py` installs a process-wide asyncio exception handler at import time to silence
Windows `ConnectionResetError`. It also registers four routes at import.

**H. `get_current_node_id()` walks CPython stack frames** looking for a local named `unique_id`, to
key a model cache. Works, but it is a hidden dependency on ComfyUI's executor internals.

**I. No validation and opaque errors.**
Bad geometry surfaces as a `KeyError` or a shape mismatch deep inside the sampler. There is no
pre-flight check on 8k+1 frame counts, divisible-by-32 dimensions, guide ordering, or whether the
selected model even supports the requested mode.

**J. No tests at all**, Python or JS.

### 3.5 Version debt in the shipped workflows

`example_workflows/LTX_Director_2_Workflow_Distilled.json` is LTX-2.3-shaped: `DualCLIPLoader`
with a separate text-projection file, `BasicScheduler("linear_quadratic")`, `VAEDecode`,
`ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors`, and two `LTXDirectorGuide`
nodes plus two `LTXDirectorCropGuides`. It also depends on `ComfyUI-KJNodes`
(`VAELoaderKJ`, `ModelPreviewOverrideKJ`). None of that is how a 2.5 graph is built.

---

## 4. Modern ComfyUI patterns to adopt

| Area | Current Director | What to do |
| --- | --- | --- |
| Node schema | V3 `io.Schema` in `ltx_director.py`; **V1 `INPUT_TYPES` dict** in `ltx_director_guide.py` | V3 `io.Schema` everywhere, registered via `ComfyExtension.get_node_list` |
| Registration | Hybrid `comfy_entrypoint` + `NODE_CLASS_MAPPINGS` | Keep both (Manager needs the static map) but generate the static map from one source |
| Workflow JSON | Hand-authored flat graph | Emit schema `0.4` with `definitions.subgraphs` |
| Frontend | One 470 KB file | ES modules under `js/director/`, one concern per file |
| Media upload | Custom chunked endpoint | Keep it (`/upload/image` caps at 100 MB) but namespace routes and add content-hash dedup |
| Big data across nodes | Raw tensors in custom sockets | Pass a small, versioned, JSON-serialisable **spec**; let LTX nodes own the tensors |
| Node state | `properties` + hidden widgets + editor fields | One serialised `project` string widget; everything derived |

---

## 5. Design questions the research settles

### 5.1 Should the Director keep executing tensor work?

**No, for the guide path.** `LTXVAddGuide`, `LTXVImgToVideoInplace`, `LTXVPreprocess`,
`EmptyLTXVLatentVideo`, `LTXVEmptyLatentAudio`, `LTXVLatentUpsampler` and
`LTXAddVideoICLoRAGuide` cover everything `LTXDirectorGuide` does by hand — and they track the
model. The Director should *compile to* them.

**Yes, for Prompt Relay.** Nothing upstream does single-pass per-segment conditioning. That stays
a Director node, but as a **small, focused node** (`model + clip + relay spec → model + conditioning`)
rather than a side effect of a 26-input mega-node.

### 5.2 Compile to a graph, or execute a graph?

Both, at different layers:

* **Director Spec** — a versioned JSON document describing the shot (timeline, references,
  prompts, audio, generation settings). Pure data, no tensors.
* **Compiler** — pure function `Spec → ComfyUI workflow JSON (schema 0.4, subgraphed)`. Testable
  without a GPU, without ComfyUI, without models. This is where LTX version adapters live.
* **Runtime nodes** — a thin set that consume the spec for the parts a static graph can't express
  (prompt relay conditioning).

The compiler being a pure function is what makes "survive the next LTX release" true: a new
release is a new adapter module plus a new preset, not a rewrite.

### 5.3 Retake / regional workflows — real or not?

**Partly real.** Upstream supports:
* temporal region preservation via `noise_mask` on latent frames (`LTXVFreezeLatent`,
  `SetLatentNoiseMask`) — this is what the existing retake mode approximates;
* spatial masking via `LTXVPreprocessMasks` + the in/outpainting IC-LoRA
  (`ltx-2.3-22b-ic-lora-in-outpainting-0.9`), with dedicated 2.5 example graphs;
* per-guide spatial attention masks via `LTXVAddGuide(attention_mask=…)`.

So "select a temporal range and re-generate it" is supported, and "select a spatial region" is
supported *through the inpaint IC-LoRA*. The existing README admits retake "is not potent enough";
the reason is that it fakes it with a hand-built mask rather than using the inpaint path.

### 5.4 Multi-shot — one generation or many?

2.5 has **native multishot**: one generation produces connected shots holding identity across cuts.
That means the Director should offer both and be explicit about which is which:

* **In-generation multishot** — one clip, shot boundaries expressed in the prompt, optionally
  reinforced by Prompt Relay segments. Best consistency, bounded by max clip length.
* **Sequence of clips** — separate generations chained by last-frame → first-frame (the
  Continuation mode). Unbounded length, weaker consistency, mitigated by the Ingredients IC-LoRA
  reference sheet.

### 5.5 Video extension

`LTXVExtendSampler` and `LTXVLoopingSampler` exist upstream for this. Rather than reimplementing
frame bookkeeping, the Director should compile continuation as *"new generation, first frame
pinned from the previous clip's last frame"* (which is just I2V) and expose
`LTXVExtendSampler` as an advanced preset.

---

## 6. Corrections to the brief

| Brief says | Reality |
| --- | --- |
| "latest LTX 2.5 / current LTX Video ecosystem" | Correct, but the upstream **README** still documents 2.3. The 2.5 truth is in `example_workflows/2.5/` and in ComfyUI core. |
| Study `ltx_director_guide.py` "JS files" | `js/ltx_director_guide.js` is 46 lines and only hides a widget. Effectively all UI is in `ltx_director.js`. |
| "GGUF interoperability" | No GGUF anywhere in upstream 2.5. The repo's own `LTX_Director_2_Workflow_GGUF.json` uses third-party GGUF loaders against **2.3**. Not a 2.5 capability. |
| "Gemma conditioning" | Two distinct things: `CLIPLoader(type="ltxv")` on a Gemma-4 encoder (conditioning) and `TextGenerateLTX2Prompt` on a small Gemma-4 (prompt enhancement). `LTXVGemmaEnhancePrompt` / `GemmaAPITextEncode` in ComfyUI-LTXVideo are a *third*, API-key path. |
| "Camera-control presets" | There **are** real camera-control LoRAs (`ltx-2-19b-lora-camera-control-{dolly-in,dolly-out,dolly-left,dolly-right,jib-up,jib-down,static}`), but they are 2-19b (LTX-2.0) files. For 2.5 the camera vocabulary should be prompt-side, with the LoRAs offered only when present and version-matched. |
| "Retake-style workflows where technically supported" | Supported — see §5.3. |
| "frame interpolation" | Not an LTX feature; ComfyUI has generic `nodes_frame_interpolation`. Out of scope. |
| Suggested `director/compiler/ltx25.py` | Adopted, with a registry so 2.3 can be added as a sibling adapter. |

---

## 7. Constraints the compiler must enforce

Derived from the node sources, not from folklore:

1. **Frame count ≡ 1 (mod 8)** — `EmptyLTXVLatentVideo` step is 8; `LTXVAddGuide` rounds
   `frame_idx` down to a multiple of 8 for videos of 9+ frames.
2. **Width and height divisible by 32** — `EmptyLTXVLatentVideo` step is 32 and the latent is
   `H//32 × W//32`.
3. **Guide `frame_idx`** — any value for single frames; multiple of 8 for multi-frame video
   guides; negative counts from the end (`-1` = last frame).
4. **IC-LoRA guides** need `frame_idx ≡ 1 (mod 8)` (`LTXAddVideoICLoRAGuide` rounds down).
5. **`LTXVCropGuides` is mandatory** whenever any `LTXVAddGuide` / IC-LoRA guide was added —
   guides are appended to the latent's temporal axis and must be cropped before decode.
6. **Audio latent frame count** derives from `LTXVEmptyLatentAudio(frames_number, frame_rate)`;
   it is *not* the video latent length.
7. **AV latents must be concatenated** (`LTXVConcatAVLatent`) before sampling and separated
   (`LTXVSeparateAVLatent`) after, in that order, on both stages.
8. **Two-stage** must re-pin I2V conditioning after `LTXVLatentUpsampler` (the upsampler drops
   `noise_mask`).

---

## 8. Reuse and licensing

* This repo: **GPL-3.0** (`LICENSE`). Derivative work stays GPL-3.0.
* `Lightricks/ComfyUI-LTXVideo`: **LTX-2 Community License Agreement**. Not vendored. Only node
  IDs, input names and documented semantics are used — interface facts, not code.
* `comfyanonymous/ComfyUI`: GPL-3.0. Compatible; still not vendored — only referenced.
* Prompt Relay derives from Kijai's `ComfyUI-PromptRelay` and the method described at
  `gordonchen19.github.io/Prompt-Relay`, and is already in-tree under this repo's GPL-3.0. Its
  provenance is recorded in the module header going forward.

---

## 9. Conclusions carried into Phase 1

1. Split into **spec → compiler → graph**, with a thin runtime node set.
2. The spec is versioned, small, JSON, and contains **no tensors and no base64**.
3. The compiler is a **pure function**, unit-testable with no GPU and no ComfyUI import.
4. LTX version support is an **adapter**, selected by a capability probe, not by `if` chains.
5. Emit **schema 0.4 subgraphs** so the compiled graph is legible and expandable.
6. Keep Prompt Relay; shrink it to one node; stop monkey-patching where a documented hook exists.
7. Delete nothing users depend on without a replacement — see the preservation list in §3.3.

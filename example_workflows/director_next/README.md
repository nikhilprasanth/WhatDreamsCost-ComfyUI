# Director Next example workflows

These are **compiler output**, not hand-built graphs. Each one is what the
LTX Director produces for a particular kind of shot, with the Director node
itself removed — open one to see exactly what the editor's Generate button
runs.

Regenerate with `python tools/build_examples.py` after changing the compiler.

Every graph expects the LTX-2.5 model set:

* `ltx-2.5-22b-distilled-transformer-bf16.safetensors` → `ComfyUI/models/diffusion_models/`
* `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` → `ComfyUI/models/text_encoders/`
* `ltx-2.5-video-vae-bf16.safetensors` → `ComfyUI/models/vae/`
* `ltx-2.5-audio-vae-bf16.safetensors` → `ComfyUI/models/vae/`
* `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` → `ComfyUI/models/latent_upscale_models/`
* `gemma4_e2b_it_bf16.safetensors` → `ComfyUI/models/text_encoders/`

| Workflow | Nodes | What it does |
| --- | --- | --- |
| [`LTX-2.5_Director_T2V.json`](./LTX-2.5_Director_T2V.json) | 22 | Text to video with synchronised audio. Start here. |
| [`LTX-2.5_Director_T2V_Two_Stage.json`](./LTX-2.5_Director_T2V_Two_Stage.json) | 31 | Generates at half size, upscales the latent, then refines. Slower, sharper. |
| [`LTX-2.5_Director_I2V.json`](./LTX-2.5_Director_I2V.json) | 25 | Opens on a still and moves from there. |
| [`LTX-2.5_Director_FFLF.json`](./LTX-2.5_Director_FFLF.json) | 29 | Pins the first and last frames and fills between them. |
| [`LTX-2.5_Director_Keyframes.json`](./LTX-2.5_Director_Keyframes.json) | 32 | Pins any number of frames along the timeline. |
| [`LTX-2.5_Director_Prompt_Relay.json`](./LTX-2.5_Director_Prompt_Relay.json) | 23 | Three prompt regions over one shot, in a single sampling pass. |
| [`LTX-2.5_Director_A2V.json`](./LTX-2.5_Director_A2V.json) | 24 | Follows a supplied soundtrack, frozen through sampling and muxed back in. |
| [`LTX-2.5_Director_T2A.json`](./LTX-2.5_Director_T2A.json) | 20 | Audio only — no video VAE, no decode. |
| [`LTX-2.5_Director_ICLoRA.json`](./LTX-2.5_Director_ICLoRA.json) | 27 | Follows a reference clip's motion through an IC-LoRA. |

Image and audio filenames in these graphs are placeholders (`first_frame.png`, `soundtrack.wav`). Point them at your own files, or build
the shot in the Director node and press Generate instead.

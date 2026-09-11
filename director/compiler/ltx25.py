"""The LTX-2.5 compiler.

Builds the graph the upstream LTX-2.5 examples and the official ComfyUI
templates build, from a Director Spec. Every topology choice here was taken from
reading those graphs — see ``docs/RESEARCH.md`` §2 for the extraction and the
exact numbers.

The spine, for every video mode::

    UNETLoader ─┐
    CLIPLoader ─┤  LTXVConditioning ─▶ [mode conditioning] ─▶ LTXVConcatAVLatent
    VAELoader ──┤                                                    │
    VAELoader ──┘                                                    ▼
                                        RandomNoise ─┐       SamplerCustomAdvanced
                            LTXVDualCFGGuider ───────┼──────────▶    │
                              KSamplerSelect ────────┤               ▼
                                ManualSigmas ────────┘       LTXVSeparateAVLatent
                                                              │            │
              [LTXVCropGuides] ◀───────────────────────────────┘            │
                     └─▶ VAEDecodeTiled ─┐                                  │
                       LTXVAudioVAEDecode ◀────────────────────────────────┘
                              └─▶ CreateVideo ─▶ SaveVideo

Three deliberate departures from the reference graphs, each for a reason:

* **No ``ResizeImageMaskNode``.** Its widget layout depends on which dynamic-combo
  variant is selected, which cannot be emitted reliably. It is also unnecessary:
  ``LTXVAddGuide`` and ``LTXVImgToVideoInplace`` both resize the image to the
  latent's exact pixel geometry with a centre crop. One fewer node, one less
  thing to break.
* **No ``ComfyMathExpression`` chains.** The reference graphs compute the frame
  count and stage-1 size inside the canvas so the user can type seconds. The
  Director has already done that arithmetic — and validated it — so the compiled
  graph carries the resolved numbers. They stay editable as plain widgets.
* **``SamplerCustomAdvanced.output``, not ``denoised_output``.** Both appear
  upstream; ``output`` is what the two-stage graphs and the T2V template use.
"""

from __future__ import annotations

from typing import Any

from ..core import time as dtime
from ..core.prompt import compile_prompt
from ..core.spec import (
    DEFAULT_SIGMAS_STAGE1,
    DEFAULT_SIGMAS_STAGE2,
    Reference,
    Spec,
)
from ..core.validate import Report
from .base import CompileResult, Compiler, register
from .graph import GraphBuilder, Layout, NodeHandle, Port, Section

__all__ = ["LTX25Compiler", "FAMILY"]

FAMILY = "ltx2.5"

#: ``seed_mode`` → litegraph's ``control_after_generate`` widget value. "list"
#: is driven by the take manager queueing one run per seed, so the node itself
#: stays fixed.
_SEED_CONTROL = {
    "fixed": "fixed",
    "random": "randomize",
    "increment": "increment",
    "decrement": "decrement",
    "list": "fixed",
}

_SECTION_COLOURS = {
    "Models": "#353535",
    "Inputs": "#3f789e",
    "Prompt": "#3f789e",
    "Canvas": "#2a6e3f",
    "Generate": "#6e3f6e",
    "Refine": "#6e5a3f",
    "Output": "#444444",
}


class _Build:
    """Mutable scratch space for one compilation.

    Keeping it out of the compiler instance is what makes ``compile`` reentrant
    and therefore genuinely pure.
    """

    def __init__(self, spec: Spec, builder: GraphBuilder) -> None:
        self.spec = spec
        self.g = builder
        self.index: dict[str, int] = {}
        self.report = Report()

        # model ports
        self.model: Port | None = None
        self.clip: Port | None = None
        self.enhancer_clip: Port | None = None
        self.vae: Port | None = None
        self.audio_vae: Port | None = None
        self.upscaler: Port | None = None
        self.iclora_downscale: Port | None = None

        # media ports, keyed by media id
        self.images: dict[str, Port] = {}
        self.video_frames: dict[str, Port] = {}
        self.audio: dict[str, Port] = {}

        # conditioning and latents
        self.positive: Port | None = None
        self.negative: Port | None = None
        self.video_latent: Port | None = None
        self.audio_latent: Port | None = None
        self.needs_crop = False
        #: First-frame image, kept so the refine stage can re-pin it.
        self.first_frame: Port | None = None
        #: Audio waveform muxed straight into the output instead of decoded.
        self.passthrough_audio: Port | None = None

    def record(self, name: str, node: NodeHandle) -> NodeHandle:
        self.index[name] = node.id
        return node

    # -- resolved geometry -------------------------------------------------

    @property
    def two_stage(self) -> bool:
        return self.spec.generation.stages > 1 and self.spec.project.mode != "t2a"

    @property
    def final_size(self) -> tuple[int, int]:
        return self.spec.project.width, self.spec.project.height

    @property
    def stage1_size(self) -> tuple[int, int]:
        """Stage 1 runs at a fraction of the output size when refining.

        The reference two-stage graphs halve both axes, lock composition and
        motion cheaply, then upsample and refine.
        """
        width, height = self.final_size
        if not self.two_stage:
            return width, height
        divisor = max(1, self.spec.generation.stage1_divisor)
        return dtime.snap_dim(width // divisor), dtime.snap_dim(height // divisor)


class LTX25Compiler(Compiler):
    family = FAMILY
    modes = ("t2v", "i2v", "fflf", "keyframes", "continue", "a2v", "t2a", "iclora")

    def compile(
        self,
        spec: Spec,
        *,
        caps: Any = None,
        layout: Layout = "flat",
    ) -> CompileResult:
        builder = self._builder(spec, layout)
        ctx = _Build(spec, builder)
        result = CompileResult(workflow={}, family=self.family, layout=layout)

        self._models(ctx, caps, result)
        self._inputs(ctx, result)
        self._prompt(ctx, caps, result)
        self._canvas(ctx, result)
        self._generate(ctx, result)
        if ctx.two_stage:
            self._refine(ctx, result)
        self._output(ctx, result)

        result.workflow = builder.build()
        result.api = builder.build_api()
        result.node_index = ctx.index
        result.report.extend(ctx.report.diagnostics)
        return result

    # ----------------------------------------------------------------------
    # models
    # ----------------------------------------------------------------------

    def _models(self, ctx: _Build, caps: Any, result: CompileResult) -> None:
        spec = ctx.spec
        with ctx.g.section("Models", color=_SECTION_COLOURS["Models"]) as s:
            unet = ctx.record("unet", s.node(
                "UNETLoader", unet_name=spec.models.unet, weight_dtype="default",
            ))
            model: Port = unet.out()

            # Plain LoRAs first, then IC-LoRAs: an IC-LoRA also reports the
            # reference downscale factor its guides need, so it has to be last
            # for that output to reflect the active one.
            for lora in spec.models.loras:
                if not lora.enabled or not lora.name or lora.kind != "lora":
                    continue
                node = s.node(
                    "LoraLoaderModelOnly",
                    model=model, lora_name=lora.name, strength_model=lora.strength,
                )
                model = node.out()

            for lora in spec.models.loras:
                if not lora.enabled or not lora.name or lora.kind != "iclora":
                    continue
                node = ctx.record("iclora", s.node(
                    "LTXICLoRALoaderModelOnly",
                    model=model, lora_name=lora.name, strength_model=lora.strength,
                ))
                model = node.out("model")
                ctx.iclora_downscale = node.out("latent_downscale_factor")

            if spec.project.mode == "t2a":
                # Severs the a2v/v2a cross-attention and skips the video stream
                # entirely, which is what makes audio-only sampling cheap.
                model = ctx.record("audio_only", s.node(
                    "LTXVAudioOnlyModel", model=model,
                )).out()

            ctx.model = model

            ctx.clip = ctx.record("clip", s.node(
                "CLIPLoader", clip_name=spec.models.clip, type="ltxv", device="default",
            )).out()

            if spec.prompt.enhance.enabled and spec.models.enhancer_clip:
                ctx.enhancer_clip = ctx.record("enhancer_clip", s.node(
                    "CLIPLoader",
                    clip_name=spec.models.enhancer_clip, type="ltxv", device="default",
                )).out()

            if spec.needs_video:
                ctx.vae = ctx.record("vae", s.node(
                    "VAELoader", vae_name=spec.models.vae,
                )).out()

            if spec.needs_audio:
                ctx.audio_vae = ctx.record("audio_vae", s.node(
                    "VAELoader", vae_name=spec.models.audio_vae,
                )).out()

            if ctx.two_stage and spec.models.upscaler:
                ctx.upscaler = ctx.record("upscaler", s.node(
                    "LatentUpscaleModelLoader", model_name=spec.models.upscaler,
                )).out()

    # ----------------------------------------------------------------------
    # media inputs
    # ----------------------------------------------------------------------

    def _inputs(self, ctx: _Build, result: CompileResult) -> None:
        spec = ctx.spec
        wanted = self._referenced_media(ctx)
        if not wanted:
            return

        with ctx.g.section("Inputs", color=_SECTION_COLOURS["Inputs"]) as s:
            for media_id in wanted:
                entry = spec.media[media_id]
                if entry.kind == "image":
                    node = s.node("LoadImage", image=entry.path, title=entry.filename)
                    ctx.images[media_id] = node.out("IMAGE")
                elif entry.kind == "video":
                    loader = s.node("LoadVideo", file=entry.path, title=entry.filename)
                    parts = s.node("GetVideoComponents", video=loader.out("VIDEO"))
                    ctx.video_frames[media_id] = parts.out("images")
                    ctx.audio[media_id] = parts.out("audio")
                else:
                    node = s.node("LoadAudio", audio=entry.path, title=entry.filename)
                    ctx.audio[media_id] = node.out("AUDIO")

    def _referenced_media(self, ctx: _Build) -> list[str]:
        """Media actually used by the compiled graph, in a stable order.

        Deduplicated, so the same file referenced twice is loaded once and the
        executor caches one decode rather than two.
        """
        spec = ctx.spec
        out: list[str] = []

        def add(media_id: str | None) -> None:
            if media_id and media_id in spec.media and media_id not in out:
                out.append(media_id)

        for ref in self._guide_references(ctx):
            add(ref.media)
        for ref in self._control_references(ctx):
            add(ref.media)
        if spec.project.mode == "a2v" or spec.audio.mode in ("import", "inpaint"):
            for clip in spec.audio.clips:
                if clip.enabled:
                    add(clip.media)
        add(spec.audio.reference)
        return out

    # ----------------------------------------------------------------------
    # prompt
    # ----------------------------------------------------------------------

    def _prompt(self, ctx: _Build, caps: Any, result: CompileResult) -> None:
        spec = ctx.spec
        text = compile_prompt(spec)

        with ctx.g.section("Prompt", color=_SECTION_COLOURS["Prompt"]) as s:
            positive_text = ctx.record("prompt_text", s.node(
                "PrimitiveStringMultiline", value=text, title="Prompt",
            )).out()
            negative_text = ctx.record("negative_text", s.node(
                "PrimitiveStringMultiline",
                value=spec.prompt.negative, title="Negative prompt",
            )).out()

            encode_source = positive_text
            if ctx.enhancer_clip is not None:
                enhance = s.node(
                    "TextGenerateLTX2Prompt",
                    clip=ctx.enhancer_clip,
                    prompt=positive_text,
                    max_length=spec.prompt.enhance.max_length,
                    seed=spec.prompt.enhance.seed,
                    title="Enhance prompt",
                )
                first = self._first_frame_media(ctx)
                if spec.prompt.enhance.use_image and first is not None:
                    # Grounding the enhancer on the opening frame is what the
                    # I2V system prompt is written for.
                    enhance.set(image=first)
                ctx.record("enhance", enhance)
                encode_source = enhance.out()
                # The enhanced text is never hidden: it is on the canvas.
                s.node("PreviewAny", source=encode_source, title="Effective prompt")

            positive = ctx.record("encode_positive", s.node(
                "CLIPTextEncode", clip=ctx.clip, text=encode_source,
            )).out()
            negative = ctx.record("encode_negative", s.node(
                "CLIPTextEncode", clip=ctx.clip, text=negative_text,
            )).out()

            conditioning = ctx.record("conditioning", s.node(
                "LTXVConditioning",
                positive=positive, negative=negative, frame_rate=spec.project.fps,
            ))
            ctx.positive = conditioning.out("positive")
            ctx.negative = conditioning.out("negative")

    def _first_frame_media(self, ctx: _Build) -> Port | None:
        for ref in ctx.spec.active_references("keyframe"):
            if ref.anchor == "start":
                return ctx.images.get(ref.media or "")
        return None

    # ----------------------------------------------------------------------
    # canvas
    # ----------------------------------------------------------------------

    def _canvas(self, ctx: _Build, result: CompileResult) -> None:
        spec = ctx.spec
        with ctx.g.section("Canvas", color=_SECTION_COLOURS["Canvas"]) as s:
            if spec.needs_video:
                width, height = ctx.stage1_size
                video = ctx.record("empty_latent", s.node(
                    "EmptyLTXVLatentVideo",
                    width=width, height=height, length=spec.project.frames, batch_size=1,
                )).out()
            else:
                # The model splits its input positionally into [video, audio], so
                # audio-only sampling still needs a video latent at index 0. This
                # placeholder is never attended to.
                video = ctx.record("empty_latent", s.node(
                    "LTXVAudioOnlyEmptyVideoLatent", title="Audio-only placeholder",
                )).out()

            builder = _CANVAS_BUILDERS[spec.project.mode]
            video = builder(self, ctx, s, video)

            audio = self._audio_latent(ctx, s)

            if audio is not None:
                concat = ctx.record("concat", s.node(
                    "LTXVConcatAVLatent", video_latent=video, audio_latent=audio,
                ))
                ctx.video_latent = concat.out()
            else:
                ctx.video_latent = video
            ctx.audio_latent = audio

    # -- mode-specific conditioning ---------------------------------------

    def _canvas_plain(self, ctx: _Build, s: Section, video: Port) -> Port:
        """Text to video, and audio-only: nothing pinned."""
        return video

    def _canvas_first_frame(self, ctx: _Build, s: Section, video: Port) -> Port:
        """Image to video and Continue Shot: pin frame 0 in place.

        ``LTXVImgToVideoInplace`` writes at index 0 only, which is exactly what
        a first frame is, and is cheaper than a full guide because it adds no
        keyframe tokens and so needs no crop afterwards.
        """
        spec = ctx.spec
        first = next(
            (r for r in spec.active_references("keyframe") if r.anchor == "start"), None
        )
        if first is None:
            return video

        image = self._guide_image(ctx, s, first)
        if image is None:
            return video
        ctx.first_frame = image

        node = ctx.record("pin_first_frame", s.node(
            "LTXVImgToVideoInplace",
            vae=ctx.vae, image=image, latent=video,
            strength=spec.generation.first_frame_strength, bypass=False,
        ))
        return node.out()

    def _canvas_guides(self, ctx: _Build, s: Section, video: Port) -> Port:
        """First/last frame and arbitrary keyframes: a chain of ``LTXVAddGuide``.

        Guides append keyframe tokens to the latent's temporal axis, so
        ``LTXVCropGuides`` after sampling is mandatory — that is what
        :attr:`_Build.needs_crop` records.
        """
        spec = ctx.spec
        references = spec.active_references("keyframe")
        if not references:
            return video

        latent = video
        for ref in references:
            image = self._guide_image(ctx, s, ref)
            if image is None:
                continue
            frame_idx = self._guide_index(ctx, ref)
            if ref.anchor == "start":
                ctx.first_frame = image
            node = s.node(
                "LTXVAddGuide",
                positive=ctx.positive, negative=ctx.negative,
                vae=ctx.vae, latent=latent, image=image,
                frame_idx=frame_idx,
                strength=ref.strength if ref.strength else spec.generation.guide_strength,
                title=ref.label or f"Guide @ {frame_idx}",
            )
            ctx.positive = node.out("positive")
            ctx.negative = node.out("negative")
            latent = node.out("latent")
            ctx.needs_crop = True
            ctx.index.setdefault("first_guide", node.id)
        return latent

    def _canvas_iclora(self, ctx: _Build, s: Section, video: Port) -> Port:
        """IC-LoRA control: the guide clip steers generation through the LoRA."""
        spec = ctx.spec
        references = self._control_references(ctx)
        if not references:
            return video

        latent = video
        for ref in references:
            frames = ctx.video_frames.get(ref.media or "") or ctx.images.get(ref.media or "")
            if frames is None:
                continue
            node = s.node(
                "LTXAddVideoICLoRAGuide",
                positive=ctx.positive, negative=ctx.negative,
                vae=ctx.vae, latent=latent, image=frames,
                frame_idx=self._iclora_index(ctx, ref),
                strength=ref.strength,
                crop="center" if ref.crop == "center" else "disabled",
                title=ref.label or f"IC-LoRA {ref.role}",
            )
            # The LoRA reports the downscale its guides were trained at; reading
            # it from the loader beats hardcoding a number per LoRA file.
            if ctx.iclora_downscale is not None:
                node.set(latent_downscale_factor=ctx.iclora_downscale)
            ctx.positive = node.out("positive")
            ctx.negative = node.out("negative")
            latent = node.out("latent")
            ctx.needs_crop = True
            ctx.index.setdefault("first_guide", node.id)

        # An IC-LoRA shot can still open on a still.
        return self._canvas_first_frame(ctx, s, latent)

    def _canvas_audio_driven(self, ctx: _Build, s: Section, video: Port) -> Port:
        """Audio to video: the supplied track is frozen, and the video follows it."""
        return self._canvas_first_frame(ctx, s, video)

    # -- guide helpers -----------------------------------------------------

    def _guide_image(self, ctx: _Build, s: Section, ref: Reference) -> Port | None:
        """The image port for a reference, compressed the way LTX expects.

        ``LTXVPreprocess`` applies the H.264 artefacts the model was trained to
        see on conditioning frames. Every reference graph does this; skipping it
        makes guides look unnaturally clean and the motion stick.
        """
        source = ctx.images.get(ref.media or "") or ctx.video_frames.get(ref.media or "")
        if source is None:
            return None
        compression = ctx.spec.generation.img_compression
        if compression <= 0:
            return source
        key = f"preprocess::{ref.media}"
        cached = ctx.index.get(key)
        if cached is not None:
            # Reuse one preprocess per file rather than per reference.
            return next(n for n in ctx.g._nodes if n.id == cached).out()
        node = s.node("LTXVPreprocess", image=source, img_compression=compression)
        ctx.index[key] = node.id
        return node.out()

    def _guide_index(self, ctx: _Build, ref: Reference) -> int:
        frames = ctx.spec.project.frames
        raw = ref.resolved_frame(frames)
        media = ctx.spec.media.get(ref.media or "")
        guide_frames = 1
        if media is not None and media.kind == "video":
            guide_frames = max(1, int(media.duration * (media.fps or ctx.spec.project.fps)))
        return dtime.snap_guide_index(raw, guide_frames)

    def _iclora_index(self, ctx: _Build, ref: Reference) -> int:
        raw = ref.resolved_frame(ctx.spec.project.frames)
        media = ctx.spec.media.get(ref.media or "")
        guide_frames = 1
        if media is not None and media.kind == "video":
            guide_frames = max(1, int(media.duration * (media.fps or ctx.spec.project.fps)))
        return dtime.snap_iclora_index(raw, guide_frames)

    def _guide_references(self, ctx: _Build) -> list[Reference]:
        mode = ctx.spec.project.mode
        if mode in ("t2v", "t2a"):
            return []
        return ctx.spec.active_references("keyframe")

    def _control_references(self, ctx: _Build) -> list[Reference]:
        if ctx.spec.project.mode != "iclora":
            return []
        return ctx.spec.active_references("motion", "control")

    # -- audio latent ------------------------------------------------------

    def _audio_latent(self, ctx: _Build, s: Section) -> Port | None:
        spec = ctx.spec
        if not spec.needs_audio or ctx.audio_vae is None:
            return None

        supplied = self._supplied_audio(ctx)
        if supplied is None:
            return ctx.record("empty_audio", s.node(
                "LTXVEmptyLatentAudio",
                audio_vae=ctx.audio_vae,
                frames_number=spec.project.frames,
                frame_rate=spec.project.fps,
            )).out()

        trimmed = s.node(
            "TrimAudioDuration",
            audio=supplied,
            start_index=self._audio_start(ctx),
            duration=spec.project.duration,
        )
        # Muxed straight into the output: re-decoding audio the user supplied
        # would only degrade it.
        ctx.passthrough_audio = trimmed.out()

        encoded = ctx.record("encode_audio", s.node(
            "LTXVAudioVAEEncode", audio=trimmed.out(), audio_vae=ctx.audio_vae,
        )).out()

        if spec.audio.mode == "inpaint":
            # Inpainting means "generate only in the gaps between the supplied
            # clips", which needs a per-audio-frame noise mask. No upstream node
            # builds one, so the compiled graph generates over the whole track.
            # Better to say that than to ship something that quietly differs
            # from what the timeline shows.
            ctx.report.warn(
                "audio.inpaint_approximated",
                "Audio inpainting is approximated: the compiled graph generates across the "
                "whole track rather than only in the gaps between your clips.",
                "Use Import to keep your audio exactly as supplied, or Generate to replace "
                "it entirely.",
                "audio.mode",
            )
            return encoded

        return ctx.record("freeze_audio", s.node(
            "LTXVFreezeLatent", latent=encoded,
        )).out()

    def _supplied_audio(self, ctx: _Build) -> Port | None:
        spec = ctx.spec
        if spec.audio.mode not in ("import", "inpaint") and spec.project.mode != "a2v":
            return None
        for clip in spec.audio.clips:
            if clip.enabled and clip.media in ctx.audio:
                return ctx.audio[clip.media]
        return None

    def _audio_start(self, ctx: _Build) -> float:
        for clip in ctx.spec.audio.clips:
            if clip.enabled and clip.media in ctx.audio:
                return round(clip.trim_start, 4)
        return 0.0

    # ----------------------------------------------------------------------
    # sampling
    # ----------------------------------------------------------------------

    def _generate(self, ctx: _Build, result: CompileResult) -> None:
        spec = ctx.spec
        sigmas = spec.generation.sigmas_stage1 or list(DEFAULT_SIGMAS_STAGE1)
        with ctx.g.section("Generate", color=_SECTION_COLOURS["Generate"]) as s:
            latent = self._sample(
                ctx, s, ctx.video_latent, sigmas,
                sampler=spec.generation.sampler,
                seed=spec.generation.seed,
                seed_control=_SEED_CONTROL.get(spec.generation.seed_mode, "fixed"),
                prefix="stage1",
            )
            separated = ctx.record("separate_stage1", s.node(
                "LTXVSeparateAVLatent", av_latent=latent,
            ))
            video = separated.out("video_latent")
            audio = separated.out("audio_latent") if ctx.audio_latent is not None else None

            if ctx.needs_crop and spec.needs_video:
                # Guides append keyframe tokens to the latent's temporal axis.
                # They have to come off before the upsampler, which would
                # otherwise scale them as if they were picture, and before the
                # refine pass, which conditions on the cleaned pair.
                video = self._crop_guides(ctx, s, video)

            ctx.video_latent = video
            ctx.audio_latent = audio

    def _sample(
        self,
        ctx: _Build,
        s: Section,
        latent: Port,
        sigmas: list[float],
        *,
        sampler: str,
        seed: int,
        seed_control: str,
        prefix: str,
    ) -> Port:
        spec = ctx.spec
        noise = s.node("RandomNoise", noise_seed=seed, control_after_generate=seed_control)
        guider = s.node(
            "LTXVDualCFGGuider",
            model=ctx.model, positive=ctx.positive, negative=ctx.negative,
            video_cfg=spec.generation.video_cfg, audio_cfg=spec.generation.audio_cfg,
        )
        sampler_node = s.node("KSamplerSelect", sampler_name=sampler)
        schedule = s.node("ManualSigmas", sigmas=_format_sigmas(sigmas))
        node = ctx.record(f"sampler_{prefix}", s.node(
            "SamplerCustomAdvanced",
            noise=noise.out(), guider=guider.out(),
            sampler=sampler_node.out(), sigmas=schedule.out(),
            latent_image=latent,
        ))
        ctx.record(f"seed_{prefix}", noise)
        return node.out("output")

    def _crop_guides(self, ctx: _Build, s: Section, video: Port) -> Port:
        """Strip the keyframe tokens guides appended, before decoding.

        Mandatory whenever any guide was added: the tokens live on the temporal
        axis of the latent and would otherwise decode as extra frames.
        """
        node = ctx.record("crop_guides", s.node(
            "LTXVCropGuides",
            positive=ctx.positive, negative=ctx.negative, latent=video,
        ))
        ctx.positive = node.out("positive")
        ctx.negative = node.out("negative")
        return node.out("latent")

    # ----------------------------------------------------------------------
    # refine
    # ----------------------------------------------------------------------

    def _refine(self, ctx: _Build, result: CompileResult) -> None:
        spec = ctx.spec
        sigmas = spec.generation.sigmas_stage2 or list(DEFAULT_SIGMAS_STAGE2)
        with ctx.g.section("Refine", color=_SECTION_COLOURS["Refine"]) as s:
            upsampled = ctx.record("upsample", s.node(
                "LTXVLatentUpsampler",
                samples=ctx.video_latent, upscale_model=ctx.upscaler, vae=ctx.vae,
            )).out()

            if ctx.first_frame is not None:
                # The upsampler drops noise_mask, so the first frame has to be
                # pinned again — at full strength this time, because stage 2 is
                # refining an image that already matches it.
                upsampled = s.node(
                    "LTXVImgToVideoInplace",
                    vae=ctx.vae, image=ctx.first_frame, latent=upsampled,
                    strength=1.0, bypass=False, title="Re-pin first frame",
                ).out()

            latent = upsampled
            if ctx.audio_latent is not None:
                latent = s.node(
                    "LTXVConcatAVLatent",
                    video_latent=upsampled, audio_latent=ctx.audio_latent,
                ).out()

            sampled = self._sample(
                ctx, s, latent, sigmas,
                sampler=spec.generation.refine_sampler,
                seed=spec.generation.seed,
                seed_control="fixed",
                prefix="stage2",
            )
            separated = ctx.record("separate_stage2", s.node(
                "LTXVSeparateAVLatent", av_latent=sampled,
            ))
            ctx.video_latent = separated.out("video_latent")
            ctx.audio_latent = (
                separated.out("audio_latent") if ctx.audio_latent is not None else None
            )

    # ----------------------------------------------------------------------
    # output
    # ----------------------------------------------------------------------

    def _output(self, ctx: _Build, result: CompileResult) -> None:
        spec = ctx.spec
        decode = spec.generation.decode
        with ctx.g.section("Output", color=_SECTION_COLOURS["Output"]) as s:
            if not spec.needs_video:
                audio = ctx.record("decode_audio", s.node(
                    "LTXVAudioVAEDecode",
                    samples=ctx.audio_latent, audio_vae=ctx.audio_vae,
                )).out()
                ctx.record("save", s.node(
                    "SaveAudioAdvanced",
                    audio=audio,
                    filename_prefix=spec.generation.save_prefix.replace("video/", "audio/"),
                    format="flac",
                ))
                return

            images = ctx.record("decode_video", s.node(
                "VAEDecodeTiled",
                samples=ctx.video_latent, vae=ctx.vae,
                tile_size=decode.tile_size, overlap=decode.overlap,
                temporal_size=decode.temporal_size,
                temporal_overlap=decode.temporal_overlap,
            )).out()

            video = s.node(
                "CreateVideo",
                images=images, fps=spec.project.fps, bit_depth=decode.bit_depth,
            )
            audio_port = self._output_audio(ctx, s)
            if audio_port is not None:
                video.set(audio=audio_port)
            ctx.record("create_video", video)

            ctx.record("save", s.node(
                "SaveVideo",
                video=video.out(),
                filename_prefix=spec.generation.save_prefix,
                format="auto", codec="auto",
            ))

    def _output_audio(self, ctx: _Build, s: Section) -> Port | None:
        if ctx.passthrough_audio is not None and ctx.spec.audio.mode == "import":
            # Supplied audio was frozen through sampling, so the original
            # waveform is still the best version of it.
            return ctx.passthrough_audio
        if ctx.audio_latent is None or ctx.audio_vae is None:
            return None
        return ctx.record("decode_audio", s.node(
            "LTXVAudioVAEDecode", samples=ctx.audio_latent, audio_vae=ctx.audio_vae,
        )).out()


# --------------------------------------------------------------------------
# mode dispatch
#
# A table rather than a chain of ``if``s, so adding a mode is adding a row.
# --------------------------------------------------------------------------

_CANVAS_BUILDERS: dict[str, Any] = {
    "t2v": LTX25Compiler._canvas_plain,
    "t2a": LTX25Compiler._canvas_plain,
    "i2v": LTX25Compiler._canvas_first_frame,
    "continue": LTX25Compiler._canvas_first_frame,
    "fflf": LTX25Compiler._canvas_guides,
    "keyframes": LTX25Compiler._canvas_guides,
    "a2v": LTX25Compiler._canvas_audio_driven,
    "iclora": LTX25Compiler._canvas_iclora,
}


def _format_sigmas(sigmas: list[float]) -> str:
    """``ManualSigmas`` takes a comma-separated string.

    Formatted with ``repr``-style trimming so ``0.99375`` stays exact and
    ``1.0`` does not become ``1.0000000000``.
    """
    return ", ".join(f"{value:g}" for value in sigmas)


register(LTX25Compiler())

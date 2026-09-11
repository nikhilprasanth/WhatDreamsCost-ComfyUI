"""Node signatures: the slot order and widget order of every node we emit.

A ComfyUI workflow references inputs by **index**, so the compiler has to know
each node type's declaration order exactly. Getting it wrong produces a graph
that loads but is wired to the wrong sockets — the worst possible failure, since
it looks fine.

So this table is not guesswork. Every entry was extracted from real upstream
graphs (``ComfyUI-LTXVideo/example_workflows/2.5/*.json`` and the official
``Comfy-Org/workflow_templates`` LTX-2.5 templates) and cross-checked against
the node source in ``comfy_extras/nodes_lt*.py``, ``nodes_textgen.py`` and
``ComfyUI-LTXVideo``.

Two rules the serialised format follows, and that :mod:`director.compiler.graph`
depends on:

1. ``widgets_values`` carries **every** widget in declaration order, always.
2. ``inputs`` carries every non-widget input, plus any widget input that is
   *linked* (marked with ``{"widget": {"name": …}}``). An unlinked widget input
   is absent from ``inputs``, which is why slot indices have to be computed from
   the links actually present rather than from the signature alone.

When ComfyUI changes a node, the fix is an edit here — not a hunt through the
adapter.

**Dynamic-combo nodes are deliberately absent.** ``ResizeImageMaskNode`` and
friends serialise a different number of widgets depending on which variant of
their dynamic combo is selected, so their layout cannot be emitted from a static
table. The compiler avoids needing them — see the note at the top of
:mod:`director.compiler.ltx25`. ``TextGenerateLTX2Prompt`` is the exception: every
LTX-2.5 reference graph serialises the same twelve values, so that one layout is
safe to reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Slot", "InputSlot", "NodeSignature", "NODE_SIGNATURES", "signature_for"]


@dataclass(frozen=True)
class Slot:
    name: str
    type: str


@dataclass(frozen=True)
class InputSlot:
    name: str
    type: str
    #: True when this input is widget-backed, i.e. it can take a literal value
    #: as well as a link.
    widget: bool = False
    #: True when the graph is valid without it.
    optional: bool = False


@dataclass(frozen=True)
class NodeSignature:
    inputs: tuple[InputSlot, ...] = ()
    outputs: tuple[Slot, ...] = ()
    #: ``(name, default)`` per entry in ``widgets_values``, in order. Includes
    #: widgets that are not inputs at all — ``control_after_generate`` being the
    #: common one.
    widgets: tuple[tuple[str, Any], ...] = ()
    #: Default node size; only affects how the graph looks when opened.
    size: tuple[int, int] = (300, 100)

    def input_index(self, name: str) -> int:
        for i, slot in enumerate(self.inputs):
            if slot.name == name:
                return i
        raise KeyError(name)

    def output_index(self, name: str) -> int:
        for i, slot in enumerate(self.outputs):
            if slot.name == name:
                return i
        raise KeyError(name)

    def input(self, name: str) -> InputSlot:
        return self.inputs[self.input_index(name)]

    @property
    def widget_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.widgets)


def _i(name: str, type_: str, *, widget: bool = False, optional: bool = False) -> InputSlot:
    return InputSlot(name, type_, widget, optional)


def _o(name: str, type_: str) -> Slot:
    return Slot(name, type_)


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

_LOADERS: dict[str, NodeSignature] = {
    "UNETLoader": NodeSignature(
        inputs=(_i("unet_name", "COMBO", widget=True), _i("weight_dtype", "COMBO", widget=True)),
        outputs=(_o("MODEL", "MODEL"),),
        widgets=(("unet_name", ""), ("weight_dtype", "default")),
        size=(340, 82),
    ),
    "CLIPLoader": NodeSignature(
        inputs=(
            _i("clip_name", "COMBO", widget=True),
            _i("type", "COMBO", widget=True),
            _i("device", "COMBO", widget=True),
        ),
        outputs=(_o("CLIP", "CLIP"),),
        widgets=(("clip_name", ""), ("type", "ltxv"), ("device", "default")),
        size=(340, 106),
    ),
    "VAELoader": NodeSignature(
        inputs=(_i("vae_name", "COMBO", widget=True),),
        outputs=(_o("VAE", "VAE"),),
        widgets=(("vae_name", ""),),
        size=(340, 58),
    ),
    "LatentUpscaleModelLoader": NodeSignature(
        inputs=(_i("model_name", "COMBO", widget=True),),
        outputs=(_o("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL"),),
        widgets=(("model_name", ""),),
        size=(340, 58),
    ),
    "LoraLoaderModelOnly": NodeSignature(
        inputs=(
            _i("model", "MODEL"),
            _i("lora_name", "COMBO", widget=True),
            _i("strength_model", "FLOAT", widget=True),
        ),
        outputs=(_o("MODEL", "MODEL"),),
        widgets=(("lora_name", ""), ("strength_model", 1.0)),
        size=(320, 82),
    ),
    # ComfyUI-LTXVideo. Also reports the LoRA's reference downscale factor,
    # which IC-LoRA guides need.
    "LTXICLoRALoaderModelOnly": NodeSignature(
        inputs=(
            _i("model", "MODEL"),
            _i("lora_name", "COMBO", widget=True),
            _i("strength_model", "FLOAT", widget=True),
        ),
        outputs=(_o("model", "MODEL"), _o("latent_downscale_factor", "FLOAT")),
        widgets=(("lora_name", ""), ("strength_model", 1.0)),
        size=(340, 82),
    ),
    "LTXVAudioOnlyModel": NodeSignature(
        inputs=(_i("model", "MODEL"),),
        outputs=(_o("model", "MODEL"),),
        size=(280, 30),
    ),
}


# --------------------------------------------------------------------------
# conditioning
# --------------------------------------------------------------------------

_CONDITIONING: dict[str, NodeSignature] = {
    "CLIPTextEncode": NodeSignature(
        inputs=(_i("clip", "CLIP"), _i("text", "STRING", widget=True)),
        outputs=(_o("CONDITIONING", "CONDITIONING"),),
        widgets=(("text", ""),),
        size=(400, 200),
    ),
    "LTXVConditioning": NodeSignature(
        inputs=(
            _i("positive", "CONDITIONING"),
            _i("negative", "CONDITIONING"),
            _i("frame_rate", "FLOAT", widget=True),
        ),
        outputs=(_o("positive", "CONDITIONING"), _o("negative", "CONDITIONING")),
        widgets=(("frame_rate", 25.0),),
        size=(300, 78),
    ),
    "ConditioningZeroOut": NodeSignature(
        inputs=(_i("conditioning", "CONDITIONING"),),
        outputs=(_o("CONDITIONING", "CONDITIONING"),),
        size=(260, 30),
    ),
    "TextGenerateLTX2Prompt": NodeSignature(
        inputs=(
            _i("clip", "CLIP"),
            _i("image", "IMAGE", optional=True),
            _i("video", "IMAGE", optional=True),
            _i("audio", "AUDIO", optional=True),
            _i("prompt", "STRING", widget=True),
            _i("max_length", "INT", widget=True),
            _i("sampling_mode", "COMFY_DYNAMICCOMBO_V3", widget=True),
            _i("sampling_mode.temperature", "FLOAT", widget=True),
            _i("sampling_mode.top_k", "INT", widget=True),
            _i("sampling_mode.top_p", "FLOAT", widget=True),
            _i("sampling_mode.min_p", "FLOAT", widget=True),
            _i("sampling_mode.repetition_penalty", "FLOAT", widget=True),
            _i("sampling_mode.seed", "INT", widget=True),
            _i("sampling_mode.presence_penalty", "FLOAT", widget=True),
            _i("thinking", "BOOLEAN", widget=True),
            _i("use_default_template", "BOOLEAN", widget=True),
        ),
        outputs=(_o("generated_text", "STRING"),),
        # The 12 values the official LTX-2.5 templates ship, in their order.
        widgets=(
            ("prompt", ""), ("max_length", 600), ("sampling_mode", "on"),
            ("temperature", 0.7), ("top_k", 64), ("top_p", 0.95), ("min_p", 0.05),
            ("repetition_penalty", 1.15), ("seed", 0), ("presence_penalty", 0),
            ("thinking", False), ("use_default_template", True),
        ),
        size=(400, 320),
    ),
}


# --------------------------------------------------------------------------
# latents and guides
# --------------------------------------------------------------------------

_LATENTS: dict[str, NodeSignature] = {
    "EmptyLTXVLatentVideo": NodeSignature(
        inputs=(
            _i("width", "INT", widget=True),
            _i("height", "INT", widget=True),
            _i("length", "INT", widget=True),
            _i("batch_size", "INT", widget=True),
        ),
        outputs=(_o("LATENT", "LATENT"),),
        widgets=(("width", 768), ("height", 512), ("length", 97), ("batch_size", 1)),
        size=(300, 130),
    ),
    "LTXVEmptyLatentAudio": NodeSignature(
        inputs=(
            _i("audio_vae", "VAE"),
            _i("frames_number", "INT", widget=True),
            _i("frame_rate", "FLOAT,INT", widget=True),
        ),
        outputs=(_o("Latent", "LATENT"),),
        widgets=(("frames_number", 97), ("frame_rate", 25), ("batch_size", 1)),
        size=(320, 106),
    ),
    "LTXVAudioOnlyEmptyVideoLatent": NodeSignature(
        outputs=(_o("latent", "LATENT"),),
        size=(300, 30),
    ),
    "LTXVImgToVideoInplace": NodeSignature(
        inputs=(
            _i("vae", "VAE"),
            _i("image", "IMAGE"),
            _i("latent", "LATENT"),
            _i("strength", "FLOAT", widget=True),
            _i("bypass", "BOOLEAN", widget=True),
        ),
        outputs=(_o("latent", "LATENT"),),
        widgets=(("strength", 1.0), ("bypass", False)),
        size=(320, 130),
    ),
    "LTXVAddGuide": NodeSignature(
        inputs=(
            _i("positive", "CONDITIONING"),
            _i("negative", "CONDITIONING"),
            _i("vae", "VAE"),
            _i("latent", "LATENT"),
            _i("image", "IMAGE"),
            _i("frame_idx", "INT", widget=True),
            _i("strength", "FLOAT", widget=True),
            _i("attention_mask", "MASK", optional=True),
            _i("iclora_parameters", "IC_LORA_PARAMETERS", optional=True),
        ),
        outputs=(
            _o("positive", "CONDITIONING"),
            _o("negative", "CONDITIONING"),
            _o("latent", "LATENT"),
        ),
        widgets=(("frame_idx", 0), ("strength", 1.0)),
        size=(340, 190),
    ),
    "LTXVCropGuides": NodeSignature(
        inputs=(
            _i("positive", "CONDITIONING"),
            _i("negative", "CONDITIONING"),
            _i("latent", "LATENT"),
        ),
        outputs=(
            _o("positive", "CONDITIONING"),
            _o("negative", "CONDITIONING"),
            _o("latent", "LATENT"),
        ),
        size=(300, 78),
    ),
    "LTXVConcatAVLatent": NodeSignature(
        inputs=(_i("video_latent", "LATENT"), _i("audio_latent", "LATENT")),
        outputs=(_o("latent", "LATENT"),),
        size=(300, 54),
    ),
    "LTXVSeparateAVLatent": NodeSignature(
        inputs=(_i("av_latent", "LATENT"),),
        outputs=(_o("video_latent", "LATENT"), _o("audio_latent", "LATENT")),
        size=(300, 54),
    ),
    "LTXVLatentUpsampler": NodeSignature(
        inputs=(
            _i("samples", "LATENT"),
            _i("upscale_model", "LATENT_UPSCALE_MODEL"),
            _i("vae", "VAE"),
        ),
        outputs=(_o("LATENT", "LATENT"),),
        size=(300, 78),
    ),
    # comfy_extras/nodes_lt_keyframes.py — replaces the SolidMask +
    # SetLatentNoiseMask pair the older A2V example still uses.
    "LTXVFreezeLatent": NodeSignature(
        inputs=(_i("latent", "LATENT"),),
        outputs=(_o("latent", "LATENT"),),
        size=(280, 30),
    ),
    "SetLatentNoiseMask": NodeSignature(
        inputs=(_i("samples", "LATENT"), _i("mask", "MASK")),
        outputs=(_o("LATENT", "LATENT"),),
        size=(280, 54),
    ),
    "SolidMask": NodeSignature(
        inputs=(
            _i("value", "FLOAT", widget=True),
            _i("width", "INT", widget=True),
            _i("height", "INT", widget=True),
        ),
        outputs=(_o("MASK", "MASK"),),
        widgets=(("value", 0.0), ("width", 512), ("height", 512)),
        size=(280, 106),
    ),
    # ComfyUI-LTXVideo
    "LTXAddVideoICLoRAGuide": NodeSignature(
        inputs=(
            _i("positive", "CONDITIONING"),
            _i("negative", "CONDITIONING"),
            _i("vae", "VAE"),
            _i("latent", "LATENT"),
            _i("image", "IMAGE"),
            _i("frame_idx", "INT", widget=True),
            _i("strength", "FLOAT", widget=True),
            _i("latent_downscale_factor", "FLOAT", widget=True),
            _i("crop", "COMBO", widget=True),
            _i("use_tiled_encode", "BOOLEAN", widget=True),
            _i("tile_size", "INT", widget=True),
            _i("tile_overlap", "INT", widget=True),
        ),
        outputs=(
            _o("positive", "CONDITIONING"),
            _o("negative", "CONDITIONING"),
            _o("latent", "LATENT"),
        ),
        widgets=(
            ("frame_idx", 0), ("strength", 1.0), ("latent_downscale_factor", 1.0),
            ("crop", "disabled"), ("use_tiled_encode", False),
            ("tile_size", 256), ("tile_overlap", 64),
        ),
        size=(340, 226),
    ),
}


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

_SAMPLING: dict[str, NodeSignature] = {
    "RandomNoise": NodeSignature(
        inputs=(_i("noise_seed", "INT", widget=True),),
        outputs=(_o("NOISE", "NOISE"),),
        widgets=(("noise_seed", 0), ("control_after_generate", "fixed")),
        size=(300, 82),
    ),
    "KSamplerSelect": NodeSignature(
        inputs=(_i("sampler_name", "COMBO", widget=True),),
        outputs=(_o("SAMPLER", "SAMPLER"),),
        widgets=(("sampler_name", "euler"),),
        size=(300, 58),
    ),
    # ComfyUI-LTXVideo: an explicit sigma list, which is how the distilled LTX
    # schedules are expressed — they are a fixed schedule, not a step count.
    "ManualSigmas": NodeSignature(
        inputs=(_i("sigmas", "STRING", widget=True),),
        outputs=(_o("SIGMAS", "SIGMAS"),),
        widgets=(("sigmas", "1.0, 0.0"),),
        size=(400, 82),
    ),
    "BasicScheduler": NodeSignature(
        inputs=(
            _i("model", "MODEL"),
            _i("scheduler", "COMBO", widget=True),
            _i("steps", "INT", widget=True),
            _i("denoise", "FLOAT", widget=True),
        ),
        outputs=(_o("SIGMAS", "SIGMAS"),),
        widgets=(("scheduler", "normal"), ("steps", 20), ("denoise", 1.0)),
        size=(300, 106),
    ),
    "CFGGuider": NodeSignature(
        inputs=(
            _i("model", "MODEL"),
            _i("positive", "CONDITIONING"),
            _i("negative", "CONDITIONING"),
            _i("cfg", "FLOAT", widget=True),
        ),
        outputs=(_o("GUIDER", "GUIDER"),),
        widgets=(("cfg", 8.0),),
        size=(300, 106),
    ),
    "LTXVDualCFGGuider": NodeSignature(
        inputs=(
            _i("model", "MODEL"),
            _i("positive", "CONDITIONING"),
            _i("negative", "CONDITIONING"),
            _i("video_cfg", "FLOAT", widget=True),
            _i("audio_cfg", "FLOAT", widget=True),
        ),
        outputs=(_o("GUIDER", "GUIDER"),),
        widgets=(("video_cfg", 3.0), ("audio_cfg", 7.0)),
        size=(300, 130),
    ),
    "SamplerCustomAdvanced": NodeSignature(
        inputs=(
            _i("noise", "NOISE"),
            _i("guider", "GUIDER"),
            _i("sampler", "SAMPLER"),
            _i("sigmas", "SIGMAS"),
            _i("latent_image", "LATENT"),
        ),
        outputs=(_o("output", "LATENT"), _o("denoised_output", "LATENT")),
        size=(300, 130),
    ),
}


# --------------------------------------------------------------------------
# media in and out
# --------------------------------------------------------------------------

_MEDIA: dict[str, NodeSignature] = {
    "LoadImage": NodeSignature(
        inputs=(_i("image", "COMBO", widget=True), _i("upload", "IMAGEUPLOAD", widget=True)),
        outputs=(_o("IMAGE", "IMAGE"), _o("MASK", "MASK")),
        widgets=(("image", ""), ("upload", "image")),
        size=(320, 340),
    ),
    "LoadAudio": NodeSignature(
        inputs=(
            _i("audio", "COMBO", widget=True),
            _i("audioUI", "AUDIO_UI", widget=True),
            _i("upload", "AUDIOUPLOAD", widget=True),
        ),
        outputs=(_o("AUDIO", "AUDIO"),),
        widgets=(("audio", ""), ("audioUI", None), ("upload", None)),
        size=(320, 136),
    ),
    "LoadVideo": NodeSignature(
        inputs=(_i("file", "COMBO", widget=True), _i("upload", "VIDEOUPLOAD", widget=True)),
        outputs=(_o("VIDEO", "VIDEO"),),
        widgets=(("file", ""), ("upload", "video")),
        size=(320, 340),
    ),
    "GetVideoComponents": NodeSignature(
        inputs=(_i("video", "VIDEO"),),
        outputs=(
            _o("images", "IMAGE"),
            _o("audio", "AUDIO"),
            _o("fps", "FLOAT"),
            _o("bit_depth", "INT"),
        ),
        size=(300, 102),
    ),
    "LTXVPreprocess": NodeSignature(
        inputs=(_i("image", "IMAGE"), _i("img_compression", "INT", widget=True)),
        outputs=(_o("output_image", "IMAGE"),),
        widgets=(("img_compression", 35),),
        size=(300, 58),
    ),
    "LTXVAudioVAEEncode": NodeSignature(
        inputs=(_i("audio", "AUDIO"), _i("audio_vae", "VAE")),
        outputs=(_o("Audio Latent", "LATENT"),),
        size=(300, 54),
    ),
    "LTXVAudioVAEDecode": NodeSignature(
        inputs=(_i("samples", "LATENT"), _i("audio_vae", "VAE")),
        outputs=(_o("Audio", "AUDIO"),),
        size=(300, 54),
    ),
    "TrimAudioDuration": NodeSignature(
        inputs=(
            _i("audio", "AUDIO"),
            _i("start_index", "FLOAT", widget=True),
            _i("duration", "FLOAT", widget=True),
        ),
        outputs=(_o("AUDIO", "AUDIO"),),
        widgets=(("start_index", 0.0), ("duration", 5.0)),
        size=(300, 82),
    ),
    "VAEDecodeTiled": NodeSignature(
        inputs=(
            _i("samples", "LATENT"),
            _i("vae", "VAE"),
            _i("tile_size", "INT", widget=True),
            _i("overlap", "INT", widget=True),
            _i("temporal_size", "INT", widget=True),
            _i("temporal_overlap", "INT", widget=True),
        ),
        outputs=(_o("IMAGE", "IMAGE"),),
        widgets=(
            ("tile_size", 512), ("overlap", 64),
            ("temporal_size", 64), ("temporal_overlap", 8),
        ),
        size=(300, 154),
    ),
    "VAEDecode": NodeSignature(
        inputs=(_i("samples", "LATENT"), _i("vae", "VAE")),
        outputs=(_o("IMAGE", "IMAGE"),),
        size=(280, 54),
    ),
    "CreateVideo": NodeSignature(
        inputs=(
            _i("images", "IMAGE"),
            _i("audio", "AUDIO", optional=True),
            _i("fps", "FLOAT", widget=True),
            _i("bit_depth", "INT", widget=True),
        ),
        outputs=(_o("VIDEO", "VIDEO"),),
        widgets=(("fps", 30.0), ("bit_depth", 8)),
        size=(300, 102),
    ),
    "SaveVideo": NodeSignature(
        inputs=(
            _i("video", "VIDEO"),
            _i("filename_prefix", "STRING", widget=True),
            _i("format", "COMBO", widget=True),
            _i("codec", "COMFY_DYNAMICCOMBO_V3", widget=True),
        ),
        outputs=(_o("video", "VIDEO"),),
        widgets=(("filename_prefix", "video/ComfyUI"), ("format", "auto"), ("codec", "auto")),
        size=(640, 400),
    ),
    "SaveAudioAdvanced": NodeSignature(
        inputs=(_i("audio", "AUDIO"),),
        outputs=(_o("audio", "AUDIO"),),
        widgets=(("filename_prefix", "audio/ComfyUI"), ("format", "flac"), ("quality", "320k")),
        size=(400, 136),
    ),
}


# --------------------------------------------------------------------------
# primitives and notes
# --------------------------------------------------------------------------

_UTILITY: dict[str, NodeSignature] = {
    "PrimitiveInt": NodeSignature(
        inputs=(_i("value", "INT", widget=True),),
        outputs=(_o("INT", "INT"),),
        widgets=(("value", 0), ("control_after_generate", "fixed")),
        size=(260, 82),
    ),
    "PrimitiveFloat": NodeSignature(
        inputs=(_i("value", "FLOAT", widget=True),),
        outputs=(_o("FLOAT", "FLOAT"),),
        widgets=(("value", 0.0),),
        size=(260, 58),
    ),
    "PrimitiveBoolean": NodeSignature(
        inputs=(_i("value", "BOOLEAN", widget=True),),
        outputs=(_o("BOOLEAN", "BOOLEAN"),),
        widgets=(("value", False),),
        size=(260, 58),
    ),
    "PrimitiveStringMultiline": NodeSignature(
        inputs=(_i("value", "STRING", widget=True),),
        outputs=(_o("STRING", "STRING"),),
        widgets=(("value", ""),),
        size=(400, 180),
    ),
    "PreviewAny": NodeSignature(
        inputs=(_i("source", "*"),),
        outputs=(_o("STRING", "STRING"),),
        size=(320, 90),
    ),
    "MarkdownNote": NodeSignature(
        widgets=(("text", ""),),
        size=(360, 180),
    ),
    "Note": NodeSignature(
        widgets=(("text", ""),),
        size=(320, 120),
    ),
}


NODE_SIGNATURES: dict[str, NodeSignature] = {
    **_LOADERS,
    **_CONDITIONING,
    **_LATENTS,
    **_SAMPLING,
    **_MEDIA,
    **_UTILITY,
}


def signature_for(node_type: str) -> NodeSignature:
    """The signature for ``node_type``.

    Raises with the node name rather than a ``KeyError``, because this fires
    during compilation and the message reaches the user.
    """
    try:
        return NODE_SIGNATURES[node_type]
    except KeyError:
        raise KeyError(
            f"No signature registered for node type {node_type!r}. Add one to "
            f"director/compiler/signatures.py — its slot order cannot be guessed."
        ) from None

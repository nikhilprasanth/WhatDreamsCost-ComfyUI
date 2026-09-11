"""Attaching the Prompt Relay mask to a model's cross-attention.

The mask is additive and the LTX cross-attention signature already accepts a
``mask`` kwarg, so the patch *wraps* whatever forward is currently installed
rather than replacing it. That is what lets Prompt Relay stack with other nodes
patching the same attention — KJNodes' NAG, for instance.

Two changes from the version this was extracted from, both about failing softly:

* :func:`inspect_model` and :func:`apply_patches` return a reason instead of
  raising. An architecture we do not recognise, or attention another node has
  already claimed, means "render without per-region prompts and say so" — not
  "refuse to render the shot".
* The attention backend is overridden only when *we* contributed a mask. Sage
  and similar backends silently drop arbitrary masks, so the override is
  necessary for our own; imposing it on a mask another node supplied would
  change that node's behaviour behind its back.

Part of Prompt Relay; see :mod:`director.relay.mask` for provenance. GPL-3.0.
"""

from __future__ import annotations

import inspect
import logging
import types
from typing import Any, Callable

import comfy.ldm.modules.attention

# Model classification deliberately lives in a module that does not import
# comfy, so "this model cannot do per-region prompts" is answerable without a
# running ComfyUI.
from .support import ModelSupport, detect_model_type, inspect_model

log = logging.getLogger(__name__)

__all__ = [
    "ModelSupport",
    "apply_patches",
    "detect_model_type",
    "get_current_node_id",
    "inspect_model",
]


def get_current_node_id() -> str:
    """ComfyUI's ``unique_id`` for the node currently executing.

    Found by walking the call stack, because the executor does not pass it to
    helpers. Used only as a cache key, so a wrong answer costs a redundant model
    clone rather than a wrong result — which is why the fallback is a constant
    rather than an exception.
    """
    try:
        frame = inspect.currentframe()
        while frame:
            if "unique_id" in frame.f_locals:
                return str(frame.f_locals["unique_id"])
            frame = frame.f_back
    except Exception:
        pass
    return "default_relay"


# --------------------------------------------------------------------------
# attention plumbing
# --------------------------------------------------------------------------

def _masked_attention(q, k, v, heads, mask, transformer_options={}, **kwargs):
    # Call attention_pytorch directly: wrap_attn may route to a backend that
    # ignores an arbitrary additive mask, which would silently drop the relay.
    return comfy.ldm.modules.attention.attention_pytorch(
        q, k, v, heads, mask=mask,
        _inside_attn_wrapper=True,
        transformer_options=transformer_options,
        **kwargs,
    )


def _make_masked_override(prev: Callable | None) -> Callable:
    """Route mask-bearing attention calls through the pytorch implementation.

    Chains to any previously installed override when no mask is present, so
    another node's backend choice survives.
    """
    def override(func, *args, **kwargs):
        if kwargs.get("mask") is not None:
            return comfy.ldm.modules.attention.attention_pytorch(*args, **kwargs)
        if prev is not None:
            return prev(func, *args, **kwargs)
        return func(*args, **kwargs)

    return override


def debug_log(msg: str) -> None:
    """Deliberately silent. Flip to ``log.debug`` when chasing a mask problem."""


# --------------------------------------------------------------------------
# Wan cross-attention
# --------------------------------------------------------------------------

def _wan_t2v_forward(self, mask_fn, x, context, transformer_options={}, **kwargs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)

    mask = mask_fn(q.shape[1], k.shape[1], q.dtype, q.device, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )
    return self.o(x)


def _wan_i2v_forward(self, mask_fn, x, context, context_img_len, transformer_options={}, **kwargs):
    context_img = context[:, :context_img_len]
    context_text = context[:, context_img_len:]

    q = self.norm_q(self.q(x))

    k_img = self.norm_k_img(self.k_img(context_img))
    v_img = self.v_img(context_img)
    img_x = comfy.ldm.modules.attention.optimized_attention(
        q, k_img, v_img, heads=self.num_heads, transformer_options=transformer_options,
    )

    k = self.norm_k(self.k(context_text))
    v = self.v(context_text)

    mask = mask_fn(q.shape[1], k.shape[1], q.dtype, q.device, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )

    return self.o(x + img_x)


class _CrossAttnPatch:
    """Binds ``(impl, mask_fn)`` as a method on a Wan cross-attention module."""

    def __init__(self, impl: Callable, mask_fn: Callable | None) -> None:
        self.impl = impl
        self.mask_fn = mask_fn

    def __get__(self, obj, objtype=None):
        impl = self.impl

        def wrapped(self_module, *args, **kwargs):
            transformer_options = kwargs.get("transformer_options", {})
            active_mask_fn = transformer_options.get("promptrelay_mask_fn", self.mask_fn)
            return impl(self_module, active_mask_fn, *args, **kwargs)

        return types.MethodType(wrapped, obj)


# --------------------------------------------------------------------------
# LTX cross-attention
# --------------------------------------------------------------------------

def _make_ltx_mask_wrapper(underlying: Callable, mask_fn: Callable | None, attr: str) -> Callable:
    """Wrap an LTX cross-attention forward, adding the relay mask.

    ``underlying`` is whatever ``get_model_object`` returned — the default bound
    forward, or another node's patch — already bound to its module. Wrapping
    rather than replacing is what makes Prompt Relay composable.
    """
    def wrapped(_self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
        debug_log(
            f"wrapped called: x.shape={list(x.shape)} "
            f"context.shape={list(context.shape) if context is not None else None} "
            f"mask_is_none={mask is None}"
        )
        contributed = False
        active_mask_fn = transformer_options.get("promptrelay_mask_fn", mask_fn)
        if active_mask_fn is not None and context is not None:
            opts = {**transformer_options, "promptrelay_attn_type": attr}
            pr_mask = active_mask_fn(x.shape[1], context.shape[1], x.dtype, x.device, opts)
            if pr_mask is not None:
                mask = pr_mask if mask is None else mask + pr_mask
                contributed = True

        if contributed:
            prev = transformer_options.get("optimized_attention_override")
            transformer_options = {
                **transformer_options,
                "optimized_attention_override": _make_masked_override(prev),
            }

        return underlying(
            x, context=context, mask=mask, pe=pe, k_pe=k_pe,
            transformer_options=transformer_options,
        )

    wrapped._promptrelay_wrapper = True  # type: ignore[attr-defined]
    return wrapped


# --------------------------------------------------------------------------
# patching
# --------------------------------------------------------------------------

def _already_patched(model_clone: Any, key: str) -> bool:
    return key in getattr(model_clone, "object_patches", {})


def apply_patches(model_clone: Any, arch: str, mask_fn: Callable | None) -> str:
    """Attach the relay mask. Returns "" on success, or a reason it could not.

    Reporting rather than raising is deliberate: losing per-region prompts
    should cost the user their per-region prompts, not their render.
    """
    diffusion_model = model_clone.get_model_object("diffusion_model")

    if arch == "wan":
        from comfy.ldm.wan.model import WanI2VCrossAttention

        keys = [
            f"diffusion_model.blocks.{idx}.cross_attn.forward"
            for idx in range(len(diffusion_model.blocks))
        ]
        # Wan's cross-attention is replaced rather than wrapped, so another node
        # already holding it is a genuine conflict rather than a composition.
        if any(_already_patched(model_clone, key) for key in keys):
            return (
                "Another node has already modified this model's cross-attention, which "
                "per-region prompts also need."
            )
        for idx, block in enumerate(diffusion_model.blocks):
            cross_attn = block.cross_attn
            impl = (
                _wan_i2v_forward
                if isinstance(cross_attn, WanI2VCrossAttention)
                else _wan_t2v_forward
            )
            model_clone.add_object_patch(
                keys[idx],
                _CrossAttnPatch(impl, mask_fn).__get__(cross_attn, cross_attn.__class__),
            )
        return ""

    if arch == "ltx":
        options = model_clone.model_options.setdefault("transformer_options", {})
        options["promptrelay_mask_fn"] = mask_fn

        attached = 0
        for idx, block in enumerate(diffusion_model.transformer_blocks):
            for attr in ("attn2", "audio_attn2"):
                module = getattr(block, attr, None)
                if module is None:
                    continue
                key = f"diffusion_model.transformer_blocks.{idx}.{attr}.forward"
                # get_model_object returns a prior patch when one exists, so
                # wrapping composes with whatever is already installed.
                underlying = model_clone.get_model_object(key)
                wrapper = _make_ltx_mask_wrapper(underlying, mask_fn, attr)
                model_clone.add_object_patch(key, types.MethodType(wrapper, module))
                attached += 1

        if attached == 0:
            return "This model exposes no cross-attention for per-region prompts to attach to."
        log.info("[Director] Prompt Relay attached to %d attention blocks.", attached)
        return ""

    return f"Per-region prompts are not implemented for {arch!r} models."

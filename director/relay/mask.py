"""Prompt Relay — the temporal cross-attention penalty.

Per-segment prompts in a **single** sampler pass. Each stretch of the timeline
gets its own text, and a Gaussian penalty added to cross-attention keeps video
tokens attending mostly to the text for the moment they belong to.

Nothing upstream does this. ComfyUI-LTXVideo's answer to multi-prompt is
``LTXVLoopingSampler`` + ``MultiPromptProvider``, which samples temporal tiles
sequentially — a different trade-off, with seams and N times the sampling cost.
A single pass has neither.

Provenance
----------
The method is Prompt Relay, described at https://gordonchen19.github.io/Prompt-Relay/
and first implemented for ComfyUI in Kijai's ``ComfyUI-PromptRelay``. This
implementation came into WhatDreamsCost-ComfyUI with LTX Director 1.3 and is
carried forward unchanged here except for its module location; the maths is
covered by a bit-for-bit regression test in ``tests/test_relay.py``.

Licensed GPL-3.0 with the rest of this repository.

The two knobs that matter
-------------------------
``epsilon``
    Sets ``sigma = 1 / ln(1/epsilon)``, which controls how sharply a segment
    boundary cuts. Anything below ~0.1 is a hard boundary; 0.5 and above blends.
``window``
    Half the segment length, minus a margin. Inside the window the penalty is
    zero — the text applies fully; outside it grows quadratically with distance.
"""

from __future__ import annotations

import logging
import math

import torch

log = logging.getLogger(__name__)

__all__ = [
    "build_temporal_cost",
    "build_temporal_cost_scaled",
    "create_mask_fn",
    "build_segments",
    "distribute_segment_lengths",
]


def build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, tokens_per_frame):
    """Gaussian penalty matrix [Lq, Lk] for video cross-attention (integer frame indexing)."""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.long) // tokens_per_frame

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames.float()[:, None] - seg["midpoint"]).abs()
        strength = seg.get("strength", 1.0)
        cost = strength * (torch.relu(d - seg["window"]) ** 2) / (2 * seg["sigma"] ** 2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


def build_temporal_cost_scaled(q_token_idx, Lq, Lk, device, dtype, latent_frames, is_audio=False):
    """Penalty matrix for queries that don't map to integer frames (e.g. LTXAV audio tokens)."""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.float32) * latent_frames / Lq

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames[:, None] - seg["midpoint"]).abs()
        if is_audio:
            sigma_val = seg.get("sigma_audio", seg["sigma"])
            window_val = seg.get("window_audio", seg["window"])
            strength_val = seg.get("strength_audio", 1.0)
        else:
            sigma_val = seg["sigma"]
            window_val = seg["window"]
            strength_val = seg.get("strength", 1.0)
        cost = strength_val * (torch.relu(d - window_val) ** 2) / (2 * sigma_val ** 2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


def debug_log(msg):
    pass


def create_mask_fn(q_token_idx, fallback_tokens_per_frame, latent_frames):
    """Closure: mask_fn(Lq, Lk, dtype, device, transformer_options) -> additive mask or None.

    Takes shapes/dtype/device instead of tensors so callers can compute the mask
    without first materializing q/k projections — required so PromptRelay can
    wrap an existing cross-attn forward (e.g. KJNodes NAG) instead of replacing it.
    """
    cache = {}
    max_token_idx = max(int(seg["local_token_idx"].max().item()) for seg in q_token_idx) + 1

    def mask_fn(Lq, Lk, dtype, device, transformer_options):
        debug_log(f"mask_fn check: Lq={Lq} Lk={Lk} max_token_idx={max_token_idx} cond_or_uncond={transformer_options.get('cond_or_uncond', [])}")
        if Lq == Lk:
            debug_log("mask_fn: Lq == Lk, returning None")
            return None

        # Only apply on conditional pass — not unconditional (negative prompt)
        cond_or_uncond = transformer_options.get("cond_or_uncond", [])
        if 1 in cond_or_uncond and 0 not in cond_or_uncond:
            debug_log("mask_fn: unconditional pass, returning None")
            return None

        grid_sizes = transformer_options.get("grid_sizes", None)
        attn_type = transformer_options.get("promptrelay_attn_type", "attn2")
        is_audio = (attn_type == "audio_attn2")

        if is_audio:
            mode = "scaled"
            video_lq = -1
        else:
            if grid_sizes is not None:
                video_tpf = int(grid_sizes[1]) * int(grid_sizes[2])
            else:
                if Lq % latent_frames == 0:
                    video_tpf = Lq // latent_frames
                else:
                    video_tpf = fallback_tokens_per_frame
            video_lq = latent_frames * video_tpf

            # Skip cross-modal attention — text keys are padded to a fixed length ≥ max_token_idx and != video_lq
            if Lk == video_lq or Lk < max_token_idx:
                debug_log(f"mask_fn: Lk == video_lq ({Lk == video_lq}) or Lk < max_token_idx ({Lk < max_token_idx}), returning None")
                return None

            mode = "video" if Lq == video_lq else "scaled"

        key = (Lq, Lk, mode, device)
        if key not in cache:
            if mode == "video":
                cost = build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, video_tpf)
            else:
                cost = build_temporal_cost_scaled(q_token_idx, Lq, Lk, device, dtype, latent_frames, is_audio=is_audio)
            log.info(
                "[PromptRelay] Built penalty matrix (%s): Lq=%d, Lk=%d, nonzero=%d/%d",
                mode, Lq, Lk, (cost > 0).sum().item(), cost.numel(),
            )
            debug_log(f"Built penalty matrix ({mode}): Lq={Lq}, Lk={Lk}, key={key}")
            cache[key] = -cost

        return cache[key].to(dtype)

    return mask_fn


def build_segments(token_ranges, segment_lengths, epsilon=1e-3, relay_options=None):
    """Per-segment metadata for the temporal penalty.

    relay_options (optional dict) overrides per-stream knobs:
        video_strength, video_window_scale,
        audio_epsilon, audio_strength, audio_window_scale
    Audio knobs only affect architectures whose cross-attention takes the scaled
    (non-integer-frame) path — currently LTX audio_attn2.
    """
    # Paper uses constant sigma = 1/ln(1/epsilon) regardless of segment length
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448

    opts = relay_options or {}
    v_strength = opts.get("video_strength", 1.0)
    v_window_scale = opts.get("video_window_scale", 1.0)
    a_epsilon = opts.get("audio_epsilon")
    a_strength = opts.get("audio_strength", 1.0)
    a_window_scale = opts.get("audio_window_scale", 1.0)

    if a_epsilon is not None and 0 < a_epsilon < 1:
        sigma_audio = 1.0 / math.log(1.0 / a_epsilon)
    else:
        sigma_audio = sigma

    if relay_options:
        log.info(
            "[PromptRelay] Advanced options active — video: strength=%.3f window_scale=%.3f | "
            "audio: epsilon=%s strength=%.3f window_scale=%.3f",
            v_strength, v_window_scale,
            f"{a_epsilon:.4f}" if a_epsilon is not None else "inherit",
            a_strength, a_window_scale,
        )

    q_token_idx = []
    frame_cursor = 0

    for (tok_start, tok_end), L in zip(token_ranges, segment_lengths):
        if L <= 0:
            frame_cursor += L
            continue
        midpoint = (2 * frame_cursor + L) // 2
        base_window = max(L // 2 - 2, 0)
        q_token_idx.append({
            "local_token_idx": torch.arange(tok_start, tok_end),
            "midpoint": midpoint,
            "window": max(base_window * v_window_scale, 0.0),
            "sigma": sigma,
            "strength": v_strength,
            "window_audio": max(base_window * a_window_scale, 0.0),
            "sigma_audio": sigma_audio,
            "strength_audio": a_strength,
        })
        frame_cursor += L

    return q_token_idx


def distribute_segment_lengths(num_segments, latent_frames, specified_lengths=None):
    """Validate or auto-distribute segment frame counts, capped to fit within latent_frames."""
    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError(
                f"Number of segment_lengths ({len(specified_lengths)}) "
                f"must match number of local prompts ({num_segments})"
            )
        lengths = specified_lengths
    else:
        # ceil division — matches reference implementation
        step = -(-latent_frames // num_segments)
        lengths = [step] * num_segments

    effective = []
    cursor = 0
    for L in lengths:
        end = min(cursor + L, latent_frames)
        effective.append(max(end - cursor, 0))
        cursor = end
    return effective

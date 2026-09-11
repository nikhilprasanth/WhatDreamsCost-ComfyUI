"""Compatibility shim — Prompt Relay moved to ``director.relay``.

The maths now lives in :mod:`director.relay.mask` and the prompt/token handling
in :mod:`director.relay.tokens`. This module re-exports both so the Director 2.x
node, and any third-party code that imported from here, keeps working.

New code should import from ``director.relay``.
"""

from __future__ import annotations

from .director.relay.mask import (
    build_segments,
    build_temporal_cost,
    build_temporal_cost_scaled,
    create_mask_fn,
    distribute_segment_lengths,
)
from .director.relay.tokens import get_raw_tokenizer, map_token_indices

__all__ = [
    "build_segments",
    "build_temporal_cost",
    "build_temporal_cost_scaled",
    "create_mask_fn",
    "distribute_segment_lengths",
    "get_raw_tokenizer",
    "map_token_indices",
]

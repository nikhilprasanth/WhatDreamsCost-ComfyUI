"""Prompt assembly and token-range mapping for Prompt Relay.

The penalty matrix needs to know which *token indices* belong to which segment.
That means building one prompt out of the global text plus every segment's text,
then finding where each segment landed after tokenisation.

Done incrementally — tokenise the prefix, then the prefix plus one more segment,
and take the difference — because SentencePiece is context-dependent: tokenising
a fragment on its own does not give the tokens it would produce in place.

Part of Prompt Relay; see :mod:`director.relay.mask` for provenance. GPL-3.0.
"""

from __future__ import annotations

__all__ = ["get_raw_tokenizer", "map_token_indices"]


def get_raw_tokenizer(clip):
    """Extract the raw SPiece/HF tokenizer from a ComfyUI CLIP object."""
    tokenizer_wrapper = clip.tokenizer
    for attr_name in dir(tokenizer_wrapper):
        if attr_name.startswith("_"):
            continue
        inner = getattr(tokenizer_wrapper, attr_name, None)
        if inner is not None and hasattr(inner, "tokenizer"):
            return inner.tokenizer

    raise RuntimeError(
        f"Could not find raw tokenizer on CLIP object. "
        f"Known attributes: {[a for a in dir(tokenizer_wrapper) if not a.startswith('_')]}"
    )


def map_token_indices(raw_tokenizer, global_prompt, local_prompts):
    """Tokenize global + space-prefixed locals; return (full_prompt, per-local token ranges).

    Uses incremental tokenization to avoid SentencePiece context-dependency issues.
    """
    prefixed_locals = [" " + lp for lp in local_prompts]
    full_prompt = global_prompt + "".join(prefixed_locals)
    
    # Detect if the tokenizer appends EOS dynamically
    has_eos = getattr(raw_tokenizer, "add_eos", False)
    if not has_eos:
        try:
            test_res = raw_tokenizer("test")
            if isinstance(test_res, dict) and "input_ids" in test_res:
                ids = test_res["input_ids"]
            elif hasattr(test_res, "input_ids"):
                ids = test_res.input_ids
            elif isinstance(test_res, list):
                ids = test_res
            else:
                ids = []
            
            if ids:
                eos_id = getattr(raw_tokenizer, "eos_token_id", None)
                if eos_id is not None and ids[-1] == eos_id:
                    has_eos = True
                elif ids[-1] == 1:
                    has_eos = True
        except Exception:
            pass

    eos_adj = 1 if has_eos else 0

    prev_len = len(raw_tokenizer(global_prompt)["input_ids"]) - eos_adj
    token_ranges = []
    built = global_prompt

    for plp in prefixed_locals:
        built += plp
        cur_len = len(raw_tokenizer(built)["input_ids"]) - eos_adj
        if cur_len <= prev_len:
            raise ValueError(f"Local prompt produced no tokens: '{plp.strip()}'")
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len

    return full_prompt, token_ranges

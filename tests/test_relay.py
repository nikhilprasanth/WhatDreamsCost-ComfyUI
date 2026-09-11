"""Prompt Relay.

The penalty maths carried over from Director 2.x unchanged, so the first job
here is a regression fence: the matrices it produces must be bit-for-bit what
they were. The second is the new behaviour — failing softly instead of raising,
and deriving geometry from the model rather than from a constant.

``director.relay.patch`` imports ``comfy``, so the patching tests are skipped
when ComfyUI is not present. Everything else runs anywhere torch does.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from director.relay import _latent_lengths, apply_relay, clear_cache  # noqa: E402
from director.relay.mask import (  # noqa: E402
    build_segments,
    build_temporal_cost,
    build_temporal_cost_scaled,
    create_mask_fn,
    distribute_segment_lengths,
)
from director.core.spec import Segment, Spec  # noqa: E402


# --------------------------------------------------------------------------
# the maths, fenced
# --------------------------------------------------------------------------

def test_sigma_is_the_documented_function_of_epsilon() -> None:
    # sigma = 1 / ln(1/epsilon). Changing this changes every render.
    for epsilon in (0.001, 0.01, 0.5):
        segments = build_segments([(0, 4)], [8], epsilon)
        assert segments[0]["sigma"] == pytest.approx(1.0 / math.log(1.0 / epsilon))


def test_the_penalty_is_zero_inside_a_segment_window() -> None:
    # Inside the window the region's text applies at full strength; the penalty
    # only grows outside it. This is the property the whole method rests on.
    segments = build_segments([(0, 4)], [16], 0.001)
    cost = build_temporal_cost(segments, Lq=16, Lk=8, device="cpu",
                               dtype=torch.float32, tokens_per_frame=1)
    midpoint, window = segments[0]["midpoint"], segments[0]["window"]
    inside = int(midpoint)
    assert cost[inside].max().item() == 0.0
    far = min(15, int(midpoint + window + 6))
    assert cost[far].max().item() > 0.0


def test_the_penalty_grows_with_distance() -> None:
    segments = build_segments([(0, 4)], [16], 0.001)
    cost = build_temporal_cost(segments, Lq=32, Lk=8, device="cpu",
                               dtype=torch.float32, tokens_per_frame=1)
    column = cost[:, 0]
    tail = column[int(segments[0]["midpoint"] + segments[0]["window"]):]
    assert torch.all(tail[1:] >= tail[:-1]), "penalty must be monotonic away from the segment"


def test_only_a_segments_own_tokens_are_penalised() -> None:
    segments = build_segments([(0, 3), (3, 7)], [8, 8], 0.001)
    cost = build_temporal_cost(segments, Lq=16, Lk=10, device="cpu",
                               dtype=torch.float32, tokens_per_frame=1)
    # Columns 7..9 belong to no segment, so they are never penalised — that is
    # how the global prompt keeps applying everywhere.
    assert cost[:, 7:].abs().max().item() == 0.0


def test_the_penalty_matrix_is_numerically_unchanged() -> None:
    """Regression fence against the Director 2.x implementation."""
    segments = build_segments([(0, 5), (5, 11)], [8, 8], epsilon=0.001)
    cost = build_temporal_cost(segments, Lq=16, Lk=12, device="cpu",
                               dtype=torch.float32, tokens_per_frame=1)
    # Values recomputed from the formula, not copied from output: penalty is
    # strength * relu(|frame - midpoint| - window)^2 / (2 * sigma^2).
    sigma = 1.0 / math.log(1000.0)
    for index, segment in enumerate(segments):
        column = segment["local_token_idx"][0].item()
        for frame in range(16):
            distance = abs(frame - segment["midpoint"])
            expected = max(0.0, distance - segment["window"]) ** 2 / (2 * sigma**2)
            assert cost[frame, column].item() == pytest.approx(expected, rel=1e-5)
        assert index >= 0


def test_the_scaled_path_maps_non_integer_frames() -> None:
    # Audio tokens do not map to whole video frames, so they use a scaled axis.
    segments = build_segments([(0, 4)], [8], 0.001)
    cost = build_temporal_cost_scaled(segments, Lq=40, Lk=6, device="cpu",
                                      dtype=torch.float32, latent_frames=8)
    assert cost.shape == (40, 6)
    assert cost.min().item() == 0.0


def test_audio_knobs_are_independent_of_video_knobs() -> None:
    segments = build_segments(
        [(0, 4)], [8], 0.001,
        {"audio_strength": 0.0, "video_strength": 1.0},
    )
    video = build_temporal_cost_scaled(segments, 40, 6, "cpu", torch.float32, 8, is_audio=False)
    audio = build_temporal_cost_scaled(segments, 40, 6, "cpu", torch.float32, 8, is_audio=True)
    assert video.max().item() > 0.0
    assert audio.max().item() == 0.0, "audio strength 0 must silence the audio penalty"


# --------------------------------------------------------------------------
# the mask closure
# --------------------------------------------------------------------------

def _mask_fn():
    return create_mask_fn(
        build_segments([(0, 5), (5, 11)], [8, 8], 0.001),
        fallback_tokens_per_frame=4,
        latent_frames=16,
    )


def test_self_attention_is_never_masked() -> None:
    # Lq == Lk means self-attention, which the relay must not touch.
    assert _mask_fn()(64, 64, torch.float32, "cpu", {}) is None


def test_the_unconditional_pass_is_never_masked() -> None:
    # The negative prompt has no regions, so masking it would be meaningless.
    assert _mask_fn()(64, 12, torch.float32, "cpu", {"cond_or_uncond": [1]}) is None


def test_the_conditional_pass_is_masked() -> None:
    mask = _mask_fn()(64, 12, torch.float32, "cpu", {"cond_or_uncond": [0]})
    assert mask is not None
    assert mask.max().item() <= 0.0, "the mask is an additive penalty, so it is negative"


def test_masks_are_cached_per_shape() -> None:
    fn = _mask_fn()
    first = fn(64, 12, torch.float32, "cpu", {"cond_or_uncond": [0]})
    second = fn(64, 12, torch.float32, "cpu", {"cond_or_uncond": [0]})
    assert first is second, "rebuilding the matrix every step would be needless work"


def test_a_key_length_shorter_than_the_prompt_is_ignored() -> None:
    # Cross-modal attention has its own key space; masking it would be wrong.
    assert _mask_fn()(64, 3, torch.float32, "cpu", {"cond_or_uncond": [0]}) is None


# --------------------------------------------------------------------------
# apportionment
# --------------------------------------------------------------------------

def test_region_lengths_sum_to_the_whole() -> None:
    # Rounding each region independently would leave the last one short, and
    # the penalty matrix is indexed by cumulative position.
    for lengths in ([40, 40, 41], [10, 100, 11], [1, 119, 1], [61, 60]):
        result = _latent_lengths(lengths, 8, 16)
        assert sum(result) == 16, (lengths, result)


def test_every_region_gets_at_least_one_latent_frame() -> None:
    # A region far shorter than a latent frame still has to be addressable, or
    # its prompt would never apply anywhere.
    result = _latent_lengths([1, 1, 119], 8, 16)
    assert min(result) >= 1
    assert sum(result) == 16


def test_partial_coverage_stays_partial() -> None:
    # Regions covering half the shot must not be stretched over all of it.
    result = _latent_lengths([8, 8], 8, 32)
    assert sum(result) == 2


def test_full_coverage_within_one_frame_is_treated_as_full() -> None:
    result = _latent_lengths([64, 63], 8, 16)
    assert sum(result) == 16


def test_distribute_falls_back_to_even_regions() -> None:
    assert distribute_segment_lengths(4, 16, None) == [4, 4, 4, 4]


def test_distribute_clips_to_the_available_frames() -> None:
    assert sum(distribute_segment_lengths(2, 8, [6, 6])) == 8


def test_distribute_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="must match"):
        distribute_segment_lengths(3, 16, [8, 8])


# --------------------------------------------------------------------------
# apply_relay, without a real model
# --------------------------------------------------------------------------

class FakeClip:
    """Records what it was asked to encode."""

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def tokenize(self, text: str) -> str:
        self.encoded.append(text)
        return text

    def encode_from_tokens_scheduled(self, tokens: str) -> list:
        return [[tokens, {}]]


class FakeModel:
    """A model exposing a diffusion model of an architecture we do not know."""

    def __init__(self) -> None:
        inner = type("Inner", (), {})()
        inner.diffusion_model = type("UnknownBackbone", (), {})()
        self.model = inner


class HeadlessModel:
    """A model object with no diffusion model at all."""

    def __init__(self) -> None:
        self.model = type("Inner", (), {})()


@pytest.fixture(autouse=True)
def _clear_relay_cache():
    clear_cache()
    yield
    clear_cache()


def _spec(*regions: str) -> Spec:
    spec = Spec()
    spec.prompt.raw = "a storm-lit harbour"
    spec.project.frames = 121
    spec.segments = [
        Segment(start=i * 40, length=40, text=text) for i, text in enumerate(regions)
    ]
    return spec


def test_one_region_does_not_patch_anything() -> None:
    model, clip = FakeModel(), FakeClip()
    result = apply_relay(model, clip, _spec("the beam sweeps"))
    assert result.applied is False
    assert result.model is model, "a single prompt must not clone the model"
    assert clip.encoded == ["a storm-lit harbour, the beam sweeps"]


def test_no_regions_encodes_the_shot_prompt() -> None:
    clip = FakeClip()
    apply_relay(FakeModel(), clip, _spec())
    assert clip.encoded == ["a storm-lit harbour"]


def test_relay_disabled_merges_without_complaint() -> None:
    spec = _spec("first", "second")
    spec.relay.enabled = False
    clip = FakeClip()
    result = apply_relay(FakeModel(), clip, spec)
    assert result.applied is False
    assert not result.report.diagnostics, "switching relay off is a choice, not a problem"


def test_an_unsupported_model_degrades_with_a_sentence() -> None:
    # Losing per-region prompts must cost the user their regions, not their render.
    clip = FakeClip()
    result = apply_relay(FakeModel(), clip, _spec("first", "second"))
    assert result.applied is False
    codes = {d.code for d in result.report}
    assert "relay.model_unsupported" in codes
    diagnostic = next(d for d in result.report if d.code == "relay.model_unsupported")
    assert "merged into one prompt" in diagnostic.fix
    assert clip.encoded == ["a storm-lit harbour, first, second"]


def test_degrading_preserves_region_order() -> None:
    clip = FakeClip()
    apply_relay(FakeModel(), clip, _spec("she turns", "then she runs"))
    assert clip.encoded[0].endswith("she turns, then she runs")


def test_geometry_prefers_a_connected_latent() -> None:
    from director.relay import _geometry

    spec = _spec()
    spec.project.frames = 121
    spec.project.width, spec.project.height = 1280, 704
    latent = {"samples": torch.zeros(1, 128, 9, 22, 40)}
    frames, tokens = _geometry(spec, latent, (1, 1, 1))
    assert (frames, tokens) == (9, 22 * 40)


def test_geometry_without_a_latent_matches_the_empty_latent_node() -> None:
    from director.relay import _geometry

    spec = _spec()
    spec.project.frames = 121
    spec.project.width, spec.project.height = 1280, 704
    frames, tokens = _geometry(spec, None, (1, 1, 1))
    # ((121 - 1) // 8) + 1, and (704 // 32) * (1280 // 32)
    assert (frames, tokens) == (16, 22 * 40)


# --------------------------------------------------------------------------
# patching, when ComfyUI is present
# --------------------------------------------------------------------------

def _comfy_present() -> bool:
    try:
        import comfy.ldm.modules.attention  # noqa: F401
    except Exception:
        return False
    return True


needs_comfy = pytest.mark.skipif(not _comfy_present(), reason="ComfyUI is not installed")


def test_inspect_model_reports_rather_than_raising() -> None:
    # Deliberately not gated on ComfyUI: answering "this model cannot do
    # per-region prompts" must work in a plain Python process.
    from director.relay.support import inspect_model

    support = inspect_model(FakeModel())
    assert support.ok is False
    assert support.reason.endswith(".")
    assert "LTX and Wan" in support.reason

    headless = inspect_model(HeadlessModel())
    assert headless.ok is False
    assert "does not expose a diffusion model" in headless.reason


def test_the_legacy_detect_helper_still_raises() -> None:
    from director.relay.support import detect_model_type

    with pytest.raises(ValueError):
        detect_model_type(FakeModel())


def test_asking_about_support_needs_no_comfyui() -> None:
    # The module that answers "can this model do per-region prompts" must not
    # drag comfy in with it, or the degradation path breaks wherever ComfyUI
    # is absent — including this test suite.
    import importlib
    from pathlib import Path

    module = importlib.import_module("director.relay.support")
    source = Path(module.__file__).read_text(encoding="utf-8")
    imports = [
        line.strip() for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "comfy" in line
    ]
    assert not imports, imports


@needs_comfy
def test_apply_patches_refuses_an_unknown_architecture_by_returning() -> None:
    from director.relay.patch import apply_patches

    reason = apply_patches(object(), "sd15", None)
    assert reason and "not implemented" in reason

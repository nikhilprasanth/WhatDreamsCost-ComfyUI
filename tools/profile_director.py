#!/usr/bin/env python3
"""Measure the Director against the budgets in ``docs/ARCHITECTURE.md`` §11.

Everything here runs without a GPU, without models and without ComfyUI, because
everything it measures is supposed to. Numbers go into ``docs/PERFORMANCE.md``.

Usage::

    python tools/profile_director.py [--json] [--runs N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from director.compiler import compile_spec  # noqa: E402
from director.core import new_spec, validate  # noqa: E402
from director.core.spec import MediaEntry, Reference, Segment, Spec  # noqa: E402


def measure(name: str, budget_ms: float | None, fn: Callable[[], Any], runs: int) -> dict[str, Any]:
    fn()  # warm up: the first call pays import and type-hint resolution costs
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    median = statistics.median(samples)
    return {
        "name": name,
        "median_ms": round(median, 3),
        "p95_ms": round(sorted(samples)[int(len(samples) * 0.95) - 1], 3),
        "budget_ms": budget_ms,
        "within_budget": budget_ms is None or median <= budget_ms,
        "runs": runs,
    }


def shot(references: int = 0, segments: int = 0, stages: int = 1) -> Spec:
    spec = new_spec()
    spec.prompt.raw = "A lighthouse beam sweeps across a storm-lit harbour."
    spec.models.family = "ltx2.5"
    spec.models.unet = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    spec.models.clip = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    spec.models.vae = "ltx-2.5-video-vae-bf16.safetensors"
    spec.models.audio_vae = "ltx-2.5-audio-vae-bf16.safetensors"
    spec.generation.stages = stages
    if stages > 1:
        spec.models.upscaler = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

    if references:
        spec.project.mode = "keyframes"
        spec.project.frames = max(spec.project.frames, references * 8 + 1)
        for index in range(references):
            entry = MediaEntry(filename=f"frame_{index:03d}.png", kind="image",
                               width=1280, height=704)
            spec.media[entry.id] = entry
            anchor = "start" if index == 0 else "index"
            reference = Reference(role="keyframe", media=entry.id, anchor=anchor)
            reference.at.frame = min(index * 8, spec.project.frames - 1)
            spec.references.append(reference)

    if segments:
        span = max(1, spec.project.frames // segments)
        spec.segments = [
            Segment(start=i * span, length=span, text=f"beat {i}") for i in range(segments)
        ]
    return spec.normalise()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    simple = shot()
    heavy = shot(references=20, segments=8, stages=2)
    huge = shot(references=200, segments=16, stages=2)

    simple_json = simple.to_json()
    heavy_json = heavy.to_json()

    results = [
        # Budget: spec serialisation < 2 ms. Hit on every keystroke, debounced.
        measure("serialise a simple shot", 2.0, simple.to_json, args.runs),
        measure("serialise a 20-reference shot", 2.0, heavy.to_json, args.runs),
        measure("parse a simple shot", 2.0, lambda: Spec.from_json(simple_json), args.runs),
        measure("parse a 20-reference shot", 2.0, lambda: Spec.from_json(heavy_json), args.runs),
        measure("settings digest", 2.0, simple.digest, args.runs),

        # Budget: validation is called as the user types.
        measure("validate a simple shot", 5.0, lambda: validate(simple), args.runs),
        measure("validate a 20-reference shot", 5.0, lambda: validate(heavy), args.runs),
        measure("validate a 200-reference shot", 50.0, lambda: validate(huge), max(5, args.runs // 5)),

        # Budget: compile < 30 ms. Runs on Generate and on Compile Workflow.
        measure("compile text-to-video", 30.0, lambda: compile_spec(simple), args.runs),
        measure("compile 20 keyframes, two stages", 30.0, lambda: compile_spec(heavy), args.runs),
        measure("compile as subgraphs", 30.0,
                lambda: compile_spec(heavy, layout="subgraphs"), args.runs),
        measure("compile 200 keyframes", 200.0,
                lambda: compile_spec(huge), max(5, args.runs // 10)),
    ]

    sizes = {
        "simple shot, bytes": len(simple_json),
        "20-reference shot, bytes": len(heavy_json),
        "200-reference shot, bytes": len(huge.to_json()),
        "compiled workflow, text-to-video, bytes": len(
            json.dumps(compile_spec(simple).workflow)
        ),
        "compiled workflow, 20 keyframes two stages, bytes": len(
            json.dumps(compile_spec(heavy).workflow)
        ),
    }
    # Budget: project JSON under 64 KB for a 20-reference shot.
    size_budget = {"20-reference shot, bytes": 64 * 1024}

    payload = {"timings": results, "sizes": sizes, "size_budgets": size_budget}

    if args.json:
        print(json.dumps(payload, indent=1))
        return 0

    width = max(len(r["name"]) for r in results) + 2
    print(f"{'measurement'.ljust(width)}{'median':>10}{'p95':>10}{'budget':>10}")
    print("-" * (width + 30))
    for row in results:
        budget = f"{row['budget_ms']:.0f} ms" if row["budget_ms"] else "—"
        flag = "" if row["within_budget"] else "  OVER"
        print(f"{row['name'].ljust(width)}{row['median_ms']:>9.2f}ms"
              f"{row['p95_ms']:>9.2f}ms{budget:>10}{flag}")

    print()
    for name, value in sizes.items():
        budget = size_budget.get(name)
        note = ""
        if budget:
            note = f"   (budget {budget // 1024} KB){'  OVER' if value > budget else ''}"
        print(f"{name.ljust(width)}{value / 1024:>9.1f} KB{note}")

    over = [r["name"] for r in results if not r["within_budget"]]
    if over:
        print(f"\nover budget: {', '.join(over)}")
        return 1
    print("\nall within budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

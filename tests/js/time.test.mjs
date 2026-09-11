/**
 * The browser-side time mirror.
 *
 * This module duplicates `director/core/time.py` so dragging a marker can snap
 * at pointer speed. A duplicate is only safe while it agrees with the original,
 * so these are the same cases the Python tests use — if the two ever disagree,
 * one of these files fails.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import * as t from "../../js/director/state/time.js";

test("frame counts snap up to 1 + 8k", () => {
  const cases = [[1, 1], [2, 9], [8, 9], [9, 9], [10, 17], [97, 97], [120, 121], [121, 121], [122, 129]];
  for (const [input, expected] of cases) {
    assert.equal(t.snapFrames(input), expected, `snapFrames(${input})`);
  }
});

test("snapping is idempotent and always valid", () => {
  for (let n = -3; n < 400; n += 1) {
    const once = t.snapFrames(n);
    assert.equal(t.snapFrames(once), once, `not idempotent at ${n}`);
    assert.ok(t.isValidFrameCount(once), `${once} is not 1 + 8k`);
  }
});

test("snapping never shortens a shot", () => {
  // "5 seconds" must never quietly become less than the user asked for.
  for (let n = 1; n < 400; n += 1) assert.ok(t.snapFrames(n) >= n);
});

test("the nearest valid counts bracket the input", () => {
  assert.deepEqual(t.nearestValidFrames(120), [113, 121]);
  assert.deepEqual(t.nearestValidFrames(121), [121, 121]);
  assert.deepEqual(t.nearestValidFrames(2), [1, 9]);
});

test("latent frames match EmptyLTXVLatentVideo", () => {
  for (const [frames, latent] of [[1, 1], [9, 2], [97, 13], [121, 16]]) {
    assert.equal(t.latentFrames(frames), latent);
  }
});

test("dimensions snap to multiples of 32", () => {
  for (const [input, expected] of [[0, 32], [31, 32], [47, 32], [48, 64], [700, 704], [1280, 1280]]) {
    assert.equal(t.snapDim(input), expected, `snapDim(${input})`);
  }
  for (let n = 0; n < 2048; n += 7) {
    const once = t.snapDim(n);
    assert.equal(t.snapDim(once), once);
    assert.ok(t.isValidDim(once));
  }
});

test("seconds round-trip through frames at every usual frame rate", () => {
  for (const fps of [24, 25, 30, 50]) {
    for (let frame = 0; frame < 200; frame += 13) {
      assert.equal(t.secondsToFrames(t.framesToSeconds(frame, fps), fps), frame);
    }
  }
});

test("percent maps the endpoints exactly and clamps outside them", () => {
  assert.equal(t.percentToFrames(0, 121), 0);
  assert.equal(t.percentToFrames(100, 121), 120);
  assert.equal(t.percentToFrames(-50, 121), 0);
  assert.equal(t.percentToFrames(500, 121), 120);
  assert.equal(t.framesToPercent(120, 121), 100);
});

test("a single-frame shot has no percent range to divide by", () => {
  assert.equal(t.percentToFrames(50, 1), 0);
  assert.equal(t.framesToPercent(0, 1), 0);
});

test("a zero frame rate does not divide by zero", () => {
  assert.equal(t.framesToSeconds(100, 0), 0);
  assert.equal(t.secondsToFrames(4, 0), 0);
});

test("unit conversion is its own inverse", () => {
  for (const unit of ["seconds", "frames", "percent"]) {
    for (const frame of [0, 1, 48, 120]) {
      const value = t.fromFrames(frame, unit, 24, 121);
      assert.equal(t.toFrames(value, unit, 24, 121), frame, `${unit} at ${frame}`);
    }
  }
});

test("timecode reads as mm:ss.cc", () => {
  assert.equal(t.timecode(0, 24), "00:00.00");
  assert.equal(t.timecode(36, 24), "00:01.50");
  assert.equal(t.timecode(24 * 65, 24), "01:05.00");
});

test("multi-frame guides snap down to the 8-frame grid, negatives do not", () => {
  assert.equal(t.snapGuideIndex(13, 25), 8);
  assert.equal(t.snapGuideIndex(16, 25), 16);
  // -1 is the documented way to anchor a last frame; snapping it would break it.
  assert.equal(t.snapGuideIndex(-1, 25), -1);
  // Single frames accept any index.
  assert.equal(t.snapGuideIndex(13, 1), 13);
});

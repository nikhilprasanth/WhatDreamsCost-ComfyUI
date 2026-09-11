/**
 * The frame and geometry rules, mirrored from `director/core/time.py`.
 *
 * This is the one place the editor duplicates Python, and it is deliberate:
 * dragging a marker has to snap at pointer speed, and a round trip per pointer
 * move is not that. Everything slower than a drag — validation, compilation —
 * still goes to the server, so the mirror is a latency optimisation rather than
 * a second implementation.
 *
 * The two constants below come from ComfyUI's `EmptyLTXVLatentVideo`: `length`
 * has step 8 and the latent is `height // 32` by `width // 32`. If they ever
 * change, they change in Python first and here second — and the golden-file
 * tests will notice before a user does.
 */

export const TEMPORAL_STRIDE = 8;
export const SPATIAL_STRIDE = 32;

/** True when `frames` is `1 + 8k`. */
export function isValidFrameCount(frames) {
  return frames >= 1 && (frames - 1) % TEMPORAL_STRIDE === 0;
}

/**
 * Round up to the next valid frame count.
 *
 * Up rather than to-nearest, so "5 seconds" never quietly becomes less than the
 * user asked for.
 */
export function snapFrames(frames) {
  if (frames <= 1) return 1;
  return Math.ceil((frames - 1) / TEMPORAL_STRIDE) * TEMPORAL_STRIDE + 1;
}

/** The valid counts either side, so a diagnostic can offer a choice. */
export function nearestValidFrames(frames) {
  const up = snapFrames(frames);
  if (up === frames) return [frames, frames];
  return [Math.max(1, up - TEMPORAL_STRIDE), up];
}

/** Latent temporal size, matching `EmptyLTXVLatentVideo`. */
export function latentFrames(frames) {
  return Math.floor((Math.max(1, frames) - 1) / TEMPORAL_STRIDE) + 1;
}

/** Round a pixel dimension to the nearest multiple of 32. */
export function snapDim(value, minimum = SPATIAL_STRIDE) {
  if (value <= minimum) return minimum;
  return Math.max(minimum, Math.round(value / SPATIAL_STRIDE) * SPATIAL_STRIDE);
}

export function isValidDim(value) {
  return value >= SPATIAL_STRIDE && value % SPATIAL_STRIDE === 0;
}

// --------------------------------------------------------------------------
// units
//
// Frames are the storage unit everywhere. Seconds and percent are display
// preferences, converted on the way in and out, so changing fps re-labels
// positions instead of moving them.
// --------------------------------------------------------------------------

export function framesToSeconds(frames, fps) {
  return fps > 0 ? frames / fps : 0;
}

export function secondsToFrames(seconds, fps) {
  return fps > 0 ? Math.round(seconds * fps) : 0;
}

export function percentToFrames(percent, total) {
  if (total <= 1) return 0;
  const ratio = Math.min(Math.max(percent, 0), 100) / 100;
  return Math.round(ratio * (total - 1));
}

export function framesToPercent(frames, total) {
  if (total <= 1) return 0;
  return Math.min(Math.max(frames / (total - 1), 0), 1) * 100;
}

export function toFrames(value, unit, fps, total) {
  if (unit === "frames") return Math.round(value);
  if (unit === "seconds") return secondsToFrames(value, fps);
  if (unit === "percent") return percentToFrames(value, total);
  return Math.round(value);
}

export function fromFrames(frames, unit, fps, total) {
  if (unit === "frames") return frames;
  if (unit === "seconds") return framesToSeconds(frames, fps);
  if (unit === "percent") return framesToPercent(frames, total);
  return frames;
}

/** `mm:ss.cc`, for rulers and inspector readouts. */
export function timecode(frames, fps) {
  const seconds = framesToSeconds(frames, fps);
  const minutes = Math.floor(Math.max(0, seconds) / 60);
  const rest = Math.max(0, seconds) - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${rest.toFixed(2).padStart(5, "0")}`;
}

/** A short label in whatever unit the user is reading in. */
export function label(frames, unit, fps, total) {
  if (unit === "frames") return `${frames}f`;
  if (unit === "percent") return `${framesToPercent(frames, total).toFixed(0)}%`;
  return `${framesToSeconds(frames, fps).toFixed(2)}s`;
}

/** Snap a guide index the way `LTXVAddGuide` would, so the UI shows the truth. */
export function snapGuideIndex(frameIdx, guideFrames = 1) {
  if (frameIdx < 0 || guideFrames <= TEMPORAL_STRIDE) return frameIdx;
  return Math.floor(frameIdx / TEMPORAL_STRIDE) * TEMPORAL_STRIDE;
}

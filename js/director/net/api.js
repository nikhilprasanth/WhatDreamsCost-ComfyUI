/**
 * The Director's HTTP client.
 *
 * Every call goes through `request`, which turns a failure into the same
 * `{message, fix}` shape the validator produces. That is what lets the editor
 * have one way to show a problem regardless of where it came from — a rule
 * violation, a missing model, or a dead server.
 */

import { api } from "../../../../scripts/api.js";

export class DirectorError extends Error {
  constructor(message, fix = "", code = "") {
    super(message);
    this.name = "DirectorError";
    this.fix = fix;
    this.code = code;
  }
}

async function request(path, { method = "GET", body = null, signal = null } = {}) {
  let response;
  try {
    response = await api.fetchApi(path, {
      method,
      signal,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new DirectorError(
      "ComfyUI did not respond.",
      "Check that the server is still running, then try again."
    );
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok || payload?.ok === false) {
    const error = payload?.error ?? {};
    throw new DirectorError(
      error.message ?? `The request to ${path} failed (${response.status}).`,
      error.fix ?? "",
      error.code ?? ""
    );
  }
  return payload ?? {};
}

// --------------------------------------------------------------------------
// capabilities and vocabulary
// --------------------------------------------------------------------------

let capabilityCache = null;

/** Feature flags, installed models and suggested defaults. Cached per page. */
export async function capabilities({ refresh = false } = {}) {
  if (capabilityCache && !refresh) return capabilityCache;
  capabilityCache = await request(`/ltxdirector/capabilities${refresh ? "?refresh=1" : ""}`);
  return capabilityCache;
}

let vocabularyCache = null;

/**
 * Camera and lens vocabulary.
 *
 * Fetched rather than duplicated in JavaScript, so the labels shown in a
 * dropdown and the prose the compiler emits cannot drift apart.
 */
export async function vocabulary() {
  vocabularyCache ??= await request("/ltxdirector/vocabulary");
  return vocabularyCache;
}

let presetCache = null;

export async function presets() {
  presetCache ??= await request("/ltxdirector/presets");
  return presetCache;
}

// --------------------------------------------------------------------------
// the shot
// --------------------------------------------------------------------------

/**
 * Validate a shot.
 *
 * The same rules the nodes run, so what the editor shows and what ComfyUI
 * executes cannot disagree.
 */
export async function validate(project, { signal = null } = {}) {
  return request("/ltxdirector/validate", { method: "POST", body: { project }, signal });
}

/** Compile a shot to a native graph. */
export async function compile(project, { layout = "flat", includeApi = false } = {}) {
  return request("/ltxdirector/compile", {
    method: "POST",
    body: { project, layout, include_api: includeApi },
  });
}

/** Compile and queue a shot, so Generate needs no canvas. */
export async function queue(project) {
  return request("/ltxdirector/queue", {
    method: "POST",
    body: { project, client_id: api.clientId ?? api.initialClientId },
  });
}

// --------------------------------------------------------------------------
// project files
// --------------------------------------------------------------------------

export async function saveProject(project, name) {
  return request("/ltxdirector/project/save", { method: "POST", body: { project, name } });
}

export async function loadProject(name) {
  return request(`/ltxdirector/project/load?name=${encodeURIComponent(name)}`);
}

export async function listProjects() {
  return request("/ltxdirector/project/list");
}

/** Convert a Director 2.x timeline. Returns the project and what was lost. */
export async function importTimeline(payload) {
  return request("/ltxdirector/project/import", { method: "POST", body: payload });
}

// --------------------------------------------------------------------------
// media
// --------------------------------------------------------------------------

export async function probeMedia(path) {
  return request(`/ltxdirector/media/probe?path=${encodeURIComponent(path)}`);
}

export async function listMedia() {
  return request("/ltxdirector/media/list");
}

export async function audioPeaks(path) {
  return request(`/ltxdirector/media/peaks?path=${encodeURIComponent(path)}`);
}

/**
 * A thumbnail URL.
 *
 * A URL rather than image data: the browser's own cache does the caching, and
 * decoded pixels never enter project state.
 */
export function thumbnailUrl(path, atSeconds = 0) {
  const query = new URLSearchParams({ path, t: String(atSeconds) });
  return api.apiURL(`/ltxdirector/media/thumb?${query}`);
}

export { request as _request };

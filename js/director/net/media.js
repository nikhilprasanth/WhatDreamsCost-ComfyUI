/**
 * Uploading media.
 *
 * Two properties matter here, and both come from content addressing:
 *
 * - **A file already on the server is never uploaded again.** The hash is
 *   computed in the browser and checked first, so dropping the same reference
 *   into a third shot costs one round trip and no bytes.
 * - **Large files work at all.** ComfyUI's own upload endpoint caps out well
 *   below the size of a usable source clip, so this chunks.
 *
 * What never happens: decoding the file in the browser. Dimensions, duration,
 * thumbnails and waveforms all come back from the server, which reads a header
 * rather than a stream.
 */

import { api } from "../../../../scripts/api.js";
import { DirectorError } from "./api.js";

/** Comfortably under the server's per-chunk limit. */
const CHUNK_BYTES = 8 * 1024 * 1024;

const IMAGE = /\.(png|jpe?g|webp|bmp|gif|tiff?|avif)$/i;
const VIDEO = /\.(mp4|webm|mkv|avi|mov|m4v|flv|wmv|mpg)$/i;
const AUDIO = /\.(wav|mp3|ogg|flac|m4a|aac|opus|wma)$/i;

export function kindFor(filename) {
  if (VIDEO.test(filename)) return "video";
  if (AUDIO.test(filename)) return "audio";
  if (IMAGE.test(filename)) return "image";
  return "unknown";
}

export function isSupported(filename) {
  return kindFor(filename) !== "unknown";
}

/**
 * SHA-256 of a File, computed in chunks.
 *
 * `crypto.subtle.digest` wants the whole buffer, so a very large file is hashed
 * by its head, tail and size instead. That is not a cryptographic hash of the
 * contents, but it is a perfectly good cache key — and it means a 4 GB clip does
 * not have to be read into memory to find out it is already on the server.
 */
export async function fileDigest(file) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return `size-${file.size}-${file.lastModified}-${file.name}`;

  const SAMPLE = 4 * 1024 * 1024;
  let buffer;
  if (file.size <= SAMPLE * 2) {
    buffer = await file.arrayBuffer();
  } else {
    const head = new Uint8Array(await file.slice(0, SAMPLE).arrayBuffer());
    const tail = new Uint8Array(await file.slice(file.size - SAMPLE).arrayBuffer());
    const size = new TextEncoder().encode(String(file.size));
    const joined = new Uint8Array(head.length + tail.length + size.length);
    joined.set(head, 0);
    joined.set(tail, head.length);
    joined.set(size, head.length + tail.length);
    buffer = joined.buffer;
  }
  const digest = await subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function post(formData) {
  const response = await api.fetchApi("/ltxdirector/media/upload", {
    method: "POST",
    body: formData,
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok || payload?.ok === false) {
    const error = payload?.error ?? {};
    throw new DirectorError(
      error.message ?? "The upload failed.",
      error.fix ?? "Check that ComfyUI can write to its input folder."
    );
  }
  return payload;
}

/**
 * Upload a file, skipping the transfer when the server already has it.
 *
 * @param {File} file
 * @param {(fraction: number) => void} [onProgress]
 * @returns {Promise<object>} a media entry: path, kind, sha256, dimensions, duration
 */
export async function upload(file, onProgress = null) {
  if (!isSupported(file.name)) {
    throw new DirectorError(
      `${file.name} is not an image, video or audio file.`,
      "Drop a PNG, JPEG, MP4, MOV, WAV or MP3."
    );
  }

  const digest = await fileDigest(file);

  // Ask before sending. This is the whole point of content addressing.
  const check = await api.fetchApi("/ltxdirector/media/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sha256: digest, filename: file.name, size: file.size }),
  }).then((r) => r.json()).catch(() => null);

  if (check?.exists) {
    onProgress?.(1);
    return { ...check, sha256: digest, reused: true };
  }

  const chunks = Math.max(1, Math.ceil(file.size / CHUNK_BYTES));
  let result = null;
  for (let index = 0; index < chunks; index += 1) {
    const slice = file.slice(index * CHUNK_BYTES, (index + 1) * CHUNK_BYTES);
    const form = new FormData();
    form.append("file", slice, file.name);
    form.append("append", index > 0 ? "true" : "false");
    form.append("final", index === chunks - 1 ? "true" : "false");
    form.append("sha256", digest);
    result = await post(form);
    onProgress?.((index + 1) / chunks);
  }

  return { ...result, reused: false };
}

/**
 * Upload several files, reporting each as it lands.
 *
 * Sequential on purpose: a browser will happily open six parallel uploads of a
 * gigabyte each and make the server unusable while they run.
 */
export async function uploadAll(files, { onFile = null, onProgress = null } = {}) {
  const results = [];
  const list = [...files];
  for (let index = 0; index < list.length; index += 1) {
    const file = list[index];
    try {
      const entry = await upload(file, (fraction) => {
        onProgress?.((index + fraction) / list.length, file.name);
      });
      results.push({ ok: true, file, entry });
      onFile?.(entry, file);
    } catch (error) {
      results.push({ ok: false, file, error });
    }
  }
  return results;
}

/** Turn an upload result into a media registry entry. */
export function toMediaEntry(result) {
  return {
    filename: result.filename ?? (result.path ?? "").split("/").pop() ?? "",
    subfolder: (result.path ?? "").includes("/")
      ? result.path.slice(0, result.path.lastIndexOf("/"))
      : "",
    kind: result.kind ?? kindFor(result.path ?? ""),
    sha256: result.sha256 ?? "",
    width: result.width ?? 0,
    height: result.height ?? 0,
    duration: result.duration ?? 0,
    fps: result.fps ?? 0,
    size: result.size ?? 0,
  };
}

/** The path a media entry refers to, as ComfyUI's loaders expect it. */
export function pathFor(entry) {
  if (!entry) return "";
  return entry.subfolder ? `${entry.subfolder}/${entry.filename}` : entry.filename;
}

/** Files from a drop or paste event, filtered to what we can use. */
export function filesFrom(event) {
  const transfer = event.dataTransfer ?? event.clipboardData;
  if (!transfer) return [];
  return [...(transfer.files ?? [])].filter((file) => isSupported(file.name));
}

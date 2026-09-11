/**
 * The project store.
 *
 * Every state bug the previous Director had came from keeping the project in
 * three places and syncing them by hand. These tests defend the replacement:
 * one document, typed mutations, undo that restores exact equality, and
 * serialisation that happens in exactly one place.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Store, emptyProject } from "../../js/director/state/store.js";

/** A store that records everything written to the node widget. */
function makeStore() {
  const writes = [];
  const store = new Store({ persist: (json) => writes.push(json) });
  return { store, writes };
}

function snapshot(store) {
  return JSON.stringify(store.state);
}

// --------------------------------------------------------------------------
// shape
// --------------------------------------------------------------------------

test("a new project matches the schema the Python side expects", () => {
  const project = emptyProject();
  assert.equal(project.schema, 3);
  assert.equal(project.project.mode, "t2v");
  assert.equal(project.project.frames, 121);
  assert.equal(project.project.width % 32, 0);
  // An untouched camera control must add nothing to the prompt.
  assert.equal(project.prompt.camera.move, "");
});

test("a project carries no media payloads", () => {
  // The whole point of the rebuild: project state stays small and portable.
  const { store } = makeStore();
  store.addMedia({ filename: "shot.png", kind: "image", sha256: "abc", width: 1280, height: 704 });
  const json = store.toJSON();
  assert.ok(!json.includes("base64"));
  assert.ok(!json.includes("data:image"));
  assert.ok(json.length < 4096);
});

// --------------------------------------------------------------------------
// serialisation
// --------------------------------------------------------------------------

test("loading and re-serialising is lossless", () => {
  const { store } = makeStore();
  store.addReference({ label: "first" });
  store.addSegment({ start: 0, length: 40, text: "the beam sweeps" });
  const json = store.toJSON();

  const reloaded = new Store({ persist: () => {} });
  reloaded.loadJSON(json, { record: false });
  assert.equal(reloaded.toJSON(), json);
});

test("an empty or malformed widget value yields a usable project", () => {
  // A freshly dropped node has nothing in it, which is not an error.
  for (const value of ["", "   ", "{not json", "null"]) {
    const { store } = makeStore();
    store.loadJSON(value, { record: false });
    assert.equal(store.state.schema, 3);
    assert.equal(store.project.frames, 121);
  }
});

test("a project missing newer fields is filled in without losing what it has", () => {
  const { store } = makeStore();
  store.load({ schema: 3, project: { fps: 30, frames: 97 }, prompt: { raw: "kept" } },
    { record: false });
  assert.equal(store.project.fps, 30);
  assert.equal(store.prompt.raw, "kept");
  // Defaults arrive for everything the saved project predates.
  assert.equal(store.generation.seed, 42);
  assert.ok(store.state.audio);
});

test("writes go to the node exactly once, on a debounce", async () => {
  const { store, writes } = makeStore();
  store.update("a", (s) => { s.prompt.raw = "one"; });
  store.update("b", (s) => { s.prompt.raw = "two"; });
  store.update("c", (s) => { s.prompt.raw = "three"; });
  assert.equal(writes.length, 0, "nothing is written while the user is still typing");

  store.flush();
  assert.equal(writes.length, 1);
  assert.equal(JSON.parse(writes[0]).prompt.raw, "three");
});

// --------------------------------------------------------------------------
// undo and redo
// --------------------------------------------------------------------------

test("undo restores exact equality", () => {
  const { store } = makeStore();
  const before = snapshot(store);
  store.update("edit", (s) => { s.prompt.raw = "changed"; s.project.frames = 249; });
  assert.notEqual(snapshot(store), before);

  assert.equal(store.undo(), true);
  assert.equal(snapshot(store), before);
});

test("redo restores exact equality too", () => {
  const { store } = makeStore();
  store.update("edit", (s) => { s.prompt.raw = "changed"; });
  const after = snapshot(store);
  store.undo();
  assert.equal(store.redo(), true);
  assert.equal(snapshot(store), after);
});

test("undo on a fresh store is a no-op, not a crash", () => {
  const { store } = makeStore();
  assert.equal(store.undo(), false);
  assert.equal(store.redo(), false);
});

test("a new edit clears the redo stack", () => {
  const { store } = makeStore();
  store.update("one", (s) => { s.prompt.raw = "a"; });
  store.undo();
  assert.equal(store.canRedo, true);
  store.update("two", (s) => { s.prompt.raw = "b"; });
  assert.equal(store.canRedo, false);
});

test("coalesced edits produce one undo entry, not sixty", () => {
  // Dragging a marker must not fill the undo stack.
  const { store } = makeStore();
  const before = snapshot(store);
  const reference = store.addReference({});
  const afterAdd = snapshot(store);

  for (let frame = 0; frame < 40; frame += 1) {
    store.updateReference(reference.id, { at: { frame, unit: "seconds" } }, { coalesce: true });
  }
  store.undo();
  assert.equal(snapshot(store), afterAdd, "one undo should unwind the whole drag");
  store.undo();
  assert.equal(snapshot(store), before);
});

test("a transaction is one undo entry", () => {
  const { store } = makeStore();
  const before = snapshot(store);
  store.transaction("preset", (state) => {
    state.generation.stages = 2;
    state.project.width = 1920;
    state.project.height = 1088;
  });
  store.undo();
  assert.equal(snapshot(store), before);
});

test("cosmetic changes are never recorded for undo", () => {
  const { store } = makeStore();
  store.setUi({ zoom: 3, playhead: 40, selection: "ref_x" });
  assert.equal(store.canUndo, false, "moving the playhead is not an edit");
});

test("the undo stack is bounded", () => {
  const store = new Store({ persist: () => {}, historyLimit: 5 });
  for (let i = 0; i < 20; i += 1) {
    store.update(`edit ${i}`, (s) => { s.generation.seed = i; });
  }
  let undos = 0;
  while (store.undo()) undos += 1;
  assert.equal(undos, 5);
});

// --------------------------------------------------------------------------
// the digest
// --------------------------------------------------------------------------

test("presentation and history never change the digest", () => {
  const { store } = makeStore();
  const before = store.digestSource();
  store.setUi({ zoom: 4, playhead: 99, expanded: ["models"] });
  store.recordTake({ seed: 1, created: "now" });
  store.update("meta", (s) => { s.meta.name = "Renamed"; }, { cosmetic: true });
  assert.equal(store.digestSource(), before);
});

test("anything that changes the output changes the digest", () => {
  const { store } = makeStore();
  const before = store.digestSource();
  store.update("seed", (s) => { s.generation.seed += 1; });
  assert.notEqual(store.digestSource(), before);
});

// --------------------------------------------------------------------------
// invariants
// --------------------------------------------------------------------------

test("duration snaps to a length LTX can generate", () => {
  const { store } = makeStore();
  store.setDuration(120);
  assert.equal(store.project.frames, 121);
  store.setDuration(2);
  assert.equal(store.project.frames, 9);
});

test("normalising clamps references into the shot", () => {
  const { store } = makeStore();
  const reference = store.addReference({ at: { frame: 9999, unit: "frames" }, strength: 4 });
  store.normalise();
  const stored = store.references.find((r) => r.id === reference.id);
  assert.equal(stored.at.frame, store.project.frames - 1);
  assert.equal(stored.strength, 1);
});

test("shortening a shot pulls markers back inside it", () => {
  const { store } = makeStore();
  store.setDuration(241);
  const reference = store.addReference({ at: { frame: 200, unit: "frames" } });
  store.setDuration(49);
  store.normalise();
  assert.ok(store.references.find((r) => r.id === reference.id).at.frame <= 48);
});

test("prompt regions stay in time order", () => {
  const { store } = makeStore();
  store.addSegment({ start: 80, length: 40, text: "c" });
  store.addSegment({ start: 0, length: 40, text: "a" });
  store.addSegment({ start: 40, length: 40, text: "b" });
  store.normalise();
  assert.deepEqual(store.segments.map((s) => s.text), ["a", "b", "c"]);
});

test("a zero frame rate is repaired rather than propagated", () => {
  const { store } = makeStore();
  store.update("fps", (s) => { s.project.fps = 0; });
  store.normalise();
  assert.equal(store.project.fps, 24);
});

// --------------------------------------------------------------------------
// identity
// --------------------------------------------------------------------------

test("ids are unique and prefixed by kind", () => {
  const { store } = makeStore();
  const ids = new Set();
  for (let i = 0; i < 200; i += 1) ids.add(store.addReference({}).id);
  assert.equal(ids.size, 200);
  for (const id of ids) assert.ok(id.startsWith("ref_"), id);
});

test("editing a reference never changes its id", () => {
  const { store } = makeStore();
  const reference = store.addReference({ label: "before" });
  store.updateReference(reference.id, { label: "after", strength: 0.5 });
  assert.equal(store.references[0].id, reference.id);
  assert.equal(store.references[0].label, "after");
});

test("removing something only removes that something", () => {
  const { store } = makeStore();
  const keep = store.addReference({ label: "keep" });
  const drop = store.addReference({ label: "drop" });
  store.removeReference(drop.id);
  assert.deepEqual(store.references.map((r) => r.id), [keep.id]);
});

// --------------------------------------------------------------------------
// subscribers
// --------------------------------------------------------------------------

test("subscribers are told what changed", () => {
  const { store } = makeStore();
  const reasons = [];
  store.subscribe((_, reason) => reasons.push(reason));
  store.update("prompt", (s) => { s.prompt.raw = "x"; });
  store.setUi({ playhead: 3 });
  assert.deepEqual(reasons, ["prompt", "ui"]);
});

test("a view that throws does not stop the others", () => {
  // One broken panel must not freeze the whole editor.
  const { store } = makeStore();
  const seen = [];
  store.subscribe(() => { throw new Error("boom"); });
  store.subscribe(() => seen.push(true));
  const originalError = console.error;
  console.error = () => {};
  try {
    store.update("prompt", (s) => { s.prompt.raw = "x"; });
  } finally {
    console.error = originalError;
  }
  assert.deepEqual(seen, [true]);
});

test("unsubscribing stops the callbacks", () => {
  const { store } = makeStore();
  let calls = 0;
  const stop = store.subscribe(() => { calls += 1; });
  store.update("a", (s) => { s.prompt.raw = "1"; });
  stop();
  store.update("b", (s) => { s.prompt.raw = "2"; });
  assert.equal(calls, 1);
});

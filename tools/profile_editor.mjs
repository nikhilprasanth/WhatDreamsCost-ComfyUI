/**
 * Editor-side timings, for docs/PERFORMANCE.md.
 *
 * Measures the store, which is what runs on every keystroke and every drag
 * frame. The canvas raster itself needs a real browser and is a manual check —
 * see docs/PERFORMANCE.md.
 *
 *   node tools/profile_editor.mjs
 */

import { Store } from "../js/director/state/store.js";

function measure(name, budgetMs, fn, runs = 200) {
  fn();
  const samples = [];
  for (let i = 0; i < runs; i += 1) {
    const start = performance.now();
    fn();
    samples.push(performance.now() - start);
  }
  samples.sort((a, b) => a - b);
  const median = samples[Math.floor(samples.length / 2)];
  return { name, median, p95: samples[Math.floor(samples.length * 0.95)], budgetMs };
}

function storeWith(references) {
  const store = new Store({ persist: () => {} });
  store.setDuration(references * 8 + 1);
  for (let i = 0; i < references; i += 1) {
    const media = store.addMedia({ filename: `f${i}.png`, kind: "image" });
    store.addReference({ media, at: { frame: i * 8, unit: "frames" } });
  }
  return store;
}

const small = storeWith(20);
const large = storeWith(200);

const rows = [
  measure("serialise 20 references", 2, () => small.toJSON()),
  measure("serialise 200 references", 8, () => large.toJSON()),
  measure("digest 200 references", 8, () => large.digestSource()),
  measure("normalise 200 references", 4, () => large.normalise()),
  measure("one drag frame, 200 references", 4, () => {
    const id = large.references[100].id;
    large.updateReference(id, { at: { frame: 40, unit: "frames" } }, { coalesce: true });
  }),
  measure("undo with 200 references", 16, () => {
    large.update("edit", (s) => { s.generation.seed += 1; });
    large.undo();
  }, 50),
];

const width = Math.max(...rows.map((r) => r.name.length)) + 2;
console.log("measurement".padEnd(width) + "median".padStart(10) + "p95".padStart(10)
  + "budget".padStart(10));
console.log("-".repeat(width + 30));
let over = 0;
for (const row of rows) {
  const flag = row.median <= row.budgetMs ? "" : "  OVER";
  if (flag) over += 1;
  console.log(row.name.padEnd(width)
    + `${row.median.toFixed(2)}ms`.padStart(10)
    + `${row.p95.toFixed(2)}ms`.padStart(10)
    + `${row.budgetMs} ms`.padStart(10) + flag);
}
console.log(over ? `\n${over} over budget` : "\nall within budget");
process.exit(over ? 1 : 0);

#!/usr/bin/env bun
/**
 * Tree-scoring micro-bench (#2307): row-at-a-time `scoreRow` against batched
 * `scoreRows`, on the real node rows of real pages, for the region and section
 * heads. Every variant's scores must equal `scoreRow`'s bit for bit; the run
 * fails otherwise.
 *
 *   bun run experiments/score-bench.ts --in name=<dir>[,name=<dir>...] \
 *     --artifact classifier-v2.json [--limit N] [--reps 5] [--out summary.json]
 *
 * Rows are built once per page outside every timer. Variants run in rotating
 * order on each page (A/B in one process, so container speed cancels out).
 */
import { readFileSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { TOKEN_BUCKETS, featurizeDocument } from "../featurize.ts";
import { type Head, scoreRow, scoreRows } from "../runtime/model-format.ts";
import { toVector, tryLoadModel } from "../runtime/predict.ts";
import { decode } from "./link-position.ts";

type Variant = {
  name: string;
  run: (head: Head, rows: Float32Array, count: number, out: Float32Array) => void;
};

function argument(name: string, fallback?: string): string {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? fallback : Bun.argv[index + 1];
  if (value === undefined || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

function quantile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  return (
    Math.round(sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))]! * 1000) / 1000
  );
}

const artifact = JSON.parse(readFileSync(argument("--artifact"), "utf8"));
const { model, reason } = tryLoadModel(artifact);
if (!model) throw new Error(`artifact unusable: ${reason}`);
const width = model.nodeFeatureCount + TOKEN_BUCKETS;
const reps = Number(argument("--reps", "5"));
const limit = Number(argument("--limit", "0"));

function perRow(head: Head, rows: Float32Array, count: number, out: Float32Array): void {
  const scores = new Float32Array(head.numClass);
  for (let r = 0; r < count; r++) {
    scoreRow(head, rows.subarray(r * width, (r + 1) * width), scores);
    out.set(scores, r * head.numClass);
  }
}

function batched(size: number) {
  return (head: Head, rows: Float32Array, count: number, out: Float32Array): void => {
    for (let start = 0; start < count; start += size) {
      const n = Math.min(size, count - start);
      scoreRows(
        head,
        rows.subarray(start * width, (start + n) * width),
        width,
        n,
        out.subarray(start * head.numClass, (start + n) * head.numClass),
      );
    }
  };
}

const variants: Variant[] = [
  { name: "perRow", run: perRow },
  { name: "batch8", run: batched(8) },
  { name: "batch32", run: batched(32) },
  { name: "batch128", run: batched(128) },
  { name: "batchAll", run: batched(1 << 30) },
];
const heads = { region: model.region, section: model.section! } as const;

const pages: { set: string; rows: Float32Array; count: number }[] = [];
for (const pair of argument("--in").split(",")) {
  const [name, dir] = pair.split("=") as [string, string];
  const manifest = readFileSync(`${dir}/manifest.jsonl`, "utf8").split("\n").filter(Boolean);
  for (const line of limit > 0 ? manifest.slice(0, limit) : manifest) {
    const row = JSON.parse(line) as { sha256: string; url: string; source?: string };
    let html: string;
    try {
      html = decode(gunzipSync(readFileSync(`${dir}/blobs/${row.sha256}.html.gz`)));
    } catch {
      continue;
    }
    const { document } = parseHTML(html);
    const featurized = featurizeDocument(document as never, row.url);
    const count = featurized.nodes.length;
    if (count === 0) continue;
    const rows = new Float32Array(count * width);
    featurized.nodes.forEach((node, i) => {
      rows.set(toVector(node.features, node.tokens, model.nodeFeatureCount, null, null), i * width);
    });
    pages.push({ set: row.source ? `${name}:${row.source}` : name, rows, count });
  }
}

const times: Record<string, Record<string, number[]>> = {};
let mismatches = 0;
for (const [headName, head] of Object.entries(heads)) {
  times[headName] = Object.fromEntries(variants.map((v) => [v.name, pages.map(() => Infinity)]));
  for (const [p, page] of pages.entries()) {
    const reference = new Float32Array(page.count * head.numClass);
    perRow(head, page.rows, page.count, reference);
    const referenceBits = new Uint32Array(reference.buffer);
    for (let rep = 0; rep < reps; rep++) {
      for (let j = 0; j < variants.length; j++) {
        const variant = variants[(j + rep + p) % variants.length]!;
        const out = new Float32Array(page.count * head.numClass);
        const started = performance.now();
        variant.run(head, page.rows, page.count, out);
        const elapsed = performance.now() - started;
        const slot = times[headName]![variant.name]!;
        slot[p] = Math.min(slot[p]!, elapsed);
        const bits = new Uint32Array(out.buffer);
        for (let i = 0; i < bits.length; i++) {
          if (bits[i] !== referenceBits[i]) {
            mismatches++;
            break;
          }
        }
      }
    }
  }
}

const summary = {
  cpu: cpus()[0]?.model ?? "unknown",
  pages: pages.length,
  rows: pages.reduce((s, p) => s + p.count, 0),
  reps,
  mismatches,
  heads: Object.fromEntries(
    Object.entries(times).map(([headName, byVariant]) => {
      const base = byVariant.perRow!.reduce((s, v) => s + v, 0);
      return [
        headName,
        Object.fromEntries(
          Object.entries(byVariant).map(([name, values]) => {
            const total = values.reduce((s, v) => s + v, 0);
            return [
              name,
              {
                totalMs: Math.round(total),
                perPageP50: quantile(values, 0.5),
                perPageP95: quantile(values, 0.95),
                speedup: Math.round((base / total) * 100) / 100,
              },
            ];
          }),
        ),
      ];
    }),
  ),
};
console.log(JSON.stringify(summary, null, 2));
const out = Bun.argv.includes("--out") ? argument("--out") : "";
if (out) writeFileSync(out, `${JSON.stringify(summary, null, 2)}\n`);
if (mismatches > 0) {
  console.error(`${mismatches} variant runs differ from scoreRow`);
  process.exit(1);
}

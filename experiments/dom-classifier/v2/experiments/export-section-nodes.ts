#!/usr/bin/env bun
/**
 * Node features and shipped-head section probabilities for sampled corpus nodes (#2307).
 *
 *   bun run experiments/export-section-nodes.ts --items items.jsonl --corpus <page-corpus> \
 *       --artifact artifact3=<classifier-v2.json> [--artifact r1=...] --out nodes.jsonl
 *
 * One row per item: {itemId, pageId, nodeId, tag, depth, f: [dense node features (NODE_FEATURE_NAMES order)],
 * k: [token buckets], p: {artifactName: [section-head probabilities in the head's class order]}}.
 * `classes` for each artifact is written to <out>.classes.json. These are the SAME vectors the shipped section head
 * reads (featurizer v2.2.0 candidate walk, numeric features + 512 hashed tokens), so a second-stage model or a retrained
 * head sees exactly the runtime's input. No page text, no urls.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { featurizeDocument, NODE_FEATURE_NAMES } from "../featurize.ts";
import { scoreRow } from "../runtime/model-format.ts";
import { tryLoadModel, toVector } from "../runtime/predict.ts";
import { decode } from "./link-position.ts";

function values(name: string): string[] {
  const out: string[] = [];
  Bun.argv.forEach((v, i) => {
    if (v === name && Bun.argv[i + 1]) out.push(Bun.argv[i + 1]!);
  });
  return out;
}
const corpus = values("--corpus")[0]!;
const outPath = values("--out")[0]!;
const items = readFileSync(values("--items")[0]!, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as { itemId: string; pageId: string; nodeId: string });
const artifacts = values("--artifact").map((spec) => {
  const [name, path] = spec.split("=") as [string, string];
  const { model, reason } = tryLoadModel(JSON.parse(readFileSync(path, "utf8")));
  if (!model?.section) throw new Error(`${name}: ${reason || "no section head"}`);
  return { name, model };
});

const wanted = new Set(items.map((i) => i.pageId));
const sha = new Map<string, string>();
for (const line of readFileSync(`${corpus}/pages/manifest.jsonl`, "utf8").split("\n")) {
  if (!line) continue;
  try {
    const row = JSON.parse(line);
    const full = row.raw?.sha256 as string | undefined;
    if (full && wanted.has(full.slice(0, 16))) sha.set(full.slice(0, 16), full);
  } catch {
    // half-written last line
  }
}
const byPage = new Map<string, typeof items>();
for (const item of items) byPage.set(item.pageId, [...(byPage.get(item.pageId) ?? []), item]);

function softmax(scores: Float32Array): number[] {
  let max = -Infinity;
  for (const s of scores) max = Math.max(max, s);
  const exps = Array.from(scores, (s) => Math.exp(s - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => Math.round((e / total) * 10000) / 10000);
}

const rows: string[] = [];
for (const [pageId, pageItems] of byPage) {
  const full = sha.get(pageId);
  if (!full) continue;
  try {
    const html = decode(gunzipSync(readFileSync(`${corpus}/pages/blobs/${full}.html.gz`)).subarray(0, 4_000_000));
    const { document } = parseHTML(html);
    const featurized = featurizeDocument(document as never, "https://x.invalid/", {});
    const byId = new Map(featurized.nodes.map((n) => [n.nodeId, n]));
    for (const item of pageItems) {
      const node = byId.get(Number(item.nodeId));
      if (!node) continue;
      const p: Record<string, number[]> = {};
      for (const { name, model } of artifacts) {
        const vector = toVector(node.features, node.tokens, model.nodeFeatureCount, null, null);
        const scores = new Float32Array(model.section!.numClass);
        scoreRow(model.section!, vector, scores);
        p[name] = softmax(scores);
      }
      rows.push(
        JSON.stringify({
          itemId: item.itemId,
          pageId,
          nodeId: node.nodeId,
          tag: node.tag,
          depth: node.depth,
          f: node.features.map((v) => Math.round(v * 10000) / 10000),
          k: node.tokens,
          p,
        }),
      );
    }
  } catch {
    // unreadable page
  }
}
writeFileSync(outPath, rows.join("\n") + "\n");
writeFileSync(
  `${outPath}.classes.json`,
  JSON.stringify({ classes: Object.fromEntries(artifacts.map((a) => [a.name, a.model.section!.classes])), featureNames: NODE_FEATURE_NAMES }),
);
console.log(JSON.stringify({ rows: rows.length, items: items.length, artifacts: artifacts.map((a) => a.name), denseFeatures: NODE_FEATURE_NAMES.length }));

#!/usr/bin/env bun
/**
 * Section-head predictions on sampled corpus nodes, thresholds off (#2307).
 *
 *   bun run experiments/predict-section-nodes.ts --items items.jsonl --corpus <page-corpus> \
 *       --artifact artifact3=<classifier-v2.json> [--artifact r1=...] --out predictions.jsonl
 *
 * Every artifact runs through `classifyPage` as shipped (comments rule included) EXCEPT that the section head's
 * class thresholds (and platform overrides) are set to 0, so any node whose top class is not `none` carries that
 * class and its probability. Per-class thresholds are then chosen offline against the labels (analysis scripts).
 * A node the head leaves at `none` has `label: null`. `source` is "rules" for the comments rule.
 * Output rows: {artifact, itemId, label, confidence, source}. No page text.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

import { classifyPage, tryLoadModel } from "../runtime/predict.ts";
import { decode } from "./link-position.ts";

function values(name: string): string[] {
  const out: string[] = [];
  Bun.argv.forEach((v, i) => {
    if (v === name && Bun.argv[i + 1]) out.push(Bun.argv[i + 1]!);
  });
  return out;
}
const corpus = values("--corpus")[0]!;
const items = readFileSync(values("--items")[0]!, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as { itemId: string; pageId: string; nodeId: string; url: string });
const artifacts = values("--artifact").map((spec) => {
  const [name, path] = spec.split("=") as [string, string];
  const { model, reason } = tryLoadModel(JSON.parse(readFileSync(path, "utf8")));
  if (!model?.section) throw new Error(`${name}: ${reason || "no section head"}`);
  model.section.classThresholds.fill(0);
  model.section.platformThresholds.clear();
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

const rows: string[] = [];
let pages = 0;
for (const [pageId, pageItems] of byPage) {
  const full = sha.get(pageId);
  if (!full) continue;
  const html = decode(gunzipSync(readFileSync(`${corpus}/pages/blobs/${full}.html.gz`)).subarray(0, 4_000_000));
  for (const { name, model } of artifacts) {
    let result;
    try {
      result = classifyPage(html, pageItems[0]!.url, { model, budgetMs: 5000 });
    } catch {
      continue;
    }
    const byNode = new Map(result.nodes.map((n) => [n.nodeRef.nodeId, n]));
    for (const item of pageItems) {
      const node = byNode.get(Number(item.nodeId));
      rows.push(
        JSON.stringify({
          artifact: name,
          itemId: item.itemId,
          label: node?.section?.label ?? null,
          confidence: node?.section ? Math.round(node.section.confidence * 1000) / 1000 : null,
          source: node?.section?.source ?? null,
        }),
      );
    }
  }
  pages++;
}
writeFileSync(values("--out")[0]!, rows.join("\n") + "\n");
console.log(JSON.stringify({ pages, items: items.length, rows: rows.length, artifacts: artifacts.map((a) => a.name) }));

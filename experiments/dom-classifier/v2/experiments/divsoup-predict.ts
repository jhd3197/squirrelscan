#!/usr/bin/env bun
/**
 * Predict region labels for the div-soup evaluation sample, for ONE artifact.
 *
 * This is the "before / after" half of G2's non-semantic evaluation. It reads
 * the sample (which nodes), never the labels: the labels live in the embargoed
 * `divsoup-eval-v1-labels/` directory and are joined in afterwards by
 * `divsoup_score.py`, so this file can be re-run against any artifact without
 * anybody having looked at the answers.
 *
 * Two fields in the sample are the MODEL'S OWN output from an earlier run
 * (`predicted`, `confidence`). They are dropped at the parse boundary here for
 * the same reason `run-divsoup-v7.ts` drops them: a prediction produced with a
 * previous prediction in view is not an independent measurement.
 *
 * Node identity is `classifyDocument`'s `nodeRef.nodeId`, which is the
 * featurizer's preorder element index — the same numbering the sample was
 * written with. It is NOT an index into `extractPage`'s candidate array.
 *
 *   bun run divsoup-predict.ts --sample <sample.jsonl> \
 *     --store <a/content-store.db>[:<b/content-store.db>] \
 *     --artifact <classifier-v2.json> --out <predictions.jsonl>
 */
import { Database } from "bun:sqlite";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { classifyDocument, tryLoadModel } from "../runtime/predict.ts";

function argument(name: string, fallback?: string): string {
  const index = Bun.argv.indexOf(name);
  if (index < 0) {
    if (fallback !== undefined) return fallback;
    throw new Error(`missing ${name}`);
  }
  const value = Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

const samplePath = resolve(argument("--sample"));
const artifactPath = resolve(argument("--artifact"));
const outPath = resolve(argument("--out"));
const storePaths = argument("--store")
  .split(":")
  .map((path) => resolve(path.trim()))
  .filter(Boolean);

type SamplePage = {
  pageId: string;
  contentHash: string;
  url: string;
  split: string;
  nodeIds: number[];
};
const sample: SamplePage[] = readFileSync(samplePath, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => {
    const row = JSON.parse(line) as {
      pageId: string;
      contentHash: string;
      url: string;
      split: string;
      candidates: Array<{ nodeId: number }>;
    };
    // `predicted` and `confidence` are not read into anything.
    return {
      pageId: row.pageId,
      contentHash: row.contentHash,
      url: row.url,
      split: row.split,
      nodeIds: row.candidates.map((candidate) => candidate.nodeId),
    };
  });

const stores = storePaths.map((path) => new Database(path, { readonly: true }));
const selects = stores.map((store) => ({
  exact: store.query<{ content: Uint8Array }, [string]>(
    "select content from content where hash = ?",
  ),
  prefix: store.query<{ content: Uint8Array }, [string, string]>(
    "select content from content where hash >= ? and hash < ? limit 2",
  ),
}));

function htmlFor(hash: string): string | null {
  let row: { content: Uint8Array } | null = null;
  const isFullHash = hash.length === 64;
  for (const select of selects) {
    if (isFullHash) row = select.exact.get(hash);
    else {
      const rows = select.prefix.all(hash, `${hash}g`);
      if (rows.length > 1) return null;
      row = rows[0] ?? null;
    }
    if (row?.content) break;
  }
  if (!row?.content) return null;
  const bytes = row.content instanceof Uint8Array ? row.content : new Uint8Array(row.content);
  try {
    return new TextDecoder().decode(gunzipSync(bytes));
  } catch {
    try {
      return new TextDecoder().decode(bytes);
    } catch {
      return null;
    }
  }
}

const { model, reason } = tryLoadModel(JSON.parse(readFileSync(artifactPath, "utf8")));
if (!model) throw new Error(`artifact would not load: ${reason || "unknown"}`);

mkdirSync(dirname(outPath), { recursive: true, mode: 0o700 });
const out: string[] = [];
const totals = { pages: 0, nodes: 0, missingHtml: 0, missingNode: 0, degraded: 0 };
for (const page of sample) {
  const html = htmlFor(page.contentHash);
  if (!html) {
    totals.missingHtml++;
    continue;
  }
  const { document } = parseHTML(html);
  const result = classifyDocument(document as never, page.url, { model });
  if (result.degraded) totals.degraded++;
  const byNode = new Map(result.nodes.map((node) => [node.nodeRef.nodeId, node]));
  totals.pages++;
  for (const nodeId of page.nodeIds) {
    const node = byNode.get(nodeId);
    if (!node) {
      totals.missingNode++;
      continue;
    }
    totals.nodes++;
    out.push(
      JSON.stringify({
        pageId: page.pageId,
        nodeId,
        split: page.split,
        label: node.label,
        confidence: node.confidence,
        source: node.source,
        tag: node.nodeRef.tag,
      }),
    );
  }
}
for (const store of stores) store.close();
writeFileSync(outPath, out.map((line) => `${line}\n`).join(""), { mode: 0o600 });
console.log(JSON.stringify({ stage: "complete", artifact: artifactPath, ...totals }));

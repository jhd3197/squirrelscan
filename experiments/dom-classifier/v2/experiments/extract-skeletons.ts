#!/usr/bin/env bun
/**
 * Skeletons and skeleton features for every training page (#2307). Read-only on all inputs.
 *
 *   bun run experiments/extract-skeletons.ts --out <dir> \
 *       --corpus /Volumes/BACKUP/page-corpus \
 *       --dataset <silver-cc dir> --store <content-store.db> [--shard i/n]
 * With `--shard i/n` a process takes the pages whose id hashes to shard i of n (run n of them, one --out each).
 *
 * Two sources, one id space: `p` is the first 16 hex of the page's content sha256 (the
 * corpus `raw.sha256`, and the `pageId` of the silver datasets and their label files).
 * A page in both sources is written once. Sealed domains and bucket >= 90 are skipped,
 * so nothing here is evaluation data.
 *
 * Output `skeletons.jsonl`, one row per page:
 *   {p, src, d (domain hash), s (train|val), f: [numbers in SKELETON_FEATURES order], c: compact skeleton}
 * `c` holds page words (headings, title): PRIVATE, never commit or copy it.
 * The tags-only skeleton is `c` with the quoted words dropped; features carry no words.
 */
import { Database } from "bun:sqlite";
import { createHash } from "node:crypto";
import { appendFileSync, createReadStream, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { createGunzip, gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import {
  buildSkeleton,
  renderCompact,
  SKELETON_FEATURES,
  skeletonFeatures,
  skeletonTokens,
} from "../runtime/skeleton.ts";
import { decode } from "./link-position.ts";

function arg(name: string): string | undefined {
  const i = Bun.argv.indexOf(name);
  return i < 0 ? undefined : Bun.argv[i + 1];
}
const shardSpec = (arg("--shard") ?? "0/1").split("/").map(Number);
const [shard, shards] = [shardSpec[0]!, shardSpec[1]!];
/** A page belongs to exactly one shard, by its id, so a page in both sources is never written twice. */
const mine = (p: string) => shards <= 1 || parseInt(p.slice(0, 4), 16) % shards === shard;
const out = arg("--out");
if (!out) throw new Error("--out is required");
mkdirSync(out, { recursive: true });

const sha16 = (text: string) => createHash("sha256").update(text).digest("hex").slice(0, 16);
const bucketOf = (domain: string) =>
  parseInt(createHash("sha256").update(domain).digest("hex").slice(0, 8), 16) % 100;

const corpus = arg("--corpus");
const sealed = new Set<string>(
  readFileSync(`${corpus ?? "/Volumes/BACKUP/page-corpus"}/splits/sealed-domains.jsonl`, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l).d as string),
);

const seen = new Set<string>();
const limit = Number(arg("--limit") ?? 0);
const sink = `${out}/skeletons.jsonl`;
writeFileSync(sink, "");
let buffer: string[] = [];
let written = 0;
const flush = () => {
  if (buffer.length) appendFileSync(sink, buffer.join("\n") + "\n");
  written += buffer.length;
  buffer = [];
};
const stats = {
  corpus: 0,
  silver: 0,
  skippedSealed: 0,
  skippedBucket: 0,
  duplicate: 0,
  failed: 0,
  notStored: 0,
};

function emit(p: string, src: string, domainHash: string, split: string, html: string): void {
  if (limit && written + buffer.length >= limit) return;
  if (seen.has(p)) {
    stats.duplicate++;
    return;
  }
  seen.add(p);
  try {
    const tree = buildSkeleton(parseHTML(html).document);
    const f = skeletonFeatures(tree);
    buffer.push(
      JSON.stringify({
        p,
        src,
        d: domainHash,
        s: split,
        f: SKELETON_FEATURES.map((k) => f[k]),
        k: skeletonTokens(tree),
        c: renderCompact(tree),
      }),
    );
    if (buffer.length >= 500) flush();
  } catch {
    stats.failed++;
  }
}

if (corpus) {
  for (const line of readFileSync(`${corpus}/pages/manifest.jsonl`, "utf8").split("\n")) {
    if (!line) continue;
    const page = JSON.parse(line);
    if (page.raw?.status !== "stored") {
      stats.notStored++;
      continue;
    }
    if (page.bucket >= 90) {
      stats.skippedBucket++;
      continue;
    }
    if (sealed.has(page.domain_hash)) {
      stats.skippedSealed++;
      continue;
    }
    if (!mine(page.raw.sha256.slice(0, 16))) continue;
    try {
      const bytes = gunzipSync(readFileSync(`${corpus}/pages/blobs/${page.raw.sha256}.html.gz`));
      emit(
        page.raw.sha256.slice(0, 16),
        "corpus",
        page.domain_hash,
        page.split,
        decode(bytes.length > 8_000_000 ? bytes.subarray(0, 8_000_000) : bytes),
      );
      stats.corpus++;
    } catch {
      stats.failed++;
    }
  }
}

const dataset = arg("--dataset");
const storePath = arg("--store");
if (dataset && storePath) {
  const db = new Database(storePath, { readonly: true });
  const select = db.query<{ content: Uint8Array }, [string, string]>(
    "SELECT content FROM content WHERE hash >= ? AND hash < ? LIMIT 2",
  );
  const input = createInterface({
    input: createReadStream(`${dataset}/page-index.jsonl.gz`).pipe(createGunzip()),
  });
  for await (const line of input) {
    if (!line) continue;
    const row = JSON.parse(line) as { p: string; domain: string; s: string };
    if (!mine(row.p)) continue;
    const digest = sha16(row.domain);
    if (bucketOf(row.domain) >= 90) {
      stats.skippedBucket++;
      continue;
    }
    if (sealed.has(digest)) {
      stats.skippedSealed++;
      continue;
    }
    const rows = select.all(row.p, `${row.p}g`);
    if (rows.length !== 1) {
      stats.notStored++;
      continue;
    }
    try {
      emit(row.p, "silver", digest, row.s, new TextDecoder().decode(gunzipSync(rows[0]!.content)));
      stats.silver++;
    } catch {
      stats.failed++;
    }
  }
}

flush();
if (stats.failed > 0.2 * (written + stats.failed)) {
  throw new Error(
    `${stats.failed} of ${written + stats.failed} pages failed to emit: refusing to write a mostly-empty shard`,
  );
}
writeFileSync(
  `${out}/features.json`,
  JSON.stringify({ order: SKELETON_FEATURES, stats, rows: written }, null, 1) + "\n",
);
console.log(JSON.stringify({ rows: written, ...stats }));

#!/usr/bin/env bun
/**
 * Parity of runtime/platform.ts with corpus/cc_corpus.py `platform_of` (#2307).
 *
 * The Python tags (`cc_corpus.py tag-pages`, {p, platform}) are what the
 * platform guard was computed on, so the runtime must name the same platform
 * for the same bytes. Reads each VAL page of a dataset from the stores (read
 * only), runs platformOf, compares. Exit code 1 on any disagreement.
 *
 *   bun run experiments/platform-parity.ts --dataset <silver-v4> --tags tags.jsonl \
 *     --store a.db:b.db [--limit N] --out report.json
 */
import { Database } from "bun:sqlite";
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

import { platformBucket, platformOf } from "../runtime/platform.ts";

function argument(name: string, fallback?: string): string {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? fallback : Bun.argv[index + 1];
  if (value === undefined || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

const tags = new Map<string, string | null>();
for (const line of readFileSync(argument("--tags"), "utf8").split("\n")) {
  if (!line) continue;
  const row = JSON.parse(line) as { p: string; platform: string | null };
  tags.set(row.p, row.platform);
}
const index = gunzipSync(readFileSync(`${argument("--dataset")}/page-index.jsonl.gz`))
  .toString("utf8")
  .split("\n")
  .filter(Boolean)
  .map((l) => JSON.parse(l) as { p: string; s: string })
  .filter((row) => row.s === "val");
const limit = Number(argument("--limit", "0"));
const selects = argument("--store")
  .split(":")
  .filter(Boolean)
  .map((p) =>
    new Database(p, { readonly: true }).query<{ content: Uint8Array }, [string, string]>(
      "SELECT content FROM content WHERE hash >= ? AND hash < ? LIMIT 2",
    ),
  );
const decoder = new TextDecoder();

let compared = 0;
let platformAgree = 0;
let bucketAgree = 0;
const mismatches: Array<{ p: string; python: string | null; ts: string }> = [];
for (const row of limit > 0 ? index.slice(0, limit) : index) {
  const python = tags.get(row.p);
  if (python === undefined || python === null) continue;
  let html: string | null = null;
  for (const select of selects) {
    const hits = select.all(row.p, `${row.p}g`);
    if (hits.length === 1) {
      html = decoder.decode(gunzipSync(hits[0]!.content));
      break;
    }
  }
  if (html === null) continue;
  const ts = platformOf(html);
  compared++;
  if (ts === python) platformAgree++;
  else if (mismatches.length < 50) mismatches.push({ p: row.p, python, ts });
  if (platformBucket(ts) === platformBucket(python)) bucketAgree++;
}
const report = { compared, platformAgree, bucketAgree, mismatches };
writeFileSync(argument("--out"), `${JSON.stringify(report, null, 2)}\n`);
console.log(
  JSON.stringify({ compared, platformAgree, bucketAgree, mismatches: mismatches.length }),
);
if (platformAgree !== compared) process.exitCode = 1;

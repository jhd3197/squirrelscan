#!/usr/bin/env bun
/**
 * Count the schema.org types each dataset page declares, for page-type v8's
 * free labels.
 *
 * The featurizer never reads schema markup (featurize.ts, rule 1), so a label
 * derived here can train the page-type head without the head ever seeing the
 * evidence the label came from: the "schema hidden" condition holds for every
 * page by construction. This file only COUNTS; which type is a page's main
 * entity, and what v8 class that means, is decided by `schema_labels_v8.py`,
 * which needs per-domain statistics (a type on most of a domain's pages is
 * site-wide chrome, not the page's subject).
 *
 * Reads the stores read-only. Writes one row per page, with no url and no text:
 *   {p, d, s, root, ld: {type: topLevelCount}, ldNested: {...}, md: {type: n}, ldErrors}
 *
 *   DOM_CLASSIFIER_DATA_ROOT=... bun run corpus/schema-types.ts \
 *     --dataset <root>/silver-v3 --store <a/content-store.db>:<b/content-store.db> \
 *     --out <root>/pagetype-v8/schema-types.jsonl
 */
import { Database } from "bun:sqlite";
import { createHash } from "node:crypto";
import { createReadStream, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";
import { createGunzip, gunzipSync } from "node:zlib";

function argument(name: string): string {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? undefined : Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}
const dataset = resolve(argument("--dataset"));
const outPath = resolve(argument("--out"));
const stores = argument("--store")
  .split(":")
  .filter(Boolean)
  .map((path) => new Database(resolve(path), { readonly: true }));
const selects = stores.map((db) =>
  db.query<{ content: Uint8Array }, [string, string]>(
    "SELECT content FROM content WHERE hash >= ? AND hash < ? LIMIT 2",
  ),
);

const LD_BLOCK =
  /<script\b[^>]*type\s*=\s*["']?application\/ld\+json["']?[^>]*>([\s\S]*?)<\/script>/gi;
const MICRODATA = /\bitemtype\s*=\s*["']?https?:\/\/(?:www\.)?schema\.org\/([A-Za-z]+)/gi;

type Counts = Record<string, number>;
const bump = (counts: Counts, type: string) => {
  counts[type] = (counts[type] ?? 0) + 1;
};
function typesOf(node: Record<string, unknown>): string[] {
  const raw = node["@type"];
  const list = Array.isArray(raw) ? raw : raw === undefined ? [] : [raw];
  return list
    .filter((t): t is string => typeof t === "string")
    .map((t) => t.replace(/^https?:\/\/(www\.)?schema\.org\//, ""));
}
/** Top-level entities (a block's root, or members of its @graph) are counted
 *  apart from nested ones: an Offer inside a Product is not the page's subject. */
function walk(value: unknown, top: Counts, nested: Counts, depth: number): void {
  if (Array.isArray(value)) {
    for (const item of value) walk(item, top, nested, depth);
    return;
  }
  if (!value || typeof value !== "object") return;
  const node = value as Record<string, unknown>;
  for (const type of typesOf(node)) bump(depth === 0 ? top : nested, type);
  for (const [key, child] of Object.entries(node)) {
    if (key === "@graph") walk(child, top, nested, depth);
    else if (key !== "@type" && key !== "@context") walk(child, top, nested, depth + 1);
  }
}

function html(pageId: string): string | null {
  for (const select of selects) {
    const rows = select.all(pageId, `${pageId}g`);
    if (rows.length !== 1) continue;
    try {
      return new TextDecoder().decode(gunzipSync(rows[0]!.content));
    } catch {
      return null;
    }
  }
  return null;
}

mkdirSync(dirname(outPath), { recursive: true, mode: 0o700 });
const lines: string[] = [];
const stats = { pages: 0, missingHtml: 0, withLd: 0, withMicrodata: 0, ldErrors: 0 };
const input = createInterface({
  input: createReadStream(`${dataset}/page-index.jsonl.gz`).pipe(createGunzip()),
});
for await (const line of input) {
  if (!line) continue;
  const row = JSON.parse(line) as { p: string; u: string; domain: string; s: string };
  stats.pages++;
  const text = html(row.p);
  if (text === null) {
    stats.missingHtml++;
    continue;
  }
  const ld: Counts = {};
  const ldNested: Counts = {};
  let ldErrors = 0;
  for (const match of text.matchAll(LD_BLOCK)) {
    try {
      walk(JSON.parse(match[1]!.trim()), ld, ldNested, 0);
    } catch {
      ldErrors++;
    }
  }
  const md: Counts = {};
  for (const match of text.matchAll(MICRODATA)) bump(md, match[1]!);
  let root = false;
  try {
    root = new URL(row.u).pathname.replace(/\/+$/, "") === "";
  } catch {
    /* an unparseable url is not a root */
  }
  if (Object.keys(ld).length) stats.withLd++;
  if (Object.keys(md).length) stats.withMicrodata++;
  stats.ldErrors += ldErrors;
  lines.push(
    JSON.stringify({
      p: row.p,
      // The domain is carried hashed, like every other dataset file.
      d: createHash("sha256").update(row.domain).digest("hex").slice(0, 16),
      s: row.s,
      root,
      ld,
      ldNested,
      md,
      ldErrors,
    }),
  );
  if (stats.pages % 10000 === 0) console.log(JSON.stringify({ stage: "progress", ...stats }));
}
for (const db of stores) db.close();
writeFileSync(outPath, `${lines.join("\n")}\n`, { mode: 0o600 });
console.log(JSON.stringify({ stage: "complete", ...stats }));

#!/usr/bin/env bun
/**
 * Every registrable domain the DOM classifier corpus has already seen, for the
 * Common Crawl diversity extension (#2307). A new domain must be new: a site
 * already in any training, eval or cloud set would leak a template across the
 * old/new comparison, or put a customer's site into a public-weights corpus.
 *
 * Inputs are private files given on the command line; nothing here names a
 * host. Each input is read by shape:
 *   - *.jsonl / *.jsonl.gz: every string field that parses as a URL or a bare
 *     domain (`u`, `url`, `final_url`, `domain`, `host`, ...);
 *   - split-manifest.json: its domain HASHES (the manifests never stored the
 *     domain string), kept as a second exclusion list;
 *   - anything else (tsv, txt): every http(s) URL and every bare hostname token.
 *
 * Output (private): domains.txt (registrable domains, tldts with private
 * suffixes, the same function build-dataset.ts splits on) and
 * domain-hashes.txt (sha256(domain)[0:16], the dataset `d` field).
 *
 *   bun run corpus/cc-exclusions.ts --out <private>/exclusions <input>...
 */
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";

import { parse as parseDomain } from "tldts";

const outIndex = Bun.argv.indexOf("--out");
if (outIndex < 0 || !Bun.argv[outIndex + 1]) throw new Error("--out <dir> is required");
const outDir = resolve(Bun.argv[outIndex + 1]!);
const inputs = Bun.argv
  .slice(2)
  .filter((_, i, all) => all[i - 1] !== "--out" && all[i] !== "--out");
if (inputs.length === 0) throw new Error("no inputs");

export function registrable(value: string): string {
  const trimmed = value.trim().toLowerCase();
  if (!trimmed || trimmed.length > 300) return "";
  const parsed = parseDomain(trimmed, { allowPrivateDomains: true });
  if (!parsed.domain || parsed.isIp) return "";
  return parsed.domain;
}

const URL_RE = /https?:\/\/[^\s"'<>\\]+/gi;
const HOST_RE = /\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b/gi;

const domains = new Set<string>();
const hashes = new Set<string>();
const perInput: Record<string, number> = {};

function addString(value: string): void {
  if (/^https?:\/\//i.test(value)) {
    const domain = registrable(value);
    if (domain) domains.add(domain);
    return;
  }
  if (/^[a-z0-9.-]+\.[a-z]{2,24}$/i.test(value)) {
    const domain = registrable(value);
    if (domain) domains.add(domain);
  }
}

function walk(value: unknown): void {
  if (typeof value === "string") addString(value);
  else if (Array.isArray(value)) for (const item of value) walk(item);
  else if (value && typeof value === "object") for (const item of Object.values(value)) walk(item);
}

for (const input of inputs) {
  const before = domains.size + hashes.size;
  const raw = readFileSync(input);
  const text = (input.endsWith(".gz") ? gunzipSync(raw) : raw).toString("utf8");
  if (input.endsWith("split-manifest.json")) {
    const manifest = JSON.parse(text) as { domains?: Record<string, string> };
    if (!manifest.domains) throw new Error(`${input}: not a domain-hash split manifest`);
    for (const hash of Object.keys(manifest.domains)) hashes.add(hash);
  } else if (/\.jsonl(\.gz)?$/.test(input)) {
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      try {
        walk(JSON.parse(line));
      } catch {
        // A truncated final line in a live ledger is not an input error.
      }
    }
  } else {
    for (const match of text.matchAll(URL_RE)) addString(match[0]);
    for (const match of text.matchAll(HOST_RE)) addString(match[0]);
  }
  perInput[input] = domains.size + hashes.size - before;
}

for (const domain of domains)
  hashes.add(createHash("sha256").update(domain).digest("hex").slice(0, 16));
mkdirSync(outDir, { recursive: true, mode: 0o700 });
writeFileSync(`${outDir}/domains.txt`, `${[...domains].sort().join("\n")}\n`, { mode: 0o600 });
writeFileSync(`${outDir}/domain-hashes.txt`, `${[...hashes].sort().join("\n")}\n`, { mode: 0o600 });
console.log(JSON.stringify({ domains: domains.size, hashes: hashes.size, inputs: inputs.length }));

#!/usr/bin/env bun
/**
 * Skeleton cost and size on real corpus pages (#2307). Read-only on the corpus.
 *
 *   bun run experiments/bench-skeleton.ts --corpus /Volumes/BACKUP/page-corpus \
 *       [--n 300] [--seed 7] [--samples <dir>]
 *
 * Reports parse time, skeleton time (headings and tags-only), skeleton size in chars and
 * approximate tokens (chars / 4), node count. `--samples` writes a few skeletons per
 * Jev class to a PRIVATE directory for eyeballing (page words appear there: never commit).
 * Sealed domains and bucket >= 90 pages are skipped.
 */
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { buildSkeleton, renderCompact, renderXml } from "../runtime/skeleton.ts";
import { decode } from "./link-position.ts";

function arg(name: string, fallback?: string): string | undefined {
  const i = Bun.argv.indexOf(name);
  return i < 0 ? fallback : Bun.argv[i + 1];
}
const corpus = arg("--corpus")!;
const n = Number(arg("--n", "300"));
let seed = Number(arg("--seed", "7"));
const samplesDir = arg("--samples");
const rand = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;

const sealed = new Set(
  readFileSync(`${corpus}/splits/sealed-domains.jsonl`, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l).d as string),
);
const pages = readFileSync(`${corpus}/pages/manifest.jsonl`, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((l) => JSON.parse(l))
  .filter((p) => p.raw?.status === "stored" && p.bucket < 90 && !sealed.has(p.domain_hash));
const picked = pages
  .map((p) => ({ p, r: rand() }))
  .sort((a, b) => a.r - b.r)
  .slice(0, n)
  .map((x) => x.p);

const pct = (xs: number[], q: number) =>
  [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(q * xs.length))]!;
const stat = (xs: number[]) =>
  `p50 ${pct(xs, 0.5).toFixed(1)}  p95 ${pct(xs, 0.95).toFixed(1)}  max ${Math.max(...xs).toFixed(1)}`;
const parseMs: number[] = [],
  skelMs: number[] = [],
  noneMs: number[] = [],
  chars: number[] = [],
  compactChars: number[] = [],
  noneChars: number[] = [],
  nodes: number[] = [],
  elements: number[] = [];
let failed = 0;
for (const page of picked) {
  try {
    const bytes = gunzipSync(readFileSync(`${corpus}/pages/blobs/${page.raw.sha256}.html.gz`));
    const html = decode(bytes.length > 8_000_000 ? bytes.subarray(0, 8_000_000) : bytes);
    let t = performance.now();
    const { document } = parseHTML(html);
    parseMs.push(performance.now() - t);
    elements.push(document.querySelectorAll("*").length);
    t = performance.now();
    const tree = buildSkeleton(document);
    const xml = renderXml(tree);
    skelMs.push(performance.now() - t);
    t = performance.now();
    const bare = renderXml(buildSkeleton(document, { text: "none" }));
    noneMs.push(performance.now() - t);
    chars.push(xml.length);
    compactChars.push(renderCompact(tree).length);
    noneChars.push(bare.length);
    nodes.push(xml.split("\n").length);
  } catch {
    failed += 1;
  }
}
console.log(`pages ${picked.length} (failed ${failed}); elements ${stat(elements)}`);
console.log(`parse ms        ${stat(parseMs)}`);
console.log(`skeleton ms     ${stat(skelMs)}   (build + xml, headings)`);
console.log(`tags-only ms    ${stat(noneMs)}`);
console.log(
  `xml chars       ${stat(chars)}  ~tokens p50 ${Math.round(pct(chars, 0.5) / 4)} p95 ${Math.round(pct(chars, 0.95) / 4)}`,
);
console.log(
  `compact chars   ${stat(compactChars)}  ~tokens p50 ${Math.round(pct(compactChars, 0.5) / 4)}`,
);
console.log(`tags-only chars ${stat(noneChars)}`);
console.log(`xml lines       ${stat(nodes)}`);

if (samplesDir) {
  mkdirSync(samplesDir, { recursive: true });
  for (const [i, page] of picked.slice(0, 12).entries()) {
    const html = decode(
      gunzipSync(readFileSync(`${corpus}/pages/blobs/${page.raw.sha256}.html.gz`)),
    );
    writeFileSync(
      `${samplesDir}/sample-${i}.xml`,
      renderXml(buildSkeleton(parseHTML(html).document)) + "\n",
    );
  }
}

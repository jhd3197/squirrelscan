#!/usr/bin/env bun
/**
 * Text packets for the teacher check of "declares no schema" candidates (#2307).
 *
 *   bun run experiments/judge-packets.ts --sample sample.jsonl --skeletons <dir> --corpus <page-corpus> --out packets.jsonl
 *
 * One packet per sampled page: language, title, headings, the first ~900 characters of body
 * text (site chrome removed), and a few counts. The teacher judges the PAGE, not the model's
 * skeleton, so it is not marked by the thing it checks. No URL, no schema, no model score and
 * no sample kind reach the packet. Words appear in the output: PRIVATE, never commit.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { decode } from "./link-position.ts";

function arg(name: string): string {
  const i = Bun.argv.indexOf(name);
  if (i < 0 || !Bun.argv[i + 1]) throw new Error(`${name} is required`);
  return Bun.argv[i + 1]!;
}
const corpus = arg("--corpus");
const sample = readFileSync(arg("--sample"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as { p: string });
const wanted = new Set(sample.map((s) => s.p));
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

const flat = (text: string | null | undefined, limit: number) => (text ?? "").replace(/\s+/g, " ").trim().slice(0, limit);
const packets: string[] = [];
for (const { p } of sample) {
  const full = sha.get(p);
  if (!full) continue;
  try {
    const html = decode(gunzipSync(readFileSync(`${corpus}/pages/blobs/${full}.html.gz`)).subarray(0, 4_000_000));
    const { document } = parseHTML(html);
    for (const el of Array.from(document.querySelectorAll("script,style,noscript,template,svg,nav,header,footer,aside,form"))) el.remove();
    const main = document.querySelector("main, article, [role=main]") ?? document.body;
    const headings = Array.from(document.querySelectorAll("h1,h2,h3"))
      .slice(0, 10)
      .map((h) => `${h.localName}: ${flat(h.textContent, 120)}`);
    const text = flat(main?.textContent, 900);
    const body = flat(document.body?.textContent, 100000);
    packets.push(
      JSON.stringify({
        itemId: p,
        packet: [
          `language: ${document.documentElement?.getAttribute("lang") ?? "unknown"}`,
          `title: ${flat(document.querySelector("title")?.textContent, 160)}`,
          `headings:\n${headings.join("\n")}`,
          `body text length: ${body.length} characters; paragraphs: ${document.querySelectorAll("p").length}; links: ${document.querySelectorAll("a").length}; images: ${document.querySelectorAll("img").length}; forms: ${document.querySelectorAll("form").length}`,
          `start of main text:\n${text}`,
        ].join("\n"),
      }),
    );
  } catch {
    // unreadable page: no packet
  }
}
writeFileSync(arg("--out"), packets.join("\n") + "\n");
console.log(JSON.stringify({ sample: sample.length, packets: packets.length }));

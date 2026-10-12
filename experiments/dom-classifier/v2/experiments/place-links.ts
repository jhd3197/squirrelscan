#!/usr/bin/env bun
/**
 * Treetop as a CLI step: which part of a page is each link in? (#2307)
 *
 *   bun run experiments/place-links.ts [--in rows.jsonl] [--out placed.jsonl]
 *       [--artifact classifier-v2.json] [--budget-ms 500]
 *
 * JSON in, JSON out, one line each, batchable, no network. stdin/stdout by default.
 *
 * Input line:
 *   {"id": "...", "url": "https://page.example/", "html": "<html>..."            // or
 *    "html_path": "/path/page.html" | "/path/page.html.gz",                       // not both
 *    "links": [{"href": "...", "anchor_text": "..."}]}                           // optional
 * With no `links`, every outbound link on the page is placed.
 *
 * Output line (input order, one per input line):
 *   {"id", "ok": true, "page_type": {label, confidence, source}, "degraded": "",
 *    "comment_sections": [{label, evidence}],
 *    "links": [{i, href, section, region, rel, conf, evidence, source, located, in_walk}]}
 *   or {"id", "ok": false, "error": "..."} for a line that failed; the rest still run.
 * See link-sections.ts for the meaning of each link field. `section` is
 * `comments` / `reviews` when the markup says the link sits in a comment
 * thread, whatever the model thinks.
 *
 * Without `--artifact` the deterministic rules answer (the comments rule,
 * landmark tags): `degraded` says "no model artifact". Page content is hostile:
 * it is parsed, never executed, and `html_path` is only read, never followed.
 */
import { createReadStream, readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { gunzipSync } from "node:zlib";

import type { Model } from "../runtime/model-format.ts";
import { tryLoadModel } from "../runtime/predict.ts";
import { placePage } from "./link-sections.ts";
import { decode, type ManifestLink } from "./link-position.ts";

export interface PlaceRequest {
  id: string;
  url: string;
  html?: string;
  html_path?: string;
  links?: ManifestLink[] | null;
}

const MAX_HTML_BYTES = 20 * 1024 * 1024;

/** One request to one output object; never throws. */
export function placeRequest(
  line: string,
  model: Model | null,
  budgetMs: number,
): Record<string, unknown> {
  let id: unknown = null;
  try {
    const request = JSON.parse(line) as PlaceRequest;
    id = request.id ?? null;
    if (typeof request.url !== "string" || !request.url) throw new Error("url is required");
    if ((request.html === undefined) === (request.html_path === undefined))
      throw new Error("give html or html_path, not both and not neither");
    let html = request.html;
    if (html === undefined) {
      const raw = readFileSync(request.html_path!);
      if (raw.length > MAX_HTML_BYTES) throw new Error("html_path larger than 20 MB");
      const bytes = request.html_path!.endsWith(".gz") ? gunzipSync(raw) : raw;
      if (bytes.length > MAX_HTML_BYTES) throw new Error("html larger than 20 MB");
      html = decode(bytes);
    } else if (html.length > MAX_HTML_BYTES) {
      throw new Error("html larger than 20 MB");
    }
    const placed = placePage(html, request.url, request.links, { model, budgetMs });
    return {
      id,
      ok: true,
      page_type: placed.pageType,
      degraded: placed.degraded,
      comment_sections: placed.commentSections,
      links: placed.links.map(({ located, inWalk, ...row }) => ({
        ...row,
        located,
        in_walk: inWalk,
      })),
    };
  } catch (error) {
    return {
      id,
      ok: false,
      error: String(error instanceof Error ? error.message : error).slice(0, 300),
    };
  }
}

function argument(name: string, fallback?: string): string | undefined {
  const index = Bun.argv.indexOf(name);
  if (index < 0) return fallback;
  const value = Bun.argv[index + 1];
  if (value === undefined || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

async function main(): Promise<void> {
  const artifact = argument("--artifact");
  const budgetMs = Number(argument("--budget-ms", "500"));
  let model: Model | null = null;
  if (artifact) {
    const loaded = tryLoadModel(JSON.parse(readFileSync(artifact, "utf8")));
    if (!loaded.model) throw new Error(`artifact unusable: ${loaded.reason}`);
    model = loaded.model;
  }
  const input = argument("--in");
  const out = argument("--out");
  const lines = createInterface({
    input: input ? createReadStream(input) : process.stdin,
    crlfDelay: Infinity,
  });
  const writer = out ? Bun.file(out).writer() : null;
  for await (const line of lines) {
    if (!line.trim()) continue;
    const text = `${JSON.stringify(placeRequest(line, model, budgetMs))}\n`;
    if (writer) writer.write(text);
    else process.stdout.write(text);
  }
  await writer?.end();
}

if (import.meta.main) await main();

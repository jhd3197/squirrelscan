#!/usr/bin/env bun
/**
 * Cross-page context for page type v8.1 (#2307): the page featurizer sees only
 * the page's own DOM and URL, but "About us" in a site's footer pointing at a
 * page says more about that page than anything on it.
 *
 * For every dataset page this writes (private output, it carries urls and
 * anchor text):
 *   {p, d, url, skeleton, links: [{t, a, r}]}
 * where `t` is a normalised same-site link target, `a` its anchor text (<=60
 * chars) and `r` the region it sits in (nav | header | footer | aside | body),
 * and `skeleton` hashes the page's structural template (tag + first class of
 * the first elements at depth <= 3), so pages rendered by one template share it.
 * `site_context_features.py` turns this into per-page features.
 */
import { Database } from "bun:sqlite";
import { createHash } from "node:crypto";
import { createReadStream, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";
import { createGunzip, gunzipSync } from "node:zlib";

import { parse as parseDomain } from "tldts";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";

function argument(name: string, fallback?: string): string {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? fallback : Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}
const dataset = resolve(argument("--dataset"));
const outPath = resolve(argument("--out"));
const limit = Number(argument("--limit", "0"));
const stores = argument("--store")
  .split(":")
  .filter(Boolean)
  .map((p) => new Database(resolve(p), { readonly: true }));
const selects = stores.map((db) =>
  db.query<{ content: Uint8Array }, [string, string]>(
    "SELECT content FROM content WHERE hash >= ? AND hash < ? LIMIT 2",
  ),
);

/** Same-site target in one canonical form: host (no www) + path, no query/hash,
 *  no trailing slash, lower-cased. */
export function normaliseUrl(href: string, base: string): string | null {
  try {
    const url = new URL(href, base);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    const path = url.pathname.replace(/\/+$/, "") || "/";
    return `${url.hostname.replace(/^www\./, "")}${path}`.toLowerCase();
  } catch {
    return null;
  }
}

type El = Element & { parentElement: El | null };
function regionOf(el: El): string {
  for (let cur: El | null = el.parentElement; cur; cur = cur.parentElement) {
    const tag = cur.localName;
    const role = (cur.getAttribute("role") ?? "").toLowerCase();
    if (tag === "nav" || role === "navigation") return "nav";
    if (tag === "footer" || role === "contentinfo") return "footer";
    if (tag === "header" || role === "banner") return "header";
    if (tag === "aside" || role === "complementary") return "aside";
  }
  return "body";
}
function skeleton(body: El | null): string {
  const parts: string[] = [];
  const walk = (el: El, depth: number) => {
    if (parts.length >= 40 || depth > 3) return;
    const first = (el.getAttribute("class") ?? "").trim().split(/\s+/)[0] ?? "";
    parts.push(`${depth}${el.localName}.${first.replace(/\d+/g, "")}`);
    for (const child of el.children as unknown as El[]) walk(child, depth + 1);
  };
  if (body) walk(body, 0);
  return createHash("sha256").update(parts.join("|")).digest("hex").slice(0, 12);
}

mkdirSync(dirname(outPath), { recursive: true, mode: 0o700 });
// Streamed: 74k pages x hundreds of links does not belong in one array.
const out = Bun.file(outPath).writer();
let pages = 0;
const input = createInterface({
  input: createReadStream(`${dataset}/page-index.jsonl.gz`).pipe(createGunzip()),
});
for await (const line of input) {
  if (!line) continue;
  if (limit > 0 && pages >= limit) break;
  const row = JSON.parse(line) as { p: string; u: string; domain: string; s: string };
  let html: string | null = null;
  for (const select of selects) {
    const found = select.all(row.p, `${row.p}g`);
    if (found.length === 1) {
      html = new TextDecoder().decode(gunzipSync(found[0]!.content));
      break;
    }
  }
  if (html === null) continue;
  pages++;
  const { document } = parseHTML(html);
  const site = parseDomain(row.u, { allowPrivateDomains: true });
  const domain = site.domain ?? site.hostname ?? "";
  const links: Array<{ t: string; a: string; r: string }> = [];
  const seen = new Set<string>();
  for (const a of document.querySelectorAll("a[href]") as unknown as El[]) {
    const target = normaliseUrl(a.getAttribute("href") ?? "", row.u);
    if (!target) continue;
    const host = target.split("/")[0]!;
    const other = parseDomain(`https://${host}/`, { allowPrivateDomains: true });
    if ((other.domain ?? other.hostname) !== domain) continue;
    const anchor = (a.textContent ?? "").replace(/\s+/g, " ").trim().toLowerCase().slice(0, 60);
    const region = regionOf(a);
    const key = `${target}|${anchor}|${region}`;
    if (seen.has(key)) continue;
    seen.add(key);
    links.push({ t: target, a: anchor, r: region });
  }
  out.write(
    `${JSON.stringify({
      p: row.p,
      d: createHash("sha256").update(row.domain).digest("hex").slice(0, 16),
      url: normaliseUrl(row.u, row.u),
      skeleton: skeleton((document as { body?: El | null }).body ?? null),
      links: links.slice(0, 600),
    })}\n`,
  );
  if (pages % 10000 === 0) console.log(JSON.stringify({ stage: "progress", pages }));
}
for (const db of stores) db.close();
await out.end();
console.log(JSON.stringify({ stage: "complete", pages }));

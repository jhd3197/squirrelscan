#!/usr/bin/env bun
/**
 * Free node labels for the section-purpose head (labels/section-purpose-v1.md).
 *
 * Three independent sources, each recorded separately on the node so their
 * agreement can be measured rather than assumed:
 *   md   microdata: an `itemscope itemtype=...` element (or its nearest
 *        candidate ancestor, when that is not much bigger) - node-aligned;
 *   ld   JSON-LD entities text-matched to the TIGHTEST candidate containing
 *        them: FAQ question names, breadcrumb names, review bodies, street
 *        address/phone, offer price + a buy control, author name;
 *   sem  ARIA/semantic markup: aria-label~breadcrumb, <address>, <details>
 *        groups, video players, rel=author/byline. `sem` is ALSO the
 *        deterministic baseline the head must beat.
 * Candidates are exactly the featurizer's (featurizeDocument), addressed by
 * nodeId, so a label lands on a row the head is trained on.
 *
 * Reads stores read-only. Output (private), one row per labelled node:
 *   {p, s, n, md?, ld?, sem?}      plus one {p, s, candidates} row per page
 * and a spot-check file with tag/class/short text for sampled nodes.
 */
import { Database } from "bun:sqlite";
import { createReadStream, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";
import { createGunzip, gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { featurizeDocument, SKIP_SUBTREE } from "../featurize.ts";

function argument(name: string, fallback?: string): string {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? fallback : Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}
const dataset = resolve(argument("--dataset"));
const typesPath = resolve(argument("--schema-types"));
const outPath = resolve(argument("--out"));
const extraPages = Number(argument("--extra-pages", "10000"));
const limit = Number(argument("--limit", "0"));
const stores = argument("--store")
  .split(":")
  .filter(Boolean)
  .map((path) => new Database(resolve(path), { readonly: true }));
const selects = stores.map((db) =>
  db.query<{ content: Uint8Array }, [string, string]>(
    "SELECT content FROM content WHERE hash >= ? AND hash < ? LIMIT 2",
  ),
);

type El = Element & { parentElement: El | null };

/** Elements in the featurizer's preorder: index i is nodeId i (the same walk as
 *  model-suggestions/featurizer-order.ts, which v2's tsconfig cannot import). */
function featurizerOrder(document: { body?: unknown }): El[] {
  const order: El[] = [];
  const body = document.body as El | null | undefined;
  const stack: El[] = body ? [body] : [];
  while (stack.length > 0) {
    const el = stack.pop()!;
    if (SKIP_SUBTREE.has(el.localName)) continue;
    order.push(el);
    const kids = [...(el.children as unknown as El[])].filter(
      (c) => !SKIP_SUBTREE.has(c.localName),
    );
    for (let i = kids.length - 1; i >= 0; i--) stack.push(kids[i]!);
  }
  return order;
}
const norm = (text: string) => text.replace(/\s+/g, " ").trim().toLowerCase();

// --- which pages -----------------------------------------------------------
const RELEVANT = new Set([
  "FAQPage",
  "Question",
  "Review",
  "AggregateRating",
  "BreadcrumbList",
  "LocalBusiness",
  "PostalAddress",
  "Offer",
  "AggregateOffer",
  "Person",
  "VideoObject",
]);
const chosen = new Set<string>();
const others: string[] = [];
for (const line of readFileSync(typesPath, "utf8").split("\n")) {
  if (!line) continue;
  const row = JSON.parse(line) as { p: string; ld: object; ldNested: object; md: object };
  const types = [...Object.keys(row.ld), ...Object.keys(row.ldNested), ...Object.keys(row.md)];
  if (types.some((t) => RELEVANT.has(t))) chosen.add(row.p);
  else others.push(row.p);
}
// A deterministic sample of pages with none of that schema: negatives, and the
// only place the semantic source is seen without schema next to it.
others.sort((a, b) => (BigInt(Bun.hash(a)) < BigInt(Bun.hash(b)) ? -1 : 1));
for (const p of others.slice(0, extraPages)) chosen.add(p);

// --- JSON-LD strings per class --------------------------------------------
type Targets = {
  faq: string[];
  crumbs: string[];
  reviews: string[];
  address: string[];
  prices: number[];
  authors: string[];
  video: boolean;
};
const LD_BLOCK =
  /<script\b[^>]*type\s*=\s*["']?application\/ld\+json["']?[^>]*>([\s\S]*?)<\/script>/gi;
function typesOf(node: Record<string, unknown>): string[] {
  const raw = node["@type"];
  return (Array.isArray(raw) ? raw : [raw]).filter((t): t is string => typeof t === "string");
}
function nameOf(value: unknown): string {
  if (typeof value === "string") return value;
  if (value && typeof value === "object") {
    const name = (value as Record<string, unknown>).name;
    if (typeof name === "string") return name;
  }
  return "";
}
function targetsOf(html: string): Targets {
  const t: Targets = {
    faq: [],
    crumbs: [],
    reviews: [],
    address: [],
    prices: [],
    authors: [],
    video: false,
  };
  const visit = (value: unknown, parentType: string): void => {
    if (Array.isArray(value)) {
      for (const v of value) visit(v, parentType);
      return;
    }
    if (!value || typeof value !== "object") return;
    const node = value as Record<string, unknown>;
    const types = typesOf(node);
    if (types.includes("Question")) t.faq.push(nameOf(node));
    if (types.includes("BreadcrumbList") && Array.isArray(node.itemListElement))
      for (const item of node.itemListElement as Record<string, unknown>[])
        t.crumbs.push(nameOf(item) || nameOf(item?.item));
    if (types.includes("Review")) {
      const body = node.reviewBody ?? node.description;
      if (typeof body === "string") t.reviews.push(body.slice(0, 80));
    }
    if (types.includes("PostalAddress") && typeof node.streetAddress === "string")
      t.address.push(node.streetAddress);
    if ((types.includes("Offer") || types.includes("AggregateOffer")) && node.price !== undefined) {
      const price = Number(String(node.price).replace(/,/g, ""));
      if (Number.isFinite(price) && price > 0) t.prices.push(price);
    }
    if (types.includes("VideoObject")) t.video = true;
    for (const [key, child] of Object.entries(node)) {
      if (key === "author") {
        for (const a of Array.isArray(child) ? child : [child]) {
          const name = nameOf(a);
          if (name) t.authors.push(name);
        }
      }
      if (key !== "@type" && key !== "@context") visit(child, types[0] ?? parentType);
    }
  };
  for (const match of html.matchAll(LD_BLOCK)) {
    try {
      visit(JSON.parse(match[1]!.trim()), "");
    } catch {
      /* malformed block: no targets from it */
    }
  }
  const clean = (list: string[], min: number) => [
    ...new Set(list.map(norm).filter((s) => s.length >= min)),
  ];
  t.faq = clean(t.faq, 8);
  t.crumbs = clean(t.crumbs, 2);
  t.reviews = clean(t.reviews, 20);
  t.address = clean(t.address, 6);
  t.authors = clean(t.authors, 4);
  return t;
}

// --- per page ------------------------------------------------------------------
const MD_CLASS: Record<string, string> = {
  FAQPage: "faq_block",
  BreadcrumbList: "breadcrumbs",
  Review: "reviews",
  AggregateRating: "reviews",
  PostalAddress: "business_info",
  LocalBusiness: "business_info",
  Offer: "product_offer",
  AggregateOffer: "product_offer",
  VideoObject: "video_embed",
};
const MAX_TEXT: Record<string, number> = {
  breadcrumbs: 400,
  business_info: 500,
  product_offer: 800,
  author_byline: 200,
  video_embed: 400,
};
const VIDEO_SRC = /youtube\.com|youtu\.be|vimeo\.com|wistia|loom\.com|dailymotion/i;

type Row = { p: string; s: string; n: number; md?: string; ld?: string; sem?: string };

function labelPage(
  pageId: string,
  split: string,
  url: string,
  html: string,
  rows: Row[],
  spot: object[],
) {
  const { document } = parseHTML(html);
  const featurized = featurizeDocument(document as never, url);
  const order = featurizerOrder(document as never);
  const candidates = new Set(featurized.nodes.map((n) => n.nodeId));
  const idOf = new Map<El, number>();
  order.forEach((el, i) => idOf.set(el, i));
  const textCache = new Map<number, string>();
  const text = (n: number) => {
    let t = textCache.get(n);
    if (t === undefined) textCache.set(n, (t = norm(order[n]!.textContent ?? "")));
    return t;
  };
  /** Nearest candidate at or above `el` that is not much bigger than it. */
  const anchor = (el: El, slack = 200): number | null => {
    const own = norm(el.textContent ?? "").length;
    for (let cur: El | null = el; cur; cur = cur.parentElement) {
      const id = idOf.get(cur);
      if (id !== undefined && candidates.has(id))
        return text(id).length <= Math.max(own * 1.5, own + slack) ? id : null;
    }
    return null;
  };
  const labels = new Map<number, Row>();
  const put = (n: number | null, source: "md" | "ld" | "sem", cls: string) => {
    if (n === null) return;
    const cap = MAX_TEXT[cls];
    if (cap && text(n).length > cap) return;
    let row = labels.get(n);
    if (!row) labels.set(n, (row = { p: pageId, s: split, n }));
    row[source] ??= cls;
  };

  // md: microdata elements.
  for (const el of document.querySelectorAll("[itemtype]") as unknown as El[]) {
    const type = (el.getAttribute("itemtype") ?? "").replace(
      /^https?:\/\/(www\.)?schema\.org\//,
      "",
    );
    let cls = MD_CLASS[type];
    if (type === "Person" && /author/i.test(el.getAttribute("itemprop") ?? ""))
      cls = "author_byline";
    if (cls) put(anchor(el), "md", cls);
  }

  // ld: text-matched JSON-LD targets. Tightest candidate holding the most.
  const t = targetsOf(html);
  const ids = [...candidates];
  const tightest = (strings: string[], minHits: number): number | null => {
    if (strings.length < minHits) return null;
    let best: number | null = null;
    let bestHits = 0;
    for (const n of ids) {
      const body = text(n);
      let hits = 0;
      for (const s of strings) if (body.includes(s)) hits++;
      if (hits < minHits) continue;
      if (hits > bestHits || (hits === bestHits && body.length < text(best!).length)) {
        best = n;
        bestHits = hits;
      }
    }
    return best;
  };
  put(tightest(t.faq, 2), "ld", "faq_block");
  put(tightest(t.crumbs, 2), "ld", "breadcrumbs");
  put(tightest(t.reviews, 1), "ld", "reviews");
  put(tightest(t.address, 1), "ld", "business_info");
  put(tightest(t.authors, 1), "ld", "author_byline");
  if (t.prices.length) {
    let best: number | null = null;
    for (const n of ids) {
      const el = order[n]!;
      if (!el.querySelector('button, input[type="submit"], [role="button"]')) continue;
      const numbers = (text(n).match(/\d[\d,]*(?:\.\d+)?/g) ?? []).map((s) =>
        Number(s.replace(/,/g, "")),
      );
      if (!t.prices.some((p) => numbers.includes(p))) continue;
      if (best === null || text(n).length < text(best).length) best = n;
    }
    put(best, "ld", "product_offer");
  }

  // sem: markup rules (also the deterministic baseline).
  for (const n of ids) {
    const el = order[n]!;
    const tag = el.localName;
    const label = (el.getAttribute("aria-label") ?? "").toLowerCase();
    const cls = `${el.getAttribute("class") ?? ""} ${el.getAttribute("id") ?? ""}`.toLowerCase();
    if (
      /breadcrumb/.test(label) ||
      (/breadcrumb/.test(cls) && el.querySelectorAll("a").length >= 2)
    )
      put(n, "sem", "breadcrumbs");
    if (tag === "address") put(n, "sem", "business_info");
    let details = 0;
    for (const child of el.children as unknown as El[])
      if (child.localName === "details" && child.querySelector("summary")) details++;
    if (details >= 2) put(n, "sem", "faq_block");
    if (
      (el.querySelector('a[rel~="author"]') || /\bbyline\b|post-author|author-name/.test(cls)) &&
      text(n).length <= 150
    )
      put(n, "sem", "author_byline");
  }
  for (const el of document.querySelectorAll("video, iframe") as unknown as El[]) {
    if (el.localName === "iframe" && !VIDEO_SRC.test(el.getAttribute("src") ?? "")) continue;
    put(anchor(el, 300), "sem", "video_embed");
  }
  void t.video;

  for (const row of labels.values()) {
    rows.push(row);
    if (spot.length < 200000 && BigInt(Bun.hash(`${pageId}:${row.n}`)) % 50n === 0n) {
      const el = order[row.n]!;
      spot.push({
        ...row,
        tag: el.localName,
        cls: (el.getAttribute("class") ?? "").slice(0, 60),
        text: text(row.n).slice(0, 100),
      });
    }
  }
  return candidates.size;
}

// --- run --------------------------------------------------------------------
mkdirSync(dirname(outPath), { recursive: true, mode: 0o700 });
const rows: Row[] = [];
const spot: object[] = [];
const pages: string[] = [];
const stats = { pages: 0, labelled: 0, missingHtml: 0, failed: 0 };
const input = createInterface({
  input: createReadStream(`${dataset}/page-index.jsonl.gz`).pipe(createGunzip()),
});
for await (const line of input) {
  if (!line) continue;
  const row = JSON.parse(line) as { p: string; u: string; s: string };
  if (!chosen.has(row.p)) continue;
  if (limit > 0 && stats.pages >= limit) break;
  let html: string | null = null;
  for (const select of selects) {
    const found = select.all(row.p, `${row.p}g`);
    if (found.length === 1) {
      html = new TextDecoder().decode(gunzipSync(found[0]!.content));
      break;
    }
  }
  if (html === null) {
    stats.missingHtml++;
    continue;
  }
  try {
    const before = rows.length;
    const count = labelPage(row.p, row.s, row.u, html, rows, spot);
    pages.push(JSON.stringify({ p: row.p, s: row.s, candidates: count }));
    stats.labelled += rows.length - before;
  } catch {
    stats.failed++;
  }
  if (++stats.pages % 5000 === 0) console.log(JSON.stringify({ stage: "progress", ...stats }));
}
for (const db of stores) db.close();
writeFileSync(outPath, `${rows.map((r) => JSON.stringify(r)).join("\n")}\n`, { mode: 0o600 });
writeFileSync(outPath.replace(/\.jsonl$/, ".pages.jsonl"), `${pages.join("\n")}\n`, {
  mode: 0o600,
});
writeFileSync(
  outPath.replace(/\.jsonl$/, ".spot.jsonl"),
  `${spot.map((r) => JSON.stringify(r)).join("\n")}\n`,
  { mode: 0o600 },
);
console.log(JSON.stringify({ stage: "complete", ...stats }));

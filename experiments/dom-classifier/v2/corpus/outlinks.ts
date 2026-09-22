#!/usr/bin/env bun
/**
 * Harvest outbound registrable domains from the HTML already in a local store.
 *
 * Why this exists: the project databases' `links` table is empty on almost
 * every site in this store (463 domains across 17,239 databases), so the
 * "outbound links" seed source has to come from the stored bytes instead. It
 * also fixes a *composition* problem rather than only a size one. The seeds
 * taken from project directories are the sites squirrelscan users audited:
 * 665 `vercel.app`, 210 `lovable.app`, 759 `.ai`. That skew is useful — those
 * are exactly the landmark-less themes this corpus is short of — but a
 * classifier trained only on them would never see WordPress, a news site, a
 * government page or a shop. What those pages LINK TO is the rest of the web.
 *
 * No network, no DOM parse: one regex over the decompressed bytes. A full
 * linkedom parse of 28k pages costs minutes and nothing here needs a tree, only
 * the host of each href.
 *
 * Emits `outlinks.jsonl` (domain, url, count) under the private data root.
 * The store is opened READ-ONLY.
 */
import { Database } from "bun:sqlite";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";

import { parse as parseDomain } from "tldts";

const dataRoot = process.env.DOM_CLASSIFIER_DATA_ROOT;
const storeRoot = process.env.SQUIRREL_STORE;
if (!dataRoot) throw new Error("DOM_CLASSIFIER_DATA_ROOT must be set");
if (!storeRoot) throw new Error("SQUIRREL_STORE must be set");

const outDir = resolve(dataRoot, "corpus");
mkdirSync(outDir, { recursive: true, mode: 0o700 });

function argument(name: string, fallback: string): string {
  const index = Bun.argv.indexOf(name);
  if (index < 0) return fallback;
  const value = Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}
const minCount = Number(argument("--min-count", "2"));

/** Hosts that are infrastructure, not websites with a theme to learn from. */
const SKIP_DOMAIN = new Set([
  "w3.org",
  "schema.org",
  "googleapis.com",
  "gstatic.com",
  "google-analytics.com",
  "googletagmanager.com",
  "doubleclick.net",
  "facebook.net",
  "cloudflare.com",
  "jsdelivr.net",
  "unpkg.com",
  "cdnjs.com",
  "bootstrapcdn.com",
  "jquery.com",
  "fontawesome.com",
  "gravatar.com",
  "wp.com",
  "wp.org",
  "licensebuttons.net",
  "creativecommons.org",
  "amazonaws.com",
  "cloudfront.net",
  "akamaized.net",
  "googleusercontent.com",
  "ggpht.com",
  "ytimg.com",
  "twimg.com",
  "fbcdn.net",
  "gmpg.org",
  "purl.org",
  "xmlns.com",
  "openweathermap.org",
  "adobe.com",
  // Beacons, ad exchanges and link shorteners: a fetch of the root returns an
  // API error or a redirect stub, never a page with a theme to learn from.
  "googlesyndication.com",
  "shopifysvc.com",
  "wa.me",
  "discord.gg",
  "t.me",
  "bit.ly",
  "buff.ly",
  "goo.gl",
  "tinyurl.com",
  "hotjar.com",
  "segment.com",
  "sentry.io",
  "intercom.io",
  "usefathom.com",
  "plausible.io",
  "clarity.ms",
]);
/** Platforms whose pages are a single shared template: one sample is enough. */
const CAP_ONE = new Set([
  "facebook.com",
  "twitter.com",
  "x.com",
  "instagram.com",
  "linkedin.com",
  "youtube.com",
  "tiktok.com",
  "pinterest.com",
  "reddit.com",
  "github.com",
  "t.me",
  "whatsapp.com",
  "threads.net",
  "medium.com",
  "vimeo.com",
]);

const HREF = /href\s*=\s*["'](https?:\/\/[^"'\s>]{4,300})["']/gi;

const db = new Database(`${storeRoot}/content-store.db`, { readonly: true });
const batch = db.query(
  "SELECT rowid AS rid, content FROM content WHERE rowid > ? AND content_type = 'text/html' ORDER BY rowid LIMIT ?",
);

const counts = new Map<string, { url: string; count: number }>();
let cursor = 0;
let pages = 0;
let hrefs = 0;
const startedAt = Date.now();

for (;;) {
  const rows = batch.all(cursor, 64) as Array<{ rid: number; content: Uint8Array }>;
  if (rows.length === 0) break;
  cursor = rows[rows.length - 1]!.rid;
  for (const row of rows) {
    let html: string;
    try {
      html = gunzipSync(row.content).toString("utf8");
    } catch {
      continue;
    }
    pages++;
    // One page must not be able to nominate a domain twice, or a footer link
    // repeated on every page of one site outvotes a hundred real sites.
    const onThisPage = new Set<string>();
    for (const match of html.matchAll(HREF)) {
      hrefs++;
      const raw = match[1]!;
      let url: URL;
      try {
        url = new URL(raw);
      } catch {
        continue;
      }
      const parsed = parseDomain(url.href, { allowPrivateDomains: false });
      const domain = parsed.domain;
      if (!domain || SKIP_DOMAIN.has(domain) || onThisPage.has(domain)) continue;
      onThisPage.add(domain);
      const existing = counts.get(domain);
      if (existing) existing.count++;
      else counts.set(domain, { url: `${url.protocol}//${url.host}/`, count: 1 });
    }
  }
  if (pages % 5000 === 0)
    console.log(JSON.stringify({ stage: "scan", pages, domains: counts.size, hrefs }));
}
db.close();

// A domain linked from only one page of one site is usually a typo, a
// throwaway or a tracker. Two independent pages is a low bar that removes most
// of that noise; a platform domain is capped at one seed because every page of
// it is the same template.
const rows: Array<{ domain: string; url: string; count: number }> = [];
for (const [domain, entry] of counts) {
  if (CAP_ONE.has(domain)) {
    rows.push({ domain, url: entry.url, count: entry.count });
    continue;
  }
  if (entry.count < minCount) continue;
  rows.push({ domain, url: entry.url, count: entry.count });
}
rows.sort((left, right) => right.count - left.count || (left.domain < right.domain ? -1 : 1));
writeFileSync(`${outDir}/outlinks.jsonl`, rows.map((r) => `${JSON.stringify(r)}\n`).join(""), {
  mode: 0o600,
});
console.log(
  JSON.stringify({
    stage: "complete",
    pages,
    hrefs,
    domainsSeen: counts.size,
    emitted: rows.length,
    minCount,
    seconds: Number(((Date.now() - startedAt) / 1000).toFixed(1)),
  }),
);

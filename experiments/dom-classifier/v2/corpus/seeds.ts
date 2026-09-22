#!/usr/bin/env bun
/**
 * Build the crawl seed list: one seed URL per registrable domain, ordered so a
 * truncated crawl is still a diverse crawl.
 *
 * Nothing here reaches the network. Every source is already on this machine:
 *
 *  1. the local squirrel store's project databases — one directory per site
 *     already crawled once, whose `pages` rows give a known-good seed url. The
 *     store is LRU-pruned, so ~128k of those pages no longer have bytes behind
 *     them: the URL survives, the HTML does not, and re-fetching is the whole
 *     point of this list.
 *  2. the `links` table of those same databases — outbound hrefs, which is
 *     where domains the store has never crawled come from.
 *  3. the 2026-09-19 launch-study files (`cloud-run-inventory.jsonl`,
 *     `discovered-links.jsonl`).
 *
 * Output is ONE line per registrable domain under the private data root. No
 * list is scraped from the web.
 *
 * Ordering is the part that matters. Domains are emitted round-robin across
 * public suffix, so cutting the list at any length keeps the TLD mix of the
 * whole: taking the first N of a sorted list would give N .com domains.
 */
import { Database } from "bun:sqlite";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { parse as parseDomain } from "tldts";

const dataRoot = process.env.DOM_CLASSIFIER_DATA_ROOT;
const storeRoot = process.env.SQUIRREL_STORE;
if (!dataRoot) throw new Error("DOM_CLASSIFIER_DATA_ROOT must be set");
if (!storeRoot) throw new Error("SQUIRREL_STORE must be set");

/** The 2026-09-19 study files, read-only inputs that are never modified. */
const studyRoot = process.env.DOM_CLASSIFIER_STUDY_ROOT ?? "";

const outDir = resolve(dataRoot, "corpus");
mkdirSync(outDir, { recursive: true, mode: 0o700 });

function argument(name: string, fallback: string): string {
  const index = Bun.argv.indexOf(name);
  if (index < 0) return fallback;
  const value = Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

/** A domain we will never crawl: not a site, or not ours to fetch. */
const HOST_DENY = /(^|\.)(localhost|local|test|invalid|example|internal|lan)$/i;
const IP_LIKE = /^\d{1,3}(\.\d{1,3}){3}$|^\[|:\d+$/;

type Seed = { domain: string; url: string; suffix: string; source: string };

const seeds = new Map<string, Seed>();
const counts: Record<string, number> = {};
let rejected = 0;

/** First writer wins, so a domain's source records where it was FIRST seen. */
function offer(rawUrl: string, source: string): void {
  if (!rawUrl || rawUrl.length > 2000) return;
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    rejected++;
    return;
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") return;
  const host = url.hostname;
  if (!host || IP_LIKE.test(host) || HOST_DENY.test(host)) {
    rejected++;
    return;
  }
  const parsed = parseDomain(url.href, { allowPrivateDomains: true });
  const domain = parsed.domain;
  if (!domain || !parsed.publicSuffix) {
    rejected++;
    return;
  }
  if (seeds.has(domain)) return;
  // Seed at the site root, not at whatever deep page happened to be indexed:
  // a root seed lets the crawler find the site's own navigation, and a deep
  // seed re-seeds from the sitemap anyway.
  //
  // For an outbound link the HOST is dropped as well as the path. The harvest
  // sees whichever subdomain a page happened to link to, which for the big
  // platforms is a CDN or a beacon endpoint (`cdn.shopify.com`,
  // `monorail-edge.shopifysvc.com`, `pagead2.googlesyndication.com`), and none
  // of those serves a page with a theme. The registrable root does, and apex
  // hosts redirect to `www` on their own.
  const seedHost = source === "outlinks" ? domain : url.host;
  seeds.set(domain, {
    domain,
    url: `${url.protocol}//${seedHost}/`,
    suffix: parsed.publicSuffix,
    source,
  });
  counts[source] = (counts[source] ?? 0) + 1;
}

// --- 1 + 2: the project databases -------------------------------------------
const projectsDir = `${storeRoot}/projects`;
const dirs = readdirSync(projectsDir)
  .filter((name) => !name.startsWith("."))
  .sort();
let opened = 0;
let failed = 0;
const startedAt = Date.now();
const wantLinks = !Bun.argv.includes("--no-links");

for (const dir of dirs) {
  const dbPath = `${projectsDir}/${dir}/project.db`;
  if (!existsSync(dbPath)) continue;
  let db: Database | null = null;
  try {
    db = new Database(dbPath, { readonly: true });
    for (const row of db
      .query("SELECT normalized_url AS u FROM pages WHERE status = 200 LIMIT 200")
      .all() as Array<{ u: string }>)
      offer(row.u, "store-pages");
    if (wantLinks)
      for (const row of db
        .query("SELECT href AS u FROM links WHERE is_internal = 0 LIMIT 3000")
        .all() as Array<{ u: string }>)
        offer(row.u, "store-links");
    opened++;
  } catch {
    failed++;
  } finally {
    db?.close();
  }
  if (opened % 2000 === 0 && opened > 0)
    console.log(
      JSON.stringify({ stage: "projects", opened, of: dirs.length, domains: seeds.size }),
    );
}
console.log(
  JSON.stringify({ stage: "projects-done", opened, failed, domains: seeds.size, rejected }),
);

// --- 3: the 2026-09-19 study files ------------------------------------------
if (studyRoot) {
  for (const [file, field, source] of [
    ["diverse-500-v1/cloud-run-inventory.jsonl", "url", "cloud-inventory"],
    ["diverse-500-v1/discovered-links.jsonl", "url", "discovered-links"],
  ] as const) {
    const path = `${studyRoot}/${file}`;
    if (!existsSync(path)) continue;
    for (const line of readFileSync(path, "utf8").split("\n")) {
      if (!line) continue;
      try {
        offer((JSON.parse(line) as Record<string, string>)[field] ?? "", source);
      } catch {
        rejected++;
      }
    }
  }
  console.log(JSON.stringify({ stage: "study-done", domains: seeds.size }));
}

// --- 4: outbound links harvested from the stored HTML ------------------------
// Last, so a domain the store has actually crawled keeps its known-good seed
// url and its own source label. This source exists for COMPOSITION: the
// project directories are squirrelscan's own users (vercel.app, lovable.app,
// .ai), and what their pages link to is where the .gov, the news sites, the
// shops and the non-English web come from.
const outlinksPath = `${outDir}/outlinks.jsonl`;
if (existsSync(outlinksPath)) {
  for (const line of readFileSync(outlinksPath, "utf8").split("\n")) {
    if (!line) continue;
    try {
      offer((JSON.parse(line) as { url: string }).url, "outlinks");
    } catch {
      rejected++;
    }
  }
  console.log(JSON.stringify({ stage: "outlinks-done", domains: seeds.size }));
}

// --- ordering ---------------------------------------------------------------
// Two passes. First round-robin across public suffix WITHIN each cohort, so the
// TLD mix of a truncated run matches the whole; then interleave the cohorts so
// the diverse one is spent evenly across the run rather than at the end.
function roundRobinBySuffix(pool: Seed[]): Seed[] {
  const bySuffix = new Map<string, Seed[]>();
  for (const seed of pool) {
    const bucket = bySuffix.get(seed.suffix);
    if (bucket) bucket.push(seed);
    else bySuffix.set(seed.suffix, [seed]);
  }
  for (const bucket of bySuffix.values())
    bucket.sort((left, right) =>
      createHash("sha256").update(left.domain).digest("hex") <
      createHash("sha256").update(right.domain).digest("hex")
        ? -1
        : 1,
    );
  // Largest suffix first, so .com (the bulk) is drawn every round and the long
  // tail is exhausted into the early part of the list rather than the end.
  const buckets = [...bySuffix.entries()].sort((left, right) => right[1].length - left[1].length);
  const out: Seed[] = [];
  for (let round = 0; out.length < pool.length; round++) {
    let moved = false;
    for (const [, bucket] of buckets) {
      const seed = bucket[round];
      if (seed) {
        out.push(seed);
        moved = true;
      }
    }
    if (!moved) break;
  }
  return out;
}

/**
 * The two cohorts, and why they are not interchangeable.
 *
 * `store-pages` seeds are the sites squirrelscan's own users audited. They are
 * 84% of the pool and they skew hard — 665 `vercel.app`, 210 `lovable.app`, 848
 * `.ai`. That skew is USEFUL, because those are the landmark-less themes the
 * classifier is short of, but a corpus that is 84% of them has never seen
 * WordPress, a news site, a shop or a government page.
 *
 * `outlinks` and `cloud-inventory` are the harvested rest of the web, and they
 * are the scarce resource. Interleaving to a share cap spends them evenly, so
 * the diversity of the corpus does not depend on the run reaching the end of
 * the list — which it will not, because the clock stops first.
 */
const userBaseOrder = roundRobinBySuffix(
  [...seeds.values()].filter((seed) => seed.source === "store-pages"),
);
const harvestedOrder = roundRobinBySuffix(
  [...seeds.values()].filter((seed) => seed.source !== "store-pages"),
);
const userBaseShare = Number(argument("--user-base-share", "0.4"));
const ordered: Seed[] = [];
let userBaseTaken = 0;
let userIndex = 0;
let harvestIndex = 0;
/** Where the harvested cohort ran out, after which the cap cannot hold. */
let harvestExhaustedAt = -1;
while (userIndex < userBaseOrder.length || harvestIndex < harvestedOrder.length) {
  const haveHarvest = harvestIndex < harvestedOrder.length;
  const haveUser = userIndex < userBaseOrder.length;
  // Take from the user-base cohort only while doing so keeps its running share
  // at or under the cap. Otherwise take a harvested seed.
  const wouldStayUnderCap = (userBaseTaken + 1) / (ordered.length + 1) <= userBaseShare;
  if (haveUser && (!haveHarvest || wouldStayUnderCap)) {
    if (!haveHarvest && harvestExhaustedAt < 0) harvestExhaustedAt = ordered.length;
    ordered.push(userBaseOrder[userIndex++]!);
    userBaseTaken++;
  } else if (haveHarvest) {
    ordered.push(harvestedOrder[harvestIndex++]!);
  } else break;
}

const limit = Number(argument("--limit", "0"));
const emitted = limit > 0 ? ordered.slice(0, limit) : ordered;
writeFileSync(
  `${outDir}/seeds.jsonl`,
  emitted.map((seed) => `${JSON.stringify(seed)}\n`).join(""),
  { mode: 0o600 },
);
const suffixCounts = new Map<string, number>();
for (const seed of emitted) suffixCounts.set(seed.suffix, (suffixCounts.get(seed.suffix) ?? 0) + 1);
const suffixTop = Object.fromEntries(
  [...suffixCounts.entries()].sort((left, right) => right[1] - left[1]).slice(0, 15),
);
/** The user-base share of each prefix of the list, so the cap can be checked
 *  against where the crawl actually stopped rather than against the whole. */
const shareAt: Record<string, number> = {};
for (const at of [1000, 2500, 5000, 10000, 13000, 15000]) {
  if (emitted.length < at) continue;
  const taken = emitted.slice(0, at).filter((seed) => seed.source === "store-pages").length;
  shareAt[String(at)] = Number((taken / at).toFixed(3));
}
writeFileSync(
  `${outDir}/seeds-summary.json`,
  `${JSON.stringify(
    {
      builtAt: new Date().toISOString(),
      domains: emitted.length,
      totalDiscovered: seeds.size,
      suffixes: suffixCounts.size,
      suffixTop,
      bySource: counts,
      userBaseShareCap: userBaseShare,
      userBaseShareAtPrefix: shareAt,
      harvestedPool: harvestedOrder.length,
      userBasePool: userBaseOrder.length,
      harvestExhaustedAt: harvestExhaustedAt < 0 ? null : harvestExhaustedAt,
      rejected,
      projectDbs: { opened, failed, of: dirs.length },
      wallMs: Date.now() - startedAt,
    },
    null,
    2,
  )}\n`,
  { mode: 0o600 },
);
console.log(
  JSON.stringify({
    stage: "complete",
    domains: emitted.length,
    suffixes: suffixCounts.size,
    userBaseShareAtPrefix: shareAt,
    harvestExhaustedAt: harvestExhaustedAt < 0 ? null : harvestExhaustedAt,
  }),
);

// Build the silver training dataset for DOM classifier v2 (#2307).
//
// Reads a LIVE content store that another process is writing to, so every
// connection here is readonly, every query is a short bounded batch keyed by
// rowid, and nothing ever holds a transaction open. No writes, no VACUUM, no
// PRAGMA that touches the file.
//
// Nothing this script reads goes into the repo: URLs, hostnames and page text
// all stay under $DOM_CLASSIFIER_DATA_ROOT. The rows carry a domain HASH, and
// the url lives only in page-index.jsonl.gz.
//
// Usage:
//   DOM_CLASSIFIER_DATA_ROOT=... SQUIRREL_STORE=... bun run build-dataset.ts [flags]
//
//   --index-only       build the content-hash -> url index and stop
//   --limit N          process only the first N selected pages (document order
//                      by rowid, so a prefix is a deterministic sample)
//   --max-per-domain N default 25
//   --rebuild-index    ignore the cached index
//
// Phases:
//   1. index   stream every project.db, read (normalized_url, content_hash)
//   2. select  dedupe by hash, cap per registrable domain, assign the split
//   3. build   stream the content store by rowid, featurize, label, write

import { Database } from "bun:sqlite";
import { createHash } from "node:crypto";
import {
  appendFileSync,
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { gunzipSync } from "node:zlib";
import { createGzip } from "node:zlib";

import { parse as parseDomain } from "tldts";

import { parseHTML } from "../../../packages/parser/src/dom.ts";
import { featureSchema, featurizeDocument, FEATURIZER_VERSION } from "./featurize.ts";
import { silverPageType, silverRegionLabels } from "./silver-labels.ts";

const DATASET_VERSION = "silver-v1";

/** Pages per registrable domain. A handful of sites have thousands of pages in
 *  the store and would otherwise decide every class boundary on their own. */
const DEFAULT_MAX_PER_DOMAIN = 25;

/** Uncompressed HTML above this is skipped: a 10 MB page is a data dump, and
 *  parsing one costs more than the rest of its domain put together. */
const MAX_HTML_BYTES = 3_000_000;

/** Rows per content-store read. Short and bounded, so a writer is never blocked
 *  behind this reader for longer than one batch. */
const BATCH_ROWS = 64;

interface Args {
  indexOnly: boolean;
  rebuildIndex: boolean;
  limit: number;
  maxPerDomain: number;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    indexOnly: argv.includes("--index-only"),
    rebuildIndex: argv.includes("--rebuild-index"),
    limit: Number.POSITIVE_INFINITY,
    maxPerDomain: DEFAULT_MAX_PER_DOMAIN,
  };
  const limit = argv.indexOf("--limit");
  if (limit !== -1 && argv[limit + 1]) args.limit = Number(argv[limit + 1]);
  const perDomain = argv.indexOf("--max-per-domain");
  if (perDomain !== -1 && argv[perDomain + 1]) args.maxPerDomain = Number(argv[perDomain + 1]);
  return args;
}

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} must be set (no path is hardcoded in this repo)`);
  return value;
}

const dataRoot = requireEnv("DOM_CLASSIFIER_DATA_ROOT");
const storeRoot = requireEnv("SQUIRREL_STORE");
const outDir = `${dataRoot}/${DATASET_VERSION}`;
const cacheDir = `${dataRoot}/cache`;
const logPath = `${dataRoot}/logs/build-dataset.log`;

function log(message: string): void {
  const line = `${new Date().toISOString()} ${message}\n`;
  process.stdout.write(line);
  try {
    appendFileSync(logPath, line);
  } catch {
    // The log is a convenience; losing it must not lose the run.
  }
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

// ---------------------------------------------------------------------------
// Phase 1: content hash -> url
// ---------------------------------------------------------------------------

interface IndexEntry {
  hash: string;
  url: string;
}

/**
 * Stream every project.db once, readonly, closing each before opening the next.
 * A page row's `content_hash` IS the content-store key on the CLI path (the
 * store is content-addressed by sha256 of the exact bytes, and upsertPage
 * writes that hash back into the column), so this is the whole join.
 */
function buildIndex(indexPath: string): void {
  const projectsDir = `${storeRoot}/projects`;
  const dirs = readdirSync(projectsDir)
    .filter((name) => !name.startsWith("."))
    .sort();
  log(`index: ${dirs.length} project dirs`);

  const seen = new Map<string, string>();
  let opened = 0;
  let failed = 0;
  let rows = 0;
  const started = Date.now();

  for (const dir of dirs) {
    const dbPath = `${projectsDir}/${dir}/project.db`;
    if (!existsSync(dbPath)) continue;
    let db: Database | null = null;
    try {
      db = new Database(dbPath, { readonly: true });
      const found = db
        .query(
          "SELECT normalized_url AS url, content_hash AS hash FROM pages WHERE content_hash IS NOT NULL AND status = 200",
        )
        .all() as IndexEntry[];
      opened++;
      for (const row of found) {
        rows++;
        // First url wins, so a re-crawl under a different alias does not make
        // the index depend on directory order.
        if (!seen.has(row.hash)) seen.set(row.hash, row.url);
      }
    } catch {
      failed++;
    } finally {
      db?.close();
    }
    if (opened % 2000 === 0 && opened > 0) {
      log(`index: ${opened}/${dirs.length} dbs, ${seen.size} hashes, ${Date.now() - started}ms`);
    }
  }

  const lines: string[] = [];
  for (const [hash, url] of [...seen].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
    lines.push(JSON.stringify({ hash, url }));
  }
  writeFileSync(indexPath, `${lines.join("\n")}\n`);
  log(
    `index: done. dbs=${opened} failed=${failed} pageRows=${rows} uniqueHashes=${seen.size} ` +
      `ms=${Date.now() - started}`,
  );
}

function loadIndex(indexPath: string): Map<string, string> {
  const map = new Map<string, string>();
  for (const line of readFileSync(indexPath, "utf8").split("\n")) {
    if (!line) continue;
    const entry = JSON.parse(line) as IndexEntry;
    map.set(entry.hash, entry.url);
  }
  return map;
}

// ---------------------------------------------------------------------------
// Phase 2: select and split
// ---------------------------------------------------------------------------

type Split = "train" | "val" | "holdout";

interface Selection {
  hash: string;
  url: string;
  domain: string;
  domainHash: string;
  split: Split;
}

/** 80 / 10 / 10 by registrable domain, so no domain's template appears on both
 *  sides of a split. Written once to split-manifest.json and never recomputed
 *  from anything but the domain string. */
function splitOf(domain: string): Split {
  const bucket = parseInt(sha256(domain).slice(0, 8), 16) % 100;
  if (bucket < 80) return "train";
  if (bucket < 90) return "val";
  return "holdout";
}

function registrableDomain(url: string): string {
  const parsed = parseDomain(url, { allowPrivateDomains: true });
  return parsed.domain ?? parsed.hostname ?? "";
}

/**
 * Every content hash the store still HOLDS. The store is LRU-pruned, so most
 * of the 156k hashes the project databases know about no longer have bytes
 * behind them — and the per-domain cap has to be applied to what is actually
 * available, or a domain with 100 indexed pages and 3 surviving blobs loses
 * all three to a cap that picked 25 of the 100.
 *
 * `hash` is the table's PRIMARY KEY, so this is answered from that index alone
 * and never touches the 1 GB of blobs.
 */
function availableHashes(): Set<string> {
  const db = new Database(`${storeRoot}/content-store.db`, { readonly: true });
  const started = Date.now();
  const hashes = new Set<string>();
  for (const row of db.query("SELECT hash FROM content").all() as Array<{ hash: string }>) {
    hashes.add(row.hash);
  }
  db.close();
  log(`store: ${hashes.size} content hashes present, ${Date.now() - started}ms`);
  return hashes;
}

function select(
  index: Map<string, string>,
  available: Set<string>,
  maxPerDomain: number,
): { selected: Map<string, Selection>; excluded: Record<string, number>; domains: number } {
  const byDomain = new Map<string, Array<{ hash: string; url: string }>>();
  const excluded: Record<string, number> = {
    no_domain: 0,
    over_domain_cap: 0,
    pruned_from_store: 0,
  };

  for (const [hash, url] of index) {
    if (!available.has(hash)) {
      excluded.pruned_from_store = (excluded.pruned_from_store ?? 0) + 1;
      continue;
    }
    const domain = registrableDomain(url);
    if (!domain) {
      excluded.no_domain = (excluded.no_domain ?? 0) + 1;
      continue;
    }
    const bucket = byDomain.get(domain);
    if (bucket) bucket.push({ hash, url });
    else byDomain.set(domain, [{ hash, url }]);
  }

  const selected = new Map<string, Selection>();
  for (const [domain, pages] of byDomain) {
    // Deterministic pick: order by the hash of the content hash, which has
    // nothing to do with crawl order, site size or url shape.
    pages.sort((a, b) => {
      const left = sha256(a.hash);
      const right = sha256(b.hash);
      return left < right ? -1 : left > right ? 1 : 0;
    });
    if (pages.length > maxPerDomain) {
      excluded.over_domain_cap = (excluded.over_domain_cap ?? 0) + (pages.length - maxPerDomain);
    }
    const domainHash = sha256(domain).slice(0, 16);
    const split = splitOf(domain);
    for (const page of pages.slice(0, maxPerDomain)) {
      selected.set(page.hash, { hash: page.hash, url: page.url, domain, domainHash, split });
    }
  }

  return { selected, excluded, domains: byDomain.size };
}

// ---------------------------------------------------------------------------
// Output writers
// ---------------------------------------------------------------------------

interface GzWriter {
  write(line: string): Promise<void>;
  close(): Promise<void>;
  lines: number;
}

function gzWriter(path: string): GzWriter {
  const file = createWriteStream(path);
  const gzip = createGzip({ level: 6 });
  gzip.pipe(file);
  const writer: GzWriter = {
    lines: 0,
    async write(line: string): Promise<void> {
      writer.lines++;
      if (!gzip.write(`${line}\n`)) {
        await new Promise<void>((resolve) => gzip.once("drain", resolve));
      }
    },
    close(): Promise<void> {
      return new Promise((resolve, reject) => {
        file.on("finish", resolve);
        file.on("error", reject);
        gzip.end();
      });
    },
  };
  return writer;
}

// ---------------------------------------------------------------------------
// Phase 3: build
// ---------------------------------------------------------------------------

interface Stats {
  datasetVersion: string;
  featurizerVersion: string;
  builtAt: string;
  maxPerDomain: number;
  pages: Record<Split, number>;
  domains: Record<Split, number>;
  nodeRows: Record<Split, number>;
  /** split -> label -> count, on the UNMASKED rows only (the masked rows are
   *  the same nodes, so counting both would double every number). */
  nodesByLabel: Record<Split, Record<string, number>>;
  /** Nodes whose label came from the landmark they sit in, not their own tag. */
  inheritedNodes: Record<Split, number>;
  trainableNodes: Record<Split, number>;
  pageTypes: Record<Split, Record<string, number>>;
  pageTypeSources: Record<string, number>;
  divSoupPages: Record<Split, number>;
  regionTrainablePages: Record<Split, number>;
  nestedChrome: {
    articleHeader: number;
    articleFooter: number;
    siteHeader: number;
    footer: number;
  };
  excluded: Record<string, number>;
  throughput: { pages: number; seconds: number; pagesPerSecond: number; bytes: number };
}

function emptyBySplit<T>(make: () => T): Record<Split, T> {
  return { train: make(), val: make(), holdout: make() };
}

async function build(
  args: Args,
  selected: Map<string, Selection>,
  excluded: Record<string, number>,
) {
  mkdirSync(outDir, { recursive: true });

  const nodeWriters = emptyBySplit(() => null as unknown as GzWriter);
  const pageWriters = emptyBySplit(() => null as unknown as GzWriter);
  for (const split of ["train", "val", "holdout"] as const) {
    nodeWriters[split] = gzWriter(`${outDir}/nodes-${split}.jsonl.gz`);
    pageWriters[split] = gzWriter(`${outDir}/pages-${split}.jsonl.gz`);
  }
  const pageIndexWriter = gzWriter(`${outDir}/page-index.jsonl.gz`);

  const stats: Stats = {
    datasetVersion: DATASET_VERSION,
    featurizerVersion: FEATURIZER_VERSION,
    builtAt: new Date().toISOString(),
    maxPerDomain: args.maxPerDomain,
    pages: emptyBySplit(() => 0),
    domains: emptyBySplit(() => 0),
    nodeRows: emptyBySplit(() => 0),
    nodesByLabel: emptyBySplit(() => ({}) as Record<string, number>),
    inheritedNodes: emptyBySplit(() => 0),
    trainableNodes: emptyBySplit(() => 0),
    pageTypes: emptyBySplit(() => ({}) as Record<string, number>),
    pageTypeSources: {},
    divSoupPages: emptyBySplit(() => 0),
    regionTrainablePages: emptyBySplit(() => 0),
    nestedChrome: { articleHeader: 0, articleFooter: 0, siteHeader: 0, footer: 0 },
    excluded: { ...excluded, not_selected: 0, too_large: 0, parse_failed: 0, no_candidates: 0 },
    throughput: { pages: 0, seconds: 0, pagesPerSecond: 0, bytes: 0 },
  };
  const domainsSeen = emptyBySplit(() => new Set<string>());

  const db = new Database(`${storeRoot}/content-store.db`, { readonly: true });
  // One prepared statement, reused: a short keyset read by rowid, which is the
  // cheapest thing a concurrent writer can be asked to share a file with.
  const batch = db.query(
    "SELECT rowid AS rid, hash, content FROM content WHERE rowid > ? AND content_type = 'text/html' ORDER BY rowid LIMIT ?",
  );

  const started = Date.now();
  let cursor = 0;
  let processed = 0;
  let bytes = 0;

  for (;;) {
    const rows = batch.all(cursor, BATCH_ROWS) as Array<{
      rid: number;
      hash: string;
      content: Uint8Array;
    }>;
    if (rows.length === 0) break;
    cursor = rows[rows.length - 1]!.rid;

    for (const row of rows) {
      const choice = selected.get(row.hash);
      if (!choice) {
        // Either no project database knows a url for these bytes, or the
        // per-domain cap did not pick them. Both are counted in `excluded`
        // already; this is the row-level tally of "seen and skipped".
        stats.excluded.not_selected = (stats.excluded.not_selected ?? 0) + 1;
        continue;
      }
      if (processed >= args.limit) break;

      let html: string;
      try {
        const raw = gunzipSync(row.content);
        if (raw.length > MAX_HTML_BYTES) {
          stats.excluded.too_large = (stats.excluded.too_large ?? 0) + 1;
          continue;
        }
        bytes += raw.length;
        html = raw.toString("utf8");
      } catch {
        stats.excluded.parse_failed = (stats.excluded.parse_failed ?? 0) + 1;
        continue;
      }

      let featurized;
      let regions;
      let pageType;
      try {
        const { document } = parseHTML(html);
        featurized = featurizeDocument(document as never, choice.url, { emitMasked: true });
        regions = silverRegionLabels(featurized.nodes);
        pageType = silverPageType(document as never, choice.url);
      } catch {
        stats.excluded.parse_failed = (stats.excluded.parse_failed ?? 0) + 1;
        continue;
      }
      if (featurized.nodes.length === 0) {
        stats.excluded.no_candidates = (stats.excluded.no_candidates ?? 0) + 1;
        continue;
      }

      const pageId = row.hash.slice(0, 16);
      const split = choice.split;
      const labelById = new Map(regions.labels.map((l) => [l.nodeId, l]));

      await pageWriters[split].write(
        JSON.stringify({
          p: pageId,
          d: choice.domainHash,
          s: split,
          l: pageType.label,
          ls: pageType.source,
          leak: pageType.leakProne,
          types: pageType.types,
          f: featurized.page.features,
          k: featurized.page.tokens,
          divSoup: !regions.regionTrainable,
          landmarkKinds: regions.landmarkKinds,
          elementCount: featurized.page.elementCount,
          nodeCount: featurized.nodes.length,
          excludedReason: regions.excludedReason,
        }),
      );
      await pageIndexWriter.write(
        JSON.stringify({ p: pageId, u: choice.url, domain: choice.domain, s: split }),
      );

      for (const node of featurized.nodes) {
        const label = labelById.get(node.nodeId)!;
        const base = {
          p: pageId,
          n: node.nodeId,
          d: choice.domainHash,
          s: split,
          l: label.label,
          ls: label.labelSource,
          c: label.containerLabel,
          t: label.trainable,
          inh: label.inherited,
          w: label.weight,
          // The true tag and role: the LABEL source, kept for the tag baseline
          // the error analysis showed beats every learned model on semantic
          // tags. NEVER a feature.
          tagRaw: node.tag,
          roleRaw: node.role,
          depth: node.depth,
          parent: node.parentCandidateId,
        };
        await nodeWriters[split].write(
          JSON.stringify({ ...base, m: 0, f: node.features, k: node.tokens }),
        );
        await nodeWriters[split].write(
          JSON.stringify({ ...base, m: 1, f: node.maskedFeatures, k: node.maskedTokens }),
        );
        stats.nodeRows[split] += 2;
        const counts = stats.nodesByLabel[split];
        counts[label.label] = (counts[label.label] ?? 0) + 1;
        if (label.trainable) stats.trainableNodes[split]++;
        if (label.inherited) stats.inheritedNodes[split]++;
        if (label.label === "article_header") stats.nestedChrome.articleHeader++;
        if (label.label === "article_footer") stats.nestedChrome.articleFooter++;
        if (label.label === "site_header") stats.nestedChrome.siteHeader++;
        if (label.label === "footer") stats.nestedChrome.footer++;
      }

      stats.pages[split]++;
      domainsSeen[split].add(choice.domainHash);
      const types = stats.pageTypes[split];
      types[pageType.label] = (types[pageType.label] ?? 0) + 1;
      stats.pageTypeSources[pageType.source] = (stats.pageTypeSources[pageType.source] ?? 0) + 1;
      if (!regions.regionTrainable) stats.divSoupPages[split]++;
      else stats.regionTrainablePages[split]++;

      processed++;
      if (processed % 500 === 0) {
        const seconds = (Date.now() - started) / 1000;
        log(
          `build: ${processed} pages, ${(processed / seconds).toFixed(1)}/s, ` +
            `rowid ${cursor}, ${(bytes / 1e6).toFixed(0)} MB read`,
        );
      }
    }
    if (processed >= args.limit) break;
  }

  db.close();

  for (const split of ["train", "val", "holdout"] as const) {
    stats.domains[split] = domainsSeen[split].size;
    await nodeWriters[split].close();
    await pageWriters[split].close();
  }
  await pageIndexWriter.close();

  const seconds = (Date.now() - started) / 1000;
  stats.throughput = {
    pages: processed,
    seconds: Number(seconds.toFixed(1)),
    pagesPerSecond: Number((processed / Math.max(seconds, 0.001)).toFixed(2)),
    bytes,
  };

  writeFileSync(`${outDir}/feature-schema.json`, `${JSON.stringify(featureSchema(), null, 2)}\n`);
  writeFileSync(`${outDir}/stats.json`, `${JSON.stringify(stats, null, 2)}\n`);
  log(`build: done. ${JSON.stringify(stats.throughput)}`);
  return stats;
}

// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(cacheDir, { recursive: true });
  mkdirSync(`${dataRoot}/logs`, { recursive: true });

  const indexPath = `${cacheDir}/hash-url-index.jsonl`;
  if (args.rebuildIndex || !existsSync(indexPath)) buildIndex(indexPath);
  else log(`index: reusing ${indexPath}`);
  if (args.indexOnly) return;

  const index = loadIndex(indexPath);
  log(`index: ${index.size} content hashes with a url`);

  const available = availableHashes();
  const { selected, excluded, domains } = select(index, available, args.maxPerDomain);
  excluded.in_store_without_url = available.size - (index.size - (excluded.pruned_from_store ?? 0));
  log(`select: ${selected.size} pages across ${domains} registrable domains`);

  // The split manifest is written ONCE. A later run with more pages must land
  // every domain in the same split, or the holdout is contaminated by anything
  // trained before it.
  const manifestPath = `${outDir}/split-manifest.json`;
  mkdirSync(outDir, { recursive: true });
  if (existsSync(manifestPath)) {
    const previous = JSON.parse(readFileSync(manifestPath, "utf8")) as {
      domains: Record<string, Split>;
    };
    let drifted = 0;
    for (const choice of selected.values()) {
      const already = previous.domains[choice.domainHash];
      if (already && already !== choice.split) drifted++;
    }
    if (drifted > 0)
      throw new Error(`split manifest drift on ${drifted} domains: refusing to build`);
    log(`split: manifest reused (${Object.keys(previous.domains).length} domains)`);
  } else {
    const domainsToSplit: Record<string, Split> = {};
    for (const choice of selected.values()) domainsToSplit[choice.domainHash] = choice.split;
    writeFileSync(
      manifestPath,
      `${JSON.stringify(
        {
          datasetVersion: DATASET_VERSION,
          createdAt: new Date().toISOString(),
          rule: "split = sha256(registrableDomain)[0:8] mod 100; <80 train, <90 val, else holdout",
          domains: domainsToSplit,
        },
        null,
        2,
      )}\n`,
    );
    log(`split: manifest written (${Object.keys(domainsToSplit).length} domains)`);
  }

  await build(args, selected, excluded);
}

await main();

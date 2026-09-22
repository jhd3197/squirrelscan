#!/usr/bin/env bun
/**
 * Build the Jev page-type work queue, and freeze the split assignment.
 *
 * `run-pagetype-v7.ts` wants rows of `{pageId, contentHash, url, split}`. That
 * is all derivable from the hash -> url index and the store's hash list, so the
 * queue does NOT wait for the featurized dataset: labelling can run while the
 * crawl is still going, and the runner is resumable by `pageId`, so a second
 * pass over a grown queue only labels what is new.
 *
 * ## The split is frozen by construction, not by a file
 *
 * `split = sha256(registrableDomain)[0:8] mod 100` — a pure function of the
 * domain string, the same one `build-dataset.ts` uses. A domain cannot change
 * split because the corpus grew, so the 2026-09-19 holdout domains stay in
 * holdout without anything having to be carried forward. `--verify-against`
 * checks that against the previous build's manifest and refuses to write if a
 * single domain disagrees, so the guarantee is measured rather than asserted.
 *
 * The per-domain cap and the selection order also match `build-dataset.ts`
 * (order by sha256 of the content hash, take the first N), so a page labelled
 * here is a page that ends up in the dataset.
 */
import { Database } from "bun:sqlite";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";

import { parse as parseDomain } from "tldts";

import { isHtmlType, type UnusableHtml, unusableHtmlReason } from "../unusable-html.ts";

const dataRoot = process.env.DOM_CLASSIFIER_DATA_ROOT;
if (!dataRoot) throw new Error("DOM_CLASSIFIER_DATA_ROOT must be set");
const storeRoots = (process.env.SQUIRREL_STORES ?? process.env.SQUIRREL_STORE ?? "")
  .split(":")
  .map((path) => path.trim())
  .filter(Boolean);
if (storeRoots.length === 0) throw new Error("SQUIRREL_STORES must be set");

function argument(name: string, fallback: string): string {
  const index = Bun.argv.indexOf(name);
  if (index < 0) return fallback;
  const value = Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}
const maxPerDomain = Number(argument("--max-per-domain", "25"));
const verifyAgainst = argument("--verify-against", "");
const outDir = resolve(dataRoot, "pagetype-jev-v7");
mkdirSync(outDir, { recursive: true, mode: 0o700 });

const sha256 = (value: string) => createHash("sha256").update(value).digest("hex");
type Split = "train" | "val" | "holdout";
function splitOf(domain: string): Split {
  const bucket = parseInt(sha256(domain).slice(0, 8), 16) % 100;
  if (bucket < 80) return "train";
  if (bucket < 90) return "val";
  return "holdout";
}

// --- hash -> url, from every store's project databases -----------------------
const index = new Map<string, string>();
/** The response content type the crawl recorded for each hash. A 200 can be a
 *  PNG, a zip or a JSON feed, and the content store files those bytes as
 *  text/html all the same. */
const contentTypes = new Map<string, string | null>();
for (const root of storeRoots) {
  const projectsDir = `${root}/projects`;
  if (!existsSync(projectsDir)) continue;
  const dirs = (
    await Array.fromAsync(new Bun.Glob("*/project.db").scan({ cwd: projectsDir }))
  ).sort();
  let opened = 0;
  for (const rel of dirs) {
    let db: Database | null = null;
    try {
      db = new Database(`${projectsDir}/${rel}`, { readonly: true });
      for (const row of db
        .query(
          "SELECT normalized_url AS url, content_hash AS hash, content_type AS type FROM pages WHERE content_hash IS NOT NULL AND status = 200",
        )
        .all() as Array<{ url: string; hash: string; type: string | null }>)
        if (!index.has(row.hash)) {
          index.set(row.hash, row.url);
          contentTypes.set(row.hash, row.type);
        }
      opened++;
    } catch {
      /* a project database mid-write is skipped, not fatal */
    } finally {
      db?.close();
    }
  }
  console.log(JSON.stringify({ stage: "index", root, dbs: opened, hashes: index.size }));
}

// --- which hashes still have bytes ------------------------------------------
const available = new Set<string>();
const contentStores: Database[] = [];
for (const root of storeRoots) {
  const path = `${root}/content-store.db`;
  if (!existsSync(path)) continue;
  const db = new Database(path, { readonly: true });
  for (const row of db.query("SELECT hash FROM content").all() as Array<{ hash: string }>)
    available.add(row.hash);
  contentStores.push(db);
}
const contentSelects = contentStores.map((db) =>
  db.query<{ content: Uint8Array }, [string]>("SELECT content FROM content WHERE hash = ?"),
);
/** Why a selected page has nothing to label, read from the same bytes the
 *  runner will read. `missingHtml` is a row pruned since the hash list above. */
function unusable(hash: string): UnusableHtml | "unreadableBlob" | "missingHtml" | null {
  for (const select of contentSelects) {
    const row = select.get(hash);
    if (!row) continue;
    // The store gzips every blob, so one that does not gunzip is unreadable,
    // not raw HTML: decoding it raw is how a bad read passed for a page.
    let html: string;
    try {
      html = new TextDecoder().decode(gunzipSync(row.content));
    } catch {
      return "unreadableBlob";
    }
    return unusableHtmlReason(html);
  }
  return "missingHtml";
}
console.log(JSON.stringify({ stage: "available", hashes: available.size }));

// --- select, exactly as build-dataset.ts does -------------------------------
const byDomain = new Map<string, Array<{ hash: string; url: string }>>();
let pruned = 0;
let noDomain = 0;
for (const [hash, url] of index) {
  if (!available.has(hash)) {
    pruned++;
    continue;
  }
  // `?? parsed.hostname`, matching build-dataset.ts exactly. Without it a url
  // whose host has no recognised public suffix is dropped here but still gets a
  // dataset row and a split there, so the page exists in the corpus with no
  // page-type label and nothing says why.
  const parsed = parseDomain(url, { allowPrivateDomains: true });
  const domain = parsed.domain ?? parsed.hostname ?? "";
  if (!domain) {
    noDomain++;
    continue;
  }
  const bucket = byDomain.get(domain);
  if (bucket) bucket.push({ hash, url });
  else byDomain.set(domain, [{ hash, url }]);
}

const previous: Record<string, Split> | null = verifyAgainst
  ? (JSON.parse(readFileSync(verifyAgainst, "utf8")) as { domains: Record<string, Split> }).domains
  : null;
let drifted = 0;
let checked = 0;

const rows: Record<Split, string[]> = { train: [], val: [], holdout: [] };
const domainsBySplit: Record<Split, number> = { train: 0, val: 0, holdout: 0 };
const manifest: Record<string, Split> = {};
// Named skip reasons, so a page with no usable HTML never reaches the runner
// and surfaces there as a crash (#2307).
const skipped: Record<UnusableHtml | "notHtml" | "unreadableBlob" | "missingHtml", number> = {
  emptyHtml: 0,
  noDocumentElement: 0,
  notHtml: 0,
  unreadableBlob: 0,
  missingHtml: 0,
};
for (const [domain, pages] of byDomain) {
  pages.sort((left, right) => (sha256(left.hash) < sha256(right.hash) ? -1 : 1));
  const split = splitOf(domain);
  const domainHash = sha256(domain).slice(0, 16);
  // The previous manifest keys by the 16-hex domain hash, not the domain.
  if (previous) {
    const already = previous[domainHash];
    if (already) {
      checked++;
      if (already !== split) drifted++;
    }
  }
  manifest[domainHash] = split;
  domainsBySplit[split]++;
  // Select first, THEN drop what cannot be labelled, with no backfill:
  // build-dataset.ts takes the same first N and applies the same content-type
  // and unusable-HTML checks, so a backfilled page would be labelled here and
  // never appear in the dataset. (It also drops pages over MAX_HTML_BYTES and
  // pages with no featurizer candidates, which a label here merely outlives.)
  for (const page of pages.slice(0, maxPerDomain)) {
    const reason = isHtmlType(contentTypes.get(page.hash)) ? unusable(page.hash) : "notHtml";
    if (reason) {
      skipped[reason]++;
      continue;
    }
    rows[split].push(
      JSON.stringify({
        pageId: page.hash.slice(0, 16),
        // The runner resolves content by a half-open range over this prefix.
        contentHash: page.hash.slice(0, 16),
        url: page.url,
        split,
      }),
    );
  }
}
for (const db of contentStores) db.close();
if (drifted > 0)
  throw new Error(
    `split drift on ${drifted} of ${checked} domains against ${verifyAgainst}: refusing to write`,
  );

for (const split of ["train", "val", "holdout"] as const)
  writeFileSync(`${outDir}/queue-${split}.jsonl`, rows[split].map((r) => `${r}\n`).join(""), {
    mode: 0o600,
  });
writeFileSync(
  `${outDir}/splits.json`,
  `${JSON.stringify(
    {
      createdAt: new Date().toISOString(),
      rule: "split = sha256(registrableDomain)[0:8] mod 100; <80 train, <90 val, else holdout",
      // `verified: false` spelled out. Without it a run that passed no
      // --verify-against writes "domainsChecked": 0, "drifted": 0, which reads
      // exactly like a clean verification of nothing.
      verified: Boolean(verifyAgainst),
      verifiedAgainst: verifyAgainst || null,
      domainsChecked: checked,
      drifted,
      domains: manifest,
    },
    null,
    2,
  )}\n`,
  { mode: 0o600 },
);
const summary = {
  stage: "complete",
  pages: { train: rows.train.length, val: rows.val.length, holdout: rows.holdout.length },
  domains: domainsBySplit,
  totalDomains: byDomain.size,
  maxPerDomain,
  prunedFromStore: pruned,
  noDomain,
  skipped,
  splitDriftChecked: checked,
  splitDrifted: drifted,
};
writeFileSync(`${outDir}/queue-summary.json`, `${JSON.stringify(summary, null, 2)}\n`, {
  mode: 0o600,
});
console.log(JSON.stringify(summary));

#!/usr/bin/env bun
/**
 * Fetch Common Crawl linking pages by WARC locator and place their backlinks
 * (#2307, DomainGenius's link-section batch).
 *
 *   bun run experiments/cc-link-batch.ts --locators <a.jsonl[,b.jsonl]> \
 *     --artifact classifier-v2.json --guard guard.json --out <dir> \
 *     [--corpus /corpus] [--concurrency 8] [--rate 4] [--budget-ms 500] [--limit N]
 *
 * Locator rows (DG): {locator_id, crawl, warc_filename, offset, length,
 *   source_url, domain_hash, bucket, sealed, links: [{href, anchor_text}],
 *   have_sha256?}. One capture (warc_filename, offset, length) is fetched once
 *   however many rows name it; their link lists are merged.
 *
 * Guard (`--guard`): {route: "train" | "holdout", sealed: [hash16], gold:
 *   [hash16]}. Every row's domain hash is recomputed from its URL with the
 *   split rule; a row whose hash disagrees with DG's, whose route is not the
 *   guard's, or (train) whose domain is a gold domain is skipped unfetched.
 *   The caller writes the output dir to the folder for that route, so the
 *   placement rule holds by construction.
 *
 * Politeness: at most `--concurrency` requests in flight (default 8), at most
 * `--rate` attempts per second across them (default 4: CC's CloudFront
 * blocked an IP at ~13/s), backoff with jitter on 403/429/503 (cc-warc.ts),
 * and a circuit breaker that stops starting fetches once throttling
 * dominates; the rest are reported unfetched.
 *
 * Output (PRIVATE: URLs, HTML):
 *   blobs/<sha256>.html.gz  the HTTP body as CC stores it (DG's convention)
 *   manifest.jsonl          one row per capture, with fetch status
 *   headers.jsonl           per capture: warc_date, WARC record headers and HTTP
 *                           response headers (cookie values dropped). With
 *                           `--headers-only`, only manifest.jsonl and this are
 *                           written (a backfill: no blobs, no labels).
 *   links.jsonl             DG's A16 shape plus locator_id and capture
 *   pages.jsonl             per-capture section counts
 *   stats.json
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { gunzipSync, gzipSync } from "node:zlib";

import { parse as parseDomain } from "tldts";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import type { Model } from "../runtime/model-format.ts";
import { classifyDocument, tryLoadModel } from "../runtime/predict.ts";
import { fetchWarcRange, parseWarcRecord, redact, WarcError } from "./cc-warc.ts";
import { a16Row } from "./link-sections.ts";

export { a16Row };
import { decode, linkPositions, percentile, type ManifestLink } from "./link-position.ts";

export interface LocatorRow {
  locator_id: string;
  crawl: string;
  warc_filename: string;
  offset: number;
  length: number;
  source_url: string;
  domain_hash: string;
  bucket: number;
  sealed: boolean;
  links?: ManifestLink[] | null;
  have_sha256?: string | null;
  /** DG's copy of `captureKey`, when the export carries it. */
  capture?: string | null;
}

export interface Guard {
  route: "train" | "holdout";
  sealed: string[];
  gold: string[];
}

export interface Capture {
  key: string;
  row: LocatorRow;
  /** Every locator_id that named this capture. */
  locatorIds: string[];
  /** Merged link lists; null when every row asked for all outbound links. */
  links: ManifestLink[] | null;
}

/** Split rule, as `splits/scripts/split_of.ts`: tldts with private suffixes. */
export function domainHashOf(url: string): string | null {
  const parsed = parseDomain(url, { allowPrivateDomains: true });
  const domain = parsed.domain ?? parsed.hostname;
  if (!domain) return null;
  return createHash("sha256").update(domain).digest("hex").slice(0, 16);
}

export const bucketOfHash = (hash16: string): number => parseInt(hash16.slice(0, 8), 16) % 100;

export function captureKey(row: Pick<LocatorRow, "warc_filename" | "offset" | "length">): string {
  return createHash("sha256")
    .update(`${row.warc_filename}|${row.offset}|${row.length}`)
    .digest("hex")
    .slice(0, 16);
}

/** One capture per byte range; rows naming the same range merge their links. */
export function groupCaptures(rows: readonly LocatorRow[]): Capture[] {
  const byKey = new Map<string, Capture>();
  for (const row of rows) {
    const key = captureKey(row);
    const rowLinks = row.links && row.links.length > 0 ? row.links : null;
    const seen = byKey.get(key);
    if (!seen) {
      byKey.set(key, {
        key,
        row,
        locatorIds: [row.locator_id],
        links: rowLinks ? [...rowLinks] : null,
      });
      continue;
    }
    if (!seen.locatorIds.includes(row.locator_id)) seen.locatorIds.push(row.locator_id);
    // An empty list means "every outbound link", which already covers any list.
    if (seen.links === null || rowLinks === null) {
      seen.links = null;
      continue;
    }
    const have = new Set(seen.links.map((l) => `${l.href}\u0000${l.anchor_text ?? ""}`));
    for (const link of rowLinks)
      if (!have.has(`${link.href}\u0000${link.anchor_text ?? ""}`)) seen.links.push(link);
  }
  return [...byKey.values()];
}

/** Spaces calls `1000 / perSecond` ms apart, however many callers wait. */
export function rateLimiter(perSecond: number): () => Promise<void> {
  const gap = 1000 / perSecond;
  let next = 0;
  return async () => {
    const now = Date.now();
    const at = Math.max(now, next);
    next = at + gap;
    if (at > now) await Bun.sleep(at - now);
  };
}

/** Why a capture must not be fetched into this route's folder, or null. */
export function routeRefusal(
  row: LocatorRow,
  guard: Guard,
  sealed: ReadonlySet<string>,
  gold: ReadonlySet<string>,
): string | null {
  if (row.capture && row.capture !== captureKey(row)) return "capture_mismatch";
  const hash = domainHashOf(row.source_url);
  if (!hash) return "no_domain";
  if (hash !== row.domain_hash) return "domain_hash_mismatch";
  const route = bucketOfHash(hash) >= 90 || sealed.has(hash) ? "holdout" : "train";
  if (route !== guard.route) return "route_mismatch";
  if (guard.route === "train" && gold.has(hash)) return "gold_domain";
  return null;
}

const HTML_TYPES = /^(?:text\/html|application\/xhtml\+xml)\b/i;

interface Outcome {
  status: string;
  sha256: string | null;
  httpStatus: number | null;
  contentType: string | null;
  bytes: number | null;
  fetched: "cc_warc" | "store" | null;
  fetchMs: number | null;
  truncated?: string | null;
  headers?: {
    warc_date: string | null;
    warc_headers: Record<string, string>;
    http_headers: Record<string, string>;
  };
}

function argument(name: string, fallback?: string): string {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? fallback : Bun.argv[index + 1];
  if (value === undefined || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

async function main(): Promise<void> {
  const outDir = argument("--out");
  const corpus = argument("--corpus", "/corpus");
  const concurrency = Math.max(1, Math.min(8, Number(argument("--concurrency", "8"))));
  const rate = Math.max(0.1, Math.min(8, Number(argument("--rate", "4"))));
  const limit_ = rateLimiter(rate);
  // Optional egress proxy from the environment (a Modal secret on Modal).
  const proxy = process.env.HTTPS_PROXY_URL || null;
  const budgetMs = Number(argument("--budget-ms", "500"));
  const limit = Number(argument("--limit", "0"));
  const guard = JSON.parse(readFileSync(argument("--guard"), "utf8")) as Guard;
  if (guard.route !== "train" && guard.route !== "holdout") throw new Error("guard.route");
  if (guard.sealed.length < 1000)
    throw new Error(`guard has ${guard.sealed.length} sealed hashes, expected ~1069`);
  const sealed = new Set(guard.sealed);
  const gold = new Set(guard.gold);
  const { model, reason } = tryLoadModel(JSON.parse(readFileSync(argument("--artifact"), "utf8")));
  if (!model) throw new Error(`artifact unusable: ${reason}`);

  const rows = argument("--locators")
    .split(",")
    .flatMap((file) => readFileSync(file, "utf8").split("\n").filter(Boolean))
    .map((line) => JSON.parse(line) as LocatorRow);
  let captures = groupCaptures(rows);
  if (limit > 0) captures = captures.slice(0, limit);

  mkdirSync(`${outDir}/blobs`, { recursive: true });
  const manifest = Bun.file(`${outDir}/manifest.jsonl`).writer();
  const linksOut = Bun.file(`${outDir}/links.jsonl`).writer();
  const pagesOut = Bun.file(`${outDir}/pages.jsonl`).writer();
  const headersOut = Bun.file(`${outDir}/headers.jsonl`).writer();
  // Backfill: fetch for the WARC and HTTP headers only, no blobs, no labels.
  const headersOnly = Bun.argv.includes("--headers-only");
  const statuses: Record<string, number> = {};
  const fetchMs: number[] = [];
  const classifyMs: number[] = [];
  const started = new Date().toISOString();

  // Circuit breaker: the last 200 fetch attempts, 1 = throttled.
  const recent: number[] = [];
  const throttles: Record<string, number> = {};
  let tripped = false;
  const note = (throttled: boolean) => {
    recent.push(throttled ? 1 : 0);
    if (recent.length > 200) recent.shift();
    if (recent.length >= 50 && recent.reduce((a, b) => a + b, 0) / recent.length > 0.5)
      tripped = true;
  };

  const obtain = async (
    capture: Capture,
  ): Promise<{ outcome: Outcome; body: Uint8Array | null }> => {
    const row = capture.row;
    const have = row.have_sha256;
    if (!headersOnly && have && existsSync(`${corpus}/pages/blobs/${have}.html.gz`)) {
      const body = gunzipSync(readFileSync(`${corpus}/pages/blobs/${have}.html.gz`));
      const sha = createHash("sha256").update(body).digest("hex");
      if (sha === have)
        return {
          outcome: {
            status: "ok",
            sha256: sha,
            httpStatus: null,
            contentType: null,
            bytes: body.length,
            fetched: "store",
            fetchMs: null,
          },
          body,
        };
    }
    if (tripped)
      return {
        outcome: {
          status: "not_fetched_throttled",
          sha256: null,
          httpStatus: null,
          contentType: null,
          bytes: null,
          fetched: null,
          fetchMs: null,
        },
        body: null,
      };
    const t = performance.now();
    try {
      const member = await fetchWarcRange(row.warc_filename, row.offset, row.length, {
        onThrottle: (status) => {
          throttles[status] = (throttles[status] ?? 0) + 1;
          note(true);
        },
        beforeAttempt: limit_,
        proxy,
      });
      note(false);
      const record = parseWarcRecord(member);
      const ms = Math.round(performance.now() - t);
      const base = {
        sha256: record.sha256,
        httpStatus: record.httpStatus,
        contentType: record.contentType,
        bytes: record.body.length,
        fetched: "cc_warc" as const,
        fetchMs: ms,
        truncated: record.truncated,
        headers: {
          warc_date: record.warcDate,
          warc_headers: record.warcHeaders,
          http_headers: record.httpHeaders,
        },
      };
      if (record.httpStatus !== 200)
        return { outcome: { ...base, status: `http_${record.httpStatus}` }, body: null };
      if (record.contentType && !HTML_TYPES.test(record.contentType))
        return { outcome: { ...base, status: "not_html" }, body: null };
      return {
        outcome: {
          ...base,
          status: record.undone.length ? `ok_undone_${record.undone.join("_")}` : "ok",
        },
        body: record.body,
      };
    } catch (error) {
      const status =
        error instanceof WarcError
          ? `warc_error: ${redact(error.message, proxy)}`
          : `error: ${redact(String(error), proxy).slice(0, 120)}`;
      return {
        outcome: {
          status: status.slice(0, 160),
          sha256: null,
          httpStatus: null,
          contentType: null,
          bytes: null,
          fetched: null,
          fetchMs: Math.round(performance.now() - t),
        },
        body: null,
      };
    }
  };

  const classify = (capture: Capture, sha: string, body: Uint8Array, model: Model) => {
    const t = performance.now();
    const html = decode(body);
    const { document } = parseHTML(html);
    const elements: unknown[] = [];
    const result = classifyDocument(document, capture.row.source_url, {
      model,
      budgetMs,
      html,
      elementsOut: elements,
    });
    const baseHref = (
      document as unknown as {
        querySelector(s: string): { getAttribute(n: string): string | null } | null;
      }
    )
      .querySelector("base[href]")
      ?.getAttribute("href");
    const positions = linkPositions(
      capture.row.source_url,
      result,
      elements,
      capture.links,
      baseHref,
    );
    classifyMs.push(performance.now() - t);
    const counts: Record<string, number> = {};
    for (const p of positions) {
      if (!p.matched) {
        counts.unlocated = (counts.unlocated ?? 0) + 1;
        continue;
      }
      const row = a16Row(p);
      counts[row.section as string] = (counts[row.section as string] ?? 0) + 1;
      linksOut.write(
        `${JSON.stringify({ page_sha256: sha, locator_id: capture.row.locator_id, capture: capture.key, ...row })}\n`,
      );
    }
    const located = positions.filter((p) => p.matched).length;
    pagesOut.write(
      `${JSON.stringify({
        page_sha256: sha,
        locator_id: capture.row.locator_id,
        capture: capture.key,
        n_links_total: located,
        n_links_by_section: counts,
        page_type: result.pageType.label,
        comment_sections: result.commentSections.length,
        degraded: result.degraded,
      })}\n`,
    );
  };

  const finish = (
    capture: Capture,
    outcome: Outcome,
    body: Uint8Array | null,
    refusal: string | null,
  ) => {
    let status = refusal ?? outcome.status;
    if (outcome.headers) {
      headersOut.write(
        `${JSON.stringify({ capture: capture.key, page_sha256: outcome.sha256, ...outcome.headers })}\n`,
      );
    }
    if (body && outcome.sha256 && !headersOnly) {
      const blob = `${outDir}/blobs/${outcome.sha256}.html.gz`;
      if (!existsSync(blob)) writeFileSync(blob, gzipSync(body));
      try {
        classify(capture, outcome.sha256, body, model);
      } catch (error) {
        status = `classify_error: ${String(error).slice(0, 120)}`;
      }
    }
    if (outcome.fetchMs !== null && outcome.fetched === "cc_warc") fetchMs.push(outcome.fetchMs);
    statuses[status.split(":")[0]!] = (statuses[status.split(":")[0]!] ?? 0) + 1;
    const r = capture.row;
    manifest.write(
      `${JSON.stringify({
        page_sha256: outcome.sha256,
        locator_id: r.locator_id,
        locator_ids: capture.locatorIds,
        capture: capture.key,
        url: r.source_url,
        domain_hash: r.domain_hash,
        bucket: r.bucket,
        route: guard.route,
        crawl: r.crawl,
        warc: [r.warc_filename, r.offset, r.length],
        status,
        fetched: outcome.fetched,
        http_status: outcome.httpStatus,
        content_type: outcome.contentType,
        bytes: outcome.bytes,
        truncated: outcome.truncated ?? null,
        n_links_asked: capture.links?.length ?? null,
      })}\n`,
    );
  };

  let next = 0;
  const worker = async () => {
    while (next < captures.length) {
      const capture = captures[next++]!;
      const refusal = routeRefusal(capture.row, guard, sealed, gold);
      if (refusal) {
        finish(
          capture,
          {
            status: refusal,
            sha256: null,
            httpStatus: null,
            contentType: null,
            bytes: null,
            fetched: null,
            fetchMs: null,
          },
          null,
          refusal,
        );
        continue;
      }
      const { outcome, body } = await obtain(capture);
      finish(capture, outcome, body, null);
    }
  };
  await Promise.all(Array.from({ length: concurrency }, worker));
  await Promise.all([manifest.end(), linksOut.end(), pagesOut.end(), headersOut.end()]);

  const round = (v: number | null) => (v === null ? null : Math.round(v));
  const stats = {
    started,
    ended: new Date().toISOString(),
    route: guard.route,
    rows: rows.length,
    captures: captures.length,
    concurrency,
    rate,
    proxy: proxy !== null,
    budgetMs,
    statuses,
    throttles,
    tripped,
    fetchMs: { p50: round(percentile(fetchMs, 50)), p95: round(percentile(fetchMs, 95)) },
    classifyMs: { p50: round(percentile(classifyMs, 50)), p95: round(percentile(classifyMs, 95)) },
  };
  writeFileSync(`${outDir}/stats.json`, `${JSON.stringify(stats, null, 2)}\n`);
  console.log(JSON.stringify(stats));
}

if (import.meta.main) await main();

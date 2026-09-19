#!/usr/bin/env bun
/**
 * Label page type with Jev under the frozen policy revision, at corpus scale.
 *
 * Why this exists: the page-type head trained on schema-derived silver labels
 * does not transfer to pages without schema markup, and 57% of the holdout
 * split has none. Classes with no schema.org source at all — `pricing`,
 * `legal`, `contact`, `docs`, `about` — cannot appear in a silver label, so
 * they collapse onto the nearest prose class. One Jev Choice per page reads the
 * page instead of its markup.
 *
 * The content store is opened **read-only** and is never written to. HTML is
 * rendered in memory, and the emitted row carries no URL and no page text: a
 * pageId, a content hash, the chosen class, the full probability distribution
 * and the provenance that binds the row to a policy revision.
 *
 *   bun --env-file="$TYPESAFE_ENV_FILE" run-pagetype-v7.ts \
 *     --pages <private>/pagetype-jev-v7/queue-holdout.jsonl \
 *     --store /Volumes/BACKUP/squirrel-store/content-store.db \
 *     --output <private>/pagetype-jev-v7/holdout.jsonl \
 *     --policy-revision v7 --concurrency 4
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import { Database } from "bun:sqlite";
import { createHttpTransport } from "./jev-adapter.ts";
import {
  buildPageTypeRequest,
  evaluateWithBackoff,
  loadPolicy,
  pageEvidenceFromHtml,
  PAGE_TYPES_V5,
  POLICY_REVISIONS,
  sha256,
  snapshotHash,
  type PolicyRevision,
} from "./jev-v5.ts";

const configuredRoot = process.env.DOM_CLASSIFIER_DATA_ROOT?.trim();
if (!configuredRoot)
  throw new Error("DOM_CLASSIFIER_DATA_ROOT must point to the private model-data root");
const PRIVATE_ROOT = resolve(configuredRoot);

function privatePath(raw: string): string {
  const path = resolve(raw);
  if (!path.startsWith(`${PRIVATE_ROOT}/`)) throw new Error(`path must be under ${PRIVATE_ROOT}`);
  return path;
}

function argument(name: string, fallback?: string): string {
  const index = Bun.argv.indexOf(name);
  if (index < 0) {
    if (fallback !== undefined) return fallback;
    throw new Error(`missing ${name}`);
  }
  const value = Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

type QueueRow = { pageId: string; contentHash: string; url: string; split: string };

const pagesPath = privatePath(argument("--pages"));
const storePath = resolve(argument("--store"));
const outputPath = privatePath(argument("--output"));
const concurrency = Number(argument("--concurrency", "4"));
const limit = Number(argument("--limit", "0"));
const policyRevision = argument("--policy-revision", "v7") as PolicyRevision;
if (!POLICY_REVISIONS.includes(policyRevision))
  throw new Error(`--policy-revision must be one of ${POLICY_REVISIONS.join(", ")}`);
if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 4)
  throw new Error("--concurrency must be 1..4");

const policy = loadPolicy(policyRevision);
const queue = readFileSync(pagesPath, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line) as QueueRow);

mkdirSync(dirname(outputPath), { recursive: true, mode: 0o700 });
const done = new Set<string>();
if (existsSync(outputPath))
  for (const line of readFileSync(outputPath, "utf8").split("\n").filter(Boolean)) {
    try {
      const row = JSON.parse(line) as { pageId: string; promptRevision: string };
      if (row.promptRevision === policy.revision) done.add(row.pageId);
    } catch {
      /* a half-written line resumes as not done */
    }
  }

// Read-only, and asserted: a labelling run must not be able to mutate the store
// it reads, whatever a later edit to this file does.
const store = new Database(storePath, { readonly: true });
// The page index carries a 16-character prefix of the store's 64-character
// hash, so an equality lookup finds nothing. A half-open range over the hex
// alphabet uses the hash index and, unlike LIKE, cannot be defeated by an
// underscore or percent in the key. Two hits means the prefix is ambiguous and
// the page is skipped rather than guessed at.
const select = store.query<{ hash: string; content: Uint8Array }, [string, string]>(
  "select hash, content from content where hash >= ? and hash < ? limit 2",
);

function resolveContent(prefix: string): { hash: string; html: string } | null {
  const rows = select.all(prefix, `${prefix}g`);
  if (rows.length !== 1) return null;
  const row = rows[0];
  if (!row?.content) return null;
  const bytes = row.content instanceof Uint8Array ? row.content : new Uint8Array(row.content);
  let html: string;
  try {
    html = new TextDecoder().decode(gunzipSync(bytes));
  } catch {
    // Not every blob is gzipped; fall back before deciding it is unreadable.
    try {
      html = new TextDecoder().decode(bytes);
    } catch {
      return null;
    }
  }
  return { hash: row.hash, html };
}

const pending = queue.filter((row) => !done.has(row.pageId));
const work = limit > 0 ? pending.slice(0, limit) : pending;
const requestTimeoutMs = Number(argument("--request-timeout-ms", "45000"));
if (!Number.isInteger(requestTimeoutMs) || requestTimeoutMs < 1000 || requestTimeoutMs > 120_000)
  throw new Error("--request-timeout-ms must be 1000..120000");
const transport = createHttpTransport({ timeoutMs: requestTimeoutMs });
const totals = { labelled: 0, missingHtml: 0, failed: 0, inputTokens: 0, retries: 0 };
const latencies: number[] = [];
const failures: Array<{ pageId: string; error: string }> = [];
let cursor = 0;
const startedAt = Date.now();

async function worker(): Promise<void> {
  for (;;) {
    const index = cursor++;
    if (index >= work.length) return;
    const row = work[index];
    // Everything for this row is inside the try, including reading the store
    // and building the evidence. When those sat outside it, one page that threw
    // — a malformed blob, a document linkedom refuses — rejected the worker's
    // promise and killed that worker silently. With four workers, four such
    // pages stopped the run dead at `Promise.all` with no error and no exit:
    // 1,606 rows in one burst and then nothing, which reads as a stall.
    try {
      const resolved = resolveContent(row.contentHash);
      if (!resolved) {
        totals.missingHtml++;
        continue;
      }
      const evidence = pageEvidenceFromHtml(resolved.html, row.url, row.pageId);
      const built = buildPageTypeRequest(evidence, policy);
      const { response, latencyMs, retries } = await evaluateWithBackoff(transport, built.request);
      const answer = response.answers.page_type as {
        choice?: string;
        probabilities?: Record<string, number>;
        confidence?: number;
      };
      if (typeof answer?.choice !== "string" || !PAGE_TYPES_V5.includes(answer.choice))
        throw new Error("page_type answer is not one of the eleven classes");
      appendFileSync(
        outputPath,
        `${JSON.stringify({
          schemaVersion: 1,
          provider: "typesafe",
          modelId: "jev-1.13.0",
          promptRevision: policy.revision,
          policySha256: policy.sha256,
          pageTypeMappingRevision: "dom-pagetype-v5",
          pageId: row.pageId,
          contentHash: resolved.hash,
          split: row.split,
          // No url and no page text: the row is a label, not a copy of the page.
          inputSha256: sha256(built.packet),
          snapshotHash: snapshotHash(built.request.state),
          evidenceLines: evidence.lines.length,
          pageType: answer.choice,
          confidence: answer.confidence ?? null,
          probabilities: answer.probabilities ?? null,
          usage: response.usage,
          latencyMs,
          createdAt: new Date().toISOString(),
        })}\n`,
        { mode: 0o600 },
      );
      totals.labelled++;
      totals.retries += retries;
      totals.inputTokens += Number(response.usage?.input_tokens ?? 0);
      latencies.push(latencyMs);
    } catch (error) {
      totals.failed++;
      if (failures.length < 20)
        failures.push({
          pageId: row.pageId,
          error: error instanceof Error ? error.message : String(error),
        });
    }
    if ((totals.labelled + totals.failed) % 500 === 0)
      console.log(
        JSON.stringify({
          stage: "progress",
          done: totals.labelled + totals.failed,
          of: work.length,
          ...totals,
          usd: Number(((totals.inputTokens * 42) / 1_000_000_000).toFixed(4)),
        }),
      );
  }
}

await Promise.all(Array.from({ length: Math.min(concurrency, work.length) }, worker));
store.close();
latencies.sort((left, right) => left - right);
console.log(
  JSON.stringify({
    stage: "complete",
    ...totals,
    resumed: done.size,
    usd: Number(((totals.inputTokens * 42) / 1_000_000_000).toFixed(4)),
    inputTokensPerPage: totals.labelled ? Math.round(totals.inputTokens / totals.labelled) : null,
    wallClockMs: Date.now() - startedAt,
    medianLatencyMs: latencies.length ? latencies[Math.floor(latencies.length / 2)] : null,
    p95LatencyMs: latencies.length ? latencies[Math.floor(latencies.length * 0.95)] : null,
    failures,
  }),
);

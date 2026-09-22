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
import { type UnusableHtml, unusableHtmlReason } from "../v2/unusable-html.ts";

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
/** One or more content stores, colon-separated. The corpus expansion crawls
 *  into a SECOND store so the live one is never written to, and a queue row can
 *  be satisfied by either. */
const storePaths = argument("--store")
  .split(":")
  .map((path) => resolve(path.trim()))
  .filter(Boolean);
if (storePaths.length === 0) throw new Error("--store resolved to no paths");
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
const stores = storePaths.map((path) => new Database(path, { readonly: true }));
// The page index carries a 16-character prefix of the store's 64-character
// hash, so an equality lookup finds nothing. A half-open range over the hex
// alphabet uses the hash index and, unlike LIKE, cannot be defeated by an
// underscore or percent in the key. Two hits means the prefix is ambiguous and
// the page is skipped rather than guessed at.
const selects = stores.map((store) =>
  store.query<{ hash: string; content: Uint8Array }, [string, string]>(
    "select hash, content from content where hash >= ? and hash < ? limit 2",
  ),
);

function resolveContent(prefix: string): { hash: string; html: string } | null {
  let row: { hash: string; content: Uint8Array } | undefined;
  for (const select of selects) {
    const rows = select.all(prefix, `${prefix}g`);
    // Ambiguous in ANY store means ambiguous: skip rather than pick one.
    if (rows.length > 1) return null;
    if (rows.length === 1) {
      row = rows[0];
      break;
    }
  }
  if (!row) return null;
  // The content store gzips every blob it writes, an empty body included, so a
  // blob that does not gunzip is a read that went wrong, not a format to guess
  // at. This used to fall back to decoding the raw bytes, which turned an
  // unreadable read into "a page with no element" and hid the real error.
  // Throwing lands it in `failed` with zlib's message, and a resume retries it.
  const bytes = row.content instanceof Uint8Array ? row.content : new Uint8Array(row.content);
  let html: string;
  try {
    html = new TextDecoder().decode(gunzipSync(bytes));
  } catch (error) {
    throw new Error(`unreadable blob: ${error instanceof Error ? error.message : String(error)}`);
  }
  return { hash: row.hash, html };
}

const pending = queue.filter((row) => !done.has(row.pageId));
const work = limit > 0 ? pending.slice(0, limit) : pending;
const requestTimeoutMs = Number(argument("--request-timeout-ms", "45000"));
if (!Number.isInteger(requestTimeoutMs) || requestTimeoutMs < 1000 || requestTimeoutMs > 120_000)
  throw new Error("--request-timeout-ms must be 1000..120000");
const transport = createHttpTransport({ timeoutMs: requestTimeoutMs });
const totals: Record<
  "labelled" | "missingHtml" | UnusableHtml | "noEvidence" | "failed" | "inputTokens" | "retries",
  number
> = {
  labelled: 0,
  missingHtml: 0,
  emptyHtml: 0,
  noDocumentElement: 0,
  noEvidence: 0,
  failed: 0,
  inputTokens: 0,
  retries: 0,
};
const latencies: number[] = [];
const failures: Array<{ pageId: string; error: string }> = [];
/** Every failure by message. `failures` keeps only the first 20, which on
 *  2026-09-22 showed 20 scattered pages and hid 5,365 consecutive ones. */
const failureCounts: Record<string, number> = {};
/**
 * Rows in a row that ended without a model call, as a failure or an unusable
 * page. Unusable pages run at about 0.3% of the corpus and a domain holds at
 * most 25 pages, so a long streak is the process or the store going wrong, not
 * the pages: stop, rather than record thousands of healthy pages as failures
 * (or, worse, skip them as empty) in a few minutes. `missingHtml` neither
 * extends nor breaks it, because a pruned store legitimately drops long runs;
 * the flip side is that a wrong `--store` misses every row without aborting,
 * which costs nothing and shows as `missingHtml` in the complete line. Run a
 * queue built by the current `pagetype-queue.ts`: an older one still holds the
 * unusable pages, and after a resume they sit next to each other.
 */
const MAX_NO_CALL_STREAK = Number(argument("--max-no-call-streak", "200"));
if (!Number.isInteger(MAX_NO_CALL_STREAK) || MAX_NO_CALL_STREAK < 1)
  throw new Error("--max-no-call-streak must be a positive integer");
/** Spend ceiling for THIS run, in USD at the $0.042/M input rate. Workers stop
 *  taking rows once it is reached; the requests already in flight (at most
 *  `--concurrency`) still land, so the overshoot is a few pages. 0 = no cap. */
const MAX_USD = Number(argument("--max-usd", "0"));
if (!Number.isFinite(MAX_USD) || MAX_USD < 0) throw new Error("--max-usd must be >= 0");
const spentUsd = () => (totals.inputTokens * 42) / 1_000_000_000;
let budgetReached = false;
let noCallStreak = 0;
let aborted: string | null = null;
/** Every row a worker finished, whatever the outcome. */
let processed = 0;
function noCall(lastError?: string): void {
  if (++noCallStreak < MAX_NO_CALL_STREAK || aborted) return;
  aborted =
    `${noCallStreak} consecutive rows without a model call ` +
    `(failed ${totals.failed}, emptyHtml ${totals.emptyHtml}, noDocumentElement ` +
    `${totals.noDocumentElement}, noEvidence ${totals.noEvidence})` +
    (lastError ? `, last error: ${lastError}` : "");
}
let cursor = 0;
const startedAt = Date.now();

async function worker(): Promise<void> {
  for (;;) {
    const index = cursor++;
    if (MAX_USD > 0 && spentUsd() >= MAX_USD) budgetReached = true;
    if (aborted || budgetReached || index >= work.length) return;
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
      // An empty body, or bytes with no element in them, is a page with nothing
      // to label: skipped by name, never counted as `failed`. On 2026-09-22 the
      // runner threw on these instead, about 100 pages across the three splits.
      const unusable = unusableHtmlReason(resolved.html);
      if (unusable) {
        totals[unusable]++;
        noCall();
        continue;
      }
      const evidence = pageEvidenceFromHtml(resolved.html, row.url, row.pageId);
      // No prose and no title: nothing for the model to read but the path.
      if (evidence.lines.length === 0 && !evidence.title) {
        totals.noEvidence++;
        noCall();
        continue;
      }
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
      noCallStreak = 0;
      totals.retries += retries;
      totals.inputTokens += Number(response.usage?.input_tokens ?? 0);
      latencies.push(latencyMs);
    } catch (error) {
      totals.failed++;
      const message = error instanceof Error ? error.message : String(error);
      failureCounts[message] = (failureCounts[message] ?? 0) + 1;
      if (failures.length < 20) failures.push({ pageId: row.pageId, error: message });
      noCall(message);
    } finally {
      // In `finally` so a skipped row counts too: `done` reaches `of`, and a
      // long run of skips still reports.
      if (++processed % 500 === 0)
        console.log(
          JSON.stringify({
            stage: "progress",
            done: processed,
            of: work.length,
            ...totals,
            usd: Number(((totals.inputTokens * 42) / 1_000_000_000).toFixed(4)),
          }),
        );
    }
  }
}

await Promise.all(Array.from({ length: Math.min(concurrency, work.length) }, worker));
for (const store of stores) store.close();
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
    aborted,
    budgetReached,
    unprocessed: Math.max(0, work.length - processed),
    failureCounts,
    failures,
  }),
);
// exitCode, not exit(): exit() can cut off the complete line on a piped stdout.
if (aborted) process.exitCode = 1;

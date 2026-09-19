#!/usr/bin/env bun
/**
 * Private runner for prompt revision `dom-suggestions-v5`.
 *
 * Emits the shared packet file first, from the frozen captures alone, so the
 * Luna reviewer can be run from the same bytes whether or not the Jev leg
 * succeeds. Then sends one request per page, bounded concurrency, append-only,
 * resumable, and never overwriting a v4 output.
 *
 *   bun --env-file="$TYPESAFE_ENV_FILE" run-jev-v5.ts \
 *     --selection <private>/pilot-selection.jsonl \
 *     --captures  <private>/final-500/captures \
 *     --packets   <private>/label-pilot-v5/pilot-packets-v5.jsonl \
 *     --output    <private>/label-pilot-v5/jev-v5.jsonl \
 *     --concurrency 4
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { createHttpTransport } from "./jev-adapter.ts";
import {
  buildRequestV5,
  evaluateWithBackoff,
  loadPolicy,
  PROMPT_REVISION_V5,
  sha256,
  snapshotHash,
  toRowsV5,
  type JevRowV5,
} from "./jev-v5.ts";
import type { CapturedNode, CapturedPage } from "../labeler/types.ts";

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

const flag = (name: string) => Bun.argv.includes(name);

type SelectionRow = {
  recordType: "node" | "page";
  split: string;
  pageId: string;
  nodeId: string | null;
  captureHash: string;
  stratum: string;
  v4InputSha256: string;
};

const selectionPath = privatePath(argument("--selection"));
const capturesDir = privatePath(argument("--captures"));
const packetsPath = privatePath(argument("--packets"));
const outputPath = privatePath(argument("--output"));
const concurrency = Number(argument("--concurrency", "4"));
const limit = Number(argument("--limit", "0"));
if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 4)
  throw new Error("--concurrency must be 1..4; the pilot never runs a parallel storm");

const selection = readFileSync(selectionPath, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line) as SelectionRow);

const byPage = new Map<
  string,
  { split: string; nodes: SelectionRow[]; page: SelectionRow | null }
>();
for (const row of selection) {
  const entry = byPage.get(row.pageId) ?? { split: row.split, nodes: [], page: null };
  if (row.recordType === "page") entry.page = row;
  else entry.nodes.push(row);
  byPage.set(row.pageId, entry);
}
const strata = new Map(
  selection.filter((row) => row.nodeId).map((row) => [row.nodeId as string, row.stratum]),
);
const policy = loadPolicy();
mkdirSync(dirname(outputPath), { recursive: true, mode: 0o700 });

const pages = [...byPage.entries()].sort(([left], [right]) => left.localeCompare(right));
const work = limit > 0 ? pages.slice(0, limit) : pages;

// Pass one: packets only, from the frozen captures. Deterministic and offline.
const packetRows: Array<Record<string, unknown>> = [];
const built = new Map<
  string,
  { page: CapturedPage; nodes: CapturedNode[]; includePage: boolean; split: string }
>();
for (const [pageId, entry] of work) {
  const page = JSON.parse(
    readFileSync(join(capturesDir, `${pageId}.json`), "utf8"),
  ) as CapturedPage;
  const byId = new Map(page.nodes.map((node) => [node.id, node]));
  const nodes = entry.nodes
    .map((row) => byId.get(row.nodeId as string))
    .filter((node): node is CapturedNode => !!node);
  if (nodes.length !== entry.nodes.length)
    throw new Error(`capture ${pageId} is missing a selected node`);
  built.set(pageId, { page, nodes, includePage: !!entry.page, split: entry.split });
  const plan = buildRequestV5(page, nodes, policy, { includePage: !!entry.page });
  if (entry.page)
    packetRows.push({
      packetRevision: PROMPT_REVISION_V5,
      recordType: "page",
      split: entry.split,
      pageId,
      nodeId: null,
      captureHash: page.captureHash,
      stratum: "page",
      input: plan.pagePacket,
      inputSha256: sha256(plan.pagePacket),
      v4InputSha256: entry.page.v4InputSha256,
      policySha256: policy.sha256,
      attributesAvailable: false,
    });
  for (const candidate of plan.candidates) {
    const source = entry.nodes.find((row) => row.nodeId === candidate.evidence.nodeId)!;
    packetRows.push({
      packetRevision: PROMPT_REVISION_V5,
      recordType: "node",
      split: entry.split,
      pageId,
      nodeId: candidate.evidence.nodeId,
      captureHash: page.captureHash,
      stratum: source.stratum,
      input: candidate.packet,
      inputSha256: sha256(candidate.packet),
      v4InputSha256: source.v4InputSha256,
      policySha256: policy.sha256,
      attributesAvailable: false,
    });
  }
}
writeFileSync(
  packetsPath,
  packetRows.map((row) => `${JSON.stringify(row, Object.keys(row).sort())}\n`).join(""),
  { mode: 0o600 },
);
console.log(
  JSON.stringify({
    stage: "packets",
    rows: packetRows.length,
    pages: work.length,
    policySha256: policy.sha256,
  }),
);
if (flag("--packets-only")) process.exit(0);

// Pass two: the model leg.
const done = new Set<string>();
if (existsSync(outputPath))
  for (const line of readFileSync(outputPath, "utf8").split("\n").filter(Boolean)) {
    try {
      const row = JSON.parse(line) as JevRowV5;
      if (row.promptRevision === PROMPT_REVISION_V5 && row.policySha256 === policy.sha256)
        done.add(row.pageId);
    } catch {
      /* a partially written line resumes as not done */
    }
  }

const transport = createHttpTransport();
const pending = work.filter(([pageId]) => !done.has(pageId));
const totals = { requests: 0, rows: 0, inputTokens: 0, outputTokens: 0, retries: 0, failed: 0 };
const latencies: number[] = [];
const failures: Array<{ pageId: string; error: string }> = [];
let cursor = 0;
const startedAt = Date.now();

async function worker(): Promise<void> {
  for (;;) {
    const index = cursor++;
    if (index >= pending.length) return;
    const [pageId] = pending[index];
    const entry = built.get(pageId)!;
    const plan = buildRequestV5(entry.page, entry.nodes, policy, {
      includePage: entry.includePage,
    });
    try {
      const { response, latencyMs, retries } = await evaluateWithBackoff(transport, plan.request);
      const rows = toRowsV5(
        entry.page,
        plan,
        response,
        { split: entry.split, strata, policySha256: policy.sha256, latencyMs },
        entry.includePage,
      );
      appendFileSync(outputPath, rows.map((row) => `${JSON.stringify(row)}\n`).join(""), {
        mode: 0o600,
      });
      totals.requests++;
      totals.rows += rows.length;
      totals.retries += retries;
      totals.inputTokens += Number(response.usage?.input_tokens ?? 0);
      totals.outputTokens += Number(response.usage?.output_tokens ?? 0);
      latencies.push(latencyMs);
    } catch (error) {
      totals.failed++;
      failures.push({ pageId, error: error instanceof Error ? error.message : String(error) });
    }
    if ((totals.requests + totals.failed) % 10 === 0)
      console.log(
        JSON.stringify({
          stage: "progress",
          done: totals.requests + totals.failed,
          of: pending.length,
          ...totals,
        }),
      );
  }
}

await Promise.all(Array.from({ length: Math.min(concurrency, pending.length) }, worker));
latencies.sort((left, right) => left - right);
console.log(
  JSON.stringify({
    stage: "complete",
    ...totals,
    resumedPages: done.size,
    wallClockMs: Date.now() - startedAt,
    medianLatencyMs: latencies.length ? latencies[Math.floor(latencies.length / 2)] : null,
    p95LatencyMs: latencies.length ? latencies[Math.floor(latencies.length * 0.95)] : null,
    snapshotSample: pending.length
      ? snapshotHash(
          buildRequestV5(built.get(pending[0][0])!.page, built.get(pending[0][0])!.nodes, policy, {
            includePage: built.get(pending[0][0])!.includePage,
          }).request.state,
        )
      : null,
    failures: failures.slice(0, 10),
  }),
);

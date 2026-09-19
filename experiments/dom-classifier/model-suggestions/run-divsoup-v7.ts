#!/usr/bin/env bun
/**
 * Label the div-soup evaluation sample under the frozen policy revision.
 *
 * The sample is a blind evaluation set for somebody else's model. Two rules
 * follow from that and are enforced here rather than promised:
 *
 * * **`predicted` and `confidence` are never read.** They are the model's own
 *   output; a label produced with them in view would not be an independent
 *   grade. They are not put in a packet, not used to order or stratify, and the
 *   fields are dropped as the sample is parsed. Stratification uses the tag,
 *   the split and the domain, which are facts about the page.
 * * **Nodes are addressed by the featurizer's preorder index**, so the packets
 *   line up with the sample's `nodeId` without this file re-deciding which
 *   elements are candidates.
 *
 * HTML comes from the read-only content store and is rendered in memory.
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import { Database } from "bun:sqlite";
import { shape } from "../extract-dom.ts";
import { parseHTML } from "../../../packages/parser/src/dom.ts";
import { featurizerOrder } from "./featurizer-order.ts";
import { createHttpTransport } from "./jev-adapter.ts";
import {
  buildRequestFromEvidence,
  COMPONENT_TYPES_V5,
  evaluateWithBackoff,
  loadPolicy,
  nodeEvidenceFromHtml,
  pageEvidenceFromHtml,
  POLICY_REVISIONS,
  PURPOSES_V5,
  REGIONS_V5,
  sha256,
  serializeNodePacket,
  snapshotHash,
  type ExtractedCandidate,
  type PageEvidence,
  type PolicyRevision,
} from "./jev-v5.ts";

const configuredRoot = process.env.DOM_CLASSIFIER_DATA_ROOT?.trim();
if (!configuredRoot) throw new Error("DOM_CLASSIFIER_DATA_ROOT must be set");
const PRIVATE_ROOT = resolve(configuredRoot);
const privatePath = (raw: string) => {
  const path = resolve(raw);
  if (!path.startsWith(`${PRIVATE_ROOT}/`)) throw new Error(`path must be under ${PRIVATE_ROOT}`);
  return path;
};
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

const samplePath = privatePath(argument("--sample"));
const storePath = resolve(argument("--store"));
const packetsPath = privatePath(argument("--packets"));
const outputPath = privatePath(argument("--output"));
const policyRevision = argument("--policy-revision", "v7") as PolicyRevision;
const concurrency = Number(argument("--concurrency", "4"));
if (!POLICY_REVISIONS.includes(policyRevision)) throw new Error("bad --policy-revision");
if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 4)
  throw new Error("--concurrency must be 1..4");

const policy = loadPolicy(policyRevision);

/** The sample, with the model's own outputs stripped at the parse boundary. */
type SamplePage = {
  pageId: string;
  contentHash: string;
  url: string;
  domain: string;
  split: string;
  nodeIds: number[];
};
const sample: SamplePage[] = readFileSync(samplePath, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => {
    const row = JSON.parse(line) as {
      pageId: string;
      contentHash: string;
      url: string;
      domain: string;
      split: string;
      candidates: Array<{ nodeId: number }>;
    };
    return {
      pageId: row.pageId,
      contentHash: row.contentHash,
      url: row.url,
      domain: row.domain,
      split: row.split,
      // Only the node index survives. `predicted`, `confidence` and `source`
      // are discarded here so nothing downstream can reach them.
      nodeIds: row.candidates.map((candidate) => candidate.nodeId),
    };
  })
  .sort((left, right) => left.pageId.localeCompare(right.pageId));

const store = new Database(storePath, { readonly: true });
const select = store.query<{ content: Uint8Array }, [string]>(
  "select content from content where hash = ?",
);
function htmlFor(hash: string): string | null {
  const row = select.get(hash);
  if (!row?.content) return null;
  const bytes = row.content instanceof Uint8Array ? row.content : new Uint8Array(row.content);
  try {
    return new TextDecoder().decode(gunzipSync(bytes));
  } catch {
    try {
      return new TextDecoder().decode(bytes);
    } catch {
      return null;
    }
  }
}

mkdirSync(dirname(outputPath), { recursive: true, mode: 0o700 });

// Pass one: packets, offline and deterministic.
type Prepared = {
  page: SamplePage;
  candidates: ExtractedCandidate[];
  nodeIds: number[];
  pageEvidence: PageEvidence;
};
const prepared: Prepared[] = [];
const packetRows: string[] = [];
let missingHtml = 0;
let missingNode = 0;
for (const page of sample) {
  const html = htmlFor(page.contentHash);
  if (!html) {
    missingHtml++;
    continue;
  }
  // The sample's nodeId is the featurizer's preorder index, not a position in
  // extractPage's candidate array: indexing the wrong list resolves about half
  // the sample to a different element without ever failing.
  const { document } = parseHTML(html);
  const order = featurizerOrder(document as never);
  const evidence = pageEvidenceFromHtml(html, page.url, page.pageId);
  const chosen: ExtractedCandidate[] = [];
  const chosenIds: number[] = [];
  for (const nodeId of page.nodeIds) {
    const element = order[nodeId];
    if (!element) {
      missingNode++;
      continue;
    }
    chosen.push(shape(element, page.url) as unknown as ExtractedCandidate);
    chosenIds.push(nodeId);
  }
  if (!chosen.length) continue;
  prepared.push({ page, candidates: chosen, nodeIds: chosenIds, pageEvidence: evidence });
  for (const [index, candidate] of chosen.entries()) {
    const nodeId = `${page.pageId}#${chosenIds[index]}`;
    const packet = serializeNodePacket(nodeEvidenceFromHtml(candidate, nodeId));
    packetRows.push(
      JSON.stringify(
        {
          packetRevision: policy.revision,
          recordType: "node",
          split: page.split,
          pageId: page.pageId,
          nodeId,
          captureHash: page.contentHash,
          // Structural facts only: the model's prediction never reaches here.
          stratum: candidate.tag,
          domain: page.domain,
          input: packet,
          inputSha256: sha256(packet),
          policySha256: policy.sha256,
          attributesAvailable: true,
        },
        Object.keys({
          attributesAvailable: 0, captureHash: 0, domain: 0, input: 0, inputSha256: 0,
          nodeId: 0, packetRevision: 0, pageId: 0, policySha256: 0, recordType: 0, split: 0, stratum: 0,
        }).sort(),
      ),
    );
  }
}
writeFileSync(packetsPath, packetRows.map((row) => `${row}\n`).join(""), { mode: 0o600 });
console.log(
  JSON.stringify({ stage: "packets", pages: prepared.length, nodes: packetRows.length, missingHtml, missingNode }),
);
if (Bun.argv.includes("--packets-only")) process.exit(0);

// Pass two: Jev, one request per page carrying that page's nodes.
const done = new Set<string>();
if (existsSync(outputPath))
  for (const line of readFileSync(outputPath, "utf8").split("\n").filter(Boolean)) {
    try {
      const row = JSON.parse(line) as { pageId: string; promptRevision: string };
      if (row.promptRevision === policy.revision) done.add(row.pageId);
    } catch {
      /* resume treats a partial line as not done */
    }
  }
const work = prepared.filter((item) => !done.has(item.page.pageId));
const transport = createHttpTransport({ timeoutMs: 45_000 });
const totals = { requests: 0, rows: 0, inputTokens: 0, failed: 0, retries: 0 };
const failures: Array<{ pageId: string; error: string }> = [];
const latencies: number[] = [];
let cursor = 0;
const startedAt = Date.now();

async function worker(): Promise<void> {
  for (;;) {
    const index = cursor++;
    if (index >= work.length) return;
    const item = work[index];
    // Inside the try for the same reason as the page-type runner: evidence
    // building that throws outside it kills the worker without a message.
    try {
      const evidences = item.candidates.map((candidate, position) =>
        nodeEvidenceFromHtml(candidate, `${item.page.pageId}#${item.nodeIds[position]}`),
      );
      const built = buildRequestFromEvidence(item.pageEvidence, evidences, policy, {
        includePage: true,
      });
      const { response, latencyMs, retries } = await evaluateWithBackoff(transport, built.request);
      const answers = response.answers as Record<string, Record<string, unknown>>;
      const snapshot = snapshotHash(built.request.state);
      const rows: string[] = [];
      const pageChoice = answers.page_type as { choice?: string } | undefined;
      rows.push(
        JSON.stringify({
          schemaVersion: 1,
          provider: "typesafe",
          modelId: "jev-1.13.0",
          promptRevision: policy.revision,
          policySha256: policy.sha256,
          recordType: "page",
          split: item.page.split,
          pageId: item.page.pageId,
          nodeId: null,
          captureHash: item.page.contentHash,
          stratum: "page",
          input: built.pagePacket,
          inputSha256: sha256(built.pagePacket),
          snapshotHash: snapshot,
          rawAnswers: { page_type: answers.page_type },
          observedAxes: {
            pageType: {
              complete: typeof pageChoice?.choice === "string",
              labels: pageChoice?.choice ?? null,
            },
          },
          usage: response.usage,
          latencyMs,
          createdAt: new Date().toISOString(),
        }),
      );
      for (const candidate of built.candidates) {
        const prefix = `node_${candidate.evidence.nodeId}_`;
        const raw = Object.fromEntries(
          Object.entries(answers).filter(([key]) => key.startsWith(prefix)),
        );
        const choice = answers[`${prefix}component_type`] as { choice?: string } | undefined;
        const noulSet = (kind: string, labels: readonly string[]) =>
          labels.filter((label) => {
            const answer = answers[`${prefix}${kind}${label}`] as { noul?: number } | undefined;
            return typeof answer?.noul === "number" && answer.noul >= 0.75;
          });
        rows.push(
          JSON.stringify({
            schemaVersion: 1,
            provider: "typesafe",
            modelId: "jev-1.13.0",
            promptRevision: policy.revision,
            policySha256: policy.sha256,
            recordType: "node",
            split: item.page.split,
            pageId: item.page.pageId,
            nodeId: candidate.evidence.nodeId,
            captureHash: item.page.contentHash,
            stratum: candidate.evidence.tag,
            input: candidate.packet,
            inputSha256: sha256(candidate.packet),
            snapshotHash: snapshot,
            rawAnswers: raw,
            observedAxes: {
              componentType: {
                complete: typeof choice?.choice === "string",
                labels: choice?.choice ?? null,
              },
              regions: { complete: true, labels: noulSet("region_", REGIONS_V5) },
              purposes: { complete: true, labels: noulSet("purpose_", PURPOSES_V5) },
            },
            createdAt: new Date().toISOString(),
          }),
        );
      }
      appendFileSync(outputPath, rows.map((row) => `${row}\n`).join(""), { mode: 0o600 });
      totals.requests++;
      totals.rows += rows.length;
      totals.retries += retries;
      totals.inputTokens += Number(response.usage?.input_tokens ?? 0);
      latencies.push(latencyMs);
    } catch (error) {
      totals.failed++;
      if (failures.length < 20)
        failures.push({
          pageId: item.page.pageId,
          error: error instanceof Error ? error.message : String(error),
        });
    }
    if ((totals.requests + totals.failed) % 50 === 0)
      console.log(JSON.stringify({ stage: "progress", done: totals.requests + totals.failed, of: work.length, ...totals }));
  }
}

await Promise.all(Array.from({ length: Math.min(concurrency, work.length) }, worker));
store.close();
latencies.sort((left, right) => left - right);
console.log(
  JSON.stringify({
    stage: "complete",
    ...totals,
    usd: Number(((totals.inputTokens * 42) / 1_000_000_000).toFixed(4)),
    wallClockMs: Date.now() - startedAt,
    medianLatencyMs: latencies.length ? latencies[Math.floor(latencies.length / 2)] : null,
    failures,
  }),
);

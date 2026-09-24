#!/usr/bin/env bun
/**
 * Label page elements with their section purpose (section-purpose-v2) with Jev.
 *
 * Only a SHORTLIST of each page's featurizer candidates is asked about, never
 * every node: the nodes a free-label source already flagged (to confirm or
 * correct it), the nodes whose class/id/aria tokens hint at one of the
 * purposes (to find the unmarked ones), and two deterministic random
 * candidates (so `none` is sampled too). One request per page carries all of
 * that page's shortlisted nodes against one short policy.
 *
 * Stores are read-only; the output row carries no url and no page text.
 *
 *   bun --env-file=<typesafe.env> run-section-v2.ts --pages <queue.jsonl> \
 *     --labels <node-labels.jsonl> --store <a.db>:<b.db> --output <out.jsonl> \
 *     --max-usd 10 --max-no-call-streak 200 --concurrency 4 [--dry-run]
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import { Database } from "bun:sqlite";

import { shape } from "../extract-dom.ts";
import { parseHTML } from "../../../packages/parser/src/dom.ts";
import { featurizeDocument } from "../v2/featurize.ts";
import { featurizerOrder } from "./featurizer-order.ts";
import { createHttpTransport } from "./jev-adapter.ts";
import type { JevQuestion, JevRequest } from "./jev-adapter.ts";
import {
  evaluateWithBackoff,
  JEV_MODEL_V5,
  nodeEvidenceFromHtml,
  pageEvidenceFromHtml,
  serializeNodePacket,
  serializePagePacket,
  sha256,
  type ExtractedCandidate,
} from "./jev-v5.ts";

const configuredRoot = process.env.DOM_CLASSIFIER_DATA_ROOT?.trim();
if (!configuredRoot)
  throw new Error("DOM_CLASSIFIER_DATA_ROOT must point to the private data root");
const PRIVATE_ROOT = resolve(configuredRoot);
function privatePath(raw: string): string {
  const path = resolve(raw);
  if (!path.startsWith(`${PRIVATE_ROOT}/`)) throw new Error(`path must be under ${PRIVATE_ROOT}`);
  return path;
}
function argument(name: string, fallback?: string): string {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? fallback : Bun.argv[index + 1];
  if (value === undefined || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

const LABELS_DIR = join(import.meta.dir, "..", "v2", "labels");
/** v2 = tightest container; v3 = region level (see policy-section-v3.md). The
 *  class list is shared. */
const policyRevision = Bun.argv.includes("--policy-revision")
  ? Bun.argv[Bun.argv.indexOf("--policy-revision") + 1]!
  : "v2";
if (!["v2", "v3"].includes(policyRevision)) throw new Error("--policy-revision must be v2 or v3");
const policyText = readFileSync(join(LABELS_DIR, `policy-section-${policyRevision}.md`), "utf8");
const POLICY = {
  revision: `section-purpose-${policyRevision}`,
  sha256: sha256(policyText),
  text: policyText,
};
export const SECTION_CLASSES = (
  JSON.parse(readFileSync(join(LABELS_DIR, "section-classes-v2.json"), "utf8")) as {
    classes: Array<{ name: string }>;
  }
).classes.map((c) => c.name);

/** Known ad-network markup (mirrors corpus/section-labels.ts AD_TOKENS): every
 *  output row says whether the node or anything inside it carries one, so the ad
 *  eval can report recall on ads WITHOUT a memorisable network token. */
const AD_TOKENS =
  /adsbygoogle|div-gpt-ad|gpt-ad|google_ads|googlead|taboola|outbrain|doubleclick|adsense|ad-slot|adslot|dfp-ad|ezoic|adthrive|mediavine/i;
/** Tokens that hint at a purpose, for the shortlist only (never a feature). */
const HINT =
  /\b(ad|ads|advert|sponsor|banner|promo|share|social|related|recommend|comment|disqus|chat|cookie|consent|gdpr|newsletter|subscribe|signup|login|signin|account|lang|language|currency|locale|toc|contents|pagination|pager|review|rating|testimonial|faq|accordion|breadcrumb|author|byline|price|cart|offer|video|player|event|recipe|address|contact)/i;

const pagesPath = privatePath(argument("--pages"));
const labelsPath = privatePath(argument("--labels"));
const outputPath = privatePath(argument("--output"));
const storePaths = argument("--store")
  .split(":")
  .filter(Boolean)
  .map((p) => resolve(p));
const maxNodes = Number(argument("--max-nodes-per-page", "10"));
const concurrency = Number(argument("--concurrency", "4"));
const maxUsd = Number(argument("--max-usd", "0"));
const maxStreak = Number(argument("--max-no-call-streak", "200"));
const limit = Number(argument("--limit", "0"));
const dryRun = Bun.argv.includes("--dry-run");

type QueueRow = { pageId: string; contentHash: string; url: string; split: string };
const queue = readFileSync(pagesPath, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((l) => JSON.parse(l) as QueueRow);
const flagged = new Map<string, Set<number>>();
for (const line of readFileSync(labelsPath, "utf8").split("\n")) {
  if (!line.startsWith('{"p"') || !line.includes('"n"')) continue;
  const row = JSON.parse(line) as { p: string; n?: number };
  if (row.n === undefined) continue;
  let set = flagged.get(row.p);
  if (!set) flagged.set(row.p, (set = new Set()));
  set.add(row.n);
}

const stores = storePaths.map((p) => new Database(p, { readonly: true }));
const selects = stores.map((db) =>
  db.query<{ content: Uint8Array }, [string, string]>(
    "select content from content where hash >= ? and hash < ? limit 2",
  ),
);
function htmlFor(prefix: string): string | null {
  for (const select of selects) {
    const rows = select.all(prefix, `${prefix}g`);
    if (rows.length !== 1) continue;
    try {
      return new TextDecoder().decode(gunzipSync(rows[0]!.content));
    } catch (error) {
      throw new Error(`unreadable blob: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  return null;
}

type El = Element & { getAttribute(name: string): string | null };
/** Deterministic: the same page always yields the same shortlist. */
function shortlist(pageId: string, candidates: number[], order: El[]): number[] {
  const chosen: number[] = [];
  const add = (n: number) => {
    if (chosen.length < maxNodes && !chosen.includes(n)) chosen.push(n);
  };
  for (const n of [...(flagged.get(pageId) ?? [])].sort((a, b) => a - b))
    if (candidates.includes(n)) add(n);
  for (const n of candidates) {
    const el = order[n];
    if (!el) continue;
    const hint = `${el.getAttribute("class") ?? ""} ${el.getAttribute("id") ?? ""} ${el.getAttribute("aria-label") ?? ""}`;
    if (HINT.test(hint)) add(n);
  }
  const ranked = [...candidates].sort(
    (a, b) =>
      Number(BigInt(Bun.hash(`${pageId}:${a}`)) % 1000n) -
      Number(BigInt(Bun.hash(`${pageId}:${b}`)) % 1000n),
  );
  for (const n of ranked.slice(0, 2)) if (chosen.length < maxNodes + 2) chosen.push(n);
  return [...new Set(chosen)];
}

function buildRequest(
  pageEvidence: ReturnType<typeof pageEvidenceFromHtml>,
  nodes: Array<{ nodeId: string; packet: string }>,
): JevRequest {
  const questions: Record<string, JevQuestion> = {};
  for (const node of nodes)
    questions[`node_${node.nodeId}_section`] = {
      type: "choice",
      instructions: {
        node: node.nodeId,
        judgment:
          policyRevision === "v2"
            ? "What is this element for? Choose exactly one section purpose from `policy.text`; label it only when the element is the tightest container of that whole section, otherwise `none`."
            : "What is this element for? Choose exactly one section purpose from `policy.text`: label it when its content IS that section (the section, a wrapper around only it, or its main list); answer `none` for a fragment of a section or an element with substantial unrelated content.",
      },
      criteria: Object.fromEntries(
        SECTION_CLASSES.map((label) => [
          label,
          // Short on purpose: the definitions live once in policy.text, and a
          // full sentence here is repeated for every class of every node.
          `policy.text: ${label}`,
        ]),
      ),
    };
  const state = {
    policy: POLICY,
    page: { pageId: pageEvidence.pageId, packet: serializePagePacket(pageEvidence) },
    candidates: nodes,
  };
  return { model: JEV_MODEL_V5, state: state as unknown as JevRequest["state"], questions };
}

mkdirSync(dirname(outputPath), { recursive: true, mode: 0o700 });
const done = new Set<string>();
if (existsSync(outputPath))
  for (const line of readFileSync(outputPath, "utf8").split("\n").filter(Boolean)) {
    try {
      done.add((JSON.parse(line) as { pageId: string }).pageId);
    } catch {
      /* a partial line resumes as not done */
    }
  }
const pending = queue.filter((row) => !done.has(row.pageId));
const work = limit > 0 ? pending.slice(0, limit) : pending;
const transport = dryRun ? null : createHttpTransport({ timeoutMs: 60_000 });
const totals = {
  pages: 0,
  nodes: 0,
  missingHtml: 0,
  failed: 0,
  inputTokens: 0,
  packetChars: 0,
  retries: 0,
};
const failureCounts: Record<string, number> = {};
let streak = 0;
let aborted: string | null = null;
let budgetReached = false;
let cursor = 0;
const usd = () => (totals.inputTokens * 42) / 1_000_000_000;

async function worker(): Promise<void> {
  for (;;) {
    if (maxUsd > 0 && usd() >= maxUsd) budgetReached = true;
    const index = cursor++;
    if (aborted || budgetReached || index >= work.length) return;
    const row = work[index]!;
    try {
      const html = htmlFor(row.contentHash);
      if (!html) {
        totals.missingHtml++;
        continue;
      }
      const { document } = parseHTML(html);
      const candidates = featurizeDocument(document as never, row.url).nodes.map((n) => n.nodeId);
      const order = featurizerOrder(document as never) as unknown as El[];
      const ids = shortlist(row.pageId, candidates, order);
      if (!ids.length) continue;
      const nodes = ids.map((n) => {
        const nodeId = `${row.pageId}#${n}`;
        const candidate = shape(order[n] as never, row.url) as unknown as ExtractedCandidate;
        const el = order[n] as unknown as El & { querySelectorAll(s: string): ArrayLike<El> };
        const inside = [el, ...Array.from(el.querySelectorAll("[class], [id]"))];
        const adToken = inside.some((e) =>
          AD_TOKENS.test(`${e.getAttribute("class") ?? ""} ${e.getAttribute("id") ?? ""}`),
        );
        return {
          nodeId,
          n,
          adToken,
          packet: serializeNodePacket(nodeEvidenceFromHtml(candidate, nodeId)),
        };
      });
      const pageEvidence = pageEvidenceFromHtml(html, row.url, row.pageId);
      const request = buildRequest(pageEvidence, nodes);
      totals.packetChars += JSON.stringify(request.state).length;
      totals.nodes += nodes.length;
      totals.pages++;
      if (dryRun) continue;
      const { response, latencyMs, retries } = await evaluateWithBackoff(transport!, request);
      const answers = response.answers as Record<
        string,
        { choice?: string; probabilities?: Record<string, number>; confidence?: number }
      >;
      const inputTokens = Number(response.usage?.input_tokens ?? 0);
      totals.inputTokens += inputTokens;
      totals.retries += retries;
      streak = 0;
      const lines = nodes.map((node) => {
        const answer = answers[`node_${node.nodeId}_section`];
        return JSON.stringify({
          schemaVersion: 1,
          provider: "typesafe",
          modelId: JEV_MODEL_V5,
          promptRevision: POLICY.revision,
          policySha256: POLICY.sha256,
          pageId: row.pageId,
          nodeId: node.n,
          split: row.split,
          adToken: node.adToken,
          section: answer?.choice ?? null,
          confidence: answer?.confidence ?? null,
          probabilities: answer?.probabilities ?? null,
          inputSha256: sha256(node.packet),
          requestInputTokens: inputTokens,
          nodesInRequest: nodes.length,
          latencyMs,
          createdAt: new Date().toISOString(),
        });
      });
      appendFileSync(outputPath, `${lines.join("\n")}\n`, { mode: 0o600 });
    } catch (error) {
      totals.failed++;
      const message = error instanceof Error ? error.message : String(error);
      failureCounts[message] = (failureCounts[message] ?? 0) + 1;
      if (++streak >= maxStreak)
        aborted = `${streak} consecutive failed requests, last: ${message}`;
    }
    if ((totals.pages + totals.failed) % 250 === 0)
      console.log(JSON.stringify({ stage: "progress", ...totals, usd: Number(usd().toFixed(4)) }));
  }
}

await Promise.all(Array.from({ length: Math.min(concurrency, Math.max(work.length, 1)) }, worker));
for (const store of stores) store.close();
console.log(
  JSON.stringify({
    stage: "complete",
    ...totals,
    usd: Number(usd().toFixed(4)),
    resumed: done.size,
    tokensPerNode: totals.nodes ? Math.round(totals.inputTokens / totals.nodes) : null,
    packetCharsPerNode: totals.nodes ? Math.round(totals.packetChars / totals.nodes) : null,
    aborted,
    budgetReached,
    failureCounts,
  }),
);
if (aborted) process.exitCode = 1;

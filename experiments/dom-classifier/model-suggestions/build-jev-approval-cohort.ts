#!/usr/bin/env bun
/**
 * A human **approval** cohort from the Jev v7 div-soup labels.
 *
 * The workflow is approve-or-reject one suggestion, not annotate from scratch:
 * a person reads the node's evidence and the boundary rule that decides its
 * class, then presses Enter to approve, X to reject, or edits. That is cheaper
 * per judgment than free annotation and it is the judgment the labels actually
 * need, because Jev is already right most of the time and the question is where
 * it is not.
 *
 * What the queue is, and is not:
 *
 * * **`unknown` is excluded from the queue and counted in the summary.** An
 *   abstention has nothing to approve; asking a person to confirm one spends
 *   attention and returns no label.
 * * **`layout_container` is capped at 25%.** It is 45% of the raw labels, so an
 *   uncapped sample would spend nearly half the session on the class the tag
 *   rule already answers.
 * * The rarer classes are taken in full up to that cap, because they are the
 *   ones with no support in any split.
 *
 * The labeller mounts a cohort as a directory holding `captures/` and
 * `model-suggestions.jsonl`, so both are written here in exactly the shapes its
 * validators accept. Its ids are narrower than the div-soup corpus's, so page
 * and node ids are rederived deterministically from the content hash and the
 * featurizer index; `provenance.json` records the mapping both ways.
 */
import { createHash, randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import { Database } from "bun:sqlite";
import { parseHTML } from "../../../packages/parser/src/dom.ts";
import { shape } from "../extract-dom.ts";
import { featurizerOrder } from "./featurizer-order.ts";
import { loadPolicy, nodeEvidenceFromHtml, type ExtractedCandidate } from "./jev-v5.ts";

const PRIVATE_ROOT = resolve(process.env.DOM_CLASSIFIER_DATA_ROOT ?? "");
if (!PRIVATE_ROOT) throw new Error("DOM_CLASSIFIER_DATA_ROOT must be set");
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

const labelsPath = privatePath(argument("--labels"));
const samplePath = privatePath(argument("--sample"));
const disagreementsPath = argument("--disagreements", "");
const outputDir = privatePath(argument("--output"));
const storePath = resolve(argument("--store"));
const target = Number(argument("--target", "300"));
const containerShare = Number(argument("--container-share", "0.25"));

const policy = loadPolicy("v7");
const hex = (value: string, length: number) =>
  createHash("sha256").update(value).digest("hex").slice(0, length);
/** The labeller's ids are `page_<24 hex>` and `node_<24 hex>`. */
const pageIdFor = (contentHash: string) => `page_${hex(`page\0${contentHash}`, 24)}`;
const nodeIdFor = (pageId: string, index: number) => `node_${hex(`${pageId}\0${index}`, 24)}`;

type JevRow = {
  recordType: "node" | "page";
  pageId: string;
  nodeId: string | null;
  captureHash: string;
  split: string;
  stratum: string;
  input: string;
  observedAxes: Record<string, { complete: boolean; labels: string[] | string | null }>;
  rawAnswers: Record<string, unknown>;
  snapshotHash: string;
  policySha256: string;
  promptRevision: string;
};

const labels = readFileSync(labelsPath, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line) as JevRow);
const nodeLabels = labels.filter((row) => row.recordType === "node");
const pageTypeByPage = new Map(
  labels
    .filter((row) => row.recordType === "page")
    .map((row) => [row.pageId, row.observedAxes.pageType?.labels as string | null]),
);

/** The source sample, read for page identity only; predictions are never touched. */
const sample = new Map(
  readFileSync(samplePath, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as { pageId: string; contentHash: string; url: string; domain: string; split: string })
    .map((row) => [row.pageId, { contentHash: row.contentHash, url: row.url, domain: row.domain, split: row.split }]),
);

const componentOf = (row: JevRow) => {
  const value = row.observedAxes.componentType?.labels;
  return typeof value === "string" ? value : Array.isArray(value) ? value[0] : null;
};
const order = (row: JevRow) => hex(`${row.pageId}\0${row.nodeId}`, 40);

// Stratify: cap the dominant class, take the rest in deterministic order.
const byClass = new Map<string, JevRow[]>();
let unknownCount = 0;
for (const row of [...nodeLabels].sort((left, right) => (order(left) < order(right) ? -1 : 1))) {
  const component = componentOf(row);
  if (!component) continue;
  if (component === "unknown") {
    unknownCount++;
    continue;
  }
  byClass.set(component, [...(byClass.get(component) ?? []), row]);
}
const containerCap = Math.round(target * containerShare);
const selected: JevRow[] = [];
const quotaUsed: Record<string, number> = {};
for (const [component, rows] of [...byClass.entries()].sort((left, right) => left[0].localeCompare(right[0]))) {
  const cap = component === "layout_container" ? containerCap : rows.length;
  const taken = rows.slice(0, cap);
  quotaUsed[component] = taken.length;
  selected.push(...taken);
}
// Trim or extend to the target without ever exceeding the container cap.
selected.sort((left, right) => (order(left) < order(right) ? -1 : 1));
let queue = selected.slice(0, target);
if (queue.length < target) {
  const seen = new Set(queue.map((row) => `${row.pageId}\0${row.nodeId}`));
  for (const row of selected) {
    if (queue.length >= target) break;
    if (!seen.has(`${row.pageId}\0${row.nodeId}`)) queue.push(row);
  }
}

// The 10 Jev-versus-reviewer disagreements go first, in 1/2/X mode.
type Disagreement = { pageId: string; nodeId: string; jev: Record<string, unknown>; reviewer: Record<string, unknown>; category: string; boundary: string };
const disagreements: Disagreement[] = disagreementsPath
  ? readFileSync(privatePath(disagreementsPath), "utf8")
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as Disagreement)
  : [];
const disagreementKeys = new Set(disagreements.map((row) => `${row.pageId}\0${row.nodeId}`));
queue = [
  ...nodeLabels.filter((row) => disagreementKeys.has(`${row.pageId}\0${row.nodeId}`)),
  ...queue.filter((row) => !disagreementKeys.has(`${row.pageId}\0${row.nodeId}`)),
];

/** The v7 text that decides this class, so the reviewer reads the rule, not a label name. */
function boundaryText(component: string): string {
  const sections: Array<[string, RegExp]> = [
    ["layout_container", /\*\*`<footer>` and `<header>` elements[\s\S]*?```/],
    ["navigation_menu", /\*\*`navigation_menu`\*\*[\s\S]*?```[\s\S]*?```/],
    ["card", /\*\*`card` versus `list`[\s\S]*?```[\s\S]*?```/],
    ["list", /\*\*`card` versus `list`[\s\S]*?```[\s\S]*?```/],
    ["hero", /\*\*`hero`\*\*[\s\S]*?```[\s\S]*?```/],
    ["article", /\*\*`article`\*\*[\s\S]*?```[\s\S]*?```/],
    ["banner", /\*\*`banner`\*\*[\s\S]*?```[\s\S]*?```/],
    ["image", /### Elements with no text[\s\S]*?```[\s\S]*?```/],
    ["media", /### Elements with no text[\s\S]*?```[\s\S]*?```/],
    ["content_section", /### `content_section` versus `layout_container`[\s\S]*?```[\s\S]*?```/],
  ];
  const found = sections.find(([name]) => name === component)?.[1];
  const text = found ? policy.text.match(found)?.[0] : undefined;
  return (text ?? policy.text.match(/### Precedence[\s\S]*?ladder above[\s\S]*?\./)?.[0] ?? "").trim();
}

const store = new Database(storePath, { readonly: true });
const select = store.query<{ content: Uint8Array }, [string]>(
  "select content from content where hash = ?",
);

mkdirSync(join(outputDir, "captures"), { recursive: true, mode: 0o700 });
const suggestions: string[] = [];
const queueRows: string[] = [];
const provenance: Array<Record<string, unknown>> = [];
const byPage = new Map<string, JevRow[]>();
for (const row of queue) byPage.set(row.pageId, [...(byPage.get(row.pageId) ?? []), row]);

let written = 0;
for (const [sourcePageId, rows] of byPage) {
  const meta = sample.get(sourcePageId);
  if (!meta) continue;
  const blob = select.get(meta.contentHash);
  if (!blob?.content) continue;
  const bytes = blob.content instanceof Uint8Array ? blob.content : new Uint8Array(blob.content);
  let html: string;
  try {
    html = new TextDecoder().decode(gunzipSync(bytes));
  } catch {
    html = new TextDecoder().decode(bytes);
  }
  const { document } = parseHTML(html);
  const elements = featurizerOrder(document as never);
  const pageId = pageIdFor(meta.contentHash);
  const captureHash = `sha256:${createHash("sha256").update(meta.contentHash).digest("hex")}`;
  const nodes: Array<Record<string, unknown>> = [];
  for (const row of rows) {
    const index = Number(String(row.nodeId).split("#")[1]);
    const element = elements[index];
    if (!element) continue;
    const candidate = shape(element, meta.url) as unknown as ExtractedCandidate;
    const evidence = nodeEvidenceFromHtml(candidate, String(row.nodeId));
    const nodeId = nodeIdFor(pageId, index);
    nodes.push({
      id: nodeId,
      parentId: null,
      tag: evidence.tag,
      role: evidence.role,
      text: evidence.text.slice(0, 280),
      selector: candidate.locator ?? "",
      rect: { x: 0, y: 0, width: 0, height: 0 },
      depth: evidence.depth,
      suggestion: null,
    });
    const component = componentOf(row);
    const isDisagreement = disagreementKeys.has(`${row.pageId}\0${row.nodeId}`);
    suggestions.push(
      JSON.stringify({
        schemaVersion: 1,
        id: `msug_${randomUUID()}`,
        pageId,
        nodeId,
        captureHash,
        provider: "typesafe",
        modelId: "jev-1.13.0",
        modelRevision: "jev-1.13.0",
        promptRevision: row.promptRevision,
        taxonomyRevision: "dom-taxonomy-v2",
        snapshotHash: row.snapshotHash,
        rawAnswers: row.rawAnswers,
        provisional: true,
        createdAt: new Date().toISOString(),
        mappedLabels: {
          componentType: component,
          regions: row.observedAxes.regions?.labels ?? [],
          purposes: row.observedAxes.purposes?.labels ?? [],
        },
      }),
    );
    queueRows.push(
      JSON.stringify(
        {
          mode: isDisagreement ? "1/2/X" : "approve/reject",
          order: isDisagreement ? 0 : 1,
          pageId,
          nodeId,
          sourcePageId: row.pageId,
          sourceNodeId: row.nodeId,
          split: row.split,
          tag: evidence.tag,
          suggestion: component,
          pageType: pageTypeByPage.get(row.pageId) ?? null,
          evidence: row.input,
          boundaryRule: boundaryText(component ?? ""),
          choices: isDisagreement
            ? { "1": "jev", "2": "reviewer", X: "neither" }
            : { Enter: "approve", X: "reject", edit: "choose another class" },
        },
        [
          "boundaryRule", "choices", "evidence", "mode", "nodeId", "order", "pageId",
          "pageType", "sourceNodeId", "sourcePageId", "split", "suggestion", "tag",
        ],
      ),
    );
    provenance.push({ pageId, nodeId, sourcePageId: row.pageId, sourceNodeId: row.nodeId, featurizerIndex: index });
    written++;
  }
  if (!nodes.length) continue;
  const capture = {
    id: pageId,
    url: meta.url,
    title: (/<title[^>]*>([\s\S]{0,200}?)<\/title>/i.exec(html)?.[1] ?? "").replace(/\s+/g, " ").trim(),
    capturedAt: new Date().toISOString(),
    contentHash: captureHash,
    captureHash,
    width: 1440,
    height: 900,
    viewport: { width: 1440, height: 900, deviceScaleFactor: 1 },
    nodes,
    screenshotUrl: "",
    split: "training-review",
  };
  writeFileSync(join(outputDir, "captures", `${pageId}.json`), `${JSON.stringify(capture)}\n`, {
    mode: 0o600,
  });
}
store.close();

writeFileSync(join(outputDir, "model-suggestions.jsonl"), suggestions.map((row) => `${row}\n`).join(""), { mode: 0o600 });
queueRows.sort((left, right) => (JSON.parse(left).order as number) - (JSON.parse(right).order as number));
writeFileSync(join(outputDir, "queue.jsonl"), queueRows.map((row) => `${row}\n`).join(""), { mode: 0o600 });
writeFileSync(
  join(outputDir, "provenance.json"),
  `${JSON.stringify({ policyRevision: policy.revision, policySha256: policy.sha256, idMapping: provenance }, null, 2)}\n`,
  { mode: 0o600 },
);
const distribution: Record<string, number> = {};
for (const row of queue) {
  const component = componentOf(row) ?? "none";
  distribution[component] = (distribution[component] ?? 0) + 1;
}
writeFileSync(
  join(outputDir, "cohort.json"),
  `${JSON.stringify(
    {
      cohortId: "divsoup-jev-approval-v7",
      createdAt: new Date().toISOString(),
      format: "Single-suggestion approval. Enter approves, X rejects, edit picks another class. The Jev-vs-reviewer disagreements come first in 1/2/X mode.",
      promptRevision: policy.revision,
      policySha256: policy.sha256,
      suggestions: written,
      target,
      disagreementsFirst: disagreements.length,
      unknownExcluded: unknownCount,
      containerCap,
      distribution,
      quotaUsed,
      note: "Suggestions are Jev's, unreviewed. Approving one makes it a human label; rejecting leaves the node unlabelled rather than guessing.",
    },
    null,
    2,
  )}\n`,
  { mode: 0o600 },
);
console.log(JSON.stringify({ suggestions: written, unknownExcluded: unknownCount, containerCap, distribution }));

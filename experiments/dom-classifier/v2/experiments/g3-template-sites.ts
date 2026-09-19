// G3: can the region head NAME the roots the deterministic grouping promotes?
//
// The deterministic component grouping forms `repeated-subtree` fix groups on
// template sites but can only call them "one shared block in the main content".
// This runs the classifier over the stored crawls of those same sites, maps
// each group's promoted root to a classified node, and compares the model's
// name with what the root actually is according to the inspection table.
//
// Strictly read-only and offline: stored audits, no refetch, no integration
// into packages/. Every path is an argument; no hostname or page content is
// written anywhere but the private output directory.
//
// Usage:
//   bun run g3-template-sites.ts \
//     --store <dir containing .squirrel>  --inspection <group-inspection.json> \
//     --artifact <classifier-v2.json>     --out <private dir> [--site <name>]
//
// A second store (the fresh squirrelscan.com crawl) is passed as another
// --store; sites are read from whatever project databases each store holds.

import { Database } from "bun:sqlite";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { featurizeDocument, SKIP_SUBTREE, type NodeFeatures } from "../featurize.ts";
import { classifyDocument, tryLoadModel, type NodePrediction } from "../runtime/predict.ts";

// ---------------------------------------------------------------------------
// Node identity
// ---------------------------------------------------------------------------

interface ElementLike {
  nodeType: number;
  localName?: string;
  nodeName: string;
  firstChild: ElementLike | null;
  nextSibling: ElementLike | null;
  textContent: string | null;
  getAttribute(name: string): string | null;
}

/**
 * Reproduce the featurizer's preorder element numbering, so an element found
 * by any other means can be named by the same `nodeId` a prediction carries.
 *
 * This mirrors `walkDocument`'s traversal exactly: same skip set, same
 * children-in-document-order DFS. It is a mirror rather than a re-export
 * because the featurizer does not retain element references — it deliberately
 * keeps only numbers — and an experiment should not change that.
 */
function indexElements(root: ElementLike): Map<ElementLike, number> {
  const out = new Map<ElementLike, number>();
  const stack: ElementLike[] = [root];
  let index = 0;
  while (stack.length > 0) {
    const el = stack.pop()!;
    const tag = (el.localName ?? el.nodeName.toLowerCase()).toLowerCase();
    if (SKIP_SUBTREE.has(tag)) continue;
    out.set(el, index++);
    const children: ElementLike[] = [];
    for (let child = el.firstChild; child; child = child.nextSibling) {
      if (child.nodeType !== 1) continue;
      const childTag = (child.localName ?? child.nodeName.toLowerCase()).toLowerCase();
      if (SKIP_SUBTREE.has(childTag)) continue;
      children.push(child);
    }
    for (let i = children.length - 1; i >= 0; i--) stack.push(children[i]!);
  }
  return out;
}

/** linkedom throws from `document.title` on a document with no
 *  documentElement, the same way it does from `document.body`. */
function safeText(read: () => string | null | undefined): string {
  try {
    return (read() ?? "").trim().slice(0, 120);
  } catch {
    return "";
  }
}

/**
 * Terra's `textLength` is the element's text with whitespace RUNS collapsed to
 * one space and the ends trimmed -- the same normalisation the featurizer uses.
 *
 * It is not the fully-squashed length. The first draft assumed squashed, and
 * matched 3 of 10 roots: the recorded 3565 for the pushengage nav is its spaced
 * length (squashed is 3129). The vibeadventures textSample hid this, because
 * that menu is concatenated link labels with no spaces in it either way.
 */
function normalizedText(el: ElementLike): string {
  return (el.textContent ?? "").replace(/\s+/g, " ").trim();
}

function classKey(value: string | null): string {
  return (value ?? "").trim().split(/\s+/).filter(Boolean).join(".");
}

interface RootSpec {
  tag: string;
  class: string;
  textLength: number;
  textSample: string;
}

/**
 * Find the element a group's root record describes.
 *
 * Tag and class are required; among the survivors the closest squashed-text
 * length wins, with the text sample as a tie-break. Returns null rather than a
 * guess when nothing is within 5% of the recorded length — a wrong root would
 * silently turn this whole experiment into noise.
 */
function findRoot(
  index: Map<ElementLike, number>,
  spec: RootSpec,
): { el: ElementLike; nodeId: number; lengthDelta: number } | null {
  const wantedClass = spec.class;
  const wantedSample = spec.textSample.replace(/\s+/g, " ").trim().slice(0, 60);
  // Rank by (text sample matched, then closeness of length). Kept as two
  // fields rather than one fused score, because a fused score is how the first
  // draft of this ended up comparing a sentinel against a distance.
  let best: { el: ElementLike; nodeId: number; delta: number; sampleHit: boolean } | null = null;
  for (const [el, nodeId] of index) {
    const tag = (el.localName ?? el.nodeName.toLowerCase()).toLowerCase();
    if (tag !== spec.tag) continue;
    if (wantedClass && classKey(el.getAttribute("class")) !== wantedClass) continue;
    const text = normalizedText(el);
    const delta = Math.abs(text.length - spec.textLength);
    const sampleHit = wantedSample.length === 0 || text.slice(0, 60) === wantedSample;
    const better = best === null ||
      (sampleHit && !best.sampleHit) ||
      (sampleHit === best.sampleHit && delta < best.delta);
    if (better) best = { el, nodeId, delta, sampleHit };
  }
  if (best === null) return null;
  const tolerance = Math.max(20, spec.textLength * 0.05);
  if (best.delta > tolerance) return null;
  return { el: best.el, nodeId: best.nodeId, lengthDelta: best.delta };
}

/** The classified node for an element: itself, else nearest ancestor, else the
 *  largest descendant candidate. Which one is recorded, never hidden. */
function nearestCandidate(
  nodeId: number,
  nodes: NodeFeatures[],
  predictions: NodePrediction[],
  parentOf: Map<number, number>,
): { prediction: NodePrediction; relation: "self" | "ancestor" | "descendant" } | null {
  const byId = new Map(predictions.map((p) => [p.nodeRef.nodeId, p]));
  const direct = byId.get(nodeId);
  if (direct) return { prediction: direct, relation: "self" };
  let current = parentOf.get(nodeId);
  let guard = 0;
  while (current !== undefined && current >= 0 && guard++ < 256) {
    const found = byId.get(current);
    if (found) return { prediction: found, relation: "ancestor" };
    current = parentOf.get(current);
  }
  // No ancestor is a candidate: fall forward to the biggest candidate inside.
  let bestNode: NodeFeatures | null = null;
  for (const node of nodes) {
    if (node.nodeId <= nodeId) continue;
    let up: number | undefined = node.nodeId;
    let hops = 0;
    while (up !== undefined && up >= 0 && hops++ < 256) {
      if (up === nodeId) break;
      up = parentOf.get(up);
    }
    if (up !== nodeId) continue;
    if (bestNode === null) bestNode = node;
    break;
  }
  if (bestNode) {
    const found = byId.get(bestNode.nodeId);
    if (found) return { prediction: found, relation: "descendant" };
  }
  return null;
}

// ---------------------------------------------------------------------------

function arg(name: string, fallback?: string): string {
  const at = process.argv.indexOf(`--${name}`);
  if (at !== -1 && process.argv[at + 1]) return process.argv[at + 1]!;
  if (fallback !== undefined) return fallback;
  throw new Error(`--${name} is required`);
}

function argAll(name: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < process.argv.length; i++) {
    if (process.argv[i] === `--${name}` && process.argv[i + 1]) out.push(process.argv[i + 1]!);
  }
  return out;
}

interface StoredPage {
  url: string;
  html: string;
}

function readSite(storeRoot: string, site: string, limit: number): StoredPage[] {
  const projectPath = `${storeRoot}/.squirrel/projects/${site}/project.db`;
  if (!existsSync(projectPath)) return [];
  const project = new Database(projectPath, { readonly: true });
  const content = new Database(`${storeRoot}/.squirrel/content-store.db`, { readonly: true });
  const get = content.query("SELECT content FROM content WHERE hash = ?");
  const pages: StoredPage[] = [];
  try {
    const rows = project
      .query(
        "SELECT normalized_url AS url, content_hash AS hash FROM pages " +
          "WHERE content_hash IS NOT NULL AND status = 200 ORDER BY normalized_url",
      )
      .all() as Array<{ url: string; hash: string }>;
    for (const row of rows) {
      if (pages.length >= limit) break;
      const blob = get.get(row.hash) as { content: Uint8Array } | undefined;
      if (!blob) continue;
      try {
        pages.push({ url: row.url, html: gunzipSync(blob.content).toString("utf8") });
      } catch {
        // A blob that will not decompress is not this experiment's problem.
      }
    }
  } finally {
    project.close();
    content.close();
  }
  return pages;
}

function listSites(storeRoot: string): string[] {
  const dir = `${storeRoot}/.squirrel/projects`;
  if (!existsSync(dir)) return [];
  return readdirSync(dir).filter((name) => !name.startsWith("."));
}

async function main(): Promise<void> {
  const stores = argAll("store");
  const inspectionPath = arg("inspection");
  const artifactPath = arg("artifact");
  const outDir = arg("out");
  const limit = Number(arg("limit", "200"));
  mkdirSync(outDir, { recursive: true });

  const { model, reason } = tryLoadModel(JSON.parse(readFileSync(artifactPath, "utf8")));
  if (!model) throw new Error(`artifact unusable: ${reason}`);

  const inspection = JSON.parse(readFileSync(inspectionPath, "utf8")) as {
    sites: Record<string, { rows: Array<Record<string, unknown>> }>;
  };

  // One entry per DISTINCT promoted root; the 23 rows collapse onto 10 roots.
  const roots = new Map<string, { site: string; key: string; row: Record<string, unknown> }>();
  for (const [site, value] of Object.entries(inspection.sites)) {
    for (const row of value.rows) {
      const key = `${site}|${row.componentRootKey as string}|${(row.root as RootSpec).textLength}`;
      if (!roots.has(key)) roots.set(key, { site, key, row });
    }
  }

  const perRoot = new Map<string, {
    site: string;
    componentRootKey: string;
    spec: RootSpec;
    regionRole: string;
    attribution: string;
    verdict: string;
    reason: string;
    pagesMatched: number;
    labels: string[];
    confidences: number[];
    sources: string[];
    relations: string[];
  }>();
  for (const [key, entry] of roots) {
    const row = entry.row;
    perRoot.set(key, {
      site: entry.site,
      componentRootKey: row.componentRootKey as string,
      spec: row.root as RootSpec,
      regionRole: row.regionRole as string,
      attribution: row.attribution as string,
      verdict: row.verdict as string,
      reason: (row.reason as string) ?? "",
      pagesMatched: 0,
      labels: [],
      confidences: [],
      sources: [],
      relations: [],
    });
  }

  const pageTypeRows: Array<Record<string, unknown>> = [];
  const regionSamples: Array<Record<string, unknown>> = [];

  for (const storeRoot of stores) {
    for (const site of listSites(storeRoot)) {
      const pages = readSite(storeRoot, site, limit);
      if (pages.length === 0) continue;
      console.log(`${site}: ${pages.length} stored pages`, { store: storeRoot });

      const siteRoots = [...perRoot.entries()].filter(([, v]) => v.site === site);

      for (const page of pages) {
        const { document } = parseHTML(page.html);
        const featurized = featurizeDocument(document as never, page.url);
        const result = classifyDocument(document, page.url, { model, budgetMs: 60_000 });
        const parentOf = new Map<number, number>();
        for (const node of featurized.nodes) parentOf.set(node.nodeId, node.parentId);

        pageTypeRows.push({
          site,
          url: page.url,
          pageType: result.pageType.label,
          confidence: Number(result.pageType.confidence.toFixed(4)),
          source: result.pageType.source,
          title: safeText(() => (document as unknown as { title: string }).title),
          h1: safeText(
            () =>
              (
                document as unknown as {
                  querySelector(s: string): { textContent: string | null } | null;
                }
              ).querySelector("h1")?.textContent ?? "",
          ),
          elements: featurized.page.elementCount,
          divSoup: featurized.page.divSoup,
        });

        if (siteRoots.length > 0) {
          let body: ElementLike | null = null;
          try {
            body = (document as unknown as { body: ElementLike | null }).body;
          } catch {
            body = null;
          }
          const index = body ? indexElements(body) : new Map<ElementLike, number>();
          for (const [, entry] of siteRoots) {
            const found = findRoot(index, entry.spec);
            if (!found) continue;
            const mapped = nearestCandidate(
              found.nodeId,
              featurized.nodes,
              result.nodes,
              parentOf,
            );
            if (!mapped) continue;
            entry.pagesMatched++;
            entry.labels.push(mapped.prediction.label);
            entry.confidences.push(mapped.prediction.confidence);
            entry.sources.push(mapped.prediction.source);
            entry.relations.push(mapped.relation);
          }
        }

        // Evidence for the hand check: the landmark-less site's top-level blocks.
        if (featurized.page.divSoup && regionSamples.length < 400) {
          for (const node of result.nodes.slice(0, 12)) {
            const feature = featurized.nodes.find((n) => n.nodeId === node.nodeRef.nodeId);
            if (!feature) continue;
            regionSamples.push({
              site,
              url: page.url,
              nodeId: node.nodeRef.nodeId,
              tag: node.nodeRef.tag,
              predicted: node.label,
              confidence: Number(node.confidence.toFixed(4)),
              source: node.source,
              depth: feature.depth,
            });
          }
        }
      }
    }
  }

  const groups = [...perRoot.values()].map((entry) => {
    const counts = new Map<string, number>();
    for (const label of entry.labels) counts.set(label, (counts.get(label) ?? 0) + 1);
    const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    const top = ranked[0];
    const share = top && entry.labels.length ? top[1] / entry.labels.length : 0;
    const modelShare = entry.sources.filter((s) => s === "model").length /
      Math.max(entry.sources.length, 1);
    return {
      site: entry.site,
      componentRootKey: entry.componentRootKey,
      rootTag: entry.spec.tag,
      rootClass: entry.spec.class,
      rootTextLength: entry.spec.textLength,
      deterministicRegionRole: entry.regionRole,
      attribution: entry.attribution,
      verdict: entry.verdict,
      inspectionReason: entry.reason,
      pagesMatched: entry.pagesMatched,
      // A label only counts when at least 80% of the pages agree.
      stableLabel: share >= 0.8 ? top![0] : null,
      agreement: Number(share.toFixed(4)),
      labelCounts: Object.fromEntries(ranked),
      meanConfidence: entry.confidences.length
        ? Number(
            (entry.confidences.reduce((a, b) => a + b, 0) / entry.confidences.length).toFixed(4),
          )
        : 0,
      modelSourceShare: Number(modelShare.toFixed(4)),
      mappedVia: Object.fromEntries(
        [...new Set(entry.relations)].map((r) => [r, entry.relations.filter((x) => x === r).length]),
      ),
    };
  });

  writeFileSync(`${outDir}/groups.json`, `${JSON.stringify(groups, null, 2)}\n`);
  writeFileSync(
    `${outDir}/page-types.jsonl`,
    `${pageTypeRows.map((r) => JSON.stringify(r)).join("\n")}\n`,
  );
  writeFileSync(
    `${outDir}/region-samples.jsonl`,
    `${regionSamples.map((r) => JSON.stringify(r)).join("\n")}\n`,
  );

  const named = groups.filter((g) => g.stableLabel && g.stableLabel !== "unknown");
  console.log(
    JSON.stringify(
      {
        distinctRoots: groups.length,
        matchedOnAtLeastOnePage: groups.filter((g) => g.pagesMatched > 0).length,
        stableLabel: named.length,
        pages: pageTypeRows.length,
        regionSamples: regionSamples.length,
      },
      null,
      2,
    ),
  );
}

await main();

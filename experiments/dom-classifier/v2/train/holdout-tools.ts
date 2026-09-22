// Things that need the ORIGINAL HTML rather than the built dataset:
//
//   masked-all  re-featurize the holdout with every label word hidden (p=1.0),
//               which is the honest div-soup estimate;
//   bench       p50/p95 for featurize + predict on real pages;
//   divsoup     the labelling sample Lyra's packet builder consumes;
//   inspect     pull the article candidates so a human can grade them.
//
// The content store is LIVE and written by another process: readonly, short
// batched reads, no transaction held open. Nothing written here leaves the
// private data root.
//
// Usage:
//   DOM_CLASSIFIER_DATA_ROOT=... SQUIRREL_STORE=... \
//     bun run holdout-tools.ts <mode> [--limit N]

import { Database } from "bun:sqlite";
import { createReadStream, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { createGunzip, gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { featurizeDocument } from "../featurize.ts";
import { classifyDocument, tryLoadModel } from "../runtime/predict.ts";
import { silverRegionLabels } from "../silver-labels.ts";

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} must be set`);
  return value;
}

const dataRoot = requireEnv("DOM_CLASSIFIER_DATA_ROOT");
const storeRoot = requireEnv("SQUIRREL_STORE");
/** The dataset directory under the data root. `silver-v1` is the 2026-09-19
 *  build; a larger corpus is built beside it rather than over it, and the
 *  Python scripts in this directory read the same variable. */
const dataset = `${dataRoot}/${process.env.DOM_CLASSIFIER_DATASET_VERSION ?? "silver-v1"}`;

interface PageIndexEntry {
  p: string;
  u: string;
  domain: string;
  s: string;
}

async function* gzLines(path: string): AsyncGenerator<string> {
  const rl = createInterface({ input: createReadStream(path).pipe(createGunzip()) });
  for await (const line of rl) if (line) yield line;
}

/** pageId (16-hex prefix) -> full content hash, from the cached index. */
function prefixToHash(): Map<string, string> {
  const path = `${dataRoot}/cache/hash-url-index.jsonl`;
  const map = new Map<string, string>();
  for (const line of readFileSync(path, "utf8").split("\n")) {
    if (!line) continue;
    const hash = (JSON.parse(line) as { hash: string }).hash;
    map.set(hash.slice(0, 16), hash);
  }
  return map;
}

async function pageIndex(): Promise<Map<string, PageIndexEntry>> {
  const map = new Map<string, PageIndexEntry>();
  for await (const line of gzLines(`${dataset}/page-index.jsonl.gz`)) {
    const entry = JSON.parse(line) as PageIndexEntry;
    map.set(entry.p, entry);
  }
  return map;
}

class Store {
  private readonly db: Database;
  private readonly get: ReturnType<Database["query"]>;

  constructor() {
    this.db = new Database(`${storeRoot}/content-store.db`, { readonly: true });
    this.get = this.db.query("SELECT content FROM content WHERE hash = ?");
  }

  html(hash: string): string | null {
    const row = this.get.get(hash) as { content: Uint8Array } | undefined;
    if (!row) return null;
    try {
      return gunzipSync(row.content).toString("utf8");
    } catch {
      return null;
    }
  }

  close(): void {
    this.db.close();
  }
}

/** Pages from the named splits, in dataset order, with their original HTML. */
async function* holdoutPages(
  limit: number,
  onlyDivSoup = false,
  splits: readonly string[] = ["holdout"],
) {
  const index = await pageIndex();
  const hashes = prefixToHash();
  const store = new Store();
  let yielded = 0;
  try {
    for (const split of splits) {
      for await (const line of gzLines(`${dataset}/pages-${split}.jsonl.gz`)) {
        if (yielded >= limit) break;
        const page = JSON.parse(line) as { p: string; d: string; divSoup: boolean };
        if (onlyDivSoup && !page.divSoup) continue;
        const entry = index.get(page.p);
        const hash = hashes.get(page.p);
        if (!entry || !hash) continue;
        const html = store.html(hash);
        if (!html) continue;
        yielded++;
        yield { page, entry, hash, html };
      }
      if (yielded >= limit) break;
    }
  } finally {
    store.close();
  }
}

// ---------------------------------------------------------------------------

async function maskedAll(limit: number): Promise<void> {
  const out: string[] = [];
  let pages = 0;
  let nodes = 0;
  for await (const { page, entry, html } of holdoutPages(limit)) {
    const { document } = parseHTML(html);
    const featurized = featurizeDocument(document as never, entry.u, {
      mask: true,
      maskAll: true,
    });
    const regions = silverRegionLabels(featurized.nodes);
    const labels = new Map(regions.labels.map((l) => [l.nodeId, l]));
    for (const node of featurized.nodes) {
      const label = labels.get(node.nodeId);
      if (!label) continue;
      out.push(
        JSON.stringify({
          p: page.p,
          n: node.nodeId,
          d: page.d,
          s: "holdout",
          l: label.label,
          m: 1,
          f: node.features,
          k: node.tokens,
        }),
      );
      nodes++;
    }
    pages++;
    if (pages % 250 === 0) console.log(`masked-all: ${pages} pages, ${nodes} nodes`);
  }
  writeFileSync(`${dataRoot}/models/holdout-maskall.jsonl`, `${out.join("\n")}\n`);
  console.log(`masked-all: ${pages} pages, ${nodes} nodes -> models/holdout-maskall.jsonl`);
}

async function bench(limit: number): Promise<void> {
  const artifact = JSON.parse(readFileSync(artifactPath, "utf8"));
  const coldStart = Bun.nanoseconds();
  const { model } = tryLoadModel(artifact);
  const coldMs = (Bun.nanoseconds() - coldStart) / 1e6;

  const parseMs: number[] = [];
  const totalMs: number[] = [];
  const classifyMs: number[] = [];
  let elements = 0;
  let pages = 0;
  for await (const { entry, html } of holdoutPages(limit)) {
    let start = Bun.nanoseconds();
    const { document } = parseHTML(html);
    const parsed = (Bun.nanoseconds() - start) / 1e6;
    start = Bun.nanoseconds();
    const result = classifyDocument(document, entry.u, { model, budgetMs: 10_000 });
    const classified = (Bun.nanoseconds() - start) / 1e6;
    parseMs.push(parsed);
    classifyMs.push(classified);
    totalMs.push(parsed + classified);
    elements += result.nodes.length;
    pages++;
  }
  const pct = (values: number[], p: number): number => {
    const sorted = [...values].sort((a, b) => a - b);
    return Number(sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))]!.toFixed(3));
  };
  const summary = {
    pages,
    avgNodesPerPage: Math.round(elements / Math.max(pages, 1)),
    coldArtifactLoadMs: Number(coldMs.toFixed(1)),
    artifact: artifactPath,
    artifactBytes: readFileSync(artifactPath).length,
    regionTrees: artifact.region.numTrees,
    featurizeAndPredictMs: { p50: pct(classifyMs, 0.5), p95: pct(classifyMs, 0.95) },
    parseMs: { p50: pct(parseMs, 0.5), p95: pct(parseMs, 0.95) },
    parseAndClassifyMs: { p50: pct(totalMs, 0.5), p95: pct(totalMs, 0.95) },
  };
  writeFileSync(
    `${artifactPath.replace(/[^/]+$/, "")}bench.json`,
    `${JSON.stringify(summary, null, 2)}\n`,
  );
  console.log(JSON.stringify(summary, null, 2));
}

async function divsoup(limit: number): Promise<void> {
  const outDir = `${dataRoot}/divsoup-eval-v1`;
  mkdirSync(outDir, { recursive: true });
  const { model } = tryLoadModel(JSON.parse(readFileSync(artifactPath, "utf8")));

  const rows: string[] = [];
  let pages = 0;
  // val AND holdout: the two splits no training row came from. Holdout alone
  // holds 205 div-soup pages, which is short of the 300 the sample needs.
  for await (const { page, entry, hash, html } of holdoutPages(limit, true, ["holdout", "val"])) {
    const { document } = parseHTML(html);
    const result = classifyDocument(document, entry.u, { model, budgetMs: 10_000 });
    // Highest-confidence node per predicted class, at most 8 per page, so a
    // packet covers the classes rather than eight variations of one.
    const bestByClass = new Map<string, (typeof result.nodes)[number]>();
    for (const node of result.nodes) {
      if (node.label === "unknown") continue;
      const held = bestByClass.get(node.label);
      if (!held || node.confidence > held.confidence) bestByClass.set(node.label, node);
    }
    const picked = [...bestByClass.values()]
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 8)
      .sort((a, b) => a.nodeRef.nodeId - b.nodeRef.nodeId);
    if (picked.length === 0) continue;
    rows.push(
      JSON.stringify({
        pageId: page.p,
        contentHash: hash,
        url: entry.u,
        domain: entry.domain,
        split: entry.s,
        // No labels here, by design: this sample is graded by the labelling
        // cohort, and anything written now would anchor it.
        candidates: picked.map((node) => ({
          nodeId: node.nodeRef.nodeId,
          tag: node.nodeRef.tag,
          elementId: node.nodeRef.elementId,
          predicted: node.label,
          confidence: Number(node.confidence.toFixed(4)),
          source: node.source,
        })),
      }),
    );
    pages++;
  }
  writeFileSync(`${outDir}/sample.jsonl`, `${rows.join("\n")}\n`);
  writeFileSync(
    `${outDir}/README.md`,
    [
      "# div-soup eval sample v1",
      "",
      `${pages} pages from the holdout and val splits (no training domain appears),`,
      "every one with fewer than two distinct",
      "landmark kinds: the pages the classifier exists for, where the",
      "deterministic path has nothing to read.",
      "",
      "One line per page. `contentHash` is the key into",
      "`$SQUIRREL_STORE/content-store.db` (`SELECT content FROM content WHERE hash = ?`,",
      "gzip), so a packet builder can render the full HTML. `nodeId` is the",
      "featurizer's preorder element index: re-running `featurizePage(html, url)`",
      "reproduces exactly these nodes and ids.",
      "",
      "`predicted` and `confidence` are the model's, included so a packet can be",
      "ordered or stratified. They are NOT labels and must not be shown to an",
      "annotator as suggestions: this sample becomes the G2 non-semantic eval,",
      "and it is only worth anything if the grading is independent of it.",
      "",
      "Nobody on the modelling side reads the labels back for tuning.",
    ].join("\n") + "\n",
  );
  console.log(`divsoup: ${pages} pages -> ${outDir}/sample.jsonl`);
}

async function inspect(limit: number): Promise<void> {
  const candidates = readFileSync(
    `${dataRoot}/models/pagetype/article-candidates-holdout.jsonl`,
    "utf8",
  )
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as { p: string; articleProb: number });
  const index = await pageIndex();
  const hashes = prefixToHash();
  const store = new Store();
  const outDir = `${dataRoot}/models/pagetype/inspection`;
  mkdirSync(outDir, { recursive: true });
  const rows: string[] = [];
  try {
    for (const candidate of candidates.slice(0, limit)) {
      const entry = index.get(candidate.p);
      const hash = hashes.get(candidate.p);
      if (!entry || !hash) continue;
      const html = store.html(hash);
      if (!html) continue;
      const { document } = parseHTML(html);
      const featurized = featurizeDocument(document as never, entry.u);
      const doc = document as unknown as {
        title: string;
        querySelector(s: string): { textContent: string | null } | null;
        querySelectorAll(s: string): ArrayLike<{ textContent: string | null }>;
      };
      const paragraphs = Array.from(doc.querySelectorAll("p"))
        .map((p) => (p.textContent ?? "").trim())
        .filter((t) => t.length > 40);
      writeFileSync(`${outDir}/${candidate.p}.html`, html);
      rows.push(
        JSON.stringify({
          pageId: candidate.p,
          articleProb: candidate.articleProb,
          url: entry.u,
          title: (doc.title ?? "").trim().slice(0, 160),
          h1: (doc.querySelector("h1")?.textContent ?? "").trim().slice(0, 160),
          longParagraphs: paragraphs.length,
          firstParagraph: paragraphs[0]?.slice(0, 300) ?? "",
          totalTextLength: featurized.page.features[featurized.page.features.length - 1],
          elementCount: featurized.page.elementCount,
        }),
      );
    }
  } finally {
    store.close();
  }
  writeFileSync(`${outDir}/candidates.jsonl`, `${rows.join("\n")}\n`);
  console.log(`inspect: ${rows.length} pages -> ${outDir}/candidates.jsonl`);
}

const mode = process.argv[2] ?? "";
const limitFlag = process.argv.indexOf("--limit");
const limit = limitFlag !== -1 ? Number(process.argv[limitFlag + 1]) : Number.POSITIVE_INFINITY;
const artifactFlag = process.argv.indexOf("--artifact");
const artifactPath =
  artifactFlag !== -1
    ? process.argv[artifactFlag + 1]!
    : `${dataRoot}/models/artifact/classifier-v2.json`;

if (!existsSync(dataset)) throw new Error(`no dataset at ${dataset}`);
switch (mode) {
  case "masked-all":
    await maskedAll(limit);
    break;
  case "bench":
    await bench(limit);
    break;
  case "divsoup":
    await divsoup(limit);
    break;
  case "inspect":
    await inspect(limit);
    break;
  default:
    throw new Error(`unknown mode "${mode}"; use masked-all | bench | divsoup | inspect`);
}

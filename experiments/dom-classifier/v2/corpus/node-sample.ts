#!/usr/bin/env bun
/**
 * Draw the node-labelling sample for Jev, weighted onto landmark-less pages.
 *
 * ## The failure this sample exists to fix
 *
 * Silver region labels come from the page's own landmarks, so a page that has
 * none produces no `footer` and no `navigation` rows — ever, at any corpus
 * size. The region head therefore learns those classes only from pages that
 * declare them, and is asked at runtime about pages that do not. That is a
 * distribution shift, and more silver data does not touch it: the labels are
 * missing exactly where the model is weak.
 *
 * Jev labels a node from its evidence rather than from its tag, so it CAN put a
 * `footer` label on a `<div class="site-foot">`. This sample is therefore drawn
 * at least 60% from div-soup pages (fewer than two landmark kinds), which is
 * the opposite of how the silver rows are distributed.
 *
 * ## Node identity
 *
 * `nodeId` is the featurizer's preorder element index, taken straight from the
 * dataset's node rows. `run-divsoup-v7.ts` resolves it with `featurizerOrder`,
 * which reproduces that same walk. It is NOT an index into `extractPage`'s
 * candidate array: indexing that instead resolved 553 of 822 nodes to a
 * different element, silently, because a wrong element still has a tag and
 * still makes a plausible packet.
 *
 * ## What is kept out
 *
 * - **holdout**, unless `--split holdout` is passed deliberately to build an
 *   EVALUATION sample. Training on labels drawn from the split that scores the
 *   result is the one mistake that cannot be undone by re-running anything, so
 *   the default is train and the other case announces itself.
 * - **`divsoup-eval-v1`'s pages**, by page id, when the embargoed sample list is
 *   passed with `--exclude-pages`. Those 822 nodes are the blind evaluation set.
 */
import { createHash } from "node:crypto";
import { createReadStream, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { resolve } from "node:path";
import { createGunzip } from "node:zlib";

const dataRoot = process.env.DOM_CLASSIFIER_DATA_ROOT;
if (!dataRoot) throw new Error("DOM_CLASSIFIER_DATA_ROOT must be set");

function argument(name: string, fallback: string): string {
  const index = Bun.argv.indexOf(name);
  if (index < 0) return fallback;
  const value = Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}
const datasetDir = resolve(argument("--dataset", `${dataRoot}/silver-v2`));
const outDir = resolve(argument("--out", `${dataRoot}/node-jev-v7`));
const targetNodes = Number(argument("--nodes", "40000"));
const divSoupShare = Number(argument("--divsoup-share", "0.6"));
const maxPerDomain = Number(argument("--max-per-domain", "40"));
const maxPerPage = Number(argument("--max-per-page", "6"));
const excludePagesPath = argument("--exclude-pages", "");
/**
 * Which split to draw from. `train` is the labelling sample. `holdout` builds
 * an EVALUATION sample instead, and the loud banner below exists because the
 * two files look identical: `sample.py` hard-fails if a Jev label from another
 * split reaches it, but nothing stops a person pointing the wrong path at the
 * wrong script.
 */
const split = argument("--split", "train");
if (!["train", "val", "holdout"].includes(split))
  throw new Error(`--split must be train | val | holdout, got ${split}`);
if (split !== "train")
  console.log(
    JSON.stringify({
      stage: "warning",
      message: `drawing from ${split}: this is an EVALUATION sample and must never be passed to sample.py --jev-labels`,
    }),
  );
mkdirSync(outDir, { recursive: true, mode: 0o700 });

const sha256 = (value: string) => createHash("sha256").update(value).digest("hex");
/** Deterministic order key: nothing to do with crawl order, page size or name. */
const orderKey = (...parts: string[]) => sha256(parts.join("|"));

// The exclusion list must fail LOUD. It is the stated defence against the
// embargoed evaluation pages reaching a training sample, and a guard that
// silently does nothing when its path is wrong is worse than no guard: the run
// succeeds and the summary reports `excludedPages: 0`, which reads like "there
// was nothing to exclude".
const excluded = new Set<string>();
if (excludePagesPath) {
  if (!existsSync(excludePagesPath))
    throw new Error(`--exclude-pages ${excludePagesPath} does not exist`);
  for (const line of readFileSync(excludePagesPath, "utf8").split("\n")) {
    if (!line) continue;
    try {
      excluded.add((JSON.parse(line) as { pageId: string }).pageId);
    } catch {
      /* a malformed line is not a page id */
    }
  }
  if (excluded.size === 0)
    throw new Error(`--exclude-pages ${excludePagesPath} yielded no page ids`);
}

async function* gzLines(path: string): AsyncGenerator<string> {
  const stream = createReadStream(path).pipe(createGunzip());
  for await (const line of createInterface({ input: stream, crlfDelay: Infinity }))
    if (line) yield line;
}

// --- pass A: which pages ----------------------------------------------------
type PageRow = {
  p: string;
  d: string;
  divSoup: boolean;
  nodeCount: number;
  l: string;
  /** The landmark kinds the page declared. div-soup means fewer than two. */
  landmarkKinds: string[];
};
const divSoupPages: PageRow[] = [];
const landmarkPages: PageRow[] = [];
for await (const line of gzLines(`${datasetDir}/pages-${split}.jsonl.gz`)) {
  const row = JSON.parse(line) as PageRow;
  if (excluded.has(row.p) || row.nodeCount === 0) continue;
  (row.divSoup ? divSoupPages : landmarkPages).push(row);
}
console.log(
  JSON.stringify({
    stage: "pages",
    divSoup: divSoupPages.length,
    divSoupWithOneLandmark: divSoupPages.filter((p) => (p.landmarkKinds?.length ?? 0) === 1).length,
    divSoupWithNone: divSoupPages.filter((p) => (p.landmarkKinds?.length ?? 0) === 0).length,
    landmark: landmarkPages.length,
    excluded: excluded.size,
  }),
);

/** Pick pages under a per-domain cap, in a deterministic order, until the node
 *  budget for this tier is met at `maxPerPage` nodes each. */
function choosePages(pages: PageRow[], nodeBudget: number, perDomain: Map<string, number>) {
  const chosen: PageRow[] = [];
  let budget = nodeBudget;
  // Among div-soup pages, the ones that declared ONE landmark kind come first.
  // "div-soup" is fewer than two kinds, which lumps together two populations
  // that are not alike: a page with zero is usually a single-page-app shell
  // with no header or footer to find, while a page with one has real site
  // chrome and simply did not declare the rest of it — a `<nav>` but no
  // `<footer>`. The second is where an undeclared footer actually exists, so
  // it is where the labelling budget belongs. Within a tier the order is still
  // the hash, so the choice is reproducible and independent of crawl order.
  const ordered = [...pages].sort((left, right) => {
    const rank = (page: PageRow) => (page.divSoup ? -(page.landmarkKinds?.length ?? 0) : 0);
    return rank(left) - rank(right) || (orderKey(left.p) < orderKey(right.p) ? -1 : 1);
  });
  for (const page of ordered) {
    if (budget <= 0) break;
    const used = perDomain.get(page.d) ?? 0;
    if (used >= maxPerDomain) continue;
    const take = Math.min(maxPerPage, page.nodeCount, maxPerDomain - used, budget);
    if (take <= 0) continue;
    perDomain.set(page.d, used + take);
    chosen.push(page);
    budget -= take;
  }
  return chosen;
}

const perDomain = new Map<string, number>();
const divSoupBudget = Math.round(targetNodes * divSoupShare);
const chosenDivSoup = choosePages(divSoupPages, divSoupBudget, perDomain);
// Whatever the div-soup tier could not fill is spent on landmark pages rather
// than left unspent: a shortfall there is a fact about the corpus, and the
// summary records it instead of hiding it.
const divSoupPlanned = [...perDomain.values()].reduce((sum, value) => sum + value, 0);
const chosenLandmark = choosePages(landmarkPages, targetNodes - divSoupPlanned, perDomain);
const chosen = new Map<string, PageRow>();
for (const page of [...chosenDivSoup, ...chosenLandmark]) chosen.set(page.p, page);
console.log(
  JSON.stringify({
    stage: "chosen",
    divSoupPages: chosenDivSoup.length,
    landmarkPages: chosenLandmark.length,
    divSoupNodesPlanned: divSoupPlanned,
  }),
);

// --- pass B: which nodes ----------------------------------------------------
/** Only the UNMASKED row of each node (`m: 0`): the two rows are the same node,
 *  and a packet is built from the real markup, never the masked features. */
type NodeRow = {
  p: string;
  n: number;
  m: number;
  l: string;
  tagRaw: string;
  inh: boolean;
  /** chrome score, derived from three feature columns and then discarded */
  chrome: number;
};
/**
 * Feature columns, resolved BY NAME from the dataset's own schema.
 *
 * They were hardcoded (1, 2, 29 at featurizer v2.2.0) and that is a silent
 * failure waiting to happen: a featurizer change that inserts a column
 * re-points `chromeScore` at three unrelated numbers, and the only symptom is
 * a worse sample — which is the very quantity this file is trying to improve,
 * so nothing would look broken.
 */
const featureSchema = JSON.parse(readFileSync(`${datasetDir}/feature-schema.json`, "utf8")) as {
  nodeFeatures: string[];
  featurizerVersion: string;
};
function featureIndex(name: string): number {
  const index = featureSchema.nodeFeatures.indexOf(name);
  if (index < 0)
    throw new Error(
      `feature "${name}" is not in ${datasetDir}/feature-schema.json ` +
        `(featurizer ${featureSchema.featurizerVersion}); chromeScore cannot be computed`,
    );
  return index;
}
const F_START_POS = featureIndex("startPos");
const F_END_POS = featureIndex("endPos");
const F_LINK_DENSITY = featureIndex("linkDensity");

/**
 * How much this node looks like site chrome, from position and link density.
 *
 * Measured, not assumed. On the probe run the div-soup nodes Jev went on to
 * call footer / navigation / site_header scored a median 1.18, against 0.43 for
 * the div-soup nodes it abstained on.
 *
 * It matters because of what the probe showed about the div-soup population:
 * of 38 div-soup nodes labelled, Jev found chrome on 4. The other 34 were app
 * shells, cards and lists on single-page-app themes that genuinely have no
 * header or footer — `layout_container` with an empty regions axis was the
 * right answer, not a failure to see one. Sampling div-soup nodes uniformly
 * therefore spends most of the budget on pages with no chrome to find; the
 * score puts it where chrome can exist.
 */
function chromeScore(features: number[]): number {
  const startPos = features[F_START_POS] ?? 0;
  const endPos = features[F_END_POS] ?? 0;
  const linkDensity = features[F_LINK_DENSITY] ?? 0;
  // A header sits at the start, a footer at the end, and both are link-dense.
  return Math.max(1 - startPos, endPos) * (0.5 + Math.min(linkDensity, 1));
}

const byPage = new Map<string, NodeRow[]>();
for await (const line of gzLines(`${datasetDir}/nodes-${split}.jsonl.gz`)) {
  const row = JSON.parse(line) as NodeRow & { f: number[] };
  if (row.m !== 0 || !chosen.has(row.p)) continue;
  // The feature vector is 197 numbers per row and there are millions of rows:
  // reduce it to the one number this file needs and drop the rest immediately,
  // or the sample of a 100k-page dataset does not fit in memory.
  const kept: NodeRow = {
    p: row.p,
    n: row.n,
    m: row.m,
    l: row.l,
    tagRaw: row.tagRaw,
    inh: row.inh,
    chrome: chromeScore(row.f),
  };
  const bucket = byPage.get(row.p);
  if (bucket) bucket.push(kept);
  else byPage.set(row.p, [kept]);
}

/** Within a page, spread the picks across TAGS before taking a second of any
 *  one tag. A page of forty `<div>`s otherwise contributes forty near-identical
 *  packets and teaches the boundary nothing. */
function pickNodes(rows: NodeRow[], take: number): NodeRow[] {
  const byTag = new Map<string, NodeRow[]>();
  for (const row of rows) {
    const bucket = byTag.get(row.tagRaw);
    if (bucket) bucket.push(row);
    else byTag.set(row.tagRaw, [row]);
  }
  for (const bucket of byTag.values())
    bucket.sort((left, right) =>
      orderKey(left.p, String(left.n)) < orderKey(right.p, String(right.n)) ? -1 : 1,
    );
  const tags = [...byTag.keys()].sort((left, right) => (orderKey(left) < orderKey(right) ? -1 : 1));
  const picked: NodeRow[] = [];
  for (let round = 0; picked.length < take; round++) {
    let moved = false;
    for (const tag of tags) {
      if (picked.length >= take) break;
      const row = byTag.get(tag)![round];
      if (row) {
        picked.push(row);
        moved = true;
      }
    }
    if (!moved) break;
  }
  return picked;
}

/**
 * On a div-soup page, spend most of the page's budget on the nodes that can BE
 * chrome, and the rest on the tag spread.
 *
 * Not all of it: a sample of nothing but suspected headers and footers would
 * teach the boundary between chrome and content only from one side, and the
 * content side is what `content_block` is. The split is two thirds / one third.
 */
function pickChromeFirst(rows: NodeRow[], take: number): NodeRow[] {
  const chromeTake = Math.max(1, Math.round(take * (2 / 3)));
  const ranked = [...rows].sort(
    (left, right) =>
      right.chrome - left.chrome ||
      (orderKey(left.p, String(left.n)) < orderKey(right.p, String(right.n)) ? -1 : 1),
  );
  const picked = ranked.slice(0, chromeTake);
  const taken = new Set(picked.map((row) => row.n));
  for (const row of pickNodes(
    rows.filter((row) => !taken.has(row.n)),
    take - picked.length,
  ))
    picked.push(row);
  return picked;
}

// --- emit -------------------------------------------------------------------
// The page index is the only file carrying a url, and the Jev runner needs one
// per page to build page evidence.
const urls = new Map<string, { u: string; domain: string; s: string }>();
for await (const line of gzLines(`${datasetDir}/page-index.jsonl.gz`)) {
  const row = JSON.parse(line) as { p: string; u: string; domain: string; s: string };
  if (chosen.has(row.p)) urls.set(row.p, row);
}

const perDomainFinal = new Map<string, number>();
const lines: string[] = [];
const silver: string[] = [];
const tagCounts: Record<string, number> = {};
let nodes = 0;
let divSoupNodes = 0;
/** div-soup nodes drawn from a page that declared ONE landmark kind: the tier
 *  where an undeclared footer or header can actually exist. */
let oneLandmarkNodes = 0;
let missingUrl = 0;
for (const [pageId, page] of chosen) {
  const rows = byPage.get(pageId);
  const meta = urls.get(pageId);
  if (!rows?.length) continue;
  if (!meta) {
    missingUrl++;
    continue;
  }
  const used = perDomainFinal.get(page.d) ?? 0;
  const take = Math.min(maxPerPage, maxPerDomain - used);
  if (take <= 0) continue;
  const picked = page.divSoup ? pickChromeFirst(rows, take) : pickNodes(rows, take);
  if (!picked.length) continue;
  perDomainFinal.set(page.d, used + picked.length);
  nodes += picked.length;
  if (page.divSoup) {
    divSoupNodes += picked.length;
    if ((page.landmarkKinds?.length ?? 0) === 1) oneLandmarkNodes += picked.length;
  }
  for (const row of picked) tagCounts[row.tagRaw] = (tagCounts[row.tagRaw] ?? 0) + 1;
  lines.push(
    JSON.stringify({
      pageId,
      // The full content hash is what the divsoup runner looks up; the dataset's
      // page id IS the first 16 hex of it, and the runner is given both stores.
      contentHash: pageId,
      url: meta.u,
      domain: meta.domain,
      split: meta.s,
      candidates: picked.map((row) => ({ nodeId: row.n })),
    }),
  );
  // The silver answer for the same nodes, written separately so the
  // silver-vs-Jev disagreement can be computed without re-reading the dataset.
  for (const row of picked)
    silver.push(
      JSON.stringify({
        pageId,
        nodeId: row.n,
        silverLabel: row.l,
        inherited: row.inh,
        tagRaw: row.tagRaw,
        divSoup: page.divSoup,
      }),
    );
}

writeFileSync(`${outDir}/sample.jsonl`, lines.map((l) => `${l}\n`).join(""), { mode: 0o600 });
writeFileSync(`${outDir}/silver-for-sample.jsonl`, silver.map((l) => `${l}\n`).join(""), {
  mode: 0o600,
});
const summary = {
  stage: "complete",
  split,
  pages: lines.length,
  nodes,
  divSoupNodes,
  divSoupShare: Number((divSoupNodes / Math.max(1, nodes)).toFixed(4)),
  divSoupNodesFromOneLandmarkPages: oneLandmarkNodes,
  targetNodes,
  targetDivSoupShare: divSoupShare,
  domains: perDomainFinal.size,
  maxPerDomain,
  maxPerPage,
  missingUrl,
  excludedPages: excluded.size,
  tagCounts: Object.fromEntries(
    Object.entries(tagCounts)
      .sort((left, right) => right[1] - left[1])
      .slice(0, 20),
  ),
};
writeFileSync(`${outDir}/sample-summary.json`, `${JSON.stringify(summary, null, 2)}\n`, {
  mode: 0o600,
});
console.log(JSON.stringify(summary));

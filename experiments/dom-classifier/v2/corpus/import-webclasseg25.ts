#!/usr/bin/env bun
/**
 * Import WebClasSeg25 as a tagged research cohort.
 *
 * ## Why this dataset
 *
 * It is the only public corpus that labels the four classes this lane is short
 * of. 852 pages from 500 domains in 20 languages, 3 to 4 human annotators per
 * page, and its functional-classification segments are `header`, `nav`,
 * `maincontent`, `footer` — which is the region axis, not an approximation of
 * it. The pages are European public-service sites (university, hospital,
 * municipality, court, tourism, insurance): exactly the CMS-shaped,
 * non-English half of the web the squirrelscan crawl does not reach.
 *
 * Licence: CC BY 4.0, from the Zenodo record's own API metadata. See
 * `research/webclasseg25/PROVENANCE.json` for the hashes and the attribution.
 * Nothing from it enters the repo: pages and URLs stay under the private root
 * and only features and labels are emitted, exactly as for the crawl.
 *
 * ## The join is exact, which is the whole reason this is usable
 *
 * Every element in the released HTML carries a `hyu="N"` attribute and each
 * annotated segment names a `hyuIndex`. So a label lands on a node by
 * ATTRIBUTE EQUALITY, not by an XPath or a re-derived index. Audited on one
 * page: 1,135 of 1,135 elements carry `hyu`, and 14 of 14 segments resolved.
 * Compare the div-soup eval set, where indexing the wrong array silently
 * resolved 553 of 822 nodes to a different element.
 *
 * ## What is mapped, and what is not
 *
 * `header`, `nav`, `maincontent`, `footer` map onto `site_header`,
 * `navigation`, `main_content`, `footer`. `title`, `image` and `advertisement`
 * are NOT mapped: this taxonomy has no class for them, and inventing one to
 * use the rows would put labels in the model that no consumer asked for.
 *
 * Several annotators labelled each page, so a node's weight is the SHARE that
 * agreed on the winning label. A node three of three annotators called `footer`
 * enters at 1.0; two of three at 0.67. That is the same weight channel the
 * inherited landmark labels and the Jev confidences use.
 *
 * Every row carries `cohort: "research/webclasseg25"` and `s: "train"`. They
 * never enter val or holdout, so every reported number stays on squirrelscan's
 * own distribution.
 */
import { createHash } from "node:crypto";
import {
  createReadStream,
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { createInterface } from "node:readline";
import { resolve } from "node:path";
import { createGunzip, createGzip } from "node:zlib";

import { parse as parseDomain } from "tldts";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import {
  featureSchema,
  featurizeDocument,
  FEATURIZER_VERSION,
  SKIP_SUBTREE,
} from "../featurize.ts";
import { silverRegionLabels } from "../silver-labels.ts";

const dataRoot = process.env.DOM_CLASSIFIER_DATA_ROOT;
if (!dataRoot) throw new Error("DOM_CLASSIFIER_DATA_ROOT must be set");

function argument(name: string, fallback: string): string {
  const index = Bun.argv.indexOf(name);
  if (index < 0) return fallback;
  const value = Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}
const sourceDir = resolve(argument("--source", `${dataRoot}/research/webclasseg25`));
const outDir = resolve(argument("--out", `${dataRoot}/research-webclasseg25`));
/** page_id -> url, extracted from the MHTML `Snapshot-Content-Location` headers. */
const urlsPath = argument("--urls", `${sourceDir}/page-urls.json`);
/** Domains that must not appear: our own val and holdout. */
const excludeManifest = argument("--exclude-split-manifest", "");
const limit = Number(argument("--limit", "0"));
mkdirSync(outDir, { recursive: true, mode: 0o700 });

const COHORT = "research/webclasseg25";
const TAG_TO_REGION: Record<string, string> = {
  header: "site_header",
  nav: "navigation",
  maincontent: "main_content",
  footer: "footer",
};

const sha256 = (value: string) => createHash("sha256").update(value).digest("hex");
function splitOf(domain: string): "train" | "val" | "holdout" {
  const bucket = parseInt(sha256(domain).slice(0, 8), 16) % 100;
  if (bucket < 80) return "train";
  if (bucket < 90) return "val";
  return "holdout";
}

// --- annotations ------------------------------------------------------------
type Segment = { tagType: string; hyuIndex: number };
type Record_ = { id: string; annotation_fc?: Segment[] };
const records = JSON.parse(readFileSync(`${sourceDir}/annotations.json`, "utf8")) as Record_[];
/** pageId -> hyuIndex -> tagType -> votes, plus the annotator count for the page. */
const votes = new Map<string, Map<number, Map<string, number>>>();
const annotators = new Map<string, Set<string>>();
for (const record of records) {
  const [pageId, annotator] = record.id.split("_annotator");
  if (!pageId) continue;
  let set = annotators.get(pageId);
  if (!set) annotators.set(pageId, (set = new Set()));
  set.add(annotator ?? "?");
  let page = votes.get(pageId);
  if (!page) votes.set(pageId, (page = new Map()));
  for (const segment of record.annotation_fc ?? []) {
    if (!TAG_TO_REGION[segment.tagType]) continue;
    let node = page.get(segment.hyuIndex);
    if (!node) page.set(segment.hyuIndex, (node = new Map()));
    node.set(segment.tagType, (node.get(segment.tagType) ?? 0) + 1);
  }
}

const pageUrls: Record<string, string> = existsSync(urlsPath)
  ? (JSON.parse(readFileSync(urlsPath, "utf8")) as Record<string, string>)
  : {};
if (!Object.keys(pageUrls).length)
  console.log(
    JSON.stringify({
      stage: "warning",
      message: `no ${urlsPath}: domains unknown, so the val/holdout collision check cannot run`,
    }),
  );

/** Our own val and holdout domain hashes. A research page from one of those
 *  domains would contaminate the split that grades the result, which no amount
 *  of "these rows are train-only" prevents. */
const forbidden = new Set<string>();
if (excludeManifest) {
  const manifest = JSON.parse(readFileSync(excludeManifest, "utf8")) as {
    domains: Record<string, string>;
  };
  for (const [hash, split] of Object.entries(manifest.domains))
    if (split !== "train") forbidden.add(hash);
}

// --- the featurizer's walk, carrying the hyu attribute ----------------------
type ElementLike = {
  localName?: string;
  nodeName: string;
  firstChild: unknown;
  nextSibling: unknown;
  nodeType: number;
  getAttribute?: (name: string) => string | null;
};
/** hyu attribute value -> the featurizer's preorder index for that element. */
function hyuToNodeId(document: { body?: unknown }): Map<string, number> {
  const out = new Map<string, number>();
  const body = document.body as ElementLike | null | undefined;
  if (!body) return out;
  const tagOf = (el: ElementLike) => (el.localName ?? el.nodeName.toLowerCase()).toLowerCase();
  const stack: ElementLike[] = [body];
  let index = 0;
  while (stack.length > 0) {
    const element = stack.pop()!;
    if (SKIP_SUBTREE.has(tagOf(element))) continue;
    const hyu = element.getAttribute?.("hyu");
    if (hyu != null && !out.has(hyu)) out.set(hyu, index);
    index++;
    const children: ElementLike[] = [];
    for (
      let child = element.firstChild as ElementLike | null;
      child;
      child = child.nextSibling as ElementLike | null
    )
      if (child.nodeType === 1 && !SKIP_SUBTREE.has(tagOf(child))) children.push(child);
    for (let at = children.length - 1; at >= 0; at--) stack.push(children[at]!);
  }
  return out;
}

// --- output -----------------------------------------------------------------
function gzWriter(path: string) {
  const gzip = createGzip({ level: 6 });
  const sink = createWriteStream(path);
  gzip.pipe(sink);
  return {
    async write(line: string) {
      if (!gzip.write(`${line}\n`)) await new Promise((r) => gzip.once("drain", r));
    },
    close: () =>
      new Promise<void>((res, rej) => {
        sink.on("finish", res);
        sink.on("error", rej);
        gzip.on("error", rej);
        gzip.end();
      }),
  };
}

async function* gzLines(path: string): AsyncGenerator<string> {
  const stream = createReadStream(path).pipe(createGunzip());
  for await (const line of createInterface({ input: stream, crlfDelay: Infinity }))
    if (line) yield line;
}

const nodeWriter = gzWriter(`${outDir}/nodes-train.jsonl.gz`);
const pageWriter = gzWriter(`${outDir}/pages-train.jsonl.gz`);
const indexWriter = gzWriter(`${outDir}/page-index.jsonl.gz`);

const stats = {
  cohort: COHORT,
  featurizerVersion: FEATURIZER_VERSION,
  builtAt: new Date().toISOString(),
  pages: 0,
  pagesSkippedNoAnnotations: 0,
  pagesSkippedForbiddenDomain: 0,
  pagesSkippedNoUrl: 0,
  parseFailed: 0,
  nodeRows: 0,
  labelledNodes: 0,
  segmentsResolved: 0,
  segmentsUnresolved: 0,
  segmentsNotACandidate: 0,
  divSoupPages: 0,
  labelledOnDivSoupPages: 0,
  byLabel: {} as Record<string, number>,
  byLabelOnDivSoup: {} as Record<string, number>,
  agreement: { unanimous: 0, majority: 0 },
  domains: 0,
  languages: {} as Record<string, number>,
};
const domainsSeen = new Set<string>();

for await (const line of gzLines(`${sourceDir}/pages.jsonl.gz`)) {
  if (limit > 0 && stats.pages >= limit) break;
  const page = JSON.parse(line) as {
    page_id: string;
    html: string;
    topic: string;
    language: string;
  };
  const pageVotes = votes.get(page.page_id);
  if (!pageVotes || pageVotes.size === 0) {
    stats.pagesSkippedNoAnnotations++;
    continue;
  }

  const url = pageUrls[page.page_id];
  // A page whose domain lands in OUR val or holdout is dropped outright. It is
  // the same web page a holdout domain would serve, so training on it is
  // contamination however the row is tagged.
  let domain = "";
  if (url) {
    domain = parseDomain(url, { allowPrivateDomains: true }).domain ?? "";
    if (domain) {
      const domainHash = sha256(domain).slice(0, 16);
      if (forbidden.has(domainHash) || splitOf(domain) !== "train") {
        stats.pagesSkippedForbiddenDomain++;
        continue;
      }
    }
  } else {
    stats.pagesSkippedNoUrl++;
  }

  let featurized;
  let regions;
  let hyuMap;
  try {
    const { document } = parseHTML(page.html);
    hyuMap = hyuToNodeId(document as never);
    featurized = featurizeDocument(
      document as never,
      url ?? `https://invalid.example/${page.page_id}`,
      {
        emitMasked: true,
      },
    );
    regions = silverRegionLabels(featurized.nodes);
  } catch {
    stats.parseFailed++;
    continue;
  }
  if (featurized.nodes.length === 0) continue;

  const candidateIds = new Set(featurized.nodes.map((node) => node.nodeId));
  /** nodeId -> {label, weight} from the human annotators. */
  const humanLabels = new Map<number, { label: string; weight: number }>();
  const annotatorCount = Math.max(1, annotators.get(page.page_id)?.size ?? 1);
  for (const [hyuIndex, tallies] of pageVotes) {
    const nodeId = hyuMap.get(String(hyuIndex));
    if (nodeId === undefined) {
      stats.segmentsUnresolved++;
      continue;
    }
    stats.segmentsResolved++;
    if (!candidateIds.has(nodeId)) {
      stats.segmentsNotACandidate++;
      continue;
    }
    let bestTag = "";
    let bestVotes = 0;
    let total = 0;
    for (const [tag, count] of tallies) {
      total += count;
      if (count > bestVotes) {
        bestVotes = count;
        bestTag = tag;
      }
    }
    const label = TAG_TO_REGION[bestTag];
    if (!label) continue;
    // Weight is votes for the winning label over the page's ANNOTATOR COUNT,
    // not over the annotators who happened to mark this node.
    //
    // Measured on the release: of 6,289 distinct labelled nodes, annotators
    // agree on the tagType 6,153 times against 136 splits — 97.8%. So label
    // disagreement is not where the uncertainty lives. It lives in WHICH node
    // gets picked: only 51% of nodes were marked by two or more annotators,
    // because each was asked for "the largest visible coherent DOM node" and
    // reasonable people choose an ancestor or a child of the same region.
    // Dividing by the markers would score 97.8% of these 1.0 and throw that
    // away; dividing by the annotators keeps it — 3 of 3 is 1.0, 1 of 3 is
    // 0.33 — and a node only one person called the footer should count for
    // less than one all three did.
    const weight = Number((bestVotes / annotatorCount).toFixed(4));
    void total;
    humanLabels.set(nodeId, { label, weight });
    if (bestVotes === total) stats.agreement.unanimous++;
    else stats.agreement.majority++;
  }

  const pageId = sha256(page.page_id).slice(0, 16);
  const domainHash = domain ? sha256(domain).slice(0, 16) : sha256(page.page_id).slice(0, 16);
  domainsSeen.add(domainHash);
  const divSoup = !regions.regionTrainable;
  if (divSoup) stats.divSoupPages++;
  stats.languages[page.language] = (stats.languages[page.language] ?? 0) + 1;

  await pageWriter.write(
    JSON.stringify({
      p: pageId,
      d: domainHash,
      s: "train",
      cohort: COHORT,
      l: "unlabelled",
      ls: "none",
      leak: false,
      types: [],
      f: featurized.page.features,
      k: featurized.page.tokens,
      divSoup,
      landmarkKinds: regions.landmarkKinds,
      elementCount: featurized.page.elementCount,
      nodeCount: featurized.nodes.length,
      excludedReason: regions.excludedReason,
      language: page.language,
      topic: page.topic,
    }),
  );
  await indexWriter.write(
    JSON.stringify({
      p: pageId,
      u: url ?? "",
      domain: domain || "(unknown)",
      s: "train",
      cohort: COHORT,
    }),
  );

  const silverById = new Map(regions.labels.map((label) => [label.nodeId, label]));
  for (const node of featurized.nodes) {
    const silver = silverById.get(node.nodeId);
    if (!silver) continue;
    const human = humanLabels.get(node.nodeId);
    // A human label replaces the silver one outright: three people looking at a
    // rendered page beat a landmark tag that may not be there. Where there is
    // no human label the silver one is kept, so the page still contributes its
    // ordinary rows.
    const label = human?.label ?? silver.label;
    const weight = human ? human.weight : silver.weight;
    const source = human ? "human" : silver.labelSource;
    if (human) {
      stats.labelledNodes++;
      stats.byLabel[human.label] = (stats.byLabel[human.label] ?? 0) + 1;
      if (divSoup) {
        stats.labelledOnDivSoupPages++;
        stats.byLabelOnDivSoup[human.label] = (stats.byLabelOnDivSoup[human.label] ?? 0) + 1;
      }
    }
    const base = {
      p: pageId,
      n: node.nodeId,
      d: domainHash,
      s: "train",
      cohort: COHORT,
      l: label,
      ls: source,
      c: silver.containerLabel,
      t: Boolean(human) || silver.trainable,
      inh: human ? false : silver.inherited,
      w: weight,
      tagRaw: node.tag,
      roleRaw: node.role,
      depth: node.depth,
      parent: node.parentCandidateId,
    };
    await nodeWriter.write(JSON.stringify({ ...base, m: 0, f: node.features, k: node.tokens }));
    await nodeWriter.write(
      JSON.stringify({ ...base, m: 1, f: node.maskedFeatures, k: node.maskedTokens }),
    );
    stats.nodeRows += 2;
  }
  stats.pages++;
  if (stats.pages % 100 === 0)
    console.log(
      JSON.stringify({ stage: "progress", pages: stats.pages, labelled: stats.labelledNodes }),
    );
}

await nodeWriter.close();
await pageWriter.close();
await indexWriter.close();
stats.domains = domainsSeen.size;
writeFileSync(`${outDir}/feature-schema.json`, `${JSON.stringify(featureSchema(), null, 2)}\n`);
writeFileSync(`${outDir}/stats.json`, `${JSON.stringify(stats, null, 2)}\n`, { mode: 0o600 });
console.log(JSON.stringify({ stage: "complete", ...stats, languages: undefined }));

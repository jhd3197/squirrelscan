// The inference entry point: HTML in, region and page-type labels out.
//
// One contract for both paths. Every result carries {label, confidence, source}
// where source is "model" or "rules", and a consumer never branches on source
// except to display it. The rules path runs when:
//
//   - no artifact was supplied, or it failed to parse;
//   - the artifact was built for a different featurizer version;
//   - the model's confidence for that node is below the artifact's threshold;
//   - the time budget ran out partway through a page.
//
// A missing model therefore degrades attribution, never detection.

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import {
  featurizeDocument,
  FEATURIZER_VERSION,
  TOKEN_BUCKETS,
  type NodeFeatures,
} from "../featurize.ts";
import { ArtifactError, loadModel, scoreRow, softmax, type Model } from "./model-format.ts";
import {
  declaresOwnRegion,
  isRootPath,
  rulePageType,
  ruleRegion,
  type PageTypeLabel,
  type RegionLabel,
} from "./rules-fallback.ts";

export type Source = "model" | "rules";

export interface Prediction<T extends string> {
  label: T;
  confidence: number;
  source: Source;
  /**
   * What the model said before the per-class threshold rejected it. Present
   * ONLY on an abstention (`label: "unknown"`, `source: "model"`), and only for
   * debugging: it is below the precision bar the thresholds enforce, so it must
   * not be displayed to a user or used to name a fix.
   */
  rawLabel?: string;
  rawConfidence?: number;
}

export interface NodePrediction extends Prediction<RegionLabel> {
  /** Stable reference to the node within this page: its preorder element
   *  index, the tag, and the id attribute when the page provided one. */
  nodeRef: { nodeId: number; tag: string; elementId: string };
}

export interface PageClassification {
  pageType: Prediction<PageTypeLabel>;
  nodes: NodePrediction[];
  /** Why any part of this page fell back, or "" when nothing did. */
  degraded: string;
}

export interface ClassifyOptions {
  /** Parsed artifact. Omit to run the deterministic path only. */
  model?: Model | null;
  /** Wall-clock budget for the whole page. Exceeding it finishes the remaining
   *  nodes through the rules path rather than returning a partial answer. */
  budgetMs?: number;
  /** Override the artifact's own threshold, for evaluation. */
  threshold?: number;
}

const DEFAULT_BUDGET_MS = 25;

/** Parse an artifact, returning null (never throwing) when it cannot be used. */
export function tryLoadModel(json: unknown): { model: Model | null; reason: string } {
  try {
    return { model: loadModel(json), reason: "" };
  } catch (error) {
    const reason = error instanceof ArtifactError ? error.message : "artifact failed to parse";
    return { model: null, reason };
  }
}

/** Parses through the same entry the CLI uses, so inference sees the same DOM
 *  the training data was built from. */
export function classifyPage(
  html: string,
  url: string,
  options: ClassifyOptions = {},
): PageClassification {
  const { document } = parseHTML(html);
  return classifyDocument(document, url, options);
}

/** For callers that already hold a parsed document, which the CLI does. */
export function classifyDocument(
  document: unknown,
  url: string,
  options: ClassifyOptions = {},
): PageClassification {
  const started = Date.now();
  const budget = options.budgetMs ?? DEFAULT_BUDGET_MS;
  const featurized = featurizeDocument(document as never, url);
  const byId = new Map<number, NodeFeatures>();
  for (const node of featurized.nodes) byId.set(node.nodeId, node);

  const model = options.model ?? null;
  let degraded = "";
  if (!model) {
    degraded = "no model artifact";
  } else if (model.featurizerVersion !== FEATURIZER_VERSION) {
    degraded = `artifact built for featurizer ${model.featurizerVersion}, this build is ${FEATURIZER_VERSION}`;
  } else if (model.tokenBuckets !== TOKEN_BUCKETS) {
    degraded = "artifact token-bucket count does not match the featurizer";
  } else if (
    featurized.nodes.length > 0 &&
    featurized.nodes[0]!.features.length !== model.nodeFeatureCount
  ) {
    degraded = "artifact node-feature width does not match the featurizer";
  }
  const usable = degraded === "" ? model : null;

  // Page type. The homepage rule runs FIRST, ahead of the model, because it is
  // the one page class whose silver label came from the URL: the model was
  // never trained on it and cannot produce it.
  let pageType: Prediction<PageTypeLabel>;
  const ruleType = rulePageType(document as never, url);
  if (isRootPath(url) && ruleType === "homepage") {
    pageType = { label: "homepage", confidence: 1, source: "rules" };
  } else if (usable) {
    const values = toVector(
      featurized.page.features,
      featurized.page.tokens,
      usable.pageFeatureCount,
      null,
      null,
    );
    const scores = new Float32Array(usable.pageType.numClass);
    scoreRow(usable.pageType, values, scores);
    const { index, probability } = softmax(scores);
    // Per class when the artifact carries them (page type v8); otherwise every
    // entry equals the head's single confidenceThreshold.
    const threshold = options.threshold ?? usable.pageType.classThresholds[index]!;
    pageType =
      probability >= threshold
        ? {
            label: usable.pageType.classes[index] as PageTypeLabel,
            confidence: probability,
            source: "model",
          }
        : { label: ruleType, confidence: probability, source: "rules" };
  } else {
    pageType = { label: ruleType, confidence: ruleType === "unknown" ? 0 : 1, source: "rules" };
  }

  const nodes: NodePrediction[] = [];
  const scores = usable ? new Float32Array(usable.region.numClass) : null;
  let values: Float32Array | null = null;
  let previousTokens: number[] | null = null;
  let budgetSpent = false;

  for (const node of featurized.nodes) {
    const nodeRef = { nodeId: node.nodeId, tag: node.tag, elementId: node.elementId };
    if (!usable || !scores || budgetSpent) {
      nodes.push({ nodeRef, ...withRules(node, byId) });
      continue;
    }
    // MARKUP WINS. A node carrying its own landmark tag or ARIA role has been
    // told what it is by the page author, and the model is not asked. "Model
    // primary, rules fallback" is about nodes where the markup is SILENT; it
    // never means overriding a correct declaration.
    //
    // This is not a style preference, it is a measured defect: with the model
    // consulted first, a <nav class="main-navigation"> came back content_block
    // at 0.74 on 66 of 68 pages of a stored site, contradicting a deterministic
    // answer that was right. See runtime/README.md.
    if (declaresOwnRegion(node)) {
      nodes.push({ nodeRef, ...withRules(node, byId) });
      continue;
    }
    // Checked every 32 nodes: Date.now() per node is itself measurable at the
    // scale this runs at (400 nodes x pages x rules).
    if ((nodes.length & 31) === 0 && Date.now() - started > budget) {
      budgetSpent = true;
      degraded = degraded || `time budget ${budget}ms exceeded`;
      nodes.push({ nodeRef, ...withRules(node, byId) });
      continue;
    }
    values = toVector(node.features, node.tokens, usable.nodeFeatureCount, values, previousTokens);
    previousTokens = node.tokens;
    scoreRow(usable.region, values, scores);
    const { index, probability } = softmax(scores);
    // Per class, not global. One threshold cannot serve ten classes whose
    // precision curves differ by 40 points, and the single value the old rule
    // selected (0.95) left two thirds of nodes unnamed.
    const threshold = options.threshold ?? usable.region.classThresholds[index]!;
    if (probability >= threshold) {
      nodes.push({
        nodeRef,
        label: usable.region.classes[index] as RegionLabel,
        confidence: probability,
        source: "model",
      });
    } else {
      // The model was asked and declined. That is a MODEL answer of "unknown",
      // not a rules answer: saying `source: "rules"` here would claim the
      // deterministic path had looked and found nothing, which it has not.
      nodes.push({
        nodeRef,
        label: "unknown",
        confidence: 0,
        source: "model",
        rawLabel: usable.region.classes[index],
        rawConfidence: probability,
      });
    }
  }

  return { pageType, nodes, degraded };
}

function withRules(
  node: NodeFeatures,
  byId: ReadonlyMap<number, NodeFeatures>,
): Prediction<RegionLabel> {
  const label = ruleRegion(node, byId);
  return { label, confidence: label === "unknown" ? 0 : 1, source: "rules" };
}

/**
 * Numeric features followed by the one-hot token buckets, as one float32 row.
 *
 * The buffer is reused across nodes, and the token half is cleared by UNSETTING
 * the handful of buckets the previous node set rather than zeroing all 512 of
 * them. A page's 400 nodes set about 8 buckets each, so the naive `fill(0)`
 * was doing ~200k redundant writes per page for ~3k useful ones. The numeric
 * half needs no clearing: every one of its slots is written every time.
 */
function toVector(
  features: number[],
  tokens: number[],
  numericCount: number,
  reuse: Float32Array | null,
  previousTokens: number[] | null,
): Float32Array {
  const width = numericCount + TOKEN_BUCKETS;
  const out = reuse && reuse.length === width ? reuse : new Float32Array(width);
  if (previousTokens && out === reuse) {
    for (const bucket of previousTokens) out[numericCount + bucket] = 0;
  } else {
    out.fill(0);
  }
  for (let i = 0; i < numericCount; i++) out[i] = features[i] ?? 0;
  for (const bucket of tokens) out[numericCount + bucket] = 1;
  return out;
}

/** The region head's raw class probabilities for one node, for evaluation and
 *  for the parity fixture. Returns a copy, not the reused buffer. */
export function regionProbabilities(model: Model, node: NodeFeatures): Float32Array {
  const values = toVector(node.features, node.tokens, model.nodeFeatureCount, null, null);
  const scores = new Float32Array(model.region.numClass);
  scoreRow(model.region, values, scores);
  softmax(scores);
  return scores;
}

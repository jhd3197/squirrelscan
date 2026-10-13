// The skeleton page-type head at runtime (#2307): page type from the page skeleton alone.
//
// The head is trained on teacher labels (and checked against declared schema) with schema, og:type and
// URL tokens kept OUT of its inputs, so "this looks like an article" is evidence a finding can use about a
// page that declares nothing. It reads the numeric skeleton features (skeleton.ts) in the order the exporter
// recorded, and walks the same packed trees as every other head (model-format.ts).
//
// A class whose precision goal was not met at any threshold abstains: its threshold is above 1, so it never
// fires, and `answered` says whether the top class cleared its own threshold.
//
//   const head = loadSkeletonHead(JSON.parse(readFileSync("skeleton-head.json", "utf8")));
//   const result = classifySkeleton(document, head); // { label: "editorial", confidence: 0.93, answered: true, ... }

import { decodeHead, scoreRow, type Head, type HeadJson } from "./model-format.ts";
import { buildSkeleton, skeletonFeatures } from "./skeleton.ts";

export const SKELETON_HEAD_FORMAT = "squirrel-skeleton-head";
export const SKELETON_HEAD_VERSION = 1;

export interface SkeletonHeadJson {
  format: string;
  headVersion: number;
  featureNames: string[];
  head: HeadJson;
  /** Per class: the threshold and the held-out precision/recall it was chosen at (null when the class abstains). */
  evaluation?: Record<string, { bar: number; precision: number; recall: number } | null>;
}

export interface SkeletonHead {
  head: Head;
  featureNames: string[];
  evaluation: SkeletonHeadJson["evaluation"];
}

export class SkeletonHeadError extends Error {}

export interface SkeletonPrediction {
  /** The most likely class. */
  label: string;
  confidence: number;
  /** True when `confidence` reaches that class's own threshold; false for an abstaining or unsure answer. */
  answered: boolean;
  probabilities: Record<string, number>;
}

export function loadSkeletonHead(json: unknown): SkeletonHead {
  const parsed = json as SkeletonHeadJson;
  if (!parsed || parsed.format !== SKELETON_HEAD_FORMAT) {
    throw new SkeletonHeadError("not a skeleton head");
  }
  if (parsed.headVersion !== SKELETON_HEAD_VERSION) {
    throw new SkeletonHeadError(`unsupported head version ${parsed.headVersion}`);
  }
  if (!Array.isArray(parsed.featureNames) || !parsed.head) {
    throw new SkeletonHeadError("skeleton head is incomplete");
  }
  return {
    head: decodeHead(parsed.head),
    featureNames: parsed.featureNames,
    evaluation: parsed.evaluation,
  };
}

/** Probabilities from raw class scores (a softmax, max-shifted like the exporter's reference). */
function probabilities(scores: Float32Array): Float64Array {
  let max = -Infinity;
  for (const score of scores) max = Math.max(max, score);
  const out = new Float64Array(scores.length);
  let total = 0;
  for (let i = 0; i < scores.length; i++) {
    out[i] = Math.exp(scores[i]! - max);
    total += out[i]!;
  }
  for (let i = 0; i < out.length; i++) out[i] = out[i]! / total;
  return out;
}

/** The head's answer for a feature row given in `head.featureNames` order. */
export function predictSkeletonRow(head: SkeletonHead, values: Float32Array): SkeletonPrediction {
  if (values.length !== head.featureNames.length) {
    throw new SkeletonHeadError(
      `expected ${head.featureNames.length} features, got ${values.length}`,
    );
  }
  const scores = new Float32Array(head.head.numClass);
  scoreRow(head.head, values, scores);
  const probs = probabilities(scores);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i]! > probs[best]!) best = i;
  return {
    label: head.head.classes[best]!,
    confidence: probs[best]!,
    answered: probs[best]! >= head.head.classThresholds[best]!,
    probabilities: Object.fromEntries(head.head.classes.map((name, i) => [name, probs[i]!])),
  };
}

/** Classify a parsed page: skeleton, features in the head's order, then the trees. */
export function classifySkeleton(document: unknown, head: SkeletonHead): SkeletonPrediction {
  const features = skeletonFeatures(buildSkeleton(document)) as Record<string, number>;
  const values = new Float32Array(head.featureNames.length);
  head.featureNames.forEach((name, i) => {
    const value = features[name];
    if (value === undefined) {
      throw new SkeletonHeadError(`feature ${name} is not produced by this skeleton version`);
    }
    values[i] = value;
  });
  return predictSkeletonRow(head, values);
}

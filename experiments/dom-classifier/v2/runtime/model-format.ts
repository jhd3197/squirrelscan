// Decoding and evaluating the packed classifier artifact, in pure TypeScript.
//
// The arithmetic here is deliberately boring, because it has to agree with
// export_model.py's reference evaluator exactly: thresholds are float32,
// compared with <=, left when true. `Math.fround` is what makes the comparison
// float32 in a language whose numbers are float64 — without it a threshold
// that rounded to float32 on the way out would be compared against a float64
// feature value, and the two runtimes would disagree on the rows that sit
// between the two representations.
//
// Layout (little-endian), one block per head:
//
//   magic "SQDC" | uint32 version | uint32 numTrees | uint32 numClass
//   uint32 perTreeInternal[numTrees]
//   uint32 perTreeLeaves[numTrees]
//   (pad to 4) uint16 feature[totalInternal]
//   (pad to 4) float32 threshold[totalInternal]
//   int32 left[totalInternal]    >= 0 internal node index, < 0 leaf -(v)-1
//   int32 right[totalInternal]
//   float32 leaf[totalLeaves]

export const ARTIFACT_FORMAT = "squirrel-dom-classifier";
export const ARTIFACT_VERSION = 1;
const MAGIC = 0x43_44_51_53; // "SQDC" little-endian

export interface HeadJson {
  classes: string[];
  featureColumns: number[] | null;
  confidenceThreshold: number;
  numClass: number;
  numTrees: number;
  trees: string;
}

export interface ArtifactJson {
  format: string;
  artifactVersion: number;
  featurizerVersion: string;
  nodeFeatureCount: number;
  pageFeatureCount: number;
  tokenBuckets: number;
  region: HeadJson;
  pageType: HeadJson;
}

export interface Head {
  classes: string[];
  /** Indices into the featurizer's full vector, or null for "all of it". */
  featureColumns: Int32Array | null;
  confidenceThreshold: number;
  numClass: number;
  numTrees: number;
  treeStart: Int32Array;
  leafStart: Int32Array;
  perInternal: Int32Array;
  feature: Uint16Array;
  threshold: Float32Array;
  left: Int32Array;
  right: Int32Array;
  leaf: Float32Array;
}

export interface Model {
  featurizerVersion: string;
  nodeFeatureCount: number;
  pageFeatureCount: number;
  tokenBuckets: number;
  region: Head;
  pageType: Head;
}

export class ArtifactError extends Error {}

function decodeBase64(value: string): ArrayBuffer {
  // A fresh, 4-byte-aligned buffer, so the typed-array views below are legal.
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function decodeHead(json: HeadJson): Head {
  const buffer = decodeBase64(json.trees);
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC) throw new ArtifactError("bad tree-block magic");
  const version = view.getUint32(4, true);
  if (version !== ARTIFACT_VERSION) {
    throw new ArtifactError(`unsupported tree-block version ${version}`);
  }
  const numTrees = view.getUint32(8, true);
  const numClass = view.getUint32(12, true);
  if (numClass !== json.numClass || numTrees !== json.numTrees) {
    throw new ArtifactError("tree block disagrees with its own header");
  }

  let offset = 16;
  const perInternal = new Int32Array(numTrees);
  const perLeaves = new Int32Array(numTrees);
  for (let i = 0; i < numTrees; i++) perInternal[i] = view.getUint32(offset + i * 4, true);
  offset += numTrees * 4;
  for (let i = 0; i < numTrees; i++) perLeaves[i] = view.getUint32(offset + i * 4, true);
  offset += numTrees * 4;

  let totalInternal = 0;
  let totalLeaves = 0;
  const treeStart = new Int32Array(numTrees);
  const leafStart = new Int32Array(numTrees);
  for (let i = 0; i < numTrees; i++) {
    treeStart[i] = totalInternal;
    leafStart[i] = totalLeaves;
    totalInternal += perInternal[i]!;
    totalLeaves += perLeaves[i]!;
  }

  const feature = new Uint16Array(buffer, offset, totalInternal);
  offset += totalInternal * 2;
  offset += (4 - (offset % 4)) % 4;
  const threshold = new Float32Array(buffer, offset, totalInternal);
  offset += totalInternal * 4;
  const left = new Int32Array(buffer, offset, totalInternal);
  offset += totalInternal * 4;
  const right = new Int32Array(buffer, offset, totalInternal);
  offset += totalInternal * 4;
  const leaf = new Float32Array(buffer, offset, totalLeaves);
  offset += totalLeaves * 4;
  if (offset > buffer.byteLength) throw new ArtifactError("tree block is truncated");

  return {
    classes: json.classes,
    featureColumns: json.featureColumns ? Int32Array.from(json.featureColumns) : null,
    confidenceThreshold: json.confidenceThreshold,
    numClass,
    numTrees,
    treeStart,
    leafStart,
    perInternal,
    feature,
    threshold,
    left,
    right,
    leaf,
  };
}

/** Parse an artifact. Throws ArtifactError on anything it cannot trust. */
export function loadModel(json: unknown): Model {
  const artifact = json as ArtifactJson;
  if (!artifact || artifact.format !== ARTIFACT_FORMAT) {
    throw new ArtifactError("not a squirrel-dom-classifier artifact");
  }
  if (artifact.artifactVersion !== ARTIFACT_VERSION) {
    throw new ArtifactError(`unsupported artifact version ${artifact.artifactVersion}`);
  }
  return {
    featurizerVersion: artifact.featurizerVersion,
    nodeFeatureCount: artifact.nodeFeatureCount,
    pageFeatureCount: artifact.pageFeatureCount,
    tokenBuckets: artifact.tokenBuckets,
    region: decodeHead(artifact.region),
    pageType: decodeHead(artifact.pageType),
  };
}

/**
 * Raw class scores for one row. `values` is the FULL featurizer vector
 * (numeric features followed by the token buckets); `featureColumns` selects
 * the subset this head was trained on, so the caller never has to know which.
 */
export function scoreRow(head: Head, values: Float32Array, out: Float32Array): void {
  out.fill(0);
  const { feature, threshold, left, right, leaf, treeStart, leafStart, perInternal } = head;
  const columns = head.featureColumns;
  for (let tree = 0; tree < head.numTrees; tree++) {
    const cls = tree % head.numClass;
    const internal = perInternal[tree]!;
    if (internal === 0) {
      // A stump: one leaf and no split, which the child encoding cannot name,
      // so it is addressed by the tree's own leaf offset.
      out[cls]! += leaf[leafStart[tree]!]!;
      continue;
    }
    let node = treeStart[tree]!;
    for (;;) {
      const column = feature[node]!;
      const value = columns === null ? values[column]! : values[columns[column]!]!;
      // float32 on both sides, exactly as the exporter's reference does.
      const next = Math.fround(value) <= threshold[node]! ? left[node]! : right[node]!;
      if (next < 0) {
        // The exporter re-bases child pointers into the CONCATENATED arrays, so
        // -next-1 is already the global leaf index. Adding leafStart here as
        // well double-counts the offset, which reads a leaf from a later tree
        // and quietly produces a plausible wrong answer: it cost 9,718 of
        // 10,000 parity rows.
        out[cls]! += leaf[-next - 1]!;
        break;
      }
      node = next;
    }
  }
}

/** Softmax in place, returning the argmax and its probability. */
export function softmax(scores: Float32Array): { index: number; probability: number } {
  let max = -Infinity;
  for (let i = 0; i < scores.length; i++) if (scores[i]! > max) max = scores[i]!;
  let total = 0;
  for (let i = 0; i < scores.length; i++) {
    const value = Math.exp(scores[i]! - max);
    scores[i] = value;
    total += value;
  }
  let best = 0;
  for (let i = 0; i < scores.length; i++) {
    scores[i] = scores[i]! / total;
    if (scores[i]! > scores[best]!) best = i;
  }
  return { index: best, probability: scores[best]! };
}

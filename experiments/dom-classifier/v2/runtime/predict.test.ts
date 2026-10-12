// G4: the TypeScript evaluator has to agree with the Python one, the artifact
// has to fit, and the fallback has to produce the same shape as the model.
//
// The parity fixture and the artifact live in the PRIVATE data root, so these
// tests skip themselves when DOM_CLASSIFIER_DATA_ROOT is unset. Everything that
// does not need them runs unconditionally.

import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync, statSync } from "node:fs";

import {
  ArtifactError,
  loadModel,
  scoreRow,
  scoreRows,
  softmax,
  thresholdsFor,
} from "./model-format.ts";
import { classifyPage, tryLoadModel, withExtraBlock } from "./predict.ts";
import { isRootPath } from "./rules-fallback.ts";

const DATA_ROOT = process.env.DOM_CLASSIFIER_DATA_ROOT ?? "";
const ARTIFACT = `${DATA_ROOT}/models/artifact/classifier-v2.json`;
const PARITY = `${DATA_ROOT}/models/artifact/parity-region.json`;
const PARITY_SECTION = `${DATA_ROOT}/models/artifact/parity-section.json`;
const PARITY_PAGETYPE = `${DATA_ROOT}/models/artifact/parity-pagetype.json`;
const HAS_ARTIFACT = DATA_ROOT !== "" && existsSync(ARTIFACT);
const HAS_PARITY = DATA_ROOT !== "" && existsSync(PARITY);

const PAGE = `<!doctype html><html lang="en"><head><title>How we test crawlers</title>
<script type="application/ld+json">{"@context":"https://schema.org","@type":"NewsArticle","headline":"x"}</script>
</head><body class="single-post">
  <header class="site-header"><nav><a href="/">Home</a> <a href="/archive">Archive</a></nav></header>
  <main><article><header class="entry-header"><h1>How we test crawlers</h1>
    <p>By Ada Lovelace</p></header>
    <p>The first thing a crawler has to get right is the queue, and the queue is harder
      than it looks once redirects are involved.</p>
    <p>We keep a fixture site with every shape we have seen in the wild.</p></article></main>
  <aside class="sidebar"><h2>Elsewhere</h2><ul><li><a href="/blog">Blog</a></li></ul></aside>
  <footer class="site-footer"><p>&copy; 2026 Example Journal, all rights reserved.</p></footer>
</body></html>`;

const URL = "https://example.com/blog/how-we-test-crawlers";

describe("the rules path, with no model at all", () => {
  test("produces the full contract and says why it degraded", () => {
    const out = classifyPage(PAGE, URL);
    expect(out.degraded).toBe("no model artifact");
    expect(out.pageType).toEqual({ label: "article", confidence: 1, source: "rules" });
    expect(out.nodes.length).toBeGreaterThan(4);
    for (const node of out.nodes) {
      expect(node.source).toBe("rules");
      expect(typeof node.label).toBe("string");
      expect(node.confidence).toBeGreaterThanOrEqual(0);
      expect(node.nodeRef.nodeId).toBeGreaterThanOrEqual(0);
    }
  });

  test("reads the landmarks, including the nested article header", () => {
    const out = classifyPage(PAGE, URL);
    const at = (tag: string, elementId: string) =>
      out.nodes.find((n) => n.nodeRef.tag === tag && n.nodeRef.elementId === elementId) ??
      out.nodes.find((n) => n.nodeRef.tag === tag);
    expect(at("header", "")!.label).toBe("site_header");
    expect(at("footer", "")!.label).toBe("footer");
    expect(at("nav", "")!.label).toBe("navigation");
    expect(at("aside", "")!.label).toBe("sidebar");
    expect(at("main", "")!.label).toBe("main_content");
    // The <header> inside <article> is the article's own chrome.
    const headers = out.nodes.filter((n) => n.nodeRef.tag === "header");
    expect(headers.map((h) => h.label).sort()).toEqual(["article_chrome", "site_header"]);
  });

  test("homepage is a URL rule, and only a rule", () => {
    const home = `<!doctype html><html><head><title>Example</title>
      <script type="application/ld+json">{"@type":"WebSite","name":"Example"}</script>
      </head><body><main><h1>Example</h1><p>We make widgets for people.</p></main></body></html>`;
    const out = classifyPage(home, "https://example.com/");
    expect(out.pageType.label).toBe("homepage");
    expect(out.pageType.source).toBe("rules");
    expect(isRootPath("https://example.com")).toBe(true);
    expect(isRootPath("https://example.com/?a=1")).toBe(true);
    expect(isRootPath("https://example.com/a")).toBe(false);
  });

  test("a page with no structured data and no landmarks is unknown, not a guess", () => {
    const soup = `<!doctype html><html><head><title>T</title></head><body>
      <div class="wrap"><div class="row"><p>Text that makes this a candidate node.</p></div></div>
    </body></html>`;
    const out = classifyPage(soup, "https://example.com/x");
    expect(out.pageType.label).toBe("unknown");
    expect(out.nodes.every((n) => n.label === "unknown")).toBe(true);
  });
});

// A WordPress thread under the article, as the comments rule reads it.
const WITH_COMMENTS = PAGE.replace(
  "</article></main>",
  `</article>
  <div id="comments" class="comments-area"><h2>2 responses</h2><ol class="comment-list">
    <li id="comment-4" class="comment"><article id="div-comment-4" class="comment-body">
      <div class="comment-author"><a rel="external nofollow ugc" href="https://spam.example.org/">Pills</a></div>
      <div class="comment-content"><p>Great post, thanks for sharing it with us.</p></div>
    </article></li></ol></div></main>`,
);

describe("the comments rule in the output (#2307)", () => {
  test("lists the thread and labels every node inside it, with no model", () => {
    const elements: unknown[] = [];
    const out = classifyPage(WITH_COMMENTS, URL, { elementsOut: elements });
    expect(out.commentSections).toEqual([
      { nodeId: expect.any(Number), label: "comments", evidence: "comments" },
    ]);
    const thread = elements[out.commentSections[0]!.nodeId] as Element;
    expect(thread.getAttribute("id")).toBe("comments");
    const inside = out.nodes.filter((n) => thread.contains(elements[n.nodeRef.nodeId] as Element));
    expect(inside.length).toBeGreaterThan(0);
    for (const node of inside) {
      expect(node.section).toMatchObject({ label: "comments", confidence: 1, source: "rules" });
    }
    for (const node of out.nodes) if (!inside.includes(node)) expect(node.section).toBeUndefined();
  });

  test("a page without a thread lists none", () => {
    expect(classifyPage(PAGE, URL).commentSections).toEqual([]);
  });
});

describe("artifact validation", () => {
  test("a non-artifact is refused without throwing", () => {
    expect(tryLoadModel({ hello: "world" }).model).toBeNull();
    expect(tryLoadModel({ hello: "world" }).reason).toContain("not a squirrel-dom-classifier");
    expect(tryLoadModel(null).model).toBeNull();
  });

  test("a future artifact version is refused", () => {
    const out = tryLoadModel({ format: "squirrel-dom-classifier", artifactVersion: 99 });
    expect(out.model).toBeNull();
    expect(out.reason).toContain("unsupported artifact version");
  });

  test("loadModel throws ArtifactError, tryLoadModel never throws", () => {
    expect(() => loadModel({ format: "nope" })).toThrow(ArtifactError);
    expect(() => tryLoadModel({ format: "nope" })).not.toThrow();
  });
});

/**
 * One single-split tree, packed by hand in the export_model.py layout:
 * feature 0 <= 0.5 goes to leaf 10, otherwise leaf 20. `flags` = null writes a
 * version-1 block; a number writes version 2 with that per-node flag byte.
 */
function oneSplitHead(flags: number | null, dropFlags = false): Record<string, unknown> {
  const bytes: number[] = [];
  const u32 = (v: number) => bytes.push(v & 255, (v >>> 8) & 255, (v >>> 16) & 255, v >>> 24);
  const f32 = (v: number) => bytes.push(...new Uint8Array(new Float32Array([v]).buffer));
  const i32 = (v: number) => u32(v >>> 0);
  bytes.push(0x53, 0x51, 0x44, 0x43); // "SQDC"
  u32(flags === null ? 1 : 2);
  u32(1); // numTrees
  u32(1); // numClass
  u32(1); // perTreeInternal
  u32(2); // perTreeLeaves
  bytes.push(0, 0, 0, 0); // feature uint16[1] = 0, padded to 4
  f32(0.5);
  i32(-1); // left -> leaf 0
  i32(-2); // right -> leaf 1
  f32(10);
  f32(20);
  if (flags !== null && !dropFlags) bytes.push(flags);
  const trees = btoa(String.fromCharCode(...bytes));
  return {
    classes: ["only"],
    featureColumns: null,
    confidenceThreshold: 0.5,
    classThresholds: null,
    numClass: 1,
    numTrees: 1,
    trees,
  };
}

function oneSplitModel(flags: number | null, dropFlags = false) {
  const head = oneSplitHead(flags, dropFlags);
  return loadModel({
    format: "squirrel-dom-classifier",
    artifactVersion: 1,
    featurizerVersion: "test",
    nodeFeatureCount: 1,
    pageFeatureCount: 1,
    tokenBuckets: 0,
    region: head,
    pageType: head,
  });
}

function leafFor(flags: number | null, value: number): number {
  const out = new Float32Array(1);
  scoreRow(oneSplitModel(flags).pageType, new Float32Array([value]), out);
  return out[0]!;
}

describe("missing-value splits (tree-block version 2, #2307)", () => {
  test("version 1 walks exactly as before, NaN included", () => {
    expect(leafFor(null, 0.2)).toBe(10);
    expect(leafFor(null, 0.9)).toBe(20);
    expect(leafFor(null, Number.NaN)).toBe(20);
  });

  test("a NaN-missing node sends NaN its default way, and compares everything else", () => {
    const nanLeft = 2 | 4;
    const nanRight = 2;
    expect(leafFor(nanLeft, Number.NaN)).toBe(10);
    expect(leafFor(nanRight, Number.NaN)).toBe(20);
    expect(leafFor(nanLeft, 0.9)).toBe(20);
    expect(leafFor(nanRight, 0.2)).toBe(10);
  });

  test("on a node without missing handling a NaN reads as 0, as in LightGBM", () => {
    expect(leafFor(0, Number.NaN)).toBe(10);
  });

  test("a zero-missing node sends 0 (and NaN, read as 0) its default way", () => {
    const zeroRight = 1;
    expect(leafFor(zeroRight, 0)).toBe(20);
    expect(leafFor(zeroRight, Number.NaN)).toBe(20);
    expect(leafFor(zeroRight, 0.2)).toBe(10);
    expect(leafFor(zeroRight | 4, 0)).toBe(10);
  });

  test("a version-2 block without its flags is truncated, and version 3 is refused", () => {
    expect(() => oneSplitModel(2, true)).toThrow(ArtifactError);
    const head = oneSplitHead(null);
    const bytes = Uint8Array.from(atob(head.trees as string), (c) => c.charCodeAt(0));
    bytes[4] = 3;
    head.trees = btoa(String.fromCharCode(...bytes));
    expect(() =>
      loadModel({
        format: "squirrel-dom-classifier",
        artifactVersion: 1,
        region: head,
        pageType: head,
      }),
    ).toThrow(ArtifactError);
  });

  test("the cross-page block is the caller's values when complete, else missing", () => {
    const base = new Float32Array([1, 2]);
    expect(withExtraBlock(base, 0, null)).toBe(base);
    expect(Array.from(withExtraBlock(base, 2, [3, 4]))).toEqual([1, 2, 3, 4]);
    const absent = withExtraBlock(base, 2, null);
    expect(absent.length).toBe(4);
    expect(Number.isNaN(absent[2]!) && Number.isNaN(absent[3]!)).toBe(true);
    const short = withExtraBlock(base, 2, [3]);
    expect(Number.isNaN(short[2]!)).toBe(true);
  });
});

describe("platform guard thresholds (#2307)", () => {
  const withGuard = (overrides: Record<string, Record<string, number>>) => {
    const head = { ...oneSplitHead(null), platformClassThresholds: overrides };
    return loadModel({
      format: "squirrel-dom-classifier",
      artifactVersion: 1,
      featurizerVersion: "test",
      nodeFeatureCount: 1,
      pageFeatureCount: 1,
      tokenBuckets: 0,
      region: oneSplitHead(null),
      pageType: head,
    });
  };

  test("a platform with an override gets it; any other platform, or none, gets the head's", () => {
    const model = withGuard({ wix: { only: 1.01 } });
    expect(thresholdsFor(model.pageType, "wix")[0]).toBe(1.01);
    expect(thresholdsFor(model.pageType, "shopify")[0]).toBe(0.5);
    expect(thresholdsFor(model.pageType, null)[0]).toBe(0.5);
    expect(thresholdsFor(model.region, "wix")[0]).toBe(0.5);
  });

  test("an override naming a class the head does not have is refused", () => {
    expect(() => withGuard({ wix: { nope: 0.9 } })).toThrow(ArtifactError);
  });
});

// The reads are LAZY. `describe.if(false)` still evaluates its body in bun, so
// a top-level readFileSync here throws when the private data root is not set --
// as an "unhandled error between tests", which fails the run without failing
// any test.
const artifactJson = (): Record<string, unknown> => JSON.parse(readFileSync(ARTIFACT, "utf8"));

describe.if(HAS_ARTIFACT)("with the real artifact", () => {
  // G4 since 2026-09-23: the 2 MB limit was dropped in favour of a reliable
  // model. The ceiling is 10 MB, and what matters at runtime is load time.
  test("it loads and fits the 10 MB ceiling", () => {
    const { model } = tryLoadModel(artifactJson());
    expect(model).not.toBeNull();
    const bytes = statSync(ARTIFACT).size;
    expect(bytes).toBeLessThanOrEqual(10_000_000);
    expect(model!.region.numClass).toBe(10);
    // 11 page-type classes since the head moved to Jev labels (the schema-derived
    // set could not express legal or pricing at all); v8 has one class per
    // expected schema.org type, minus the rare ones merged away.
    expect(model!.pageType.numClass).toBeGreaterThanOrEqual(11);
    expect(model!.pageType.classThresholds.length).toBe(model!.pageType.numClass);
  });

  test("it loads (JSON parse + init) within 100 ms", () => {
    const text = readFileSync(ARTIFACT, "utf8");
    // Best of three, so one scheduling hiccup on a loaded machine is not a
    // failure; a model that is genuinely slow to load is slow every time.
    let best = Number.POSITIVE_INFINITY;
    for (let run = 0; run < 3; run++) {
      const started = Bun.nanoseconds();
      const { model } = tryLoadModel(JSON.parse(text));
      best = Math.min(best, (Bun.nanoseconds() - started) / 1e6);
      expect(model).not.toBeNull();
    }
    expect(best).toBeLessThanOrEqual(100);
  });

  test("predictions carry a source, and low confidence falls back to rules", () => {
    const { model } = tryLoadModel(artifactJson());
    const out = classifyPage(PAGE, URL, { model });
    expect(out.degraded).toBe("");
    expect(out.nodes.some((n) => n.source === "model")).toBe(true);
    for (const node of out.nodes) {
      expect(["model", "rules"]).toContain(node.source);
      expect(node.confidence).toBeGreaterThanOrEqual(0);
      expect(node.confidence).toBeLessThanOrEqual(1);
    }
    // A threshold of 1.01 can never be met. Every node is then answered either
    // by the markup (a declared landmark, source "rules") or by the model
    // declining (source "model", label "unknown") -- never by a rules guess
    // dressed up as a model answer.
    const forced = classifyPage(PAGE, URL, { model, threshold: 1.01 });
    expect(forced.nodes.length).toBe(out.nodes.length);
    for (const node of forced.nodes) {
      if (node.source === "model") expect(node.label).toBe("unknown");
      else expect(node.source).toBe("rules");
    }
    expect(forced.nodes.some((n) => n.source === "rules")).toBe(true);
  });

  test("a featurizer-version mismatch degrades instead of predicting nonsense", () => {
    const stale = { ...artifactJson(), featurizerVersion: "v1.9.9" };
    const out = classifyPage(PAGE, URL, { model: tryLoadModel(stale).model });
    expect(out.degraded).toContain("v1.9.9");
    expect(out.nodes.every((n) => n.source === "rules")).toBe(true);
  });

  test("the time budget hands the rest of the page to the rules path", () => {
    // Div soup on purpose. PAGE is all landmarks, and a node that declares its
    // own landmark is answered by the rules before the budget is ever
    // consulted -- so on PAGE this test would pass for the wrong reason.
    const soup = `<!doctype html><html><head><title>T</title></head><body>${Array.from(
      { length: 40 },
      (_, i) =>
        `<div class="row r${i}"><p>Block ${i} with enough text to be a candidate.</p></div>`,
    ).join("")}</body></html>`;
    const { model } = tryLoadModel(artifactJson());
    // The precondition is that the model PATH is reachable, not that the model
    // wins the confidence threshold on this synthetic page.
    const normal = classifyPage(soup, URL, { model });
    expect(normal.nodes.length).toBeGreaterThan(10);
    expect(normal.nodes.every((n) => n.nodeRef.tag === "div")).toBe(true);
    expect(normal.degraded).toBe("");
    const out = classifyPage(soup, URL, { model, budgetMs: -1 });
    expect(out.degraded).toContain("time budget");
    expect(out.nodes.every((n) => n.source === "rules")).toBe(true);
  });

  test("a node that declares its own landmark is answered by the markup", () => {
    const { model } = tryLoadModel(artifactJson());
    const out = classifyPage(PAGE, URL, { model });
    const at = (tag: string) => out.nodes.filter((n) => n.nodeRef.tag === tag);
    for (const tag of ["nav", "footer", "aside", "main"]) {
      for (const node of at(tag)) {
        expect([tag, node.source]).toEqual([tag, "rules"]);
        expect([tag, node.confidence]).toEqual([tag, 1]);
      }
    }
    expect(at("nav")[0]!.label).toBe("navigation");
    expect(at("footer")[0]!.label).toBe("footer");
  });
});

describe.if(HAS_ARTIFACT && HAS_PARITY)("G4 parity with the Python evaluator", () => {
  test("10k holdout rows: identical argmax, probabilities within 1e-5", () => {
    const model = loadModel(artifactJson());
    const fixture = JSON.parse(readFileSync(PARITY, "utf8")) as {
      rows: number;
      width: number;
      x: Array<{ i: number[]; v: number[] }>;
      probs: number[][];
    };
    expect(fixture.rows).toBeGreaterThanOrEqual(10_000);

    const head = model.region;
    const values = new Float32Array(fixture.width);
    const scores = new Float32Array(head.numClass);
    let maxDiff = 0;
    let argmaxMatches = 0;
    for (let row = 0; row < fixture.rows; row++) {
      values.fill(0);
      const sparse = fixture.x[row]!;
      for (let k = 0; k < sparse.i.length; k++) values[sparse.i[k]!] = sparse.v[k]!;
      scoreRow(head, values, scores);
      const { index } = softmax(scores);
      const expected = fixture.probs[row]!;
      let expectedArgmax = 0;
      for (let c = 1; c < expected.length; c++) {
        if (expected[c]! > expected[expectedArgmax]!) expectedArgmax = c;
      }
      if (index === expectedArgmax) argmaxMatches++;
      for (let c = 0; c < expected.length; c++) {
        const diff = Math.abs(scores[c]! - expected[c]!);
        if (diff > maxDiff) maxDiff = diff;
      }
    }
    expect(argmaxMatches).toBe(fixture.rows);
    expect(maxDiff).toBeLessThan(1e-5);
  });
});

describe.if(HAS_ARTIFACT && HAS_PARITY && existsSync(PARITY_SECTION))(
  "batched scoring is the row-at-a-time walk, bit for bit (#2307)",
  () => {
    test("scoreRows equals scoreRow on every parity row, in batches of 1, 7, 32 and all", () => {
      const model = loadModel(artifactJson());
      for (const [path, head] of [
        [PARITY, model.region],
        [PARITY_SECTION, model.section!],
      ] as const) {
        const fixture = JSON.parse(readFileSync(path, "utf8")) as {
          rows: number;
          width: number;
          x: Array<{ i: number[]; v: number[] }>;
        };
        const rows = Math.min(fixture.rows, 2_000);
        const matrix = new Float32Array(rows * fixture.width);
        for (let row = 0; row < rows; row++) {
          const sparse = fixture.x[row]!;
          for (let k = 0; k < sparse.i.length; k++) {
            matrix[row * fixture.width + sparse.i[k]!] = sparse.v[k]!;
          }
        }
        const expected = new Float32Array(rows * head.numClass);
        const scores = new Float32Array(head.numClass);
        for (let row = 0; row < rows; row++) {
          scoreRow(head, matrix.subarray(row * fixture.width, (row + 1) * fixture.width), scores);
          expected.set(scores, row * head.numClass);
        }
        for (const size of [1, 7, 32, rows]) {
          const out = new Float32Array(rows * head.numClass);
          for (let start = 0; start < rows; start += size) {
            const n = Math.min(size, rows - start);
            scoreRows(
              head,
              matrix.subarray(start * fixture.width, (start + n) * fixture.width),
              fixture.width,
              n,
              out.subarray(start * head.numClass, (start + n) * head.numClass),
            );
          }
          expect(Buffer.from(out.buffer).equals(Buffer.from(expected.buffer))).toBe(true);
        }
      }
    });

    test("classifyPage answers the same with any scoringBatch", () => {
      const { model } = tryLoadModel(artifactJson());
      // 120 candidate blocks, so a batch of 32 spans several chunks.
      const blocks = Array.from(
        { length: 120 },
        (_, i) =>
          `<section class="block-${i % 7}"><h2>Part ${i}</h2><p>Text ${i} about crawling.</p><a href="https://other${i}.example/">x</a></section>`,
      ).join("");
      for (const html of [
        PAGE,
        `<!doctype html><html><body><main>${blocks}</main></body></html>`,
      ]) {
        const base = classifyPage(html, URL, { model, budgetMs: 1e9, scoringBatch: 1 });
        for (const batch of [7, 32, 400]) {
          const batched = classifyPage(html, URL, { model, budgetMs: 1e9, scoringBatch: batch });
          expect(batched).toEqual(base);
        }
      }
    });
  },
);

describe.if(HAS_ARTIFACT && existsSync(PARITY_SECTION))("section head parity (#2307)", () => {
  test("holdout rows: identical argmax to the Python reference, probabilities within 1e-5", () => {
    const model = loadModel(artifactJson());
    expect(model.section).not.toBeNull();
    const fixture = JSON.parse(readFileSync(PARITY_SECTION, "utf8")) as {
      rows: number;
      width: number;
      x: Array<{ i: number[]; v: number[] }>;
      probs: number[][];
    };
    const head = model.section!;
    const values = new Float32Array(fixture.width);
    const scores = new Float32Array(head.numClass);
    let maxDiff = 0;
    let argmaxMatches = 0;
    for (let row = 0; row < fixture.rows; row++) {
      values.fill(0);
      const sparse = fixture.x[row]!;
      for (let k = 0; k < sparse.i.length; k++) values[sparse.i[k]!] = sparse.v[k]!;
      scoreRow(head, values, scores);
      const { index } = softmax(scores);
      const expected = fixture.probs[row]!;
      let expectedArgmax = 0;
      for (let c = 1; c < expected.length; c++) {
        if (expected[c]! > expected[expectedArgmax]!) expectedArgmax = c;
      }
      if (index === expectedArgmax) argmaxMatches++;
      for (let c = 0; c < expected.length; c++)
        maxDiff = Math.max(maxDiff, Math.abs(scores[c]! - expected[c]!));
    }
    expect(argmaxMatches).toBe(fixture.rows);
    expect(maxDiff).toBeLessThan(1e-5);
  });

  test("a node carries a section only when a class clears its threshold, never `none`", () => {
    const { model } = tryLoadModel(artifactJson());
    const out = classifyPage(PAGE, URL, { model });
    for (const node of out.nodes) {
      if (node.section?.source !== "model") continue;
      expect(node.section.label).not.toBe("none");
      const index = model!.section!.classes.indexOf(node.section.label);
      expect(node.section.confidence).toBeGreaterThanOrEqual(
        model!.section!.classThresholds[index]!,
      );
    }
  });

  test("the comments rule wins over the head inside a thread", () => {
    const { model } = tryLoadModel(artifactJson());
    const elements: unknown[] = [];
    const out = classifyPage(WITH_COMMENTS, URL, { model, elementsOut: elements });
    const thread = elements[out.commentSections[0]!.nodeId] as Element;
    const inside = out.nodes.filter((n) => thread.contains(elements[n.nodeRef.nodeId] as Element));
    expect(inside.length).toBeGreaterThan(0);
    for (const node of inside) expect(node.section?.source).toBe("rules");
  });
});

describe.if(HAS_ARTIFACT && existsSync(PARITY_PAGETYPE))(
  "page-type head parity, with and without the link graph (#2307)",
  () => {
    const fixture = () =>
      JSON.parse(readFileSync(PARITY_PAGETYPE, "utf8")) as {
        rows: number;
        width: number;
        baseWidth: number;
        x: Array<{ i: number[]; v: number[] }>;
        probsGraph: number[][];
        probsNoGraph: number[][];
      };

    for (const variant of ["graph", "noGraph"] as const) {
      test(`${variant}: identical argmax to the Python reference, probabilities within 1e-5`, () => {
        const model = loadModel(artifactJson());
        const data = fixture();
        const head = model.pageType;
        expect(head.extraFeatureCount).toBe(data.width - data.baseWidth);
        const values = new Float32Array(data.width);
        const scores = new Float32Array(head.numClass);
        const expectedAll = variant === "graph" ? data.probsGraph : data.probsNoGraph;
        let maxDiff = 0;
        let argmaxMatches = 0;
        for (let row = 0; row < data.rows; row++) {
          values.fill(0);
          const sparse = data.x[row]!;
          for (let k = 0; k < sparse.i.length; k++) values[sparse.i[k]!] = sparse.v[k]!;
          if (variant === "noGraph") values.fill(Number.NaN, data.baseWidth);
          scoreRow(head, values, scores);
          const { index } = softmax(scores);
          const expected = expectedAll[row]!;
          let expectedArgmax = 0;
          for (let c = 1; c < expected.length; c++)
            if (expected[c]! > expected[expectedArgmax]!) expectedArgmax = c;
          if (index === expectedArgmax) argmaxMatches++;
          for (let c = 0; c < expected.length; c++)
            maxDiff = Math.max(maxDiff, Math.abs(scores[c]! - expected[c]!));
        }
        expect(argmaxMatches).toBe(data.rows);
        expect(maxDiff).toBeLessThan(1e-5);
      });
    }

    test("classifyPage works with no link graph and with a complete one", () => {
      const { model } = tryLoadModel(artifactJson());
      const none = classifyPage(PAGE, URL, { model });
      const graph = classifyPage(PAGE, URL, {
        model,
        siteFeatures: Array.from({ length: model!.pageType.extraFeatureCount }, () => 0),
      });
      for (const out of [none, graph]) {
        expect(out.degraded).toBe("");
        expect(typeof out.pageType.label).toBe("string");
      }
    });
  },
);

describe.if(HAS_ARTIFACT)("platform guard applied by classifyPage (#2307)", () => {
  test("the page's platform picks the override row; another platform's row is ignored", () => {
    const json = artifactJson() as Record<string, Record<string, unknown>>;
    const abstainAll = (head: { classes: string[] }) =>
      Object.fromEntries(head.classes.map((c) => [c, 1.01]));
    for (const name of ["pageType", "region"])
      json[name]!.platformClassThresholds = {
        custom: abstainAll(json[name] as unknown as { classes: string[] }),
      };
    const { model } = tryLoadModel(json);
    // The fixture page carries no platform markers: its bucket is "custom".
    const guarded = classifyPage(PAGE, URL, { model });
    expect(guarded.platform).toBe("custom");
    expect(guarded.pageType.source).not.toBe("model");
    for (const node of guarded.nodes)
      if (node.source === "model") expect(node.label).toBe("unknown");
    const other = classifyPage(PAGE, URL, { model, platform: "wix" });
    expect(other.platform).toBe("wix");
    expect(other.nodes.some((n) => n.source === "model" && n.label !== "unknown")).toBe(true);
  });
});

describe("page-type rules", () => {
  const withJson = (json: string): string =>
    `<!doctype html><html><head><title>T</title>
     <script type="application/ld+json">${json}</script></head><body><p>x</p></body></html>`;
  const label = (html: string, url = "https://example.com/x") => {
    const out = classifyPage(html, url);
    return out.pageType.label;
  };

  test("reads @graph, arrays and nested mainEntity", () => {
    expect(label(withJson('{"@graph":[{"@type":"WebPage"},{"@type":"BlogPosting"}]}'))).toBe(
      "article",
    );
    expect(label(withJson('[{"@type":["Product","Thing"]}]'))).toBe("product_detail");
    expect(label(withJson('{"@type":"WebPage","mainEntity":{"@type":"TechArticle"}}'))).toBe(
      "docs",
    );
  });

  test("a CollectionPage of Products is the listing, not an invented class", () => {
    expect(
      label(
        withJson(
          '{"@type":"CollectionPage","mainEntity":{"@type":"ItemList","itemListElement":[{"@type":"Product"}]}}',
        ),
      ),
    ).toBe("listing_index");
  });

  test("malformed JSON-LD does not throw", () => {
    expect(() => label(withJson("{not json"))).not.toThrow();
  });
});

describe.if(HAS_ARTIFACT)("per-class thresholds", () => {
  test("the artifact carries one threshold per class, and abstentions are above 1", () => {
    const model = loadModel(artifactJson());
    expect(model.region.classThresholds.length).toBe(model.region.numClass);
    for (const t of model.region.classThresholds) expect(t).toBeGreaterThan(0);
    // At least one class abstains; its threshold can never be met.
    expect([...model.region.classThresholds].some((t) => t > 1)).toBe(true);
  });

  test("a rejected prediction is an `unknown` from the MODEL, carrying the raw answer", () => {
    const soup = `<!doctype html><html><head><title>T</title></head><body>${Array.from(
      { length: 40 },
      (_, i) =>
        `<div class="row r${i}"><p>Block ${i} with enough text to be a candidate.</p></div>`,
    ).join("")}</body></html>`;
    const { model } = tryLoadModel(artifactJson());
    // A threshold of 1.01 rejects everything the model says.
    const out = classifyPage(soup, URL, { model, threshold: 1.01 });
    const rejected = out.nodes.filter((n) => n.label === "unknown" && n.source === "model");
    expect(rejected.length).toBeGreaterThan(0);
    for (const node of rejected) {
      // Saying "rules" here would claim the deterministic path had looked.
      expect(node.source).toBe("model");
      expect(node.confidence).toBe(0);
      expect(typeof node.rawLabel).toBe("string");
      expect(node.rawConfidence).toBeGreaterThan(0);
    }
  });
});

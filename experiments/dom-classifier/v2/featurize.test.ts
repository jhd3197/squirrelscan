// What this file is for: the four properties the rest of the pipeline assumes
// but cannot check for itself.
//
//  1. Determinism, including under attribute reordering. The dataset is built
//     once and the model is evaluated later, in another process; a featurizer
//     that drifts makes every measurement meaningless.
//  2. Masking really hides the label source. The silver labels ARE the landmark
//     tags, so if a masked <footer> vector differs from a masked <div> vector
//     with the same subtree, the model can read the answer off its own input
//     and the holdout score is fiction.
//  3. The hash is pinned. Training and inference bucket tokens in different
//     processes and, eventually, different runtimes.
//  4. The cost is linear and small enough for the < 5 ms/page budget in PLAN.md.

import { describe, expect, test } from "bun:test";

import { parseHTML } from "../../../packages/parser/src/dom.ts";
import {
  CURATED_VOCAB,
  featurizeDocument,
  featurizePage,
  featureSchema,
  fnv1a32,
  NODE_FEATURE_NAMES,
  PAGE_FEATURE_NAMES,
  TOKEN_BUCKETS,
  tokenBucket,
  tokenize,
} from "./featurize.ts";

const PAGE = `<!doctype html><html lang="en"><head><title>Test</title></head>
<body class="page single">
  <header class="site-header"><nav class="navbar"><ul>
    <li><a href="/a">Alpha</a></li><li><a href="/b">Beta</a></li></ul></nav></header>
  <main><article class="post"><h1>Heading</h1>
    <p>By Ada Lovelace</p><p>Body text long enough to register as prose on this page.</p>
  </article></main>
  <footer class="site-footer"><p>&copy; 2026 Example Inc</p>
    <a href="https://twitter.com/example">Follow</a></footer>
</body></html>`;

const URL = "https://example.com/blog/post-1";

describe("determinism", () => {
  test("the same input twice produces identical output", () => {
    const a = featurizePage(PAGE, URL, { emitMasked: true });
    const b = featurizePage(PAGE, URL, { emitMasked: true });
    expect(JSON.stringify(b)).toBe(JSON.stringify(a));
  });

  test("attribute order does not change any feature", () => {
    const reordered = PAGE.replace(
      '<header class="site-header">',
      '<header data-x="1" class="site-header">',
    ).replace('<footer class="site-footer">', '<footer class="site-footer" data-y="2">');
    const a = featurizePage(PAGE, URL);
    const b = featurizePage(reordered, URL);
    // data-* attributes are not read, so the vectors have to match exactly.
    expect(b.nodes.map((n) => n.features)).toEqual(a.nodes.map((n) => n.features));
    expect(b.page.features).toEqual(a.page.features);
  });

  test("class token order inside one attribute is preserved, not sorted", () => {
    // Determinism means "the same input gives the same output", not "any input
    // ordering gives the same output" — the bag is order-insensitive by
    // construction because it is a set of buckets.
    const a = featurizePage(PAGE.replace('class="site-footer"', 'class="site footer"'), URL);
    const b = featurizePage(PAGE.replace('class="site-footer"', 'class="footer site"'), URL);
    const footerA = a.nodes.find((n) => n.tag === "footer")!;
    const footerB = b.nodes.find((n) => n.tag === "footer")!;
    expect(footerB.tokens).toEqual(footerA.tokens);
  });
});

describe("masking", () => {
  // The same subtree twice: once under the landmark tags the silver labels are
  // read from, once under plain divs carrying the same classes.
  const landmarks = `<!doctype html><html><head><title>T</title></head><body class="page">
    <header class="site-header" id="top"><p>Masthead text goes here.</p></header>
    <main class="content-main"><p>Body text that is long enough to count.</p></main>
    <footer class="site-footer" id="bottom"><p>&copy; 2026 Example Inc, all rights reserved.</p></footer>
  </body></html>`;
  const divs = landmarks
    .replace("<header ", "<div ")
    .replace("</header>", "</div>")
    .replace("<main ", "<div ")
    .replace("</main>", "</div>")
    .replace("<footer ", "<div ")
    .replace("</footer>", "</div>");

  test("a masked landmark is byte-identical to a masked div with the same subtree", () => {
    const withLandmarks = featurizePage(landmarks, URL, { mask: true });
    const withDivs = featurizePage(divs, URL, { mask: true });
    expect(withDivs.nodes.length).toBe(withLandmarks.nodes.length);
    for (let i = 0; i < withLandmarks.nodes.length; i++) {
      const a = withLandmarks.nodes[i]!;
      const b = withDivs.nodes[i]!;
      expect(a.nodeId).toBe(b.nodeId);
      // Named per feature so a regression says WHICH one leaked.
      for (let f = 0; f < NODE_FEATURE_NAMES.length; f++) {
        expect([NODE_FEATURE_NAMES[f], a.features[f]]).toEqual([
          NODE_FEATURE_NAMES[f],
          b.features[f],
        ]);
      }
      expect(a.tokens).toEqual(b.tokens);
    }
  });

  test("without masking the landmark IS visible, so the test above is not vacuous", () => {
    const withLandmarks = featurizePage(landmarks, URL);
    const withDivs = featurizePage(divs, URL);
    const tagIndex = NODE_FEATURE_NAMES.indexOf("tagId");
    const roleIndex = NODE_FEATURE_NAMES.indexOf("roleId");
    const a = withLandmarks.nodes[0]!;
    const b = withDivs.nodes[0]!;
    expect(a.features[tagIndex]).not.toBe(b.features[tagIndex]);
    expect(a.features[roleIndex]).not.toBe(b.features[roleIndex]);
  });

  test("a label word in a class is dropped about half the time, deterministically", () => {
    // 200 nodes each carrying class="footer", so the dropout rate is visible.
    const rows = Array.from(
      { length: 200 },
      (_, i) => `<div class="footer"><p>Row ${i} with enough text to be a candidate.</p></div>`,
    ).join("");
    const page = `<!doctype html><html><head><title>T</title></head><body>${rows}</body></html>`;
    const first = featurizePage(page, URL, { mask: true });
    const second = featurizePage(page, URL, { mask: true });
    const index = NODE_FEATURE_NAMES.indexOf("self_footer");
    const kept = first.nodes.filter((n) => n.features[index] === 1).length;
    const candidates = first.nodes.filter((n) => n.tag === "div").length;
    expect(candidates).toBeGreaterThan(100);
    expect(kept / candidates).toBeGreaterThan(0.3);
    expect(kept / candidates).toBeLessThan(0.7);
    // Deterministic: the same nodes keep it every time.
    expect(second.nodes.map((n) => n.features[index])).toEqual(
      first.nodes.map((n) => n.features[index]),
    );
    // And unmasked, every one of them keeps it.
    const unmasked = featurizePage(page, URL);
    expect(unmasked.nodes.filter((n) => n.features[index] === 1).length).toBe(candidates);
  });

  test("known asymmetry: masking hides the tag from the VECTOR, not from the gate", () => {
    // A <footer> is always a candidate; the same subtree under a <div> is one
    // only if it clears the size gate. So masking equalises the vectors of the
    // nodes that ARE selected, and does NOT equalise which nodes get selected.
    // That is a population difference, not a leak — but it is worth pinning,
    // because it means a tiny div-soup footer never becomes a training row.
    const tiny = `<!doctype html><html><head><title>T</title></head><body>
      <TAG class="site-footer"><p>&copy; 2026</p></TAG></body></html>`;
    const asFooter = featurizePage(tiny.replaceAll("TAG", "footer"), URL, { mask: true });
    const asDiv = featurizePage(tiny.replaceAll("TAG", "div"), URL, { mask: true });
    expect(asFooter.nodes.length).toBe(1);
    expect(asDiv.nodes.length).toBe(0);
  });

  test("a non-label class word survives masking untouched", () => {
    const page = `<!doctype html><html><head><title>T</title></head><body>
      <div class="product card"><p>Some text that makes this a candidate node.</p></div>
    </body></html>`;
    const masked = featurizePage(page, URL, { mask: true });
    const product = NODE_FEATURE_NAMES.indexOf("self_product");
    const card = NODE_FEATURE_NAMES.indexOf("self_card");
    expect(masked.nodes[0]!.features[product]).toBe(1);
    expect(masked.nodes[0]!.features[card]).toBe(1);
  });
});

describe("the token hash", () => {
  test("golden values", () => {
    // Pinned so a rewrite in another language (or another JS engine) can be
    // checked against these five numbers rather than against a whole dataset.
    expect(fnv1a32("")).toBe(2166136261);
    expect(fnv1a32("a")).toBe(3826002220);
    expect(fnv1a32("footer")).toBe(290031026);
    expect(fnv1a32("s:footer")).toBe(1896507515);
    expect(fnv1a32("p:wrapper")).toBe(2292061302);
  });

  test("golden buckets", () => {
    expect(tokenBucket("s:footer")).toBe(1896507515 % TOKEN_BUCKETS);
    expect(tokenBucket("s:footer")).toBe(123);
    expect(tokenBucket("p:wrapper")).toBe(118);
    expect(tokenBucket("b:home")).toBe(tokenBucket("b:home"));
  });

  test("every bucket is in range", () => {
    for (const word of CURATED_VOCAB) {
      const bucket = tokenBucket(`s:${word}`);
      expect(bucket).toBeGreaterThanOrEqual(0);
      expect(bucket).toBeLessThan(TOKEN_BUCKETS);
    }
  });
});

describe("tokenize", () => {
  test("splits on separators and camelCase, lowercases, strips digits", () => {
    expect(tokenize("site-header siteHeader site_header")).toEqual(["site", "header"]);
    expect(tokenize("col-md-6 w-1/2")).toEqual(["col", "md"]);
    expect(tokenize("ProductCard__title")).toEqual(["product", "card", "title"]);
    expect(tokenize(null)).toEqual([]);
    expect(tokenize("")).toEqual([]);
  });

  test("caps the tokens taken from one element", () => {
    const many = Array.from(
      { length: 100 },
      (_, i) => `utility-${String.fromCharCode(97 + (i % 26))}${i}`,
    ).join(" ");
    expect(tokenize(many).length).toBeLessThanOrEqual(32);
  });
});

describe("candidates", () => {
  test("a div needs two element children or twenty characters of text", () => {
    const page = `<!doctype html><html><head><title>T</title></head><body>
      <div id="tiny">x</div>
      <div id="texty">twenty characters of text right here</div>
      <div id="kids"><span>a</span><span>b</span></div>
    </body></html>`;
    const ids = featurizePage(page, URL).nodes.map((n) => n.elementId);
    expect(ids).not.toContain("tiny");
    expect(ids).toContain("texty");
    expect(ids).toContain("kids");
  });

  test("the cap is respected and landmarks survive it", () => {
    const rows = Array.from(
      { length: 900 },
      (_, i) =>
        `<section id="s${i}"><p>Section ${i} body text for the candidate gate.</p></section>`,
    ).join("");
    const page = `<!doctype html><html><head><title>T</title></head><body>
      ${rows}<footer id="keep"><p>&copy; 2026</p></footer></body></html>`;
    const out = featurizePage(page, URL, { maxCandidates: 50 });
    expect(out.nodes.length).toBeLessThanOrEqual(50);
    expect(out.nodes.map((n) => n.elementId)).toContain("keep");
    // Deterministic under a repeat run.
    const again = featurizePage(page, URL, { maxCandidates: 50 });
    expect(again.nodes.map((n) => n.nodeId)).toEqual(out.nodes.map((n) => n.nodeId));
  });
});

describe("the feature vectors", () => {
  test("every candidate emits the full, finite, ordered vector", () => {
    const out = featurizePage(PAGE, URL, { emitMasked: true });
    expect(out.nodes.length).toBeGreaterThan(3);
    for (const node of out.nodes) {
      expect(node.features.length).toBe(NODE_FEATURE_NAMES.length);
      expect(node.maskedFeatures!.length).toBe(NODE_FEATURE_NAMES.length);
      for (const value of node.features) expect(Number.isFinite(value)).toBe(true);
      for (const bucket of node.tokens) expect(bucket).toBeLessThan(TOKEN_BUCKETS);
    }
    expect(out.page.features.length).toBe(PAGE_FEATURE_NAMES.length);
    for (const value of out.page.features) expect(Number.isFinite(value)).toBe(true);
  });

  test("no feature is derived from JSON-LD, microdata or og:type", () => {
    // Adding the schema block must not move a single number: it is the page
    // type LABEL, and a model that can see it has learned nothing.
    const withSchema = PAGE.replace(
      "</head>",
      `<script type="application/ld+json">{"@type":"NewsArticle"}</script>
       <meta property="og:type" content="article" />
       </head>`,
    ).replace("<article ", '<article itemscope itemtype="https://schema.org/Article" ');
    const a = featurizePage(PAGE, URL);
    const b = featurizePage(withSchema, URL);
    expect(b.page.features).toEqual(a.page.features);
    expect(b.nodes.map((n) => n.features)).toEqual(a.nodes.map((n) => n.features));
  });

  test("every accumulator survives the internal array growth", () => {
    // The walk starts with 1024-element typed arrays and doubles them. A new
    // accumulator added without an entry in `grow`'s key list keeps its ORIGINAL
    // array, and writes past 1024 then vanish silently — a TypedArray ignores
    // an out-of-range index rather than throwing. So: 1200 identical units,
    // each touching every counter, checked against arithmetic.
    const units = 1200;
    const unit =
      `<section class="u"><h2>Head</h2><p>Paragraph text long enough to be prose here.</p>` +
      `<ul><li><a href="/in">In</a></li><li><a href="https://other.test/x">Out</a></li></ul>` +
      `<img src="/i.png" alt="i" /><iframe src="/f"></iframe><video src="/v"></video>` +
      `<table><tr><td>c</td></tr></table><pre>code</pre><time datetime="2026-01-01">Jan</time>` +
      `<form><input type="search" name="q" /><button>Go</button></form>` +
      `<a href="https://twitter.com/x">s</a><a href="mailto:a@b.test">m</a></section>`;
    const page = `<!doctype html><html><head><title>T</title></head><body>${unit.repeat(units)}</body></html>`;
    const out = featurizePage(page, URL);
    const get = (name: string): number => out.page.features[PAGE_FEATURE_NAMES.indexOf(name)]!;
    expect(out.page.elementCount).toBeGreaterThan(1024);
    expect(get("sectionCount")).toBe(units);
    expect(get("h2Count")).toBe(units);
    expect(get("paragraphCount")).toBe(units);
    expect(get("listCount")).toBe(units);
    expect(get("listItemCount")).toBe(units * 2);
    expect(get("imageCount")).toBe(units);
    expect(get("iframeCount")).toBe(units);
    expect(get("videoCount")).toBe(units);
    expect(get("tableCount")).toBe(units);
    expect(get("codeBlockCount")).toBe(units);
    expect(get("timeElementCount")).toBe(units);
    expect(get("formCount")).toBe(units);
    expect(get("inputCount")).toBe(units);
    expect(get("buttonCount")).toBe(units);
    expect(get("searchInputCount")).toBe(units);
    expect(get("linkCount")).toBe(units * 4);
    expect(get("socialLinkCount")).toBe(units);
    expect(get("contactLinkCount")).toBe(units);
    expect(get("dateHitCount")).toBe(units);
  });

  test("a subtree occupies a contiguous span, and ancestors contain descendants", () => {
    // `endPos` is computed as startIndex + subtreeElements - 1, which is only
    // true because the walk is a strict preorder DFS. If that ever stops
    // holding, every position feature silently becomes wrong rather than
    // throwing, so it is asserted against the candidate ancestry.
    const page = `<!doctype html><html><head><title>T</title></head><body>
      <header><nav><ul><li><a href="/a">A</a></li><li><a href="/b">B</a></li></ul></nav></header>
      <main><article><section><div class="card"><p>Text long enough to be a candidate.</p></div>
      <div class="card"><p>Another block of text that is long enough.</p></div></section></article></main>
      <footer><div><p>&copy; 2026 Example Inc, all rights reserved here.</p></div></footer>
    </body></html>`;
    const out = featurizePage(page, URL);
    const start = NODE_FEATURE_NAMES.indexOf("startPos");
    const end = NODE_FEATURE_NAMES.indexOf("endPos");
    const byId = new Map(out.nodes.map((n) => [n.nodeId, n]));
    let checked = 0;
    for (const node of out.nodes) {
      expect(node.features[end]!).toBeGreaterThanOrEqual(node.features[start]!);
      const parent = byId.get(node.parentCandidateId);
      if (!parent) continue;
      checked++;
      expect(parent.features[start]!).toBeLessThanOrEqual(node.features[start]!);
      expect(parent.features[end]!).toBeGreaterThanOrEqual(node.features[end]!);
      expect(parent.nodeId).toBeLessThan(node.nodeId);
    }
    expect(checked).toBeGreaterThan(5);
  });

  test("the page vector reads the URL shape", () => {
    const blog = featurizePage(PAGE, "https://example.com/blog/2026/03/a-post");
    const root = featurizePage(PAGE, "https://example.com/");
    const get = (page: number[], name: string): number => page[PAGE_FEATURE_NAMES.indexOf(name)]!;
    expect(get(blog.page.features, "urlSlug_blog")).toBe(1);
    expect(get(blog.page.features, "urlHasDate")).toBe(1);
    expect(get(blog.page.features, "urlIsRoot")).toBe(0);
    expect(get(root.page.features, "urlIsRoot")).toBe(1);
    expect(get(root.page.features, "urlSlug_blog")).toBe(0);
  });

  test("the structural features say what they claim", () => {
    const out = featurizePage(PAGE, URL);
    const get = (tag: string, name: string): number => {
      const node = out.nodes.find((n) => n.tag === tag)!;
      return node.features[NODE_FEATURE_NAMES.indexOf(name)]!;
    };
    expect(get("footer", "hasCopyright")).toBe(1);
    expect(get("footer", "socialLinkCount")).toBe(1);
    expect(get("footer", "isLastBodyChild")).toBe(1);
    expect(get("nav", "linkCount")).toBe(2);
    expect(get("nav", "listItemCount")).toBe(2);
    expect(get("nav", "linkDensity")).toBeGreaterThan(0.9);
    expect(get("article", "hasByline")).toBe(1);
    expect(get("article", "headingCount")).toBe(1);
    expect(get("article", "minHeadingLevel")).toBe(1);
    expect(get("article", "paragraphCount")).toBe(2);
  });

  test("a repeated card grid is visible as a repeated sibling shape", () => {
    const cards = Array.from(
      { length: 6 },
      () =>
        `<li class="card"><img src="/i.png" alt="i" /><h3>Title</h3><a href="/x">Read more</a></li>`,
    ).join("");
    const page = `<!doctype html><html><head><title>T</title></head><body><ul id="grid">${cards}</ul></body></html>`;
    const out = featurizePage(page, URL);
    const grid = out.nodes.find((n) => n.elementId === "grid")!;
    expect(grid.features[NODE_FEATURE_NAMES.indexOf("maxRepeatedSiblingRun")]).toBe(6);
  });
});

describe("cost", () => {
  test("a 20k-element page featurizes well inside the smoke ceiling", () => {
    const blocks = Array.from(
      { length: 2000 },
      (_, i) =>
        `<section class="block b${i}"><h2>Section ${i}</h2>` +
        `<p>Paragraph text for section ${i}, long enough to count as prose.</p>` +
        `<ul><li><a href="/a${i}">One</a></li><li><a href="/b${i}">Two</a></li></ul>` +
        `<div class="card"><span>meta</span><span>more</span></div></section>`,
    ).join("");
    const page = `<!doctype html><html><head><title>Big</title></head><body class="big">${blocks}</body></html>`;
    // Parse OUTSIDE the timer. The budget in PLAN.md is the featurizer's own,
    // because the CLI already holds a parsed document when it calls this: the
    // linkedom parse of a 2 MB page is several seconds and is not a cost this
    // file can do anything about.
    // This is a SMOKE ceiling, not the performance bound. The real number,
    // measured on an idle machine, is ~40 ms of featurizer for 22k elements
    // against ~76 ms for the linkedom parse the CLI pays anyway. The ceiling
    // here is 30x that, because a wall-clock assertion on a shared machine
    // measures the scheduler: the same code read 43 ms repeatedly on an idle
    // box and 1883 ms while three other lanes were building.
    //
    // Two things were tried and rejected. A tighter absolute bound flaked
    // roughly one run in three under load. Calibrating against a parse of the
    // same page in the measurement loop was WORSE than useless: each parse
    // allocates 22k DOM nodes, and collecting that garbage lands inside the
    // next featurize measurement, which inflated it from 40 ms to 199 ms and
    // made the ratio look 6x worse than it is.
    //
    // The guard that actually catches an algorithmic regression is the
    // linear-scaling test below, which compares two sizes in one process with
    // the same garbage profile.
    const { document } = parseHTML(page);
    featurizeDocument(document as never, URL); // warm
    let best = Number.POSITIVE_INFINITY;
    let out = featurizeDocument(document as never, URL);
    for (let run = 0; run < 5; run++) {
      const start = Bun.nanoseconds();
      out = featurizeDocument(document as never, URL);
      best = Math.min(best, (Bun.nanoseconds() - start) / 1e6);
    }
    expect(out.page.elementCount).toBeGreaterThan(20000);
    expect(out.nodes.length).toBeLessThanOrEqual(400);
    expect(best).toBeLessThan(1200);
  });

  test("cost grows linearly, not quadratically, in element count", () => {
    const build = (count: number): string =>
      `<!doctype html><html><head><title>T</title></head><body>` +
      Array.from(
        { length: count },
        (_, i) => `<section class="b${i}"><p>Text ${i} for the candidate gate here.</p></section>`,
      ).join("") +
      `</body></html>`;
    // Min of five, alternating between the two sizes: a back-to-back pair on a
    // loaded machine measures whatever else was scheduled in between.
    const time = (document: unknown): number => {
      const start = Bun.nanoseconds();
      featurizeDocument(document as never, URL);
      return (Bun.nanoseconds() - start) / 1e6;
    };
    const small = parseHTML(build(1000)).document;
    const large = parseHTML(build(8000)).document;
    // Warm both shapes first: a cold first call measures JIT, not the algorithm.
    time(small);
    time(large);
    let smallBest = Number.POSITIVE_INFINITY;
    let largeBest = Number.POSITIVE_INFINITY;
    for (let run = 0; run < 5; run++) {
      smallBest = Math.min(smallBest, time(small));
      largeBest = Math.min(largeBest, time(large));
    }
    const ratio = largeBest / Math.max(0.5, smallBest);
    // 8x the elements. Quadratic would be ~64x; allow generous headroom for a
    // loaded machine and still fail on anything super-linear.
    expect(ratio).toBeLessThan(20);
  });
});

describe("the exported schema", () => {
  test("matches the vectors it describes", () => {
    const schema = featureSchema();
    expect(schema.nodeFeatureCount).toBe(NODE_FEATURE_NAMES.length);
    expect(schema.pageFeatureCount).toBe(PAGE_FEATURE_NAMES.length);
    expect(schema.tokenBuckets).toBe(TOKEN_BUCKETS);
    expect(new Set(NODE_FEATURE_NAMES).size).toBe(NODE_FEATURE_NAMES.length);
    expect(new Set(PAGE_FEATURE_NAMES).size).toBe(PAGE_FEATURE_NAMES.length);
  });
});

// The silver labels are the supervision. If they are wrong the model learns
// the wrong thing and every downstream number is measuring that instead.
//
// The two failures this file exists to catch, both drawn from
// diverse-corpus/ERROR-ANALYSIS-2026-09-19.md:
//
//  - a <header> or <footer> INSIDE an article read as site chrome, which is
//    the single most common nesting mistake and would teach the region head
//    that an article's byline block is a masthead;
//  - a page type read off a schema graph without walking @graph / arrays /
//    mainEntity, which silently labels most of the corpus "unlabelled".

import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { parseHTML } from "../../../packages/parser/src/dom.ts";
import { featurizeDocument } from "./featurize.ts";
import { silverPageType, silverRegionLabels } from "./silver-labels.ts";

import expected from "./fixtures/expected.json" with { type: "json" };

const FIXTURES = new URL("./fixtures/", import.meta.url).pathname;

function labelFixture(name: string, url: string) {
  const html = readFileSync(`${FIXTURES}${name}`, "utf8");
  const { document } = parseHTML(html);
  const featurized = featurizeDocument(document as never, url);
  const regions = silverRegionLabels(featurized.nodes);
  const pageType = silverPageType(document as never, url);
  const byElementId = new Map<string, (typeof regions.labels)[number]>();
  for (const node of featurized.nodes) {
    if (!node.elementId) continue;
    const label = regions.labels.find((l) => l.nodeId === node.nodeId);
    if (label) byElementId.set(node.elementId, label);
  }
  return { featurized, regions, pageType, byElementId };
}

describe("the synthetic fixtures", () => {
  for (const [name, spec] of Object.entries(expected.pages)) {
    describe(name, () => {
      test("page type", () => {
        const { pageType } = labelFixture(name, spec.url);
        // The expectations come from JSON, so they are plain strings; compare
        // as strings rather than widening the label union to accept them.
        expect(pageType.label as string).toBe(spec.pageType.label);
        expect(pageType.source).toBe(spec.pageType.source);
        expect(pageType.leakProne).toBe(spec.pageType.leakProne);
      });

      test("regions", () => {
        const { byElementId, regions, featurized } = labelFixture(name, spec.url);
        for (const [elementId, label] of Object.entries(spec.regions)) {
          expect([elementId, byElementId.get(elementId)?.label]).toEqual([elementId, label]);
        }
        expect(regions.landmarkKinds).toEqual(spec.landmarkKinds);
        expect(featurized.page.divSoup).toBe(spec.divSoup);
      });

      test("containers", () => {
        const { byElementId } = labelFixture(name, spec.url);
        for (const [elementId, container] of Object.entries(spec.containers)) {
          expect([elementId, byElementId.get(elementId)?.containerLabel]).toEqual([
            elementId,
            container,
          ]);
        }
      });
    });
  }
});

describe("nesting decides header and footer", () => {
  const page = (inner: string): string =>
    `<!doctype html><html><head><title>T</title></head><body>
      <header id="site"><nav id="nav"><a href="/">Home</a></nav></header>
      <main id="main">${inner}</main>
      <aside id="aside"><p>Sidebar content goes here.</p></aside>
      <footer id="site-foot"><p>&copy; 2026 Example Inc</p></footer>
    </body></html>`;

  const label = (html: string, elementId: string): string | undefined => {
    const { document } = parseHTML(html);
    const featurized = featurizeDocument(document as never, "https://example.com/x");
    const regions = silverRegionLabels(featurized.nodes);
    const node = featurized.nodes.find((n) => n.elementId === elementId);
    return regions.labels.find((l) => l.nodeId === node?.nodeId)?.label;
  };

  test("inside an <article> it is the article's own chrome", () => {
    const html = page(
      `<article id="post"><header id="h"><h1>Title</h1><p>By Ada Lovelace</p></header>
       <p>Body text for the article that is long enough.</p>
       <footer id="f"><p>Filed under testing today.</p></footer></article>`,
    );
    expect(label(html, "h")).toBe("article_header");
    expect(label(html, "f")).toBe("article_footer");
  });

  test("inside a <section> it is the section's own chrome", () => {
    const html = page(
      `<section id="sec"><header id="h"><h2>Part one</h2></header>
       <p>Section body text that is long enough to count.</p></section>`,
    );
    expect(label(html, "h")).toBe("article_header");
  });

  test("directly under <main> it is still site chrome", () => {
    // <main> is sectioning ROOT, not sectioning content: a <header> there is
    // the page header, which is the distinction the HTML spec draws and the
    // one the error analysis needs.
    const html = page(`<header id="h"><h1>Page title</h1></header><p>Body text here.</p>`);
    expect(label(html, "h")).toBe("site_header");
  });

  test("the site header and footer are site chrome", () => {
    const html = page(`<p>Body text that is long enough to count.</p>`);
    expect(label(html, "site")).toBe("site_header");
    expect(label(html, "site-foot")).toBe("footer");
  });

  test("a <nav> inside a <header> is still navigation", () => {
    const html = page(`<p>Body text that is long enough to count.</p>`);
    expect(label(html, "nav")).toBe("navigation");
  });

  test("an explicit role beats the tag", () => {
    const html = `<!doctype html><html><head><title>T</title></head><body>
      <div id="a" role="banner"><p>Masthead text goes here.</p></div>
      <div id="b" role="navigation"><a href="/">Home</a><a href="/x">X</a></div>
      <div id="c" role="main"><p>Body text that is long enough.</p></div>
      <section id="d" role="contentinfo"><p>&copy; 2026 Example Inc</p></section>
    </body></html>`;
    expect(label(html, "a")).toBe("site_header");
    expect(label(html, "b")).toBe("navigation");
    expect(label(html, "c")).toBe("main_content");
    expect(label(html, "d")).toBe("footer");
  });
});

describe("containment labels", () => {
  const build = (landmarks: string): string =>
    `<!doctype html><html><head><title>T</title></head><body>${landmarks}</body></html>`;

  test("a non-landmark inside <main> is a content_block, outside it is unlabelled", () => {
    const html = build(
      `<header><nav><a href="/">Home</a></nav></header>
       <main><div id="inside"><p>Body text that is long enough to count.</p></div></main>
       <footer><p>&copy; 2026 Example Inc</p></footer>
       <div id="outside"><p>An orphan block sitting outside every landmark.</p></div>`,
    );
    const { document } = parseHTML(html);
    const featurized = featurizeDocument(document as never, "https://example.com/x");
    const regions = silverRegionLabels(featurized.nodes);
    const at = (elementId: string) => {
      const node = featurized.nodes.find((n) => n.elementId === elementId)!;
      return regions.labels.find((l) => l.nodeId === node.nodeId)!;
    };
    expect(at("inside").label).toBe("content_block");
    expect(at("inside").trainable).toBe(true);
    expect(at("inside").inherited).toBe(false);
    expect(at("inside").weight).toBe(1);
    // Outside every landmark there is nothing to inherit, so it stays out.
    expect(at("outside").label).toBe("unlabelled");
    expect(at("outside").trainable).toBe(false);
    expect(at("outside").labelSource).toBe("outside-landmarks");
    expect(at("outside").weight).toBe(0);
  });

  test("a page with fewer than three landmark kinds gets no containment labels", () => {
    const html = build(
      `<main><div id="inside"><p>Body text that is long enough to count.</p></div></main>
       <footer><p>&copy; 2026 Example Inc</p></footer>`,
    );
    const { document } = parseHTML(html);
    const featurized = featurizeDocument(document as never, "https://example.com/x");
    const regions = silverRegionLabels(featurized.nodes);
    const inside = featurized.nodes.find((n) => n.elementId === "inside")!;
    const label = regions.labels.find((l) => l.nodeId === inside.nodeId)!;
    expect(regions.regionTrainable).toBe(true); // two kinds: main + footer
    expect(regions.containmentTrainable).toBe(false); // but not three
    expect(label.label).toBe("unlabelled");
  });

  test("a page with fewer than two landmark kinds is excluded from region training", () => {
    const html = build(`<div id="a"><p>Div soup with nothing semantic at all.</p></div>`);
    const { document } = parseHTML(html);
    const featurized = featurizeDocument(document as never, "https://example.com/x");
    const regions = silverRegionLabels(featurized.nodes);
    expect(regions.regionTrainable).toBe(false);
    expect(regions.excludedReason).toBe("landmark_kinds_0");
    expect(featurized.page.divSoup).toBe(true);
    // The rows are kept, for the eval pool.
    expect(regions.labels.length).toBeGreaterThan(0);
    expect(regions.labels.every((l) => !l.trainable)).toBe(true);
  });

  test("a node inside a landmark INHERITS it, at half weight", () => {
    const html = build(
      `<header><nav><a href="/">Home</a></nav></header>
       <main><p>Body text that is long enough to count.</p></main>
       <aside><p>Sidebar content that is long enough.</p></aside>
       <footer id="f"><div id="cols"><p>&copy; 2026 Example Inc, all rights.</p></div></footer>`,
    );
    const { document } = parseHTML(html);
    const featurized = featurizeDocument(document as never, "https://example.com/x");
    const regions = silverRegionLabels(featurized.nodes);
    const cols = featurized.nodes.find((n) => n.elementId === "cols")!;
    const label = regions.labels.find((l) => l.nodeId === cols.nodeId)!;
    // Previously this was `unlabelled` and dropped from training, which is why
    // the model could name a menu root and then call every item inside it a
    // content_block: it had never been shown one.
    expect(label.label).toBe("footer");
    expect(label.containerLabel).toBe("footer");
    expect(label.labelSource).toBe("inherited:footer");
    expect(label.inherited).toBe(true);
    expect(label.trainable).toBe(true);
    expect(label.weight).toBe(0.5);
  });
});

describe("page type from structured data", () => {
  const withJsonLd = (
    json: string,
    url = "https://example.com/x",
  ): ReturnType<typeof silverPageType> => {
    const html = `<!doctype html><html><head><title>T</title>
      <script type="application/ld+json">${json}</script></head><body><p>x</p></body></html>`;
    const { document } = parseHTML(html);
    return silverPageType(document as never, url);
  };

  test("reads @graph", () => {
    const out = withJsonLd(
      '{"@context":"https://schema.org","@graph":[{"@type":"WebPage"},{"@type":"BlogPosting"}]}',
    );
    expect(out.label).toBe("article");
    expect(out.source).toBe("jsonld");
  });

  test("reads an array of blocks and an array @type", () => {
    expect(withJsonLd('[{"@type":["Product","Thing"]}]').label).toBe("product_detail");
  });

  test("reads a nested mainEntity", () => {
    expect(withJsonLd('{"@type":"WebPage","mainEntity":{"@type":"TechArticle"}}').label).toBe(
      "docs",
    );
  });

  test("normalises a fully qualified type", () => {
    expect(withJsonLd('{"@type":"https://schema.org/NewsArticle"}').label).toBe("article");
  });

  test("a CollectionPage of Products is a product listing, not a conflict", () => {
    const out = withJsonLd(
      '{"@type":"CollectionPage","mainEntity":{"@type":"ItemList","itemListElement":[{"@type":"Product"}]}}',
    );
    expect(out.label).toBe("product_listing");
    expect(out.source).toBe("jsonld");
  });

  test("a genuine conflict is unlabelled, not a coin flip", () => {
    const out = withJsonLd('[{"@type":"NewsArticle"},{"@type":"Product"}]');
    expect(out.label).toBe("unlabelled");
    expect(out.source).toBe("conflict");
  });

  test("an FAQPage next to a Product does not conflict", () => {
    expect(withJsonLd('[{"@type":"FAQPage"},{"@type":"Product"}]').label).toBe("product_detail");
  });

  test("malformed JSON-LD is ignored rather than throwing", () => {
    expect(withJsonLd("{not json at all").label).toBe("unlabelled");
  });

  test("homepage needs the URL as well, and says so", () => {
    const root = withJsonLd('{"@type":"WebSite","name":"X"}', "https://example.com/");
    expect(root.label).toBe("homepage");
    expect(root.source).toBe("jsonld+url_root");
    expect(root.leakProne).toBe(true);
    const deep = withJsonLd('{"@type":"WebSite","name":"X"}', "https://example.com/a/b");
    expect(deep.label).toBe("unlabelled");
  });

  test("og:type is the last resort", () => {
    const html = `<!doctype html><html><head><title>T</title>
      <meta property="og:type" content="article" /></head><body><p>x</p></body></html>`;
    const { document } = parseHTML(html);
    const out = silverPageType(document as never, "https://example.com/x");
    expect(out.label).toBe("article");
    expect(out.source).toBe("og");
  });

  test("no structured data at all is unlabelled, not 'other'", () => {
    const html = `<!doctype html><html><head><title>T</title></head><body><p>x</p></body></html>`;
    const { document } = parseHTML(html);
    const out = silverPageType(document as never, "https://example.com/x");
    expect(out.label).toBe("unlabelled");
    expect(out.source).toBe("none");
    expect(out.types).toEqual([]);
  });
});

describe("inherited landmark labels", () => {
  const page = `<!doctype html><html><head><title>T</title></head><body>
    <header id="hdr"><div id="bar"><a href="/">Logo</a><a href="/x">About us here</a></div>
      <nav id="nav"><ul id="menu"><li id="item"><a href="/a">Products and services</a></li>
      <li><a href="/b">Pricing information</a></li></ul></nav></header>
    <main id="main"><div id="block"><p>Body text that is long enough to count.</p></div></main>
    <aside id="side"><div id="widget"><p>Sidebar widget text goes here.</p></div></aside>
    <footer id="foot"><div id="col"><p>&copy; 2026 Example Inc, all rights reserved.</p></div></footer>
  </body></html>`;

  const labels = () => {
    const { document } = parseHTML(page);
    const featurized = featurizeDocument(document as never, "https://example.com/x");
    const regions = silverRegionLabels(featurized.nodes);
    const at = (elementId: string) => {
      const node = featurized.nodes.find((n) => n.elementId === elementId)!;
      return regions.labels.find((l) => l.nodeId === node.nodeId)!;
    };
    return at;
  };

  test("menu members inherit navigation from the nav they sit in", () => {
    const at = labels();
    expect(at("nav").label).toBe("navigation");
    expect(at("nav").inherited).toBe(false);
    for (const id of ["menu", "item"]) {
      expect([id, at(id).label]).toEqual([id, "navigation"]);
      expect([id, at(id).inherited]).toEqual([id, true]);
      expect([id, at(id).weight]).toEqual([id, 0.5]);
    }
  });

  test("the nearest landmark wins, not the outermost", () => {
    // #menu is inside <nav> which is inside <header>: navigation, not site_header.
    const at = labels();
    expect(at("menu").label).toBe("navigation");
    // #bar is in the header but not in the nav.
    expect(at("bar").label).toBe("site_header");
    expect(at("bar").inherited).toBe(true);
  });

  test("footer columns and sidebar widgets inherit too", () => {
    const at = labels();
    expect(at("col").label).toBe("footer");
    expect(at("widget").label).toBe("sidebar");
  });

  test("inside main it is still content_block, at full weight", () => {
    const at = labels();
    expect(at("block").label).toBe("content_block");
    expect(at("block").inherited).toBe(false);
    expect(at("block").weight).toBe(1);
  });
});

// The page skeleton (#2307): what it keeps, what it must never emit, and that hostile
// markup cannot make it slow or break its output.

import { describe, expect, test } from "bun:test";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { bucket, buildSkeleton, renderCompact, renderXml, skeleton } from "./skeleton.ts";

const doc = (html: string) => parseHTML(html).document;
const page = (body: string, head = "") =>
  `<!doctype html><html lang="en"><head><title>Best shoes</title>${head}</head><body>${body}</body></html>`;

describe("skeleton", () => {
  test("keeps landmarks, headings and length buckets, collapses wrappers", () => {
    const out = skeleton(
      doc(
        page(
          "<div><div><main><div><h1>Best shoes of 2026</h1><p>" +
            "word ".repeat(120) +
            "</p></div></main></div></div>",
        ),
      ),
    );
    expect(out).toContain("<main>");
    expect(out).toContain("<h1>Best shoes of 2026</h1>");
    expect(out).toContain('<p len="500"/>');
    expect(out).not.toContain("<div");
  });

  test("repeated siblings become one run", () => {
    const items = Array.from({ length: 24 }, () => '<li><a href="/p">Item</a></li>').join("");
    const out = skeleton(doc(page(`<main><ul>${items}</ul></main>`)));
    expect(out).toContain('items="24"');
    expect(out).toContain('links="24"');
  });

  test("repeated cards collapse into one group, a lone wrapper is hoisted", () => {
    const card = '<div class="x"><img src="a"><h2>Name</h2><div class="byline">by</div></div>';
    const out = skeleton(doc(page(`<div>${card.repeat(5)}</div>`)), "compact", { text: "none" });
    expect(out).toContain("g x5");
    const lone = skeleton(
      doc(page('<div><div><img src="a"><h2>Name</h2></div></div>')),
      "compact",
      { text: "none" },
    );
    expect(lone).not.toMatch(/^\s*g\b/m);
  });

  test("schema, og:type, itemprop and every script stay out", () => {
    const head =
      '<meta property="og:type" content="article"><script type="application/ld+json">{"@type":"Article"}</script>';
    const out = skeleton(
      doc(
        page(
          '<main itemscope itemtype="https://schema.org/Article"><h1 itemprop="headline">Hi</h1></main><script>var Article=1</script>',
          head,
        ),
      ),
      "xml",
    );
    for (const leak of ["Article", "og:type", "itemprop", "itemtype", "ld+json", "schema.org"])
      expect(out).not.toContain(leak);
  });

  test("hostile page text is escaped and cannot add markup", () => {
    const out = skeleton(
      doc(page("<main><h1>&lt;/h1&gt;&lt;script&gt;alert(1)&lt;/script&gt;</h1></main>")),
    );
    expect(out).not.toContain("<script");
    expect(out).toContain("&lt;/h1&gt;");
    const titled = skeleton(
      doc(
        `<html><head><title>"x"&lt;b&gt;</title></head><body><main><p>a</p></main></body></html>`,
      ),
    );
    expect(titled).toContain("&quot;x&quot;&lt;b&gt;");
  });

  test("text none variant has no page words", () => {
    const out = skeleton(
      doc(page("<main><h1>Secret headline words</h1><p>body</p></main>")),
      "xml",
      { text: "none" },
    );
    expect(out).not.toContain("Secret");
    expect(out).not.toContain("Best shoes");
    expect(out).toContain('<h1 len="');
  });

  test("flags: price, buy button, form fields, comments, byline, time", () => {
    const out = skeleton(
      doc(
        page(
          '<main><h1>Shoe</h1><p class="price">$129.00</p><button>Add to cart</button><form><input type="text"><input type="email"></form>' +
            '<time>2026</time><div class="byline">by a</div><section id="comments"><p>nice</p></section></main>',
        ),
      ),
    );
    expect(out).toContain('price="1"');
    expect(out).toContain('buy="1"');
    expect(out).toContain('fields="2"');
    expect(out).toContain("<time/>");
    expect(out).toContain('kind="byline"');
    expect(out).toContain('kind="comments"');
  });

  test("XML output is balanced and compact is one node per line", () => {
    const html = page(
      '<header><nav><ul><li><a href="/">Home</a></li></ul></nav></header><main><article><h1>T</h1><p>x</p></article></main><footer><p>c</p></footer>',
    );
    const xml = skeleton(doc(html));
    const stack: string[] = [];
    for (const match of xml.matchAll(/<(\/?)([a-z0-9]+)([^>]*?)(\/?)>/g)) {
      if (match[4]) continue;
      if (match[1]) expect(stack.pop()).toBe(match[2]!);
      else stack.push(match[2]!);
    }
    expect(stack.length).toBe(0);
    const tree = buildSkeleton(doc(html));
    expect(renderCompact(tree).split("\n").length).toBeGreaterThan(3);
    expect(renderXml(tree)).toBe(xml);
  });

  test("deterministic, and a 20k-element page respects the node budget in good time", () => {
    const big = page(
      "<main>" +
        Array.from(
          { length: 20000 },
          (_, i) => `<section><h2>H${i}</h2><p>${"x".repeat(i % 700)}</p></section>`,
        ).join("") +
        "</main>",
    );
    const dom = doc(big);
    const started = performance.now();
    const first = skeleton(dom);
    const elapsed = performance.now() - started;
    expect(skeleton(dom)).toBe(first);
    const count = (node: { kids: unknown[] }): number =>
      1 + (node.kids as { kids: unknown[] }[]).reduce((sum, kid) => sum + count(kid), 0);
    expect(count(buildSkeleton(dom))).toBeLessThanOrEqual(302); // 300 nodes + page + more
    expect(first).toContain("<more");
    expect(elapsed).toBeLessThan(1500);
  });

  test("deep nesting does not overflow the stack", () => {
    const deep = "<div>".repeat(3000) + "<p>hi</p>" + "</div>".repeat(3000);
    expect(() => skeleton(doc(page(`<main>${deep}</main>`)))).not.toThrow();
  });

  test("bucket", () => {
    expect([0, 5, 30, 80, 150, 400, 900, 1500, 4000, 9000].map(bucket)).toEqual([
      0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
    ]);
  });
});

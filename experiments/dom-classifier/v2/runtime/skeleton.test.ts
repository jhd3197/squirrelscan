// The page skeleton (#2307): what it keeps, what it must never emit, and that hostile
// markup cannot make it slow or break its output.

import { describe, expect, test } from "bun:test";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import {
  bucket,
  buildSkeleton,
  renderCompact,
  renderXml,
  SKELETON_FEATURES,
  skeleton,
  skeletonFeatures,
  skeletonTokens,
  TOKEN_BUCKETS,
} from "./skeleton.ts";

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

  test("features: fixed keys, counts runs and flags", () => {
    const card =
      '<div><img src="a"><h2>N</h2><p class="price">$5</p><button>Add to cart</button></div>';
    const f = skeletonFeatures(
      buildSkeleton(
        doc(page(`<main><h1>Shop</h1>${card.repeat(6)}<form><input type="text"></form></main>`)),
      ),
    );
    expect(Object.keys(f)).toEqual([...SKELETON_FEATURES]);
    expect(f.h1).toBe(1);
    expect(f.group).toBe(6);
    expect(f.maxRun).toBe(6);
    expect(f.buy).toBeGreaterThan(0);
    expect(f.price).toBeGreaterThan(0);
    expect(f.field).toBe(1);
    expect(f.main).toBe(1);
  });

  test("contact signals: tel and mailto links, email and phone text, map embed, form", () => {
    const out = skeleton(
      doc(
        page(
          '<main><h1>Contact us</h1><address>1 Main St</address><a href="tel:+61400000000">Call</a><a href="mailto:hi@example.com">Email</a>' +
            '<p>Write to hello@example.com or call +61 2 9000 0000</p><iframe src="https://www.google.com/maps/embed?pb=1"></iframe>' +
            '<form><input type="text"><input type="email"><textarea></textarea><button>Send</button></form></main>',
        ),
      ),
    );
    for (const flag of ['tel="1"', 'mailto="1"', 'emailText="1"', 'phoneText="1"', 'map="1"'])
      expect(out).toContain(flag);
    const f = skeletonFeatures(
      buildSkeleton(
        doc(page('<main><a href="tel:1">x</a><form><input type="text"></form></main>')),
      ),
    );
    expect(f.tel).toBe(1);
    expect(f.form).toBe(1);
  });

  test("other page-kind signals: social links, code blocks, pricing terms, updated line", () => {
    const f = skeletonFeatures(
      buildSkeleton(
        doc(
          page(
            '<main><a href="https://twitter.com/x">t</a><pre>code</pre><p>$9 per month, billed annually</p><p>Last updated 1 June 2026</p></main>',
          ),
        ),
      ),
    );
    expect(f.social).toBe(1);
    expect(f.code).toBe(1);
    expect(f.priceTerms).toBe(1);
    expect(f.updated).toBe(1);
  });

  test("a date or a price is not a phone number", () => {
    const f = skeletonFeatures(
      buildSkeleton(
        doc(page("<main><p>Posted 2026-06-01 and 12/06/2026</p><p>$1,299.00 in stock</p></main>")),
      ),
    );
    expect(f.phoneText).toBe(0);
  });

  test("post vs page vs attachment, comment volume, dates, read-more and h1 = title", () => {
    const comments = Array.from(
      { length: 5 },
      (_, i) => `<li class="comment even" id="comment-${i}"><p>spam</p></li>`,
    ).join("");
    const f = (body: string, bodyClass: string, title = "T") =>
      skeletonFeatures(
        buildSkeleton(
          parseHTML(
            `<!doctype html><html><head><title>${title}</title></head><body class="${bodyClass}">${body}</body></html>`,
          ).document,
        ),
      );
    expect(f("<main><p>x</p></main>", "single single-post postid-12 type-post").wpPost).toBe(1);
    expect(f("<main><p>x</p></main>", "page page-id-4 page-template-default").wpPage).toBe(1);
    expect(f("<main><p>x</p></main>", "attachment attachmentid-7").wpAttachment).toBe(1);
    expect(f("<main><p>x</p></main>", "archive category-news").wpArchive).toBe(1);
    expect(f("<main><p>x</p></main>", "single-product woocommerce-page").wpProduct).toBe(1);
    expect(f(`<main><p>x</p><ol>${comments}</ol></main>`, "").commentItems).toBe(5);
    expect(
      f('<main><span class="entry-date">1 June</span><a href="/a">Read more</a></main>', "")
        .dateClass,
    ).toBe(1);
    expect(
      f('<main><span class="entry-date">1 June</span><a href="/a">Read more</a></main>', "")
        .readMore,
    ).toBe(1);
    expect(
      f("<main><h1>Best shoes of 2026</h1></main>", "", "Best shoes of 2026 | My Blog").h1EqTitle,
    ).toBe(1);
    expect(f("<main><h1>Something else</h1></main>", "", "Best shoes").h1EqTitle).toBe(0);
  });

  test("tokens: hashed title and heading words, deterministic, none without words", () => {
    const html = page(
      "<main><h1>Buy running shoes online</h1><h2>Free delivery</h2><p>body words are not tokens</p></main>",
    );
    const tokens = skeletonTokens(buildSkeleton(doc(html)));
    expect(tokens.length).toBeGreaterThan(5);
    expect(tokens.every((t) => t >= 0 && t < TOKEN_BUCKETS)).toBe(true);
    expect(tokens).toEqual([...tokens].sort((a, b) => a - b));
    expect(skeletonTokens(buildSkeleton(doc(html)))).toEqual(tokens);
    expect(skeletonTokens(buildSkeleton(doc(html), { text: "none" }))).toEqual([]);
    const other = skeletonTokens(buildSkeleton(doc(page("<main><h1>Zebra quartz</h1></main>"))));
    expect(other).not.toEqual(tokens);
  });

  test("keyword flags from the title and first heading, across languages, never from body text", () => {
    const f = (title: string, h1: string, body = "") =>
      skeletonFeatures(
        buildSkeleton(
          parseHTML(
            `<!doctype html><html><head><title>${title}</title></head><body><main><h1>${h1}</h1><p>${body}</p></main></body></html>`,
          ).document,
        ),
      );
    expect(f("Contact us | Acme", "Contact us").kwContact).toBe(1);
    expect(f("Kontakt - Firma", "Kontakt").kwContact).toBe(1);
    expect(f("Privacy Policy", "Privacy Policy").kwLegal).toBe(1);
    expect(f("Datenschutzerklärung", "Datenschutz").kwLegal).toBe(1);
    expect(f("About us", "Our story").kwAbout).toBe(1);
    expect(f("Pricing", "Plans and pricing").kwPricing).toBe(1);
    expect(f("FAQ", "Frequently asked questions").kwFaq).toBe(1);
    expect(f("Documentation", "Getting started").kwDocs).toBe(1);
    const none = f("Best shoes", "Best running shoes", "contact privacy about pricing faq docs");
    expect([
      none.kwContact,
      none.kwLegal,
      none.kwAbout,
      none.kwPricing,
      none.kwFaq,
      none.kwDocs,
    ]).toEqual([0, 0, 0, 0, 0, 0]);
  });
});

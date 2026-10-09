// The runtime platform fingerprint (#2307). Parity with the Python tagger on
// real val pages is experiments/platform-parity.ts (9,147/9,147 on 2026-10-09);
// these pin the rules on hand-written markup.

import { describe, expect, test } from "bun:test";

import { platformBucket, platformOf } from "./platform.ts";

const page = (head: string, body = "") =>
  `<!doctype html><html><head>${head}</head><body>${body}</body></html>`;

describe("platformOf", () => {
  test("names each platform from its own markers", () => {
    expect(platformOf(page('<script src="https://cdn.shopify.com/x.js"></script>'))).toBe(
      "shopify",
    );
    expect(platformOf(page('<link href="/wp-content/themes/a.css">'))).toBe("wordpress");
    expect(platformOf(page('<link href="/wp-content/a.css">', '<div class="woocommerce">'))).toBe(
      "woocommerce",
    );
    expect(platformOf(page('<img src="https://static.wixstatic.com/a.png">'))).toBe("wix");
    expect(platformOf(page('<link href="https://static1.squarespace.com/a.css">'))).toBe(
      "squarespace",
    );
    expect(platformOf('<html data-wf-page="1" data-wf-site="2"><body></body></html>')).toBe(
      "webflow",
    );
    expect(platformOf(page('<script id="__NEXT_DATA__" type="application/json">{}</script>'))).toBe(
      "nextjs",
    );
    expect(platformOf(page("<title>plain</title>"))).toBe("custom");
  });

  test("WooCommerce needs WordPress; Shopify wins over everything", () => {
    expect(platformOf(page('<div class="woocommerce"></div>'))).toBe("custom");
    expect(
      platformOf(
        page('<script src="https://cdn.shopify.com/a.js"></script><link href="/wp-content/a">'),
      ),
    ).toBe("shopify");
  });

  test("a generator meta tag counts, case-insensitively", () => {
    expect(platformOf(page('<META NAME="generator" CONTENT="WordPress 6.6">'))).toBe("wordpress");
    expect(platformOf(page('<meta name=generator content="Wix.com Website Builder">'))).toBe("wix");
  });

  test("markers past the first 300k code points are not read", () => {
    expect(platformOf(`${"x".repeat(300_000)}cdn.shopify.com`)).toBe("custom");
    // Astral characters count as ONE code point each, as in Python's slicing.
    expect(platformOf(`${"\u{1F600}".repeat(299_980)}cdn.shopify.com`)).toBe("shopify");
  });
});

describe("platformBucket", () => {
  test("named platforms stay, everything else is custom", () => {
    expect(platformBucket("wix")).toBe("wix");
    expect(platformBucket("drupal")).toBe("custom");
    expect(platformBucket(null)).toBe("custom");
  });
});

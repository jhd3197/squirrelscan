// The deterministic TOC rule (#2307): what it finds, what it refuses, and that
// hostile markup cannot make it slow.

import { describe, expect, test } from "bun:test";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { findTableOfContents, normaliseHeadingText } from "./toc-rule.ts";

const article = (toc: string) => `<html><body><main>${toc}
  <h2 id="a">Getting started</h2><p>x</p><h2 id="b">Install the CLI</h2><p>x</p>
  <h3 id="c">Run your first audit</h3><p>x</p>${"<p>filler</p>".repeat(40)}</main></body></html>`;
const find = (html: string) => findTableOfContents(parseHTML(html).document);

describe("findTableOfContents", () => {
  test("a list of anchors to the headings, before the content, is a TOC", () => {
    const found = find(
      article(
        '<ul class="toc"><li><a href="#a">1. Getting started</a></li><li><a href="#b">Install the CLI</a></li><li><a href="#c">Run your first audit</a></li></ul>',
      ),
    );
    expect(found.length).toBe(1);
    expect(found[0]!.localName).toBe("ul");
  });

  test("anchors whose text does not match the headings are not", () => {
    expect(
      find(
        article(
          '<ul><li><a href="#a">Top</a></li><li><a href="#b">Back</a></li><li><a href="#c">More</a></li></ul>',
        ),
      ).length,
    ).toBe(0);
  });

  test("two matching anchors are not enough", () => {
    expect(
      find(
        article(
          '<ul><li><a href="#a">Getting started</a></li><li><a href="#b">Install the CLI</a></li></ul>',
        ),
      ).length,
    ).toBe(0);
  });
});

describe("hostile markup stays fast (#2307)", () => {
  test("2,000 matching anchor groups (3 anchors each, outside nav): < 1 s", () => {
    // Every group clears the anchor test and reaches the placement check,
    // which used to re-walk the main element once per group.
    const headings = Array.from(
      { length: 30 },
      (_, i) => `<h2 id="h${i}">Section number ${i}</h2>`,
    ).join("");
    const group = (g: number) =>
      `<p>${[0, 1, 2].map((k) => `<a href="#h${(g + k) % 30}">Section number ${(g + k) % 30}</a>`).join(" ")}</p>`;
    const anchors = Array.from({ length: 2_000 }, (_, g) => group(g)).join("");
    const started = performance.now();
    const found = find(`<html><body><main>${anchors}${headings}</main></body></html>`);
    expect(performance.now() - started).toBeLessThan(1_000);
    expect(found.length).toBeGreaterThan(0);
  });

  test("a 100k-character numbering run normalises in < 50 ms", () => {
    const text = `${"1.".repeat(50_000)}x`;
    const started = performance.now();
    normaliseHeadingText(text);
    expect(performance.now() - started).toBeLessThan(50);
  });
});

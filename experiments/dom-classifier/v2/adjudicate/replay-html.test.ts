import { describe, expect, test } from "bun:test";
import { promoteDeferredStyles } from "./replay-html.ts";

describe("promoteDeferredStyles", () => {
  test("promotes the loadCSS preload pattern, quoted or not", () => {
    for (const tag of [
      `<link rel="preload" href="/a.css" as="style" onload="this.onload=null;this.rel='stylesheet'">`,
      `<link rel=preload as=style href=/a.css onload="this.rel='stylesheet'">`,
    ]) {
      const { html, promoted } = promoteDeferredStyles(tag);
      expect(promoted).toBe(1);
      expect(html).toContain('rel="stylesheet"');
      expect(html).not.toContain("onload");
    }
  });
  test("promotes the media=print onload pattern", () => {
    const { html, promoted } = promoteDeferredStyles(
      `<link rel="stylesheet" href="/b.css" media="print" onload="this.media='all'">`,
    );
    expect(promoted).toBe(1);
    expect(html).toBe(`<link rel="stylesheet" href="/b.css" media="all">`);
  });
  test("leaves other links alone", () => {
    for (const tag of [
      `<link rel="preload" href="/f.woff2" as="font" crossorigin>`,
      `<link rel="stylesheet" href="/c.css">`,
      `<link rel="preload" as="image" href="/i.png" onload="x()">`,
      `<link rel="stylesheet" media="print" href="/p.css">`,
    ])
      expect(promoteDeferredStyles(tag)).toEqual({ html: tag, promoted: 0 });
  });
});

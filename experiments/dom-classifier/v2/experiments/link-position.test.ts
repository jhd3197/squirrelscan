import { describe, expect, test } from "bun:test";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { classifyDocument, type PageClassification } from "../runtime/predict.ts";
import { buildPacket } from "./link-check-packets.ts";
import {
  hrefKey,
  linkPositions,
  percentile,
  regionGroup,
  resolveHref,
  summarise,
  tldGroup,
  unwrapWayback,
  type LinkRecord,
  type PageRecord,
} from "./link-position.ts";

const PAGE = "https://www.example.com/blog/post";

const HTML = `<!doctype html><html><head><title>t</title></head><body>
<header id="top"><nav id="menu"><ul>
  <li><a href="https://partner.example.org/">Partner</a></li>
  <li><a href="/about">About</a></li>
</ul></nav></header>
<main id="main"><div id="story"><p>Read
  <a href="https://news.example.net/item?id=7">the report</a> and
  <a href="https://news.example.net/item?id=7">again</a>.
  <a href="mailto:someone@example.org">mail</a>
  <a href="/web/20050101000000/http://old.example.org/page">archived</a>
</p></div></main>
<footer id="foot"><p><a href="https://shop.example.org/x">Shop</a></p></footer>
<svg><a href="https://inside-svg.example.org/">svg link</a></svg>
</body></html>`;

function classify(html = HTML, url = PAGE): { result: PageClassification; elements: unknown[] } {
  const { document } = parseHTML(html);
  const elements: unknown[] = [];
  const result = classifyDocument(document, url, { elementsOut: elements });
  return { result, elements };
}

describe("elementsOut", () => {
  test("is indexed by the same nodeIds the labels carry", () => {
    const { result, elements } = classify();
    expect(result.nodes.length).toBeGreaterThan(0);
    for (const node of result.nodes) {
      const el = elements[node.nodeRef.nodeId] as { localName: string };
      expect(el.localName.toLowerCase()).toBe(node.nodeRef.tag);
    }
  });

  test("is reset per call", () => {
    const { document } = parseHTML("<html><body><p>x</p></body></html>");
    const elements: unknown[] = ["stale"];
    classifyDocument(document, PAGE, { elementsOut: elements });
    expect(elements.includes("stale")).toBe(false);
  });
});

describe("linkPositions without manifest links", () => {
  const { result, elements } = classify();
  const rows = linkPositions(PAGE, result, elements);
  const byHref = (fragment: string) => rows.filter((r) => r.href.includes(fragment));

  test("keeps only http(s) anchors to another registrable domain, outside skipped subtrees", () => {
    const hrefs = rows.map((r) => r.href);
    expect(hrefs.some((h) => h.includes("/about"))).toBe(false);
    expect(hrefs.some((h) => h.startsWith("mailto:"))).toBe(false);
    expect(hrefs.some((h) => h.includes("inside-svg"))).toBe(false);
    expect(rows.length).toBe(5);
  });

  test("maps each link to its nearest confident region", () => {
    expect(byHref("partner")[0]!.region).toBe("navigation");
    expect(byHref("partner")[0]!.regionGroup).toBe("chrome");
    expect(byHref("shop")[0]!.region).toBe("footer");
    for (const row of byHref("news")) {
      expect(["content_block", "main_content"]).toContain(row.region);
      expect(row.regionHops).toBeGreaterThan(0);
    }
  });

  test("unwraps a Wayback href to the original target", () => {
    const archived = byHref("old.example.org");
    expect(archived.length).toBe(1);
    expect(archived[0]!.href).toBe("http://old.example.org/page");
  });

  test("counts anchors sharing a resolved href", () => {
    expect(byHref("news").every((r) => r.candidates === 2)).toBe(true);
  });
});

describe("linkPositions with manifest links", () => {
  test("disambiguates by anchor text and reports links it cannot find", () => {
    const { result, elements } = classify();
    const rows = linkPositions(PAGE, result, elements, [
      { href: "http://news.example.net/item?id=7", anchor_text: "again", link_type: "dofollow" },
      { href: "https://missing.example.org/", anchor_text: "nowhere" },
    ]);
    expect(rows[0]!.matched).toBe(true);
    expect(rows[0]!.anchorText).toBe("again");
    expect(rows[0]!.linkType).toBe("dofollow");
    const again = (elements[rows[0]!.anchorNodeId!] as { textContent: string }).textContent;
    expect(again).toBe("again");
    expect(rows[1]!.matched).toBe(false);
    expect(rows[1]!.region).toBe("unknown");
    expect(rows[1]!.regionNodeId).toBeNull();
  });
});

describe("section lookup", () => {
  test("is independent of the region lookup", () => {
    const { result, elements } = classify();
    const footer = result.nodes.find((n) => n.nodeRef.elementId === "foot")!;
    footer.section = { label: "newsletter_signup", confidence: 0.97, source: "model" };
    const shop = linkPositions(PAGE, result, elements).find((r) => r.href.includes("shop"))!;
    expect(shop.section).toBe("newsletter_signup");
    expect(shop.sectionSource).toBe("model");
    expect(shop.sectionNodeId).toBe(footer.nodeRef.nodeId);
    expect(shop.sectionHops).toBe(shop.regionHops);
    const partner = linkPositions(PAGE, result, elements).find((r) => r.href.includes("partner"))!;
    expect(partner.section).toBeNull();
  });
});

describe("comments rule", () => {
  test("marks a link inside comment markup, with its rel tokens, whatever the model says", () => {
    const html = HTML.replace(
      "</main>",
      `<div id="comments" class="comments-area"><ol class="comment-list"><li id="comment-3" class="comment">
        <div class="comment-author"><a rel="external nofollow ugc" href="https://spam.example.org/">Spammer</a></div>
      </li></ol></div></main>`,
    );
    const { result, elements } = classify(html);
    const rows = linkPositions(PAGE, result, elements);
    const spam = rows.find((r) => r.href.includes("spam"))!;
    expect(spam.commentsRule).toBe("comments");
    expect(spam.commentsRuleToken).toBe("comment-author");
    expect(spam.commentsRuleHops).toBe(1);
    expect(spam.rel).toEqual(["ugc", "nofollow"]);
    expect(spam.section).toBe("comments");
    expect(spam.sectionSource).toBe("rules");
    const story = rows.find((r) => r.href.includes("news"))!;
    expect(story.commentsRule).toBeNull();
    expect(story.rel).toEqual([]);
  });
});

describe("helpers", () => {
  test("resolveHref and unwrapWayback", () => {
    expect(unwrapWayback("https://web.archive.org/web/2004id_/http://a.example.org/x")).toBe(
      "http://a.example.org/x",
    );
    expect(resolveHref("javascript:void(0)", PAGE)).toBeNull();
    expect(resolveHref("#top", PAGE)).toBeNull();
    expect(
      resolveHref(
        "/web/19991231235959/https://b.example.org/",
        "https://web.archive.org/web/1999/x",
      ),
    ).toBe("https://b.example.org/");
    expect(hrefKey("https://WWW.Example.org/a/#frag")).toBe(hrefKey("http://example.org/a"));
  });

  test("tldGroup and regionGroup", () => {
    expect(tldGroup("https://a.example.com/")).toBe("com");
    expect(tldGroup("https://a.example.org/")).toBe("org_net");
    expect(tldGroup("https://a.example.de/")).toBe("cc");
    expect(tldGroup("https://a.example.co.uk/")).toBe("cc");
    expect(tldGroup("not a url")).toBe("other");
    expect(regionGroup("sidebar")).toBe("chrome");
    expect(regionGroup("search")).toBe("other");
  });

  test("percentile and summarise", () => {
    expect(percentile([], 50)).toBeNull();
    expect(percentile([5, 1, 3, 2, 4], 50)).toBe(3);
    expect(percentile([5, 1, 3, 2, 4], 95)).toBe(5);
    const page = (key: string, ms: number, extra: Partial<PageRecord> = {}): PageRecord => ({
      sha256: "x",
      key,
      ms,
      parseMs: 0,
      missing: false,
      parseFailure: false,
      budgetFailure: false,
      nodes: 1,
      pageType: "homepage",
      pageTypeSource: "rules",
      degraded: "",
      links: 0,
      matched: 0,
      ...extra,
    });
    const link = (
      key: string,
      region: string,
      section: string | null,
      matched = true,
    ): LinkRecord =>
      ({
        sha256: "x",
        key,
        region,
        regionGroup: regionGroup(region),
        section,
        matched,
      }) as LinkRecord;
    const summary = summarise(
      [
        page("a|2005|com|live", 10),
        page("a|2005|com|live", 30, { budgetFailure: true }),
        page("b|na|cc|na", 5, { parseFailure: true }),
      ],
      [
        link("a|2005|com|live", "footer", null),
        link("a|2005|com|live", "unknown", null),
        {
          ...link("a|2005|com|live", "content_block", "comments"),
          commentsRule: "comments",
          rel: ["ugc", "nofollow"],
        },
        link("a|2005|com|live", "unknown", null, false),
      ],
    ) as Record<string, any>;
    const a = summary["a|2005|com|live"];
    expect(a.pages).toBe(2);
    expect(a.budgetFailures).toBe(1);
    expect(a.linksLocated).toBe(3);
    expect(a.regionAnswered).toBe(0.667);
    expect(a.regionGroups.chrome).toBe(0.333);
    expect(a.sections).toEqual({ comments: 1 });
    expect(a.commentsRule).toBe(0.333);
    expect(a.relUgc).toBe(0.333);
    expect(summary["a|*|*|*"].pages).toBe(2);
    expect(summary["*|*|*|*"].parseFailures).toBe(1);
    expect(summary["*|*|*|*"].msP50).toBe(10);
  });
});

describe("link-check packets", () => {
  test("outline the anchor the probe located, with no classifier output in it", () => {
    const { result, elements } = classify();
    const shop = linkPositions(PAGE, result, elements).find((r) => r.href.includes("shop"))!;
    const packet = buildPacket(HTML, {
      itemId: "t",
      sha256: "x",
      url: PAGE,
      anchorNodeId: shop.anchorNodeId!,
      href: shop.href,
      anchorText: shop.anchorText,
    })!;
    expect(packet).toContain("anchor text: Shop");
    expect(packet).toContain("⟦LINK: Shop⟧");
    expect(packet).toMatch(/footer#foot \[spans \d+-\d+% of the document\]/);
    expect(
      packet
        .split("\n")
        .find((l) => l.includes("<- THE LINK"))!
        .trim()
        .startsWith("a "),
    ).toBe(true);
    for (const label of [
      "navigation",
      "content_block",
      "main_content",
      "unknown",
      "newsletter_signup",
    ]) {
      expect(packet).not.toContain(label);
    }
  });

  test("hide the id and class tokens a markup rule reads, and keep the rest", () => {
    const html = HTML.replace(
      "</main>",
      `<div id="comments" class="comments-area wide"><ol class="comment-list"><li id="comment-3" class="comment odd">
        <a rel="ugc" href="https://spam.example.org/">Spammer</a></li></ol></div></main>`,
    );
    const { result, elements } = classify(html);
    const spam = linkPositions(PAGE, result, elements).find((r) => r.href.includes("spam"))!;
    const row = {
      itemId: "t",
      sha256: "x",
      url: PAGE,
      anchorNodeId: spam.anchorNodeId!,
      href: spam.href,
      anchorText: spam.anchorText,
    };
    expect(buildPacket(html, row)!).toContain("#comments.comments-area.wide");
    const blind = buildPacket(html, row, { hideTokens: /comment/i })!;
    expect(blind).not.toMatch(/comment/i);
    expect(blind).toContain("div.wide");
    expect(blind).toContain("li.odd");
  });

  test("name an image link by its alt text, and refuse a nodeId that is not an anchor", () => {
    const html = `<html><body><footer><a href="https://x.example.org/"><img alt="Our profile"></a></footer></body></html>`;
    const { document } = parseHTML(html);
    const elements: unknown[] = [];
    classifyDocument(document, PAGE, { elementsOut: elements });
    const anchorId = elements.findIndex((e) => (e as { localName: string }).localName === "a");
    const row = {
      itemId: "t",
      sha256: "x",
      url: PAGE,
      anchorNodeId: anchorId,
      href: "https://x.example.org/",
      anchorText: "",
    };
    expect(buildPacket(html, row)).toContain("anchor text: Our profile (image alt)");
    expect(buildPacket(html, { ...row, anchorNodeId: 0 })).toBeNull();
  });
});

describe("review fixes", () => {
  test("an explicitly empty manifest list means no links, not every anchor", () => {
    const { result, elements } = classify();
    expect(linkPositions(PAGE, result, elements, [])).toEqual([]);
  });

  test("a link listed twice takes two different anchors", () => {
    const html = `<html><body><header><a href="https://t.example.org/">Go</a></header>
      <footer><a href="https://t.example.org/">Go</a></footer></body></html>`;
    const { result, elements } = classify(html);
    const rows = linkPositions(PAGE, result, elements, [
      { href: "https://t.example.org/", anchor_text: "Go" },
      { href: "https://t.example.org/", anchor_text: "Go" },
    ]);
    expect(rows[0]!.anchorNodeId).not.toBe(rows[1]!.anchorNodeId);
    expect(rows.map((r) => r.region).sort()).toEqual(["footer", "site_header"]);
    expect(rows.every((r) => r.matchedBy === "text")).toBe(true);
  });

  test("relative hrefs resolve against <base href>", () => {
    const html = `<html><head><base href="https://cdn.example.org/a/"></head><body>
      <main><p><a href="x">rel</a></p></main></body></html>`;
    const { result, elements } = classify(html);
    expect(linkPositions(PAGE, result, elements, null, "https://cdn.example.org/a/")[0]!.href).toBe(
      "https://cdn.example.org/a/x",
    );
    expect(linkPositions(PAGE, result, elements, null)).toEqual([]);
  });

  test("trailing slashes trim in linear time", () => {
    expect(hrefKey("https://example.org/a///")).toBe("example.org/a");
    const started = performance.now();
    hrefKey(`https://example.org/${"/".repeat(100_000)}x`);
    expect(performance.now() - started).toBeLessThan(50);
  });
});

// The HTML report is the one humans open, and it showed none of the
// component-aware findings (#2307). These pin the policy it now shares with
// markdown/llm/text: a group REPLACES the legacy page list rather than adding
// to it, a one-page group is not a group, and a report with nothing to say
// renders exactly as it did before.

import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, test } from "bun:test";

import type { AuditReport, CheckResult, ComponentOccurrence, ReportRuleResult } from "../src/types";
import type { ComponentFixGroup } from "../src/component-fix-groups";
import { FixGroupCards, renderHtml } from "../src/output/html";

function occurrence(
  pageUrl: string,
  overrides: Partial<ComponentOccurrence> = {},
): ComponentOccurrence {
  return {
    version: 1,
    pageUrl,
    siteOrigin: "https://example.test",
    provenance: { source: "page-dom", rendered: true },
    groupable: true,
    confidence: "observed",
    region: { role: "footer", nestedIn: "none", structuralSignature: "s128:aaaa" },
    family: { key: "s128:bbbb", structuralSignature: "s128:bbbb" },
    variant: { key: "s128:cccc", structuralSignature: "s128:cccc", contentHash: "s128:dddd" },
    element: { locator: "footer:1>footer>ul:1>a:2", structuralSignature: "s128:eeee" },
    defect: {
      kind: "link-text-generic",
      values: { nameSource: "text", normalizedText: "click here" },
      valueHashes: { normalizedText: "s128:ffff" },
    },
    ...overrides,
  };
}

function check(pageUrl: string, evidence?: ComponentOccurrence[]): CheckResult {
  return {
    name: "generic-link-text",
    status: "warn",
    message: "1 link(s) with generic text",
    pageUrl,
    pages: [pageUrl],
    items: [{ id: "click here" }],
    ...(evidence ? { componentOccurrences: evidence } : {}),
  };
}

function report(checks: CheckResult[]): AuditReport {
  const rule: ReportRuleResult = {
    meta: {
      id: "a11y/link-text",
      name: "Link text",
      description: "desc",
      solution: "sol",
      category: "accessibility",
      scope: "page",
      severity: "warning",
      weight: 1,
    },
    checks,
  };
  return {
    baseUrl: "https://example.test",
    timestamp: "2026-09-19T00:00:00.000Z",
    totalPages: 4,
    passed: 0,
    warnings: checks.length,
    failed: 0,
    ruleResults: { "a11y/link-text": rule },
    healthScore: {
      overall: 80,
      categories: [],
      groups: [],
      errorCount: 0,
      warningCount: 1,
      passedCount: 0,
    },
  } as unknown as AuditReport;
}

const PAGES = ["https://example.test/", "https://example.test/api", "https://example.test/guides"];

function countOccurrences(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

/** The rendered document only. The stylesheet in <head> names every class. */
function bodyOf(html: string): string {
  return html.slice(html.indexOf("<body>"));
}

describe("full coverage: the card replaces the legacy page list", () => {
  const full = report(PAGES.map((page) => check(page, [occurrence(page)])));
  const html = renderHtml(full);

  test("one card, worded exactly as the other surfaces word it", () => {
    expect(html).toContain(
      '<span class="fix-badge">Fix once</span>Fix once in the site footer: ' +
        "link text &quot;click here&quot; is generic, 3 pages",
    );
    expect(countOccurrences(html, '<li class="fix-group"')).toBe(1);
  });

  test("the legacy per-check page list is gone, the card's own list is not", () => {
    // The rule header keeps its own "3 pages affected" count; what goes is the
    // per-check list that repeated those same pages under the card.
    expect(html).not.toContain("<summary>3 pages affected</summary>");
    expect(countOccurrences(html, "pages affected")).toBe(1);
    expect(html).toContain("<summary>Affected pages (3)</summary>");
    // Every affected page is still reachable, as a real link.
    for (const page of PAGES) expect(html).toContain(`href="${page}"`);
  });

  test("no hashes, signatures or internal keys reach the page", () => {
    expect(html).not.toContain("s128:");
    expect(html).not.toContain("structuralSignature");
    expect(html).not.toContain("contentHash");
    expect(html).not.toContain("valueHashes");
    expect(html).not.toContain("component-fix:");
  });

  test("the locator appears once, as a muted developer hint", () => {
    expect(countOccurrences(html, 'class="fix-group-slot"')).toBe(1);
    expect(html).toContain('<div class="fix-group-slot">ul:1 &gt; a:2</div>');
  });

  test("no em-dashes in the rendered copy", () => {
    const body = html.slice(html.indexOf("<body>"));
    expect(body).not.toContain("—");
  });
});

describe("partial coverage: the remainder is labelled, never hidden", () => {
  const html = renderHtml(
    report([...PAGES.map((page) => check(page, [occurrence(page)])), check("https://example.test/legal")]),
  );

  test("the uncovered page keeps a list of its own, explicitly named", () => {
    expect(html).toContain("<summary>Pages not attributed to a shared component (1)</summary>");
    expect(html).toContain('href="https://example.test/legal"');
  });

  test("the covered pages are not repeated in that remainder", () => {
    const start = html.indexOf("Pages not attributed to a shared component (1)");
    const remainder = html.slice(start, html.indexOf("</details>", start));
    expect(remainder).toContain('href="https://example.test/legal"');
    expect(remainder).not.toContain('href="https://example.test/api"');
    expect(remainder).not.toContain('href="https://example.test/guides"');
  });

  test("the fix card is still there", () => {
    expect(html).toContain("Fix once in the site footer");
  });
});

describe("a one-page group is not a group", () => {
  test("HTML is byte-identical to the same report with no evidence at all", () => {
    const singlePage = report([check(PAGES[0]!, [occurrence(PAGES[0]!)])]);
    const legacy = report([check(PAGES[0]!)]);
    expect(renderHtml(singlePage)).toBe(renderHtml(legacy));
    expect(bodyOf(renderHtml(singlePage))).not.toContain("fix-group");
    // The defect is still reported the legacy way, not swallowed.
    expect(renderHtml(singlePage)).toContain("<summary>1 page affected</summary>");
  });
});

describe("attribution changes what the card claims", () => {
  // A block repeating inside main content, promoted on cross-page evidence.
  // The rule stays uncertain; the group carries the weaker attribution.
  const repeated = report(
    ["https://example.test/", "https://example.test/api", "https://example.test/guides"].map(
      (page) =>
        check(page, [
          occurrence(page, {
            groupable: false,
            confidence: "uncertain",
            uncertainReason: "region-main",
            region: { role: "main", nestedIn: "none", structuralSignature: "s128:aaaa" },
            subtreeCandidates: [
              {
                level: 2,
                skeletonSignature: "s128:block",
                locatorFromRoot: "div:1>a:1",
                textShareOfRegion: 0.2,
              },
            ],
          }),
        ]),
    ),
  );
  const html = renderHtml(repeated);

  test("it reads as an observation, not an instruction", () => {
    expect(html).toContain(
      "Likely one shared block in the main content: " +
        "link text &quot;click here&quot; is generic, seen on 3 pages",
    );
    expect(html).not.toContain("Fix once");
  });

  test("the badge is the weaker one, and says so in words", () => {
    expect(html).toContain('<span class="fix-badge fix-badge-weak">Likely shared block</span>');
    expect(html).toContain('class="fix-group fix-group-weak"');
  });

  test("an unknown attribution renders no card and keeps the legacy list", () => {
    const legacy = report(PAGES.map((page) => check(page)));
    // `ruleFixCoverage` drops an unknown attribution upstream (mode "none").
    // If one ever reaches the card renderer anyway, it must render NOTHING
    // rather than narrate a future signal with borrowed confidence.
    const markup = renderToStaticMarkup(
      React.createElement(FixGroupCards, {
        coverage: {
          mode: "full" as const,
          groups: [
            {
              id: "component-fix:0",
              ruleId: "a11y/link-text",
              check: { name: "generic-link-text", status: "warn" },
              attribution: "some-future-signal",
              affectedPages: PAGES,
              affectedPageCount: 3,
              region: { role: "footer", nestedIn: "none", structuralSignature: "s128:aaaa" },
              semanticSlot: "footer:1>footer>ul:1>a:2",
              defect: { kind: "link-text-generic", values: { normalizedText: "x" }, valueHashes: {} },
              occurrences: [],
            } as unknown as ComponentFixGroup,
          ],
          uncoveredPages: [],
        },
      }),
    );
    expect(markup).toBe("");
    // And the report it came from still renders the pages the legacy way.
    expect(renderHtml(legacy)).toContain("<summary>3 pages affected</summary>");
  });
});

describe("a rollup is one card, however many faulty elements the block holds", () => {
  // squirrelscan.com's install block: seven platform links, one component, 14
  // pages. The groups are built here rather than crawled into existence: the
  // whole grouping stage would have to run to reach the one rendering branch
  // that differs, and `componentRootKey` is the only input that decides it.
  const ROLLUP_PAGES = Array.from({ length: 14 }, (_, i) => `https://example.test/r/${i}`);
  const groups = ["download", "get it", "install", "grab", "fetch", "copy", "run"].map(
    (text, index) =>
      ({
        id: `component-fix:${index}`,
        ruleId: "a11y/link-text",
        check: { name: "generic-link-text", status: "warn" },
        attribution: "repeated-subtree",
        affectedPages: ROLLUP_PAGES,
        affectedPageCount: ROLLUP_PAGES.length,
        region: { role: "main", nestedIn: "none", structuralSignature: "s128:aaaa" },
        element: { locator: "main:1>div:1>a:1", structuralSignature: "s128:eeee" },
        semanticSlot: `div:1>a:${index + 1}`,
        defect: {
          kind: "link-text-generic",
          values: { normalizedText: text },
          valueHashes: {},
        },
        occurrences: ROLLUP_PAGES.map(() => ({}) as ComponentOccurrence),
        componentRootKey: "s128:install-block",
      }) as unknown as ComponentFixGroup,
  );

  const markup = renderToStaticMarkup(
    React.createElement(FixGroupCards, {
      coverage: { mode: "full" as const, groups, uncoveredPages: [] },
    }),
  );

  test("seven groups collapse into one card naming the block, not the buttons", () => {
    expect(countOccurrences(markup, "<li class=")).toBe(1);
    // The defect clause is `describeRollupDefect`, the same one markdown, text,
    // llm and the console print, so the four surfaces cannot drift. It counts
    // faulty ELEMENTS (7 slots x 14 pages = 98), not slots.
    expect(markup).toContain(
      "Likely one shared block in the main content: 98 links with generic text " +
        "&quot;download&quot;, &quot;get it&quot;, &quot;install&quot; (+4 more), seen on 14 pages",
    );
  });

  test("the card lists all 14 pages, collapsed", () => {
    expect(markup).toContain("<summary>Affected pages (14)</summary>");
    expect(countOccurrences(markup, 'href="https://example.test/r/')).toBe(14);
  });

  test("without a component root the groups stay separate cards", () => {
    const separate = groups.map((group) => {
      const { componentRootKey: _root, ...rest } = group as ComponentFixGroup & {
        componentRootKey: string;
      };
      return rest as ComponentFixGroup;
    });
    const plain = renderToStaticMarkup(
      React.createElement(FixGroupCards, {
        coverage: { mode: "full" as const, groups: separate, uncoveredPages: [] },
      }),
    );
    expect(countOccurrences(plain, "<li class=")).toBe(7);
  });
});

describe("site-controlled strings are neutralised", () => {
  const hostile = "\"><script>alert(1)</script>";
  const html = renderHtml(
    report(
      PAGES.map((page) =>
        check(page, [
          occurrence(page, {
            defect: {
              kind: "link-text-generic",
              values: { normalizedText: hostile },
              valueHashes: { normalizedText: "s128:ffff" },
            },
          }),
        ]),
      ),
    ),
  );

  test("a defect value cannot open a tag or break out of an attribute", () => {
    expect(html).toContain("&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;");
    expect(html).not.toContain("<script>alert(1)</script>");
  });

  test("a javascript: page URL never reaches an href", () => {
    const evil = "javascript:alert(1)";
    const withEvilPage = renderHtml(
      report([
        ...PAGES.map((page) => check(page, [occurrence(page)])),
        { ...check(evil), pages: [evil], pageUrl: evil },
      ]),
    );
    expect(withEvilPage).not.toContain('href="javascript:');
    expect(withEvilPage).toContain('href="#"');
  });
});

describe("no component evidence: nothing changes", () => {
  // Uncertain evidence with no ladder produces no groups at all. Any difference
  // here is the feature leaking into reports it has nothing to say about.
  test("HTML is byte-identical with and without the evidence", () => {
    const withEvidence = report(
      PAGES.map((page) =>
        check(page, [
          occurrence(page, { groupable: false, confidence: "uncertain", uncertainReason: "no-region" }),
        ]),
      ),
    );
    const withoutEvidence = report(PAGES.map((page) => check(page)));
    expect(renderHtml(withEvidence)).toBe(renderHtml(withoutEvidence));
    expect(renderHtml(withoutEvidence)).toContain("<summary>3 pages affected</summary>");
    expect(bodyOf(renderHtml(withoutEvidence))).not.toContain("fix-group");
  });
});

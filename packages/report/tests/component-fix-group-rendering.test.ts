// A fix group has to REPLACE output, not add to it. These pin the three states
// a rule can be in, and the hard one: a rule with no groups must render exactly
// as it did before the feature existed.

import { describe, expect, test } from "bun:test";
import type { AuditReport, CheckResult, ComponentOccurrence, ReportRuleResult } from "../src/types";
import { renderLlm } from "../src/output/llm";
import { renderMarkdown } from "../src/output/markdown";
import { renderText } from "../src/output/text";

function occurrence(pageUrl: string, overrides: Partial<ComponentOccurrence> = {}): ComponentOccurrence {
  return {
    version: 1,
    pageUrl,
    siteOrigin: "https://example.test",
    provenance: { source: "page-dom", rendered: true },
    groupable: true,
    confidence: "observed",
    region: { role: "header", nestedIn: "none", structuralSignature: "s128:aaaa" },
    family: { key: "s128:bbbb", structuralSignature: "s128:bbbb" },
    variant: { key: "s128:cccc", structuralSignature: "s128:cccc", contentHash: "s128:dddd" },
    element: { locator: "header:1>header>nav:1>a:1", structuralSignature: "s128:eeee" },
    defect: {
      kind: "link-text-generic",
      values: { nameSource: "text", normalizedText: "website" },
      valueHashes: { normalizedText: "s128:ffff" },
    },
    ...overrides,
  };
}

function report(checks: CheckResult[]): AuditReport {
  const rule: ReportRuleResult = {
    meta: {
      id: "a11y/link-text", name: "Link text", description: "desc", solution: "sol",
      category: "accessibility", scope: "page", severity: "warning", weight: 1,
    },
    checks,
  };
  return {
    baseUrl: "https://example.test", timestamp: "2026-09-19T00:00:00.000Z", totalPages: 4,
    passed: 0, warnings: checks.length, failed: 0, ruleResults: { "a11y/link-text": rule },
    healthScore: { overall: 80, categories: [], groups: [], errorCount: 0, warningCount: 1, passedCount: 0 },
  } as unknown as AuditReport;
}

function check(pageUrl: string, evidence?: ComponentOccurrence[]): CheckResult {
  return {
    name: "generic-link-text",
    status: "warn",
    message: "1 link(s) with generic text",
    pageUrl,
    pages: [pageUrl],
    items: [{ id: "website" }],
    ...(evidence ? { componentOccurrences: evidence } : {}),
  };
}

const PAGES = ["https://example.test/", "https://example.test/api", "https://example.test/guides"];

describe("full coverage: the group replaces the legacy lines", () => {
  const full = report(PAGES.map((page) => check(page, [occurrence(page)])));

  test("llm emits one fix-once line and drops Pages and Items", () => {
    const out = renderLlm(full);
    expect(out).toContain(
      '<fix-once pages="3">Fix once in the site header: link text &quot;website&quot; is generic, 3 pages: /, /api, /guides [nav:1 &gt; a:1]</fix-once>',
    );
    expect(out).not.toContain("Pages (");
    expect(out).not.toContain("Items (");
  });

  test("text and markdown say the same thing in the same words", () => {
    expect(renderText(full)).toContain(
      'Fix once in the site header (nav:1 > a:1): link text "website" is generic, 3 pages: /, /api, /guides',
    );
    expect(renderMarkdown(full)).toContain(
      '**Fix once in the site header (nav:1 > a:1): link text "website" is generic, 3 pages: /, /api, /guides**',
    );
  });

  test("no hashes, signatures or internal keys reach a human surface", () => {
    for (const out of [renderLlm(full), renderText(full), renderMarkdown(full)]) {
      expect(out).not.toContain("s128:");
      expect(out).not.toContain("structuralSignature");
      expect(out).not.toContain("contentHash");
      expect(out).not.toContain("valueHashes");
      // No em-dashes in public copy.
      expect(out).not.toContain("—");
    }
  });
});

describe("partial coverage: groups first, remainder explicitly labelled", () => {
  const partial = report([
    ...PAGES.map((page) => check(page, [occurrence(page)])),
    check("https://example.test/legal"),
  ]);

  test("llm labels the uncovered pages rather than hiding them", () => {
    const out = renderLlm(partial);
    expect(out).toContain('<fix-once pages="3"');
    expect(out).toContain("Pages not attributed to a shared component (1): /legal");
    // The covered pages are NOT repeated in that remainder list.
    const remainder = out.split("Pages not attributed to a shared component (1): ")[1]!.split("\n")[0]!;
    expect(remainder).toBe("/legal");
    expect(out).not.toContain("Pages (4)");
  });

  test("text labels the remainder too", () => {
    const out = renderText(partial);
    expect(out).toContain("Fix once in the site header");
    expect(out).toContain("Not attributed to a shared component (1):");
    expect(out).toContain("-> /legal");
  });
});

describe("no groups: byte-identical to the pre-feature rendering", () => {
  // The same report with the evidence stripped is what v0.0.96 produced. Any
  // difference here is the feature leaking into reports it has nothing to say
  // about, which is the regression this whole lane exists to prevent.
  const withEvidence = report(PAGES.map((page) => check(page, [occurrence(page, { groupable: false, confidence: "uncertain", uncertainReason: "region-main" })])));
  const withoutEvidence = report(PAGES.map((page) => check(page)));

  test("llm", () => {
    expect(renderLlm(withEvidence)).toBe(renderLlm(withoutEvidence));
    expect(renderLlm(withoutEvidence)).toContain("Pages (3): /, /api, /guides");
    expect(renderLlm(withoutEvidence)).toContain("Items (");
  });

  test("text", () => {
    expect(renderText(withEvidence)).toBe(renderText(withoutEvidence));
  });

  test("markdown", () => {
    expect(renderMarkdown(withEvidence)).toBe(renderMarkdown(withoutEvidence));
  });
});

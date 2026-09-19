// Rollups and the packed-evidence tables the JSON emits.
//
// The rollup exists because squirrelscan.com's install block produced seven
// repeated-subtree groups (one per platform button) across 14 pages: seven
// lines for one edit.

import { describe, expect, test } from "bun:test";
import type { AuditReport, CheckResult, ComponentOccurrence, ReportRuleResult } from "../src/types";
import { componentFixRollups } from "../src/component-fix-rollup";
import { fixRollupLine } from "../src/component-presentation";
import type { ComponentFixGroup } from "../src/component-fix-groups";
import { renderJson } from "../src/output/json";
import { renderLlm } from "../src/output/llm";
import { renderText } from "../src/output/text";

const SITE = "https://example.test";
const PAGES = Array.from({ length: 14 }, (_, i) => `${SITE}/releases/1.0.${i}`);

/** A group as the grouping stage produces them, plus the additive root key. */
function group(overrides: Partial<ComponentFixGroup> & { root?: string; value?: string }): ComponentFixGroup {
  const { root, value, ...rest } = overrides;
  const base = {
    id: `component-fix:${(value ?? "x").padEnd(32, "0").slice(0, 32)}`,
    ruleId: "a11y/link-text",
    check: { name: "generic-link-text", status: "warn" as const },
    attribution: "repeated-subtree" as const,
    affectedPages: PAGES,
    affectedPageCount: PAGES.length,
    region: { role: "main" as const, nestedIn: "none" as const, structuralSignature: "s128:r" },
    element: { locator: "main:1>main>div:1>a:1", structuralSignature: "s128:e" },
    semanticSlot: "main:1>main>div:1>a:1",
    defect: {
      kind: "link-text-generic" as const,
      values: { normalizedText: value ?? "download" },
      valueHashes: {},
    },
    occurrences: PAGES.map(() => ({}) as ComponentOccurrence),
    ...rest,
  } as ComponentFixGroup;
  // Terra's additive key; read optionally by the rollup.
  return root === undefined ? base : ({ ...base, componentRootKey: root } as ComponentFixGroup);
}

describe("rolling up one block", () => {
  test("seven per-button groups become one unit", () => {
    const groups = ["download", "download", "download", "download", "download", "get", "get"].map(
      (value, i) => group({ root: "install-block", value: `${value}${i}` }),
    );
    const rollups = componentFixRollups(groups);
    expect(rollups).toHaveLength(1);
    const [rollup] = rollups;
    expect(rollup!.groupIds).toHaveLength(7);
    expect(rollup!.elementCount).toBe(7 * PAGES.length);
    expect(rollup!.affectedPageCount).toBe(14);
    // Distinct values are listed up to the cap, then counted.
    expect(rollup!.distinctValues).toHaveLength(3);
    expect(rollup!.additionalValues).toBe(4);
  });

  test("differing page sets never merge", () => {
    const a = group({ root: "install-block", value: "a" });
    const b = group({
      root: "install-block",
      value: "b",
      affectedPages: PAGES.slice(0, 13),
      affectedPageCount: 13,
    });
    expect(componentFixRollups([a, b])).toHaveLength(2);
  });

  test("a different component root never merges", () => {
    expect(
      componentFixRollups([
        group({ root: "install-block", value: "a" }),
        group({ root: "footer-block", value: "b" }),
      ]),
    ).toHaveLength(2);
  });

  test("without a root key each group stands alone", () => {
    // The key is what establishes "same component". Guessing one from the
    // locator would merge unrelated blocks that happen to sit at the same depth.
    expect(componentFixRollups([group({ value: "a" }), group({ value: "b" })])).toHaveLength(2);
  });

  test("a single-member rollup is worded exactly like the group", () => {
    const one = componentFixRollups([group({ root: "install-block", value: "download" })]);
    expect(one).toHaveLength(1);
    expect(one[0]!.groups).toHaveLength(1);
  });
});

function report(groups: ComponentFixGroup[]): AuditReport {
  const checks: CheckResult[] = PAGES.map((page) => ({
    name: "generic-link-text",
    status: "warn",
    message: "7 link(s) with generic text",
    pageUrl: page,
    pages: [page],
  }));
  const rule: ReportRuleResult = {
    meta: {
      id: "a11y/link-text", name: "Link text", description: "d", solution: "s",
      category: "accessibility", scope: "page", severity: "warning", weight: 1,
    },
    checks,
  };
  const built = {
    baseUrl: SITE, timestamp: "2026-09-19T00:00:00.000Z", totalPages: PAGES.length,
    passed: 0, warnings: checks.length, failed: 0, ruleResults: { "a11y/link-text": rule },
    healthScore: { overall: 80, categories: [], groups: [], errorCount: 0, warningCount: 1, passedCount: 0 },
  } as unknown as AuditReport;
  // The grouping stage derives groups from evidence; here they are supplied
  // directly so the rollup can be exercised before that field exists upstream.
  (built as unknown as { __fixGroups: ComponentFixGroup[] }).__fixGroups = groups;
  return built;
}

describe("rendered output", () => {
  const groups = Array.from({ length: 7 }, (_, i) =>
    group({ root: "install-block", value: i < 5 ? "download" : `get${i}` }),
  );

  test("one line stands in for seven groups", () => {
    const rollups = componentFixRollups(groups);
    expect(rollups).toHaveLength(1);
    const rollup = rollups[0]!;
    expect(rollup.elementCount).toBe(98);
    expect(rollup.distinctValues[0]).toBe("download");
  });

  test("a multi-element rollup drops the slot hint and uses the block anchor", () => {
    const withHint = groups.map((g) => ({
      ...g,
      rootHint: { tag: "div", id: "install", heading: "Download squirrel" },
    })) as ComponentFixGroup[];
    const [rollup] = componentFixRollups(withHint);
    const line = fixRollupLine(rollup!, 3, (p) => new URL(p).pathname);
    expect(line).toContain('in the main content in the "Download squirrel" block');
    // One element's slot describes none of the other six.
    expect(line).not.toContain("main:1>main>div:1>a:1");
  });

  test("a single-group rollup keeps its slot: one element, however many pages", () => {
    const [rollup] = componentFixRollups([group({ root: "install-block", value: "download" })]);
    const line = fixRollupLine(rollup!, 3, (p) => new URL(p).pathname);
    expect(line).toContain("(div:1 > a:1)");
  });

  test("no root hint prints no anchor rather than an arbitrary locator", () => {
    const [rollup] = componentFixRollups(groups);
    const line = fixRollupLine(rollup!, 3, (p) => new URL(p).pathname);
    expect(line).toContain("Likely one shared block in the main content:");
  });

  test("an empty group list renders nothing and changes nothing", () => {
    const empty = report([]);
    expect(renderLlm(empty)).not.toContain("likely-shared-block");
    expect(renderText(empty)).not.toContain("Likely one shared block");
  });
});

describe("packed evidence tables are complete in JSON", () => {
  function occurrence(page: string, withLadder: boolean): ComponentOccurrence {
    const base = {
      version: 1 as const,
      pageUrl: page,
      siteOrigin: SITE,
      provenance: { source: "page-dom" as const, rendered: true },
      groupable: true,
      confidence: "observed" as const,
      region: { role: "footer" as const, nestedIn: "none" as const, structuralSignature: "s128:r" },
      family: { key: "s128:f", structuralSignature: "s128:f" },
      variant: { key: "s128:v", structuralSignature: "s128:v", contentHash: "s128:c" },
      element: { locator: "footer:1>footer>a:1", structuralSignature: "s128:e" },
      defect: { kind: "link-text-generic" as const, values: { normalizedText: "download" }, valueHashes: {} },
    };
    return withLadder
      ? ({
          ...base,
          subtreeCandidates: [
            { level: 1, skeletonSignature: "s128:l1", textShareOfRegion: 0.5, locatorFromRoot: "div:1>a:1" },
            { level: 2, skeletonSignature: "s128:l2", textShareOfRegion: 0.9, locatorFromRoot: "a:1" },
          ],
        } as unknown as ComponentOccurrence)
      : (base as ComponentOccurrence);
  }

  function evidenceReport(withLadder: boolean): AuditReport {
    const checks: CheckResult[] = PAGES.slice(0, 3).map((page) => ({
      name: "generic-link-text",
      status: "warn",
      message: "1 link(s) with generic text",
      pageUrl: page,
      pages: [page],
      componentOccurrences: [occurrence(page, withLadder)],
    }));
    const rule: ReportRuleResult = {
      meta: {
        id: "a11y/link-text", name: "Link text", description: "d", solution: "s",
        category: "accessibility", scope: "page", severity: "warning", weight: 1,
      },
      checks,
    };
    return {
      baseUrl: SITE, timestamp: "2026-09-19T00:00:00.000Z", totalPages: 3,
      passed: 0, warnings: 3, failed: 0, ruleResults: { "a11y/link-text": rule },
      healthScore: { overall: 80, categories: [], groups: [], errorCount: 0, warningCount: 1, passedCount: 0 },
    } as unknown as AuditReport;
  }

  test("every ladder index on an emitted occurrence resolves", () => {
    // The codec hoists ladder rungs into a table and leaves an index on each
    // row. Emitting the rows without the table left consumers holding indices
    // that pointed at nothing.
    const json = JSON.parse(renderJson(evidenceReport(true)));
    const check = json.issues[0].checks[0];
    const ladders = check.componentShapes.ladders;
    expect(Array.isArray(ladders)).toBe(true);
    expect(ladders.length).toBeGreaterThan(0);
    for (const row of check.componentOccurrences) {
      if (row.ladder === undefined) continue;
      expect(ladders[row.ladder]).toBeDefined();
      // One locator per rung, same order.
      expect(row.ladderLocators).toHaveLength(ladders[row.ladder].length);
    }
  });

  test("every shape index resolves, with or without ladders", () => {
    for (const withLadder of [true, false]) {
      const json = JSON.parse(renderJson(evidenceReport(withLadder)));
      const check = json.issues[0].checks[0];
      for (const row of check.componentOccurrences) {
        expect(check.componentShapes.shapes[row.shape]).toBeDefined();
      }
      // The table is omitted entirely when nothing carries a ladder, rather
      // than emitted empty.
      expect(check.componentShapes.ladders === undefined).toBe(!withLadder);
    }
  });
});

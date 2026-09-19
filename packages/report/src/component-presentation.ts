// Turning component fix groups into something that REPLACES output rather than
// adding to it.
//
// The first cut rendered a group block above the legacy `Pages (5/12)` and
// `Items` lines and filled it with `s128:` hashes. That is strictly worse than
// no feature at all: more lines to read, and the added lines carry values no
// reader can act on. A fix group only earns its place when it says the thing
// the legacy lines were approximating — "this is one defect, fix it once" — and
// then takes their place.
//
// Hashes, structural signatures and family/variant keys stay OUT of every
// human and LLM surface. They exist to make grouping decisions reproducible,
// not to be read. JSON keeps them; it is the machine format.

import type { ComponentFixGroup } from "./component-fix-groups";
import type { GroupedRule } from "./grouping";
import { ruleAffectedPages } from "./affected-pages";

/**
 * How much of a rule's affected-page set its fix groups account for.
 *
 * - `none`   — no groups. Renderers must emit byte-identical legacy output.
 * - `full`   — every affected page is attributed to a group, so the legacy
 *              page/item lines are redundant and are suppressed.
 * - `partial`— groups first, then the remainder, explicitly labelled so a
 *              reader never mistakes "not grouped" for "not affected".
 */
export type FixCoverageMode = "none" | "full" | "partial";

export interface FixCoverage {
  mode: FixCoverageMode;
  groups: ComponentFixGroup[];
  /** Affected pages NOT attributed to any group. Empty when `full`. */
  uncoveredPages: string[];
}

export function ruleFixCoverage(rule: GroupedRule): FixCoverage {
  const groups = rule.componentFixGroups ?? [];
  if (groups.length === 0) return { mode: "none", groups: [], uncoveredPages: [] };

  const covered = new Set<string>();
  for (const group of groups) for (const page of group.affectedPages) covered.add(page);

  const affected = ruleAffectedPages(rule.checks);
  const uncoveredPages = [...affected]
    .filter((page) => !covered.has(page))
    .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));

  // A rule whose pages are unknown (site-scope checks with no page list) cannot
  // be said to be fully covered by anything, so it stays partial rather than
  // silently swallowing the legacy lines.
  if (affected.size === 0) return { mode: "partial", groups, uncoveredPages: [] };
  return { mode: uncoveredPages.length === 0 ? "full" : "partial", groups, uncoveredPages };
}

/** Short words for a repair target: "site footer", "navigation in the header". */
export function describeRegion(group: ComponentFixGroup): string {
  const { role, nestedIn } = group.region;
  const slot = regionSlotIndex(group.semanticSlot);
  const base =
    role === "footer"
      ? "site footer"
      : role === "header"
        ? "site header"
        : role === "navigation"
          ? "navigation"
          : role === "main"
            ? "main content"
            : "page";
  const scoped =
    nestedIn === "article"
      ? `${base} inside an article`
      : nestedIn === "main"
        ? `${base} inside the main content`
        : base;
  return slot > 1 ? `${scoped} #${slot}` : scoped;
}

/** `footer:2>footer>ul:1>a:3` -> 2. Absent or unparseable -> 1. */
function regionSlotIndex(semanticSlot: string): number {
  const first = semanticSlot.split(">")[0] ?? "";
  const index = Number(first.split(":")[1]);
  return Number.isFinite(index) && index > 0 ? index : 1;
}

/**
 * The element's position inside its region, as a short trailing hint. The raw
 * locator stays in JSON; here it is a breadcrumb, not an identifier.
 */
export function describeSlot(group: ComponentFixGroup): string {
  const steps = group.semanticSlot.split(">").slice(1);
  const tail = steps.slice(-2);
  return tail.length > 0 ? tail.join(" > ") : group.semanticSlot;
}

function quote(value: unknown): string {
  return `"${String(value)}"`;
}

/** What is wrong, in the words a person fixing it would use. */
export function describeDefect(group: ComponentFixGroup): string {
  const { kind, values } = group.defect;
  if (kind === "stale-copyright") {
    const year = values.year;
    const current = values.currentYear;
    return current === undefined
      ? `copyright year ${year} is out of date`
      : `copyright year ${year} is behind ${current}`;
  }
  if (kind === "link-text-generic") {
    // `normalizedText` is what the rule emits; `text` covers evidence recorded
    // before that key settled, so the line never silently loses the wording.
    const text = values.normalizedText ?? values.text;
    return text === undefined ? "link text is generic" : `link text ${quote(text)} is generic`;
  }
  if (kind === "link-text-empty") {
    const reason = values.reason;
    return reason === undefined
      ? "link has no accessible text"
      : `link has no accessible text (${reason})`;
  }
  return String(kind);
}

export interface PageSample {
  shown: string[];
  total: number;
  hasMore: boolean;
}

/** A bounded, explicitly labelled page sample. `format` compresses each URL. */
export function fixGroupPages(
  group: ComponentFixGroup,
  limit: number,
  format: (page: string) => string,
): PageSample {
  const shown = group.affectedPages.slice(0, limit).map(format);
  return {
    shown,
    total: group.affectedPageCount,
    hasMore: group.affectedPageCount > shown.length,
  };
}

/** "12 pages: /, /api, /guides (+9 more)" — or "1 page: /" when there is one. */
export function formatPageSample(sample: PageSample): string {
  const noun = sample.total === 1 ? "page" : "pages";
  const list = sample.shown.join(", ");
  return sample.hasMore
    ? `${sample.total} ${noun}: ${list} (+${sample.total - sample.shown.length} more)`
    : `${sample.total} ${noun}: ${list}`;
}

/**
 * The single actionable line a fix group exists to produce.
 *
 * "Fix once in the site footer (footer > a:1): link text "website" is generic,
 *  12 pages: /, /api (+10 more)"
 */
export function fixGroupLine(
  group: ComponentFixGroup,
  limit: number,
  format: (page: string) => string,
): string {
  const where = describeRegion(group);
  const slot = describeSlot(group);
  const pages = formatPageSample(fixGroupPages(group, limit, format));
  return `Fix once in the ${where} (${slot}): ${describeDefect(group)}, ${pages}`;
}

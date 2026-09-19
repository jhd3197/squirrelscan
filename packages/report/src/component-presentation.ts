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
import type { ComponentFixRollup } from "./component-fix-rollup";
import { describeRollupDefect } from "./component-fix-rollup";
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

/**
 * Attributions a renderer knows how to phrase. An unknown value renders as
 * nothing at all rather than being narrated with borrowed confidence: a reader
 * on an older build must not be told "fix once" about evidence that did not
 * mean that. Keep in step with `ComponentFixAttribution`.
 */
const RENDERABLE_ATTRIBUTIONS = new Set<string>([
  "observed-component-evidence",
  "repeated-subtree",
]);

/**
 * Whether a group earns a line of its own.
 *
 * A one-page group is not a group. "Fix once, 1 page" says nothing the legacy
 * per-check line did not already say, and real template sites produce them in
 * bulk: one WordPress audit emitted two separate single-page footer groups and
 * a single-page nav group, three lines of pure noise. JSON keeps them, flagged
 * as they are; the human and LLM surfaces do not.
 */
export function isRenderableFixGroup(group: ComponentFixGroup): boolean {
  return group.affectedPageCount >= 2 && RENDERABLE_ATTRIBUTIONS.has(group.attribution);
}

export function ruleFixCoverage(rule: GroupedRule): FixCoverage {
  // Coverage counts only the groups that will actually be RENDERED. Counting a
  // group we then decline to print would suppress the legacy lines for its
  // pages and leave the defect described nowhere at all.
  const groups = (rule.componentFixGroups ?? []).filter(isRenderableFixGroup);
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

/** "12 pages: /, /api, /guides (+9 more)". */
export function formatPageSample(sample: PageSample, noun = "pages"): string {
  const list = sample.shown.join(", ");
  return sample.hasMore
    ? `${sample.total} ${noun}: ${list} (+${sample.total - sample.shown.length} more)`
    : `${sample.total} ${noun}: ${list}`;
}

/**
 * The line for a ROLLUP: one repair, however many faulty elements it contains.
 * A single-member rollup is worded exactly like the group it wraps, so adding
 * rollups changes nothing for reports that have no repeated blocks.
 */
export function fixRollupLine(
  rollup: ComponentFixRollup,
  limit: number,
  format: (page: string) => string,
  /** false when the caller appends its own trailing slot hint (llm does). */
  includeSlot = true,
): string {
  const group = rollup.representative;
  if (rollup.groups.length === 1) return fixGroupLine(group, limit, format, includeSlot);
  const where = includeSlot
    ? `${describeRegion(group)} (${describeSlot(group)})`
    : describeRegion(group);
  const sample = fixGroupPages(group, limit, format);
  const pages = formatPageSample(sample);
  if (group.attribution === "repeated-subtree") {
    return `Likely one shared block in the ${where}: ${describeRollupDefect(rollup)}, seen on ${pages}`;
  }
  return `Fix once in the ${where}: ${describeRollupDefect(rollup)}, ${pages}`;
}

/**
 * The single actionable line a fix group exists to produce.
 *
 * The wording tracks the ATTRIBUTION, because the two say different things.
 * Observed landmark evidence supports an instruction ("fix once here").
 * A repeated subtree supports an observation ("likely one shared block"), and
 * phrasing it as an instruction would overstate what was established.
 */
export function fixGroupLine(
  group: ComponentFixGroup,
  limit: number,
  format: (page: string) => string,
  includeSlot = true,
): string {
  const where = includeSlot
    ? `${describeRegion(group)} (${describeSlot(group)})`
    : describeRegion(group);
  const sample = fixGroupPages(group, limit, format);
  if (group.attribution === "repeated-subtree") {
    return `Likely one shared block in the ${where}: ${describeDefect(group)}, seen on ${formatPageSample(sample)}`;
  }
  return `Fix once in the ${where}: ${describeDefect(group)}, ${formatPageSample(sample)}`;
}

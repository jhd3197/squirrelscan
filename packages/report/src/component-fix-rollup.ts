// One block, one line.
//
// On squirrelscan.com the install block produced SEVEN repeated-subtree groups,
// one per platform button, across 14 pages: seven lines describing a single
// edit to a single component. Grouping got the components right and the output
// still read as seven problems.
//
// A rollup is a presentation unit over groups that are demonstrably the same
// repair: same rule, same check, same attribution, the same component root, the
// same defect kind, and the SAME affected pages. Anything less is left alone,
// because merging groups whose page sets differ would invent a claim about
// where the defect appears.

import type { ComponentFixGroup } from "./component-fix-groups";

/** Distinct defect values to name inside a rolled-up block before counting. */
export const ROLLUP_MAX_DISTINCT_VALUES = 3;

export interface ComponentFixRollup {
  /** The groups this unit stands for, in their original order. */
  groups: ComponentFixGroup[];
  /** Ids of those groups, for the JSON cross-reference. */
  groupIds: string[];
  /** The representative, for region/slot/attribution wording. */
  representative: ComponentFixGroup;
  /** Pages the whole unit affects. Identical to every member's, by construction. */
  affectedPages: string[];
  affectedPageCount: number;
  /** Faulty elements in the block: the sum of the members' occurrences. */
  elementCount: number;
  /** Distinct defect values across the members, in first-seen order. */
  distinctValues: string[];
  /** Distinct values beyond {@link ROLLUP_MAX_DISTINCT_VALUES}. */
  additionalValues: number;
}

/**
 * The component a group belongs to.
 *
 * `componentRootKey` is supplied by the grouping stage; it is read optionally so
 * this module works before that field exists and degrades to "no rollup" rather
 * than guessing a root from the locator, which would merge unrelated blocks.
 */
function componentRootKey(group: ComponentFixGroup): string | undefined {
  return (group as { componentRootKey?: string }).componentRootKey;
}

/** The value a defect is about, as a short string for the rolled-up list. */
function defectValue(group: ComponentFixGroup): string {
  const { values } = group.defect;
  const preferred = values.normalizedText ?? values.text ?? values.reason ?? values.year;
  return preferred === undefined ? group.defect.kind : String(preferred);
}

function pageSetKey(group: ComponentFixGroup): string {
  return [...group.affectedPages].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)).join("\u0000");
}

/**
 * Collapse groups that are one repair into single units, preserving order.
 *
 * Groups without a component root, or alone in their bucket, come back as
 * one-member units so callers can render a single list without special cases.
 */
export function componentFixRollups(groups: readonly ComponentFixGroup[]): ComponentFixRollup[] {
  const buckets = new Map<string, ComponentFixGroup[]>();
  const order: string[] = [];

  groups.forEach((group, index) => {
    const root = componentRootKey(group);
    // No root key, no claim that two groups are the same component. Each such
    // group gets a bucket of its own, keyed by position so it cannot merge.
    const key =
      root === undefined
        ? `solo\u0000${index}`
        : [
            group.ruleId,
            group.check.name,
            group.check.status,
            group.attribution,
            root,
            group.defect.kind,
            pageSetKey(group),
          ].join("\u0001");
    const bucket = buckets.get(key);
    if (bucket) bucket.push(group);
    else {
      buckets.set(key, [group]);
      order.push(key);
    }
  });

  return order.map((key) => {
    const members = buckets.get(key)!;
    const representative = members[0]!;
    const seen = new Set<string>();
    const distinct: string[] = [];
    for (const member of members) {
      const value = defectValue(member);
      if (seen.has(value)) continue;
      seen.add(value);
      distinct.push(value);
    }
    return {
      groups: members,
      groupIds: members.map((member) => member.id),
      representative,
      affectedPages: representative.affectedPages,
      affectedPageCount: representative.affectedPageCount,
      elementCount: members.reduce((sum, member) => sum + member.occurrences.length, 0),
      distinctValues: distinct.slice(0, ROLLUP_MAX_DISTINCT_VALUES),
      additionalValues: Math.max(0, distinct.length - ROLLUP_MAX_DISTINCT_VALUES),
    };
  });
}

/** "7 links with generic text "download", "docs" (+2 more)". */
export function describeRollupDefect(rollup: ComponentFixRollup): string {
  const { representative, elementCount, distinctValues, additionalValues } = rollup;
  const noun =
    representative.defect.kind === "stale-copyright"
      ? elementCount === 1
        ? "copyright notice"
        : "copyright notices"
      : elementCount === 1
        ? "link"
        : "links";
  const what =
    representative.defect.kind === "stale-copyright"
      ? "behind the current year"
      : representative.defect.kind === "link-text-empty"
        ? "with no accessible text"
        : "with generic text";
  const values = distinctValues.map((value) => `"${value}"`).join(", ");
  const tail = additionalValues > 0 ? `${values} (+${additionalValues} more)` : values;
  // A single element reads as the plain defect; the count only helps in bulk.
  return values.length === 0 ? `${elementCount} ${noun} ${what}` : `${elementCount} ${noun} ${what} ${tail}`;
}

// Conservative, evidence-backed actionable groups for component-aware findings.
// These are additive report views: `occurrences` stays the immutable observed
// evidence; a group only says the supplied component and defect identities agree.

import type { ComponentOccurrence } from "@squirrelscan/core-contracts";
import type { CheckResult } from "./types";

export type ComponentFixAttribution = "observed-component-evidence" | "repeated-subtree";

export interface ComponentFixGroup {
  /**
   * `component-fix:<32 hex>` — a 128-bit digest of the canonical identity below,
   * disambiguated with a `-N` suffix if two distinct identities ever collide.
   * The full identity is NOT the id: it runs to ~800 bytes of hashes, and this
   * value rides on every group in every serialized report.
   */
  id: string;
  ruleId: string;
  check: { name: string; status: CheckResult["status"] };
  /** This is derived from observations and never replaces the raw checks. */
  /**
   * How the grouping was established. Renderers treat these differently and an
   * UNKNOWN value conservatively (legacy rendering), so a future producer can
   * add one without a stale reader overstating its confidence.
   *
   * - `observed-component-evidence` — the elements share an observed landmark
   *   region identity. Strong enough to say "fix once here".
   * - `repeated-subtree` — a block recurring in page content, or on a theme
   *   with no landmarks. Weaker: likely one shared block, not established.
   */
  attribution: ComponentFixAttribution;
  affectedPages: string[];
  affectedPageCount: number;
  /** Safe repair-target context from the representative observed element. */
  region: ComponentOccurrence["region"];
  element: ComponentOccurrence["element"];
  /** The relative element locator is the semantic repair slot in this slice. */
  semanticSlot: string;
  defect: ComponentOccurrence["defect"];
  /**
   * Complete, traceable occurrence evidence. Serialized reports reference these
   * by position in the owning check rather than repeating them (see the JSON
   * renderer); renderers for humans sample pages only.
   */
  occurrences: ComponentOccurrence[];
}

/** Codepoint order. `localeCompare` is ICU/locale dependent and would make the
 * serialized order of a report depend on the machine that rendered it. */
function compare(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => compare(a, b))
      .map(([key, entry]) => `${JSON.stringify(key)}:${stable(entry)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

const FNV_128_OFFSET = 0x6c62272e07bb014262b821756295c58dn;
const FNV_128_PRIME = 0x0000000001000000000000000000013bn;
const FNV_128_MASK = (1n << 128n) - 1n;
const utf8 = new TextEncoder();

/**
 * FNV-1a, 128-bit, over the UTF-8 bytes of `value`. Pure JS on purpose: this
 * module is imported by browser report consumers, which have no `node:crypto`.
 * 128 bits makes an accidental collision negligible, and `componentFixGroups`
 * still checks for one rather than trusting the arithmetic.
 */
export function componentFixGroupDigest(value: string): string {
  let hash = FNV_128_OFFSET;
  for (const byte of utf8.encode(value)) {
    hash = ((hash ^ BigInt(byte)) * FNV_128_PRIME) & FNV_128_MASK;
  }
  return hash.toString(16).padStart(32, "0");
}

/**
 * Identity intentionally includes every non-page component and defect datum.
 * A message is deliberately absent: matching prose is not component evidence.
 * Rules make uncertain evidence page-specific before it arrives here.
 */
export function componentOccurrenceIdentity(occurrence: ComponentOccurrence): string {
  return stable({
    version: occurrence.version,
    region: occurrence.region,
    family: occurrence.family,
    // STRUCTURE only. `variant.contentHash` is the region's rendered text: it
    // is reported as evidence, but including it here would split one component
    // on any per-page string in the region (a date, a counter, a breadcrumb),
    // which is exactly the merge this feature exists to make.
    variant: { key: occurrence.variant.key, structuralSignature: occurrence.variant.structuralSignature },
    element: occurrence.element,
    defect: occurrence.defect,
  });
}

/**
 * One observation's full identity: the component it belongs to, the page it was
 * seen on, and how it was seen. Two genuinely distinct elements differ in their
 * locator or structural signature and so keep distinct keys; only a re-merge of
 * the very same observation collapses.
 */
export function componentOccurrenceKey(occurrence: ComponentOccurrence): string {
  return stable({
    component: componentOccurrenceIdentity(occurrence),
    pageUrl: occurrence.pageUrl,
    provenance: occurrence.provenance,
  });
}

/** A subtree must recur on at least this many pages to be called a component. */
export const REPEATED_SUBTREE_MIN_PAGES = 3;
/**
 * Above this share of the region's text the candidate IS the page's content.
 * A blog index's article list recurs on every index page and covers all of
 * `main`; promoting it would claim one template fix for text authored per post.
 */
export const REPEATED_SUBTREE_MAX_TEXT_SHARE = 0.5;

/**
 * What must match for two occurrences to be the same defect in the same slot.
 *
 * Deliberately NOT the destination hash. On the site this was built for, the
 * same download button carries a different href on every release page (107
 * distinct destinations across 114 occurrences), so including it yields zero
 * groups. Per-page data is exactly what a shared block is expected to vary;
 * what must match is the slot (`locatorFromRoot`) and the defect itself.
 */
function promotionDefectIdentity(occurrence: ComponentOccurrence): string {
  return stable({ kind: occurrence.defect.kind, values: occurrence.defect.values });
}

type Promotable = {
  occurrence: ComponentOccurrence;
  check: ComponentFixGroup["check"];
  scope: string;
};

/**
 * Promote occurrences the rules could not judge, using the ancestor ladders
 * they recorded.
 *
 * A rule sees one page and so cannot know whether a block repeats; it emits the
 * ladder and stays uncertain. Here, with every page in hand, the highest
 * ancestor level whose skeleton recurs on enough pages becomes the component.
 * The occurrences themselves are never rewritten: they are still `uncertain`
 * and `groupable: false`, and the group carries the weaker attribution.
 */
function promoteRepeatedSubtrees(
  ruleId: string,
  promotable: Promotable[],
): Map<string, { occurrences: ComponentOccurrence[]; check: ComponentFixGroup["check"] }> {
  const byScope = new Map<string, Promotable[]>();
  for (const entry of promotable) {
    const bucket = byScope.get(entry.scope) ?? [];
    bucket.push(entry);
    byScope.set(entry.scope, bucket);
  }

  const promoted = new Map<
    string,
    { occurrences: ComponentOccurrence[]; check: ComponentFixGroup["check"] }
  >();
  for (const [scope, entries] of byScope) {
    // Pages per (level, skeleton), counting only occurrences that pass the
    // text-share guard: a page where the block is most of the content simply
    // does not join, and cannot drag the level up on its own.
    const pagesByRung = new Map<string, Set<string>>();
    for (const { occurrence } of entries) {
      for (const candidate of occurrence.subtreeCandidates ?? []) {
        if (candidate.textShareOfRegion > REPEATED_SUBTREE_MAX_TEXT_SHARE) continue;
        const rung = `${candidate.level}\u0000${candidate.skeletonSignature}`;
        const pages = pagesByRung.get(rung) ?? new Set<string>();
        pages.add(occurrence.pageUrl);
        pagesByRung.set(rung, pages);
      }
    }
    const eligible = [...pagesByRung.entries()]
      .filter(([, pages]) => pages.size >= REPEATED_SUBTREE_MIN_PAGES)
      .map(([rung]) => {
        const separator = rung.indexOf("\u0000");
        return { level: Number(rung.slice(0, separator)), skeleton: rung.slice(separator + 1) };
      });
    if (eligible.length === 0) continue;
    // Highest level wins: it is the largest subtree that still recurs, so the
    // repair target is the whole shared block rather than one row inside it.
    // Ties resolve by signature codepoint order, never by input order.
    eligible.sort((a, b) => b.level - a.level || compare(a.skeleton, b.skeleton));
    const chosen = eligible[0]!;

    for (const { occurrence, check } of entries) {
      const candidate = (occurrence.subtreeCandidates ?? []).find(
        (item) =>
          item.level === chosen.level &&
          item.skeletonSignature === chosen.skeleton &&
          item.textShareOfRegion <= REPEATED_SUBTREE_MAX_TEXT_SHARE,
      );
      if (!candidate) continue;
      // The slot inside the block is part of the identity, so seven different
      // buttons in one block stay seven repair targets rather than one.
      const identity = stable({
        scope,
        repeatedSubtree: {
          level: candidate.level,
          skeletonSignature: candidate.skeletonSignature,
          locatorFromRoot: candidate.locatorFromRoot,
        },
      });
      const entry = promoted.get(identity) ?? { occurrences: [], check };
      entry.occurrences.push(occurrence);
      promoted.set(identity, entry);
    }
  }
  // A promoted group that ended up on fewer pages than the floor (because the
  // guard removed occurrences after the level was chosen) is not a component.
  for (const [identity, entry] of [...promoted]) {
    const pages = new Set(entry.occurrences.map((occurrence) => occurrence.pageUrl));
    if (pages.size < REPEATED_SUBTREE_MIN_PAGES) promoted.delete(identity);
  }
  void ruleId;
  return promoted;
}

/**
 * Derive deterministic, evidence-backed fix groups from the complete selected
 * check evidence. Checks without this additive field retain legacy rendering.
 */
export function componentFixGroups(ruleId: string, checks: CheckResult[]): ComponentFixGroup[] {
  const groups = new Map<
    string,
    { occurrences: ComponentOccurrence[]; check: ComponentFixGroup["check"] }
  >();
  const promotable: Promotable[] = [];
  for (const check of checks) {
    for (const occurrence of check.componentOccurrences ?? []) {
      // Unknown, article-nested and otherwise uncertain evidence stays in the
      // legacy page-scoped display. It is never promoted to an actionable
      // cross-page component claim merely because its message happens to match.
      if (!occurrence.groupable || occurrence.confidence !== "observed") {
        // …except a main-region occurrence carrying an ancestor ladder, which
        // the pass below may still promote on cross-page evidence.
        if (
          occurrence.uncertainReason === "region-main" &&
          (occurrence.subtreeCandidates?.length ?? 0) > 0
        )
          promotable.push({
            occurrence,
            check: { name: check.name, status: check.status },
            scope: stable({
              ruleId,
              origin: occurrence.siteOrigin,
              check: { name: check.name, status: check.status },
              defect: promotionDefectIdentity(occurrence),
            }),
          });
        continue;
      }
      // Read the site the rule recorded rather than re-deriving one here: a
      // second derivation could only disagree with the one that already scoped
      // `family.key`. Keeping it an explicit term means a group cannot span two
      // origins even if the family keys it was handed are indistinct.
      const identity = stable({
        ruleId,
        origin: occurrence.siteOrigin,
        check: { name: check.name, status: check.status },
        component: componentOccurrenceIdentity(occurrence),
      });
      const entry = groups.get(identity) ?? {
        occurrences: [],
        check: { name: check.name, status: check.status },
      };
      // Do not deduplicate observations: the occurrence list is evidence, not
      // a display set, and two observed elements can have identical locators.
      entry.occurrences.push(occurrence);
      groups.set(identity, entry);
    }
  }

  // Cross-page promotion runs after every page's evidence is in hand. Its
  // identities are disjoint from the observed ones (they are keyed on the
  // subtree ladder, not on family/variant), so they share the id space safely.
  const promotedIdentities = new Set<string>();
  for (const [identity, entry] of promoteRepeatedSubtrees(ruleId, promotable)) {
    promotedIdentities.add(identity);
    groups.set(identity, entry);
  }

  // Assign ids over the identities in codepoint order, so a digest collision
  // between two DISTINCT identities resolves the same way on every run and on
  // every machine rather than by map insertion order.
  const identities = [...groups.keys()].sort(compare);
  const idByIdentity = new Map<string, string>();
  const seenDigests = new Map<string, number>();
  for (const identity of identities) {
    const digest = componentFixGroupDigest(identity);
    const collisions = seenDigests.get(digest) ?? 0;
    seenDigests.set(digest, collisions + 1);
    idByIdentity.set(
      identity,
      collisions === 0 ? `component-fix:${digest}` : `component-fix:${digest}-${collisions}`,
    );
  }

  return identities
    .map((identity) => {
      const entry = groups.get(identity)!;
      // Decorate-sort-undecorate: the key is a canonical serialization of the
      // whole observation, so computing it inside the comparator would rebuild
      // it O(n log n) times. A 1000-occurrence check made that the single
      // hottest path in the rule run.
      const occurrences = entry.occurrences
        .map((occurrence) => ({ occurrence, key: componentOccurrenceKey(occurrence) }))
        .sort((a, b) => compare(a.key, b.key))
        .map((entry) => entry.occurrence);
      const affectedPages = [
        ...new Set(occurrences.map((occurrence) => occurrence.pageUrl)),
      ].sort(compare);
      const representative = occurrences[0]!;
      return {
        id: idByIdentity.get(identity)!,
        ruleId,
        check: entry.check,
        attribution: (promotedIdentities.has(identity)
          ? "repeated-subtree"
          : "observed-component-evidence") as ComponentFixAttribution,
        affectedPages,
        affectedPageCount: affectedPages.length,
        region: representative.region,
        element: representative.element,
        semanticSlot: representative.element.locator,
        defect: representative.defect,
        occurrences,
      };
    })
    .sort((a, b) => compare(a.id, b.id));
}

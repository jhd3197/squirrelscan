/**
 * Which part of a page is each link in? One call, no network (#2307).
 *
 * `placePage` parses a page, runs the runtime (`classifyDocument`, so the
 * comments rule and the heads) and places each link with `elementPlacer`, in
 * DG's per-link row shape (`a16Row`):
 *   {i, href, section, region, rel, conf, evidence, source}
 *   section: the comments rule's `comments`/`reviews` when it fires, else the
 *            section head's label, else the link's region label, else `unknown`;
 *   region:  the region label on its own (`unknown` when nothing answered);
 *   rel:     ugc / nofollow / sponsored on the link;
 *   conf:    1.0 for the rule, else the answering head's confidence;
 *   evidence: the rule's matched id/class token (digits folded), never page text;
 *   source:  `rule` (the comments rule, or a region read from markup) | `model` | null.
 * Without a model the rules path answers and `degraded` says so.
 *
 * Page content is hostile: it is parsed, never executed or fetched.
 */
import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import type { Model } from "../runtime/model-format.ts";
import { classifyDocument } from "../runtime/predict.ts";
import { linkPositions, type LinkPosition, type ManifestLink } from "./link-position.ts";

export interface LinkRow {
  i: number;
  href: string;
  section: string;
  region: string;
  rel: string[];
  conf: number | null;
  evidence: string | null;
  source: string | null;
}

/** DG's A16 link row: the comments rule, else the section, else the region. */
export function a16Row(p: LinkPosition): LinkRow {
  let section = "unknown";
  let conf: number | null = null;
  let evidence: string | null = null;
  let source: string | null = null;
  if (p.commentsRule) {
    [section, conf, evidence, source] = [p.commentsRule, 1, p.commentsRuleToken, "rule"];
  } else if (p.section) {
    [section, conf, source] = [p.section, p.sectionConfidence, p.sectionSource ?? "model"];
  } else if (p.region !== "unknown") {
    [section, conf, source] = [p.region, p.regionConfidence, p.regionSource];
  }
  return {
    i: p.i,
    href: p.href,
    section,
    region: p.region,
    rel: p.rel,
    conf: conf === null ? null : Math.round(conf * 10_000) / 10_000,
    evidence,
    // Markup answers (the comments rule, a <footer>) are deterministic.
    source: source === "rules" ? "rule" : source,
  };
}

export interface PlacedLink extends LinkRow {
  /** False when the link was not found on the page. */
  located: boolean;
  /** False when the anchor sits outside the walk (a second <body>, or after </body>). */
  inWalk: boolean;
}

export interface PlacedPage {
  pageType: { label: string; confidence: number; source: string };
  commentSections: Array<{ label: string; evidence: string }>;
  degraded: string;
  links: PlacedLink[];
}

export interface PlaceOptions {
  model?: Model | null;
  budgetMs?: number;
}

/** Place `links` on a page (or every outbound link when `links` is null/undefined). */
export function placePage(
  html: string,
  url: string,
  links: readonly ManifestLink[] | null | undefined,
  options: PlaceOptions = {},
): PlacedPage {
  const { document } = parseHTML(html);
  const elements: unknown[] = [];
  const result = classifyDocument(document, url, {
    model: options.model ?? null,
    budgetMs: options.budgetMs ?? 500,
    html,
    elementsOut: elements,
  });
  const baseHref = (
    document as unknown as {
      querySelector(s: string): { getAttribute(n: string): string | null } | null;
    }
  )
    .querySelector("base[href]")
    ?.getAttribute("href");
  const positions = linkPositions(url, result, elements, links, baseHref);
  return {
    pageType: {
      label: result.pageType.label,
      confidence: Math.round(result.pageType.confidence * 10_000) / 10_000,
      source: result.pageType.source,
    },
    commentSections: result.commentSections.map((c) => ({ label: c.label, evidence: c.evidence })),
    degraded: result.degraded,
    links: positions.map((p) => ({ ...a16Row(p), located: p.matched, inWalk: p.inWalk })),
  };
}

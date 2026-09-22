/**
 * Prompt revision `dom-suggestions-v5`: the shared-definition rebuild of the
 * Jev request, kept in a separate module so `jev-adapter.ts` (revision v4)
 * stays byte-identical and every v4 output remains reproducible.
 *
 * Three things change, each because v4 measured a failure:
 *
 * 1. **Shared definitions.** `v2/labels/policy-v5.md` is read at run time,
 *    hashed, and placed in request state. The Luna reviewer prompt embeds the
 *    same bytes. Under v4 neither annotator had definitions at all, and 33 of
 *    34 `layout_container`/`content_section` disagreements were `<footer>` and
 *    `<main>` alone.
 * 2. **One page type, not sixty-five Nouls.** v4's multi-label page-type
 *    battery agreed 1.3% of the time on the test split. v5 asks one Choice over
 *    the eleven collapsed classes in `v2/labels/page-type-map-v5.json`.
 * 3. **Purposes with support only.** Fifteen of twenty-seven purposes were used
 *    fewer than ten times across 375 pool nodes; `v2/labels/purposes-v5.json`
 *    records which were kept and what they cost.
 *
 * Node evidence is built from an immutable capture and carries no hostname, no
 * class or id token, and no attribute the capture did not already hold.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  COMPONENT_TYPES,
  REGIONS,
  TAXONOMY_REVISION,
  type CapturedNode,
  type CapturedPage,
} from "../labeler/types.ts";
import { stripStyleResidue, structureSummary } from "../extract-dom.ts";
import { parseHTML } from "../../../packages/parser/src/dom.ts";
import type { JevQuestion, JevRequest, JevResponse, TypeSafeTransport } from "./jev-adapter.ts";
import { validateResponse } from "./jev-adapter.ts";
import pageTypeMap from "../v2/labels/page-type-map-v5.json" with { type: "json" };
import purposeMap from "../v2/labels/purposes-v5.json" with { type: "json" };

export const JEV_MODEL_V5 = "jev-1.13.0";
// `as const` keeps the literal type through an object property, so a row whose
// promptRevision drifts from this module is a type error rather than a string.
export const PROMPT_REVISION_V5 = "dom-suggestions-v5" as const;
/** Revisions this module can run. Each has its own policy file; none overwrites another. */
export const POLICY_REVISIONS = ["v5", "v6", "v7"] as const;
export type PolicyRevision = (typeof POLICY_REVISIONS)[number];
export type PromptRevision = `dom-suggestions-${PolicyRevision}`;
export const PACKET_REVISION_V5 = "runtime-dom-text-v2";
/** Packets built from full HTML, which carry attributes a capture cannot hold. */
export const HTML_PACKET_REVISION_V5 = "runtime-dom-html-v1";
export const PAGE_PACKET_REVISION_V5 = "dom-page-text-v2";
export const POLICY_PATH = join(import.meta.dir, "..", "v2", "labels", "policy-v5.md");
export function policyPath(revision: PolicyRevision): string {
  return join(import.meta.dir, "..", "v2", "labels", `policy-${revision}.md`);
}

export const PAGE_TYPES_V5 = pageTypeMap.classes as readonly string[];
export const PAGE_TYPE_COLLAPSE = pageTypeMap.map as Record<string, string>;
export const PURPOSES_V5 = Object.keys(purposeMap.retained).sort() as readonly string[];
export const REGIONS_V5 = REGIONS as readonly string[];
export const COMPONENT_TYPES_V5 = COMPONENT_TYPES as readonly string[];

const MAX_TEXT = 320;
const MAX_TITLE = 240;
const MAX_PAGE_EVIDENCE = 12;
const EMPTY_TEXT = "[empty]";
/** Tags whose text is never page prose; mirrors the extractor's text walk. */
const PAGE_TEXT_SKIP: ReadonlySet<string> = new Set([
  "script",
  "style",
  "template",
  "noscript",
  "svg",
  "canvas",
  "iframe",
  "object",
  "embed",
  "head",
  "title",
  "link",
  "meta",
  "base",
  "math",
  "map",
  "area",
  "track",
  "source",
  "param",
]);

const tokenPattern = /(?:bearer\s+|api[_-]?key\s*[=:]\s*|password\s*[=:]\s*)[^\s,;]+/gi;
const emailPattern = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi;

export function sha256(value: string): string {
  return `sha256:${createHash("sha256").update(value).digest("hex")}`;
}

/**
 * Redact, then scrub stylesheet residue with the current extractor, then bound.
 *
 * Order matters. `stripStyleResidue` reads brace structure, so it has to see the
 * text before whitespace collapsing rewrites it, and the capture's own 280
 * character truncation means a trailing declaration often arrives with its
 * closing brace already gone. The scrubber handles that tail; it cannot recover
 * a selector prelude that was cut mid-token, and those rows are reported as
 * residual rather than claimed clean.
 */
export function cleanText(value: string, maximum = MAX_TEXT): string {
  const redacted = value
    .replaceAll(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, "�")
    .replace(tokenPattern, "[REDACTED]")
    .replace(emailPattern, "[EMAIL]");
  return stripStyleResidue(redacted)
    .replaceAll(/\s+/g, " ")
    .trim()
    .slice(0, maximum)
    .replace(/[\uD800-\uDBFF]$/, "�");
}

export type NodeEvidence = {
  nodeId: string;
  tag: string;
  role: string | null;
  ancestors: string[];
  siblingPosition: number;
  siblingCount: number;
  childTags: string[];
  /** `div*4,a*3,img` over the captured subtree; the capture is not a full DOM. */
  subtree: string;
  width: number;
  height: number;
  depth: number;
  text: string;
  emptyText: boolean;
  /**
   * True only when the source was full HTML. A legacy capture holds no element
   * attributes, so the fields below are absent there and are never guessed. The
   * flag travels on the packet because a labelling rule that depends on `alt`
   * is unanswerable without it, which is exactly how the v5 pilot turned 25
   * usable `image` labels into abstentions.
   */
  attributesAvailable: boolean;
  /** `alt`, `title`, `aria-label`, `role`, `type`, `name`, `placeholder`. */
  attributes?: Record<string, string>;
  /** `same-site`, a provider slug such as `youtube`, or `other-third-party`. */
  srcCategory?: string | null;
  /** Media inside the node, so an empty wrapper still describes its content. */
  mediaDescendants?: Array<{
    tag: string;
    attributes: Record<string, string>;
    srcCategory: string | null;
  }>;
  linkCount?: number;
  /** `same_site`/`external`/`anchor` counts over direct child links. */
  linkCategories?: Record<string, number>;
  /** How many siblings share this node's tag-and-children shape; the `card` signal. */
  repeatedSiblingPatternCount?: number;
  /** Curated, secret-filtered class and id tokens. See the caution in the docs. */
  classTokens?: string[];
  idTokens?: string[];
};

function finite(value: number | undefined): number {
  return Number.isFinite(value) ? Math.round(value as number) : 0;
}

/**
 * Node evidence from an immutable capture.
 *
 * The frozen captures hold no HTML and no element attributes, so the extractor's
 * `semanticAttributes` and `srcCategory` cannot be recovered here and are not
 * invented: a packet built from a capture records `attributesAvailable: false`.
 * What the capture does hold is the node graph, so an element that reached v4 as
 * the bare token `[empty]` now carries its role, its position among siblings,
 * its child tags and its subtree shape.
 */
export function nodeEvidence(page: CapturedPage, node: CapturedNode): NodeEvidence {
  const byId = new Map(page.nodes.map((item) => [item.id, item]));
  const chain: string[] = [];
  for (let current = node.parentId ? byId.get(node.parentId) : undefined; current; ) {
    chain.push(current.tag.toLowerCase());
    current = current.parentId ? byId.get(current.parentId) : undefined;
    if (chain.length >= 8) break;
  }
  const siblings = page.nodes.filter((item) => item.parentId === node.parentId);
  const children = page.nodes.filter((item) => item.parentId === node.id);
  const descendants: CapturedNode[] = [];
  const queue = [...children];
  while (queue.length && descendants.length < 400) {
    const item = queue.shift()!;
    descendants.push(item);
    queue.push(...page.nodes.filter((child) => child.parentId === item.id));
  }
  const counts: Record<string, number> = {};
  for (const item of descendants) {
    const tag = item.tag.toLowerCase();
    counts[tag] = (counts[tag] ?? 0) + 1;
  }
  const text = cleanText(node.text ?? "");
  return {
    nodeId: node.id,
    tag: node.tag.toLowerCase(),
    role: node.role ? node.role.toLowerCase() : null,
    ancestors: chain.reverse(),
    siblingPosition: Math.max(
      0,
      siblings.findIndex((item) => item.id === node.id),
    ),
    siblingCount: siblings.length,
    childTags: children.slice(0, 12).map((item) => item.tag.toLowerCase()),
    subtree: structureSummary(counts, node.tag.toLowerCase()),
    width: finite(node.rect?.width),
    height: finite(node.rect?.height),
    depth: finite(node.depth),
    text,
    emptyText: text.length === 0,
    attributesAvailable: false,
  };
}

/** One `extractPage` candidate, as far as this module reads it. */
export type ExtractedCandidate = {
  tag: string;
  roles?: string[];
  ancestorTags?: string[];
  siblingPosition?: number;
  siblingCount?: number;
  childTags?: string[];
  structureSummary?: string;
  text?: string;
  emptyText?: boolean;
  semanticAttributes?: Record<string, string>;
  srcCategory?: string | null;
  mediaDescendants?: Array<{
    tag: string;
    attributes: Record<string, string>;
    srcCategory: string | null;
  }>;
  linkCount?: number;
  linkCategories?: Record<string, number>;
  repeatedSiblingPatternCount?: number;
  classTokens?: string[];
  idTokens?: string[];
  locator?: string;
};

/**
 * Node evidence from **full HTML**, via the current `extract-dom.ts`.
 *
 * The next labelling corpus is featurized from full HTML rather than from the
 * frozen captures, so `alt`, `aria-label`, `role`, the `src` category and the
 * repeated-sibling count are all available. That is what makes the media and
 * card rules answerable: a product photo, a logo and a tracking pixel stop
 * being the same input.
 *
 * No hostname is carried. `srcCategory` reduces a URL to `same-site`, a known
 * provider slug or `other-third-party`, and class and id tokens arrive already
 * filtered by the extractor's secret-word denylist.
 *
 * Geometry is absent by construction: HTML has no layout, so `size` is omitted
 * rather than faked. Packets from the two sources are therefore not
 * byte-comparable, which is why they carry different revision strings.
 */
export function nodeEvidenceFromHtml(
  candidate: ExtractedCandidate,
  nodeId: string,
  options: { includeClassTokens?: boolean } = {},
): NodeEvidence {
  const text = cleanText(candidate.text ?? "");
  const includeClassTokens = options.includeClassTokens ?? true;
  return {
    nodeId,
    tag: candidate.tag.toLowerCase(),
    role: candidate.roles?.[0]?.toLowerCase() ?? candidate.semanticAttributes?.role ?? null,
    ancestors: (candidate.ancestorTags ?? []).map((tag) => tag.toLowerCase()),
    siblingPosition: Math.max(0, candidate.siblingPosition ?? 0),
    siblingCount: Math.max(1, candidate.siblingCount ?? 1),
    childTags: (candidate.childTags ?? []).slice(0, 12).map((tag) => tag.toLowerCase()),
    subtree: candidate.structureSummary ?? "",
    width: 0,
    height: 0,
    depth: (candidate.ancestorTags ?? []).length,
    text,
    emptyText: text.length === 0,
    attributesAvailable: true,
    attributes: candidate.semanticAttributes ?? {},
    srcCategory: candidate.srcCategory ?? null,
    mediaDescendants: (candidate.mediaDescendants ?? []).slice(0, 6),
    linkCount: candidate.linkCount ?? 0,
    linkCategories: candidate.linkCategories ?? {},
    repeatedSiblingPatternCount: candidate.repeatedSiblingPatternCount ?? 1,
    ...(includeClassTokens
      ? { classTokens: candidate.classTokens ?? [], idTokens: candidate.idTokens ?? [] }
      : {}),
  };
}

function attributePairs(attributes: Record<string, string> | undefined): string {
  const entries = Object.entries(attributes ?? {});
  return entries.length
    ? entries.map(([name, value]) => `${name}="${value.replaceAll('"', "'")}"`).join(" ")
    : "none";
}

/**
 * The one evidence rendering both annotators read. Jev gets the same fields as
 * state. A capture-sourced packet stops after `text=`; an HTML-sourced packet
 * adds the attribute lines and declares a different revision, so a reader can
 * never mistake a missing `alt` for an absent one.
 */
export function serializeNodePacket(evidence: NodeEvidence): string {
  const lines = [
    evidence.attributesAvailable ? HTML_PACKET_REVISION_V5 : PACKET_REVISION_V5,
    `tag=${evidence.tag}`,
    `role=${evidence.role ?? "none"}`,
    `ancestors=${evidence.ancestors.join(">") || "none"}`,
    `position=${evidence.siblingPosition + 1} of ${Math.max(1, evidence.siblingCount)}`,
  ];
  if (!evidence.attributesAvailable) lines.push(`size=${evidence.width}x${evidence.height}`);
  lines.push(
    `children=${evidence.childTags.join(",") || "none"}`,
    `subtree=${evidence.subtree || "none"}`,
  );
  if (evidence.attributesAvailable) {
    lines.push(
      `attributes=${attributePairs(evidence.attributes)}`,
      `src=${evidence.srcCategory ?? "none"}`,
      `media=${
        (evidence.mediaDescendants ?? [])
          .map(
            (item) =>
              `${item.tag}[${attributePairs(item.attributes)}|${item.srcCategory ?? "none"}]`,
          )
          .join(" ") || "none"
      }`,
      `links=${evidence.linkCount ?? 0} ${JSON.stringify(evidence.linkCategories ?? {})}`,
      `repeatedSiblings=${evidence.repeatedSiblingPatternCount ?? 1}`,
    );
    if (evidence.classTokens) lines.push(`classTokens=${evidence.classTokens.join(",") || "none"}`);
    if (evidence.idTokens) lines.push(`idTokens=${evidence.idTokens.join(",") || "none"}`);
  }
  lines.push(`text=${evidence.emptyText ? EMPTY_TEXT : evidence.text}`);
  return lines.join("\n");
}

export type PageEvidence = {
  pageId: string;
  title: string;
  path: string;
  lines: string[];
};

/** URL path only: the host identifies the site and never leaves this step. */
export function safePath(raw: string): string {
  try {
    const parsed = new URL(raw);
    return cleanText(parsed.pathname || "/", 200);
  } catch {
    return "/";
  }
}

/** Tags whose text is prose a reader sees, in preference order. */
const PROSE_TAGS = [
  "h1",
  "h2",
  "h3",
  "p",
  "li",
  "article",
  "main",
  "section",
  "figcaption",
  "blockquote",
  "td",
  "dd",
  "header",
  "footer",
  "nav",
  "a",
];

/**
 * Script and structured-data residue that the CSS scrubber is not built to see.
 *
 * `stripStyleResidue` discriminates CSS from JSON deliberately, because docs
 * pages show JSON as visible content. A page serialization spans the whole
 * document, so it also picks up JSON-LD blocks and minified bundles that no
 * reader sees; 9.3% of v4 page inputs carried this. Dropping a whole candidate
 * line is safe here in a way that editing node text is not: another line takes
 * its place.
 */
export function looksLikeCode(value: string): boolean {
  const text = value.trim();
  if (!text) return true;
  if (/^[({[]|^!function|^var\s|^function\s*\(|^\s*<\w/.test(text)) return true;
  if (/"@context"|"@type"|window\.|document\.|=>\s*\{|\bfunction\s*\(/.test(text)) return true;
  const punctuation = (text.match(/[{}();=<>]/g) ?? []).length;
  return punctuation / text.length > 0.06;
}

export function pageEvidence(page: CapturedPage): PageEvidence {
  const rank = (tag: string) => {
    const index = PROSE_TAGS.indexOf(tag.toLowerCase());
    return index < 0 ? PROSE_TAGS.length : index;
  };
  const lines = page.nodes
    .map((node) => ({ rank: rank(node.tag), text: cleanText(node.text ?? "") }))
    .filter((item) => item.text.length >= 40 && !looksLikeCode(item.text))
    .sort((left, right) => left.rank - right.rank)
    .map((item) => item.text)
    .filter((text, index, all) => all.indexOf(text) === index)
    .slice(0, MAX_PAGE_EVIDENCE);
  return {
    pageId: page.id,
    title: cleanText(page.title ?? "", MAX_TITLE),
    path: safePath(page.url),
    lines,
  };
}

export function serializePagePacket(evidence: PageEvidence): string {
  return [
    PAGE_PACKET_REVISION_V5,
    `title=${evidence.title}`,
    `path=${evidence.path}`,
    `text=${evidence.lines.join("\n")}`,
  ].join("\n");
}

/**
 * Page evidence from full HTML, for labelling a page type at corpus scale.
 *
 * The same shape and the same bounds as the capture path, so a page packet
 * reads identically whichever source it came from. What a page type needs is
 * the title, the URL path and some body prose; the host is dropped by
 * `safePath` and script, JSON-LD and minified bundles by `looksLikeCode`.
 *
 * Schema.org markup is deliberately NOT read here. The silver page labels are
 * derived from JSON-LD, og:type and microdata, so a model that saw those would
 * agree with them by construction and could say nothing about the pages that
 * carry no markup at all, which are 57% of the holdout split and the whole
 * reason this labelling exists.
 */
export function pageEvidenceFromHtml(html: string, pageUrl: string, pageId: string): PageEvidence {
  // A single linear walk collecting each element's OWN text. Going through
  // `extractPage` instead would shape every candidate, and shaping runs
  // `querySelectorAll` over each node's subtree: quadratic in page size, which
  // on the largest pages in the corpus stalls a run that only ever needed the
  // prose. Own text also avoids the duplication subtree text produces, where a
  // section and its paragraphs each contribute the same sentences.
  const { document } = parseHTML(html);
  // linkedom's `body` getter destructures `documentElement.firstElementChild`,
  // so on a document with no element at all it throws instead of returning
  // null. Check first; `unusableHtmlReason` is how a caller skips such a page.
  const body = (
    (document as { documentElement?: unknown }).documentElement
      ? (document as { body?: unknown }).body
      : null
  ) as
    | {
        firstChild: unknown;
        nextSibling: unknown;
        nodeType: number;
        localName?: string;
        nodeName: string;
      }
    | null
    | undefined;
  const collected: Array<{ rank: number; text: string }> = [];
  if (body) {
    const stack = [body];
    while (stack.length > 0 && collected.length < MAX_PAGE_EVIDENCE * 8) {
      const element = stack.pop()!;
      const tag = (element.localName ?? element.nodeName.toLowerCase()).toLowerCase();
      if (PAGE_TEXT_SKIP.has(tag)) continue;
      let own = "";
      const children: typeof stack = [];
      for (
        let child = element.firstChild as (typeof body & { data?: string }) | null;
        child;
        child = child.nextSibling as (typeof body & { data?: string }) | null
      ) {
        if (child.nodeType === 3) own += ` ${child.data ?? ""}`;
        else if (child.nodeType === 1) children.push(child);
      }
      const index = PROSE_TAGS.indexOf(tag);
      const text = cleanText(own);
      if (text.length >= 40 && !looksLikeCode(text))
        collected.push({ rank: index < 0 ? PROSE_TAGS.length : index, text });
      for (let position = children.length - 1; position >= 0; position--)
        stack.push(children[position]!);
    }
  }
  const lines = collected
    .sort((left, right) => left.rank - right.rank)
    .map((item) => item.text)
    .filter((text, index, all) => all.indexOf(text) === index)
    .slice(0, MAX_PAGE_EVIDENCE);
  const title = /<title[^>]*>([\s\S]{0,400}?)<\/title>/i.exec(html)?.[1] ?? "";
  return { pageId, title: cleanText(title, MAX_TITLE), path: safePath(pageUrl), lines };
}

/**
 * One Choice over the eleven page-type classes, and nothing else.
 *
 * Page type is the axis where v4's sixty-five-Noul battery agreed 1.3% of the
 * time and v7's single Choice agrees 87%. At corpus scale the single question
 * is also what makes the run affordable: one judgment per page instead of
 * seventy-nine.
 */
export function buildPageTypeRequest(
  evidence: PageEvidence,
  policy: PolicyDocument,
): { request: JevRequest; packet: string } {
  const packet = serializePagePacket(evidence);
  const state = {
    policy: { revision: policy.revision, sha256: policy.sha256, text: policy.text },
    page: {
      pageId: evidence.pageId,
      title: evidence.title,
      path: evidence.path,
      evidence: evidence.lines,
      packet,
    },
  };
  return {
    request: {
      model: JEV_MODEL_V5,
      state: state as unknown as JevRequest["state"],
      questions: {
        page_type: {
          type: "choice",
          instructions: {
            judgment:
              "What is this page primarily for? Choose exactly one class. Section 4 of `policy.text` defines every class and the tie-breaks; navigation labels and footer links are not evidence about this page.",
            evidence: "page.packet",
          },
          criteria: Object.fromEntries(
            PAGE_TYPES_V5.map((label) => [label, definitionFor("section 4 pageType", label)]),
          ),
        },
      },
    },
    packet,
  };
}

export type PolicyDocument = { text: string; sha256: string; revision: PromptRevision };

/**
 * Read one revision's policy and bind it to its own prompt revision.
 *
 * The revision travels on the document rather than as a module constant, so a
 * row can never claim one revision while carrying another's bytes; both the
 * hash and the revision are written on every emitted row.
 */
export function loadPolicy(revision: PolicyRevision = "v5"): PolicyDocument {
  if (!POLICY_REVISIONS.includes(revision)) throw new Error(`unknown policy revision ${revision}`);
  const text = readFileSync(policyPath(revision), "utf8");
  if (!text.includes("## 1. componentType") || !text.includes("## 4. pageType"))
    throw new Error(`policy-${revision}.md is missing a required section`);
  if (!text.startsWith(`# Shared annotation policy, revision \`dom-suggestions-${revision}\``))
    throw new Error(`policy-${revision}.md does not declare revision ${revision}`);
  return { text, sha256: sha256(text), revision: `dom-suggestions-${revision}` };
}

function noul(
  instructions: string | Record<string, unknown>,
  yes: string,
  no: string,
): JevQuestion {
  return { type: "noul", instructions, criteria: { true: yes, false: no } };
}

function definitionFor(section: string, label: string): string {
  return `Apply the ${section} rules in \`policy.text\` exactly as written. Label: ${label}.`;
}

export type V5Candidate = { evidence: NodeEvidence; packet: string };

export type JevStateV5 = {
  policy: { revision: string; sha256: string; text: string };
  page: {
    pageId: string;
    title: string;
    path: string;
    evidence: string[];
    packet: string;
  };
  candidates: Array<NodeEvidence & { packet: string }>;
};

/**
 * One request per page carrying only the pilot's own nodes.
 *
 * v4 sent six adapter-chosen candidates per page and 361 questions. v5 sends the
 * nodes the sample asked for and, per node, nineteen region Nouls, twelve
 * purpose Nouls and one component Choice, plus one page Choice: the question
 * count falls with the vocabulary, not with the evidence.
 */
/**
 * The request, assembled from evidence that is already built.
 *
 * Both sources land here: a capture supplies `nodeEvidence`, full HTML supplies
 * `nodeEvidenceFromHtml`, and neither one gets to re-decide the questions. That
 * matters for the div-soup evaluation, where the packets must be built from
 * HTML but must ask exactly the questions the pilot measured.
 */
export function buildRequestFromEvidence(
  pageEvidenceValue: PageEvidence,
  nodeEvidences: NodeEvidence[],
  policy: PolicyDocument,
  options: { includePage?: boolean } = {},
): { request: JevRequest; candidates: V5Candidate[]; pagePacket: string } {
  const includePage = options.includePage ?? true;
  const evidence = pageEvidenceValue;
  const pagePacket = serializePagePacket(evidence);
  const candidates = nodeEvidences.map((value) => ({
    evidence: value,
    packet: serializeNodePacket(value),
  }));
  const questions: Record<string, JevQuestion> = {};
  if (includePage) {
    questions.page_type = {
      type: "choice",
      instructions: {
        judgment:
          "What is this page primarily for? Choose exactly one class. Section 4 of `policy.text` defines every class and the tie-breaks; navigation labels and footer links are not evidence about this page.",
        evidence: "page.packet",
      },
      criteria: Object.fromEntries(
        PAGE_TYPES_V5.map((label) => [label, definitionFor("section 4 pageType", label)]),
      ),
    };
  }
  for (const candidate of candidates) {
    const prefix = `node_${candidate.evidence.nodeId}`;
    for (const label of REGIONS_V5)
      questions[`${prefix}_region_${label}`] = noul(
        {
          node: candidate.evidence.nodeId,
          judgment: `Is this node in the ${label} region?`,
          rules: "Section 2 of `policy.text`.",
        },
        `The node is inside or is the ${label} region, as section 2 of \`policy.text\` defines it.`,
        `The node is not inside the ${label} region.`,
      );
    for (const label of PURPOSES_V5)
      questions[`${prefix}_purpose_${label}`] = noul(
        {
          node: candidate.evidence.nodeId,
          judgment: `Does this node itself serve the ${label} purpose?`,
          rules: "Section 3 of `policy.text`.",
        },
        `The node's own content or controls serve the ${label} purpose.`,
        `The node does not serve the ${label} purpose; a neighbour doing so is not enough.`,
      );
    questions[`${prefix}_component_type`] = {
      type: "choice",
      instructions: {
        node: candidate.evidence.nodeId,
        judgment:
          "What is this node's component type? Work down the precedence ladder in section 1 of `policy.text` and stop at the first rule that fires. `unknown` is a valid answer and is never penalised.",
      },
      criteria: Object.fromEntries(
        COMPONENT_TYPES_V5.map((label) => [label, definitionFor("section 1 componentType", label)]),
      ),
    };
  }
  const state: JevStateV5 = {
    policy: { revision: policy.revision, sha256: policy.sha256, text: policy.text },
    page: {
      pageId: evidence.pageId,
      title: evidence.title,
      path: evidence.path,
      evidence: evidence.lines,
      packet: pagePacket,
    },
    candidates: candidates.map((candidate) => ({
      ...candidate.evidence,
      packet: candidate.packet,
    })),
  };
  return {
    request: { model: JEV_MODEL_V5, state: state as unknown as JevRequest["state"], questions },
    candidates,
    pagePacket,
  };
}

/** The capture-shaped entry point, unchanged for every existing caller. */
export function buildRequestV5(
  page: CapturedPage,
  nodes: CapturedNode[],
  policy: PolicyDocument,
  options: { includePage?: boolean } = {},
): { request: JevRequest; candidates: V5Candidate[]; pagePacket: string } {
  return buildRequestFromEvidence(
    pageEvidence(page),
    nodes.map((node) => nodeEvidence(page, node)),
    policy,
    options,
  );
}

export function snapshotHash(state: unknown): string {
  return sha256(JSON.stringify(state));
}

export type JevRowV5 = {
  schemaVersion: 1;
  provider: "typesafe";
  modelId: string;
  promptRevision: PromptRevision;
  taxonomyRevision: typeof TAXONOMY_REVISION;
  pageTypeMappingRevision: string;
  purposeMappingRevision: string;
  policySha256: string;
  pageId: string;
  nodeId: string | null;
  captureHash: string;
  snapshotHash: string;
  input: string;
  inputSha256: string;
  recordType: "node" | "page";
  split: string;
  stratum: string;
  rawAnswers: Record<string, unknown>;
  observedAxes: Record<string, { complete: boolean; labels: string[] | string | null }>;
  createdAt: string;
  usage?: JevResponse["usage"];
  latencyMs?: number;
};

const NOUL_THRESHOLD = 0.75;

function nounSet(
  answers: Record<string, unknown>,
  prefix: string,
  labels: readonly string[],
  threshold: number,
): string[] {
  return labels.filter((label) => {
    const answer = answers[`${prefix}${label}`] as { noul?: number } | undefined;
    return typeof answer?.noul === "number" && answer.noul >= threshold;
  });
}

/**
 * Rows in the shape `luna-complete-800.jsonl` uses, so the two annotators join
 * on identity and the scorer never has to know which side it is reading.
 */
export function toRowsV5(
  page: CapturedPage,
  built: ReturnType<typeof buildRequestV5>,
  response: JevResponse,
  context: {
    split: string;
    strata: Map<string, string>;
    policy: PolicyDocument;
    latencyMs: number;
  },
  includePage: boolean,
  createdAt = new Date().toISOString(),
): JevRowV5[] {
  const answers = response.answers as Record<string, Record<string, unknown>>;
  const snapshot = snapshotHash(built.request.state);
  const base = {
    schemaVersion: 1 as const,
    provider: "typesafe" as const,
    modelId: JEV_MODEL_V5,
    promptRevision: context.policy.revision,
    taxonomyRevision: TAXONOMY_REVISION,
    pageTypeMappingRevision: pageTypeMap.mappingRevision as string,
    purposeMappingRevision: purposeMap.mappingRevision as string,
    policySha256: context.policy.sha256,
    pageId: page.id,
    captureHash: page.captureHash,
    snapshotHash: snapshot,
    split: context.split,
    createdAt,
  };
  const rows: JevRowV5[] = [];
  if (includePage) {
    const choice = answers.page_type as { choice?: string } | undefined;
    rows.push({
      ...base,
      nodeId: null,
      recordType: "page",
      stratum: "page",
      input: built.pagePacket,
      inputSha256: sha256(built.pagePacket),
      rawAnswers: { page_type: answers.page_type },
      observedAxes: {
        pageType: { complete: typeof choice?.choice === "string", labels: choice?.choice ?? null },
      },
      usage: response.usage,
      latencyMs: context.latencyMs,
    });
  }
  for (const candidate of built.candidates) {
    const prefix = `node_${candidate.evidence.nodeId}_`;
    const raw = Object.fromEntries(
      Object.entries(answers).filter(([key]) => key.startsWith(prefix)),
    );
    const choice = answers[`${prefix}component_type`] as { choice?: string } | undefined;
    rows.push({
      ...base,
      nodeId: candidate.evidence.nodeId,
      recordType: "node",
      stratum: context.strata.get(candidate.evidence.nodeId) ?? "unknown",
      input: candidate.packet,
      inputSha256: sha256(candidate.packet),
      rawAnswers: raw,
      observedAxes: {
        componentType: {
          complete: typeof choice?.choice === "string",
          labels: choice?.choice ?? null,
        },
        regions: {
          complete: true,
          labels: nounSet(answers, `${prefix}region_`, REGIONS_V5, NOUL_THRESHOLD),
        },
        purposes: {
          complete: true,
          labels: nounSet(answers, `${prefix}purpose_`, PURPOSES_V5, NOUL_THRESHOLD),
        },
      },
    });
  }
  return rows;
}

/** Exponential backoff on 429/529; the pilot never runs more than four at once. */
export async function evaluateWithBackoff(
  transport: TypeSafeTransport,
  request: JevRequest,
  attempts = 5,
): Promise<{ response: JevResponse; latencyMs: number; retries: number }> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt++) {
    const started = Date.now();
    try {
      const response = await transport(request);
      return { response, latencyMs: Date.now() - started, retries: attempt };
    } catch (error) {
      lastError = error;
      const message = error instanceof Error ? error.message : String(error);
      const retryable = /HTTP (429|5\d\d)|timed out/.test(message);
      if (!retryable || attempt === attempts - 1) break;
      await Bun.sleep(Math.min(30_000, 1000 * 2 ** attempt) + Math.floor(Math.random() * 250));
    }
  }
  throw lastError instanceof Error ? lastError : new Error("TypeSafe evaluation failed");
}

export { validateResponse };

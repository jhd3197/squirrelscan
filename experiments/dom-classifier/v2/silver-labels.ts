// Silver labels for DOM classifier v2 (#2307).
//
// Free supervision, from the page's own semantics:
//
//  - REGION comes from landmark elements and ARIA landmark roles. The trap the
//    error analysis found is nesting: a <header> inside an <article> is that
//    article's header, not the site masthead, and labelling it site_header
//    teaches the model the opposite of what it needs. Containment decides it.
//  - PAGE TYPE comes from JSON-LD @type, microdata itemtype and og:type. Those
//    three never enter the features (see featurize.ts), which is the whole
//    point: the model has to learn the page's shape, so it can then label the
//    pages that carry no schema at all.
//
// Nothing here is imported by featurize.ts, and nothing here is a feature.

import type { NodeFeatures } from "./featurize.ts";
import { SECTIONING_TAGS } from "./featurize.ts";

// ---------------------------------------------------------------------------
// Region
// ---------------------------------------------------------------------------

export type RegionLabel =
  | "site_header"
  | "footer"
  | "navigation"
  | "sidebar"
  | "main_content"
  | "article_header"
  | "article_footer"
  | "search"
  | "form"
  | "dialog"
  | "content_block"
  | "unlabelled";

/** Labels a landmark can produce directly; the rest come from containment. */
const LANDMARK_REGIONS: ReadonlySet<RegionLabel> = new Set<RegionLabel>([
  "site_header",
  "footer",
  "navigation",
  "sidebar",
  "main_content",
  "article_header",
  "article_footer",
  "search",
  "form",
  "dialog",
]);

export interface NodeLabel {
  nodeId: number;
  label: RegionLabel;
  /** Where the label came from: "tag:footer", "role:contentinfo",
   *  "containment:main", "none". Kept per row so a training run can drop a
   *  provenance it does not trust. */
  labelSource: string;
  /** The nearest ancestor landmark's region, or "". Not the label: it is the
   *  answer to "which region is this div INSIDE", which is what the div-soup
   *  head ultimately has to predict, and the training lane decides whether to
   *  use it. */
  containerLabel: RegionLabel | "";
  /** False for rows kept only for analysis (unlabelled, or a page with too few
   *  landmarks to trust). */
  trainable: boolean;
  /** True when the label came from the landmark this node SITS IN rather than
   *  from the node's own markup. A menu item inside a <nav> is navigation, but
   *  it is weaker evidence than the <nav> itself. */
  inherited: boolean;
  /** Training weight. Inherited rows carry half, because containment is a
   *  weaker claim than a declaration and there are far more of them. */
  weight: number;
}

export interface RegionLabelling {
  labels: NodeLabel[];
  /** Distinct landmark kinds on the page. */
  landmarkKinds: string[];
  /** >= 2 kinds: the page says enough about itself to supervise a region head. */
  regionTrainable: boolean;
  /** >= 3 kinds: containment labels (content_block / unlabelled) are trusted. */
  containmentTrainable: boolean;
  /** Why the page was excluded from region training, or "". */
  excludedReason: string;
}

/**
 * The region a node's OWN tag or role declares, ignoring where it sits.
 * `null` for a node that declares nothing.
 */
function ownRegion(tag: string, role: string): { label: RegionLabel; source: string } | null {
  // An explicit role wins over the tag, which is what a screen reader does.
  switch (role) {
    case "banner":
      return { label: "site_header", source: "role:banner" };
    case "contentinfo":
      return { label: "footer", source: "role:contentinfo" };
    case "navigation":
      return { label: "navigation", source: "role:navigation" };
    case "complementary":
      return { label: "sidebar", source: "role:complementary" };
    case "main":
      return { label: "main_content", source: "role:main" };
    case "search":
      return { label: "search", source: "role:search" };
    case "form":
      return { label: "form", source: "role:form" };
    case "dialog":
    case "alertdialog":
      return { label: "dialog", source: `role:${role}` };
    default:
      break;
  }
  switch (tag) {
    // header/footer are decided by nesting, by the caller.
    case "nav":
      return { label: "navigation", source: "tag:nav" };
    case "aside":
      return { label: "sidebar", source: "tag:aside" };
    case "main":
      return { label: "main_content", source: "tag:main" };
    case "search":
      return { label: "search", source: "tag:search" };
    case "form":
      return { label: "form", source: "tag:form" };
    case "dialog":
      return { label: "dialog", source: "tag:dialog" };
    default:
      return null;
  }
}

/**
 * Silver region labels for one page's candidate nodes.
 *
 * The rules, in order:
 *
 *  1. An explicit ARIA landmark role, then the landmark tag, names the node.
 *  2. <header>/<footer> are site chrome ONLY when no sectioning ancestor
 *     (article/section/aside/nav) stands between them and <body>. Inside one
 *     they are article_header / article_footer. <nav> inside a <header> stays
 *     navigation: a menu is a menu wherever it lives.
 *  3. Every other candidate, on a page with >= 3 distinct landmark kinds:
 *     inside <main> it is a content_block, outside every landmark it is
 *     unlabelled (kept, not trained). Inside header/footer/nav/aside it is
 *     unlabelled too, but `containerLabel` records which one.
 *  4. A page with < 2 landmark kinds is the div-soup target: excluded from
 *     region training, kept for the eval pool.
 */
export function silverRegionLabels(nodes: readonly NodeFeatures[]): RegionLabelling {
  const byId = new Map<number, NodeFeatures>();
  for (const node of nodes) byId.set(node.nodeId, node);

  // Resolved own-region per node, and the ancestor chain walk both need this
  // first, so it is computed for every candidate before anything is decided.
  const own = new Map<number, { label: RegionLabel; source: string }>();
  for (const node of nodes) {
    const declared = ownRegion(node.tag, node.role);
    if (declared) {
      own.set(node.nodeId, declared);
      continue;
    }
    if (node.tag !== "header" && node.tag !== "footer") continue;
    const nested = hasSectioningAncestor(node, byId);
    if (node.tag === "header") {
      own.set(node.nodeId, {
        label: nested ? "article_header" : "site_header",
        source: nested ? "tag:header/nested" : "tag:header",
      });
    } else {
      own.set(node.nodeId, {
        label: nested ? "article_footer" : "footer",
        source: nested ? "tag:footer/nested" : "tag:footer",
      });
    }
  }

  const landmarkKinds = new Set<string>();
  for (const [, declared] of own) {
    if (LANDMARK_REGIONS.has(declared.label)) landmarkKinds.add(declared.label);
  }
  // article_header / article_footer are a section's own chrome, so they do not
  // count towards "this page declares its landmarks".
  landmarkKinds.delete("article_header");
  landmarkKinds.delete("article_footer");

  const regionTrainable = landmarkKinds.size >= 2;
  const containmentTrainable = landmarkKinds.size >= 3;

  const labels: NodeLabel[] = nodes.map((node) => {
    const container = nearestLandmark(node, byId, own);
    const declared = own.get(node.nodeId);
    if (declared) {
      return {
        nodeId: node.nodeId,
        label: declared.label,
        labelSource: declared.source,
        containerLabel: container,
        trainable: regionTrainable,
        inherited: false,
        weight: 1,
      };
    }
    if (!containmentTrainable) {
      return {
        nodeId: node.nodeId,
        label: "unlabelled",
        labelSource: "none",
        containerLabel: container,
        trainable: false,
        inherited: false,
        weight: 0,
      };
    }
    if (container === "main_content") {
      return {
        nodeId: node.nodeId,
        label: "content_block",
        labelSource: "containment:main",
        containerLabel: container,
        trainable: true,
        inherited: false,
        weight: 1,
      };
    }
    // A candidate inside a landmark INHERITS it. Previously these were
    // `unlabelled` and dropped from training, which is why the model could
    // find a menu root and then call every item inside it a content_block:
    // it had never been shown one. DOM containment means the node's whole
    // subtree is inside the landmark, so the claim is sound; it is weaker
    // evidence than a declaration, so it carries half the weight.
    if (container && container !== "article_header" && container !== "article_footer") {
      return {
        nodeId: node.nodeId,
        label: container,
        labelSource: `inherited:${container}`,
        containerLabel: container,
        trainable: true,
        inherited: true,
        weight: 0.5,
      };
    }
    return {
      nodeId: node.nodeId,
      label: "unlabelled",
      labelSource: container ? `containment:${container}` : "outside-landmarks",
      containerLabel: container,
      trainable: false,
      inherited: false,
      weight: 0,
    };
  });

  return {
    labels,
    landmarkKinds: [...landmarkKinds].sort(),
    regionTrainable,
    containmentTrainable,
    excludedReason: regionTrainable ? "" : `landmark_kinds_${landmarkKinds.size}`,
  };
}

/** True when a sectioning element stands between this node and <body>. */
function hasSectioningAncestor(
  node: NodeFeatures,
  byId: ReadonlyMap<number, NodeFeatures>,
): boolean {
  let current = byId.get(node.parentCandidateId);
  let guard = 0;
  while (current && guard++ < 256) {
    if (SECTIONING_TAGS.has(current.tag)) return true;
    // A <main> or a plain wrapper between the header and the body is still site
    // chrome territory, so keep walking rather than stopping at the first
    // landmark. <main> is sectioning ROOT content in HTML, not sectioning
    // content, and a <header> directly inside <main> is a page header.
    current = byId.get(current.parentCandidateId);
  }
  return false;
}

/** The region of the nearest ancestor landmark, or "" when there is none. */
function nearestLandmark(
  node: NodeFeatures,
  byId: ReadonlyMap<number, NodeFeatures>,
  own: ReadonlyMap<number, { label: RegionLabel; source: string }>,
): RegionLabel | "" {
  let current = byId.get(node.parentCandidateId);
  let guard = 0;
  while (current && guard++ < 256) {
    const declared = own.get(current.nodeId);
    if (declared && LANDMARK_REGIONS.has(declared.label)) return declared.label;
    current = byId.get(current.parentCandidateId);
  }
  return "";
}

// ---------------------------------------------------------------------------
// Page type
// ---------------------------------------------------------------------------

export type PageTypeLabel =
  | "article"
  | "product_detail"
  | "product_listing"
  | "docs"
  | "homepage"
  | "pricing"
  | "contact"
  | "about"
  | "legal"
  | "listing_index"
  | "other"
  | "unlabelled";

export interface PageTypeLabelling {
  label: PageTypeLabel;
  /** "jsonld" | "microdata" | "og" | "jsonld+url_root" | "conflict" | "none" */
  source: string;
  /** The schema type strings that were read, deduped and sorted. */
  types: string[];
  /** True when the label used the URL as well as the schema. The URL shape IS
   *  a feature, so these rows leak and a training run may want to drop them.
   *  Only `homepage` is ever flagged. */
  leakProne: boolean;
}

/** schema.org type -> page class. Only types that map unambiguously are here;
 *  a type that is not listed contributes nothing rather than guessing. */
const SCHEMA_TYPE_MAP: Record<string, PageTypeLabel> = {
  article: "article",
  newsarticle: "article",
  blogposting: "article",
  liveblogposting: "article",
  report: "article",
  scholarlyarticle: "article",
  socialmediaposting: "article",
  advertisercontentarticle: "article",
  satiricalarticle: "article",
  techarticle: "docs",
  apireference: "docs",
  howto: "docs",
  product: "product_detail",
  productgroup: "product_detail",
  individualproduct: "product_detail",
  productmodel: "product_detail",
  vehicle: "product_detail",
  car: "product_detail",
  aboutpage: "about",
  contactpage: "contact",
  collectionpage: "listing_index",
  blog: "listing_index",
  itemlist: "listing_index",
  faqpage: "other",
  qapage: "other",
  profilepage: "other",
  searchresultspage: "other",
  checkoutpage: "other",
  realestatelisting: "other",
  medicalwebpage: "other",
};

/** og:type -> page class. `website` alone says nothing beyond "a page". */
const OG_TYPE_MAP: Record<string, PageTypeLabel> = {
  article: "article",
  "article:published_time": "article",
  book: "article",
  product: "product_detail",
  "product.item": "product_detail",
  "product.group": "product_detail",
};

interface SchemaDocument {
  querySelector(selector: string): { getAttribute(name: string): string | null } | null;
  querySelectorAll(selector: string): Iterable<{
    textContent: string | null;
    getAttribute(name: string): string | null;
  }>;
}

/**
 * Page-type silver label from structured data only.
 *
 * Precedence JSON-LD > microdata > og:type. Two JSON-LD types that map to
 * different classes is a conflict, and a conflict is `unlabelled` rather than
 * a coin flip: those pages are exactly the ones the model is supposed to
 * resolve later, so poisoning them here would be circular.
 */
export function silverPageType(document: SchemaDocument, url: string): PageTypeLabelling {
  const jsonLdTypes = collectJsonLdTypes(document);
  const fromJsonLd = classify(jsonLdTypes);
  if (fromJsonLd.classes.size > 1) {
    return {
      label: "unlabelled",
      source: "conflict",
      types: [...jsonLdTypes].sort(),
      leakProne: false,
    };
  }
  if (fromJsonLd.classes.size === 1) {
    return {
      label: [...fromJsonLd.classes][0]!,
      source: "jsonld",
      types: [...jsonLdTypes].sort(),
      leakProne: false,
    };
  }

  // A site that declares only WebSite/Organization on its root is declaring a
  // homepage. The URL is part of that call, and the URL shape is a feature, so
  // the row is flagged and the training lane decides.
  if (isRootPath(url) && (jsonLdTypes.has("website") || jsonLdTypes.has("organization"))) {
    return {
      label: "homepage",
      source: "jsonld+url_root",
      types: [...jsonLdTypes].sort(),
      leakProne: true,
    };
  }

  const microdataTypes = collectMicrodataTypes(document);
  const fromMicrodata = classify(microdataTypes);
  if (fromMicrodata.classes.size === 1) {
    return {
      label: [...fromMicrodata.classes][0]!,
      source: "microdata",
      types: [...microdataTypes].sort(),
      leakProne: false,
    };
  }
  if (fromMicrodata.classes.size > 1) {
    return {
      label: "unlabelled",
      source: "conflict",
      types: [...microdataTypes].sort(),
      leakProne: false,
    };
  }

  const og = (document.querySelector('meta[property="og:type"]')?.getAttribute("content") ?? "")
    .trim()
    .toLowerCase();
  const fromOg = OG_TYPE_MAP[og];
  if (fromOg) return { label: fromOg, source: "og", types: [og], leakProne: false };
  if (og === "website" && isRootPath(url)) {
    return { label: "homepage", source: "og+url_root", types: [og], leakProne: true };
  }

  return { label: "unlabelled", source: "none", types: [], leakProne: false };
}

function isRootPath(url: string): boolean {
  const withoutScheme = url.replace(/^[a-z][a-z0-9+.-]*:\/\//i, "");
  const slash = withoutScheme.indexOf("/");
  if (slash === -1) return true;
  const path = withoutScheme.slice(slash).split(/[?#]/)[0] ?? "/";
  return path === "/" || path === "";
}

function classify(types: ReadonlySet<string>): { classes: Set<PageTypeLabel> } {
  const classes = new Set<PageTypeLabel>();
  for (const type of types) {
    const mapped = SCHEMA_TYPE_MAP[type];
    if (mapped) classes.add(mapped);
  }
  // "other" never conflicts with a real class: a FAQPage that also declares a
  // Product is a product page with an FAQ on it.
  if (classes.size > 1) classes.delete("other");
  // A CollectionPage / ItemList whose members are Products IS the product
  // listing class, not two conflicting answers. This is the single most common
  // multi-type shape in the corpus, so it is resolved rather than dropped.
  if (classes.has("listing_index") && classes.has("product_detail")) {
    classes.delete("listing_index");
    classes.delete("product_detail");
    classes.add("product_listing");
  }
  // A CollectionPage that also declares Article-shaped items is still an index.
  if (classes.has("listing_index") && classes.has("article") && types.has("blog")) {
    classes.delete("article");
  }
  return { classes };
}

/** Every @type in every JSON-LD block, including @graph, arrays and the one
 *  level of nesting that matters (mainEntity / about / itemListElement). */
function collectJsonLdTypes(document: SchemaDocument): Set<string> {
  const types = new Set<string>();
  for (const script of document.querySelectorAll('script[type="application/ld+json"]')) {
    const raw = (script.textContent ?? "").trim();
    if (!raw) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      // Malformed JSON-LD is common and is not a labelling signal either way.
      continue;
    }
    collectTypes(parsed, types, 0);
  }
  return types;
}

const NESTED_KEYS = ["@graph", "mainentity", "mainentityofpage", "about", "itemlistelement"];

function collectTypes(value: unknown, into: Set<string>, depth: number): void {
  if (depth > 6 || value === null || typeof value !== "object") return;
  if (Array.isArray(value)) {
    for (const entry of value.slice(0, 50)) collectTypes(entry, into, depth + 1);
    return;
  }
  const record = value as Record<string, unknown>;
  for (const [key, entry] of Object.entries(record)) {
    const lowerKey = key.toLowerCase();
    if (lowerKey === "@type" || lowerKey === "type") {
      for (const type of asStrings(entry)) into.add(normalizeType(type));
      continue;
    }
    if (NESTED_KEYS.includes(lowerKey)) collectTypes(entry, into, depth + 1);
  }
}

function asStrings(value: unknown): string[] {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.filter((v): v is string => typeof v === "string");
  return [];
}

/** "https://schema.org/NewsArticle" and "schema:NewsArticle" are both that. */
function normalizeType(type: string): string {
  const tail = type.split(/[/#:]/).pop() ?? type;
  return tail.trim().toLowerCase();
}

function collectMicrodataTypes(document: SchemaDocument): Set<string> {
  const types = new Set<string>();
  let seen = 0;
  for (const el of document.querySelectorAll("[itemtype]")) {
    if (seen++ > 200) break;
    const itemtype = el.getAttribute("itemtype");
    if (!itemtype) continue;
    for (const one of itemtype.split(/\s+/)) {
      if (one) types.add(normalizeType(one));
    }
  }
  return types;
}

// Page skeleton: a pruned, annotated tree between the full DOM and plain text (#2307).
//
// Plain text drops what makes a page type (forms, buttons, repeated cards, nesting),
// the full DOM is mostly noise (wrappers, utility classes, scripts). The skeleton keeps
// tag and landmark structure, collapses wrappers and inline markup, run-length collapses
// repeated siblings (`li n=24`), turns body text into length buckets, and flags the
// things that say what a page is (price, buy button, form fields, byline, comments).
//
// One tree, two renderers: `xml` for LLM prompts (teacher and judge), `compact` for
// training small models. Both come from the same tree, so they cannot drift apart.
//
// Rules, same as featurize.ts:
//  1. NOTHING derived from JSON-LD, microdata, og:type, itemprop or any <script>/<meta>
//     may appear in the output: those are the page-type LABEL source. Only <title> and
//     <html lang> are read from <head>.
//  2. Page text is hostile: it is escaped, truncated, never emitted as markup, and the
//     `text: "none"` variant has no page words at all (language independent).
//  3. Deterministic and O(n) in element count: one bottom-up pass, budgets on nodes.

export interface SkeletonOptions {
  /** `headings` keeps title and heading words (<= 60 chars); `none` keeps no words. */
  text?: "headings" | "none";
  /** Hard cap on emitted nodes; the rest becomes `<more n=K/>`. */
  maxNodes?: number;
  /** Elements nested deeper than this fold into their parent. */
  maxDepth?: number;
}

export interface SkeletonNode {
  tag: string;
  attrs: [string, string | number][];
  text?: string;
  kids: SkeletonNode[];
  /** Cached shape signature (run-length collapse); built once, bottom-up. */
  sig?: string;
  /** Run length: consecutive identical siblings collapsed into one node. */
  n: number;
}

/** Minimal DOM surface used here, so any linkedom node fits. */
interface DomNode {
  nodeType: number;
  nodeValue?: string | null;
  localName?: string;
  childNodes?: ArrayLike<DomNode>;
  getAttribute?(name: string): string | null;
  hasAttribute?(name: string): boolean;
  textContent?: string | null;
}

const DEFAULT_MAX_NODES = 300;
const DEFAULT_MAX_DEPTH = 14;
const MAX_TEXT = 60;

const SKIP = new Set([
  "script",
  "style",
  "noscript",
  "template",
  "svg",
  "math",
  "head",
  "meta",
  "link",
  "base",
  "map",
  "object",
  "embed",
  "param",
  "source",
  "track",
  "datalist",
  "slot",
]);
const INLINE = new Set([
  "a",
  "span",
  "strong",
  "em",
  "b",
  "i",
  "u",
  "s",
  "small",
  "mark",
  "abbr",
  "cite",
  "q",
  "sub",
  "sup",
  "code",
  "kbd",
  "samp",
  "var",
  "label",
  "bdi",
  "bdo",
  "wbr",
  "br",
  "font",
  "big",
  "dfn",
  "ins",
  "del",
  "data",
  "output",
  "legend",
  "caption",
  "summary",
]);
const LANDMARK = new Set([
  "header",
  "nav",
  "main",
  "article",
  "section",
  "aside",
  "footer",
  "form",
  "dialog",
  "details",
  "figure",
  "blockquote",
  "pre",
  "address",
  "table",
  "ul",
  "ol",
  "dl",
]);
const ROLE_TAG: Record<string, string> = {
  banner: "header",
  contentinfo: "footer",
  navigation: "nav",
  main: "main",
  complementary: "aside",
  search: "form",
  article: "article",
  dialog: "dialog",
};
const HEADING = /^h[1-6]$/;

/** class/id tokens that name a part of the page; the first match wins, in this order. */
const KINDS: [string, RegExp][] = [
  [
    "comments",
    /(?:^|[\s_-])(?:comments?|comment-?list|comment-?thread|comment-?section|disqus_thread|fb-comments|commento|respond)(?:$|[\s_-])/,
  ],
  ["breadcrumb", /(?:^|[\s_-])breadcrumbs?(?:$|[\s_-])/],
  ["pagination", /(?:^|[\s_-])(?:pagination|pager|page-numbers)(?:$|[\s_-])/],
  ["price", /(?:^|[\s_-])(?:price|prices|pricing|amount)(?:$|[\s_-])/],
  ["cart", /(?:^|[\s_-])(?:cart|basket|add-to-cart|addtocart|buy-?box)(?:$|[\s_-])/],
  ["byline", /(?:^|[\s_-])(?:byline|author|posted-by|entry-meta|post-meta)(?:$|[\s_-])/],
  ["share", /(?:^|[\s_-])(?:share|sharing|social)(?:$|[\s_-])/],
  ["related", /(?:^|[\s_-])(?:related|recommended|you-may-like|more-posts)(?:$|[\s_-])/],
  ["cookie", /(?:^|[\s_-])(?:cookie|consent|gdpr)(?:$|[\s_-])/],
  ["search", /(?:^|[\s_-])search(?:$|[\s_-])/],
  ["product", /(?:^|[\s_-])(?:product|product-card|product-item)(?:$|[\s_-])/],
];
const PRICE = /(?:[$€£¥₹]\s?\d[\d.,]*|\b\d[\d.,]*\s?(?:USD|EUR|GBP|AUD|CAD|JPY)\b)/;
const BUY = /\b(?:add to (?:cart|bag|basket)|buy now|checkout|add to trolley)\b/i;

/** Coarse length buckets so repeated paragraphs of similar size collapse into one run. */
export function bucket(length: number): number {
  if (length <= 0) return 0;
  if (length < 20) return 10;
  if (length < 50) return 20;
  if (length < 100) return 50;
  if (length < 200) return 100;
  if (length < 500) return 200;
  if (length < 1000) return 500;
  if (length < 2000) return 1000;
  if (length < 5000) return 2000;
  return 5000;
}

/** Page-level counters for signals that say what a page is but have no node of their own. */
interface PageStats {
  tel: number;
  mailto: number;
  emailText: number;
  phoneText: number;
  social: number;
  map: number;
  code: number;
  priceTerms: number;
  updated: number;
  wpPost: number;
  wpPage: number;
  wpAttachment: number;
  wpArchive: number;
  wpProduct: number;
  commentItems: number;
  dateClass: number;
  readMore: number;
  h1EqTitle: number;
  titleLen: number;
}
type Context = Required<SkeletonOptions> & { stats: PageStats };

const SOCIAL_HOST =
  /^https?:\/\/(?:[a-z0-9-]+\.)*(?:facebook|instagram|twitter|x|linkedin|youtube|tiktok|pinterest|github)\.com\//i;
const MAP_SRC =
  /(?:google\.[a-z.]+\/maps|maps\.google|openstreetmap|mapbox|maps\.apple|goo\.gl\/maps)/i;
const EMAIL_TEXT = /[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,4}/;
const PHONE_TEXT = /(?:\+\d{1,3}[\s.-]?)?\(?\d{1,4}\)?(?:[\s.-]\d{1,4}){2,4}/;
const DATES = /\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b/g;

/** A phone-number-shaped run of at least 8 digits, once dates are blanked out. */
function hasPhone(text: string): boolean {
  const match = PHONE_TEXT.exec(text.replace(DATES, " "));
  return match !== null && match[0].replace(/\D/g, "").length >= 8;
}
const PRICE_TERMS =
  /\b(?:per (?:month|year|user|seat)|\/(?:mo|month|yr|year)|monthly|annually|billed (?:monthly|annually|yearly)|free trial|most popular)\b/i;
const UPDATED =
  /\b(?:last (?:updated|modified|revised)|effective (?:date|as of)|revised|date of last update)\b/i;

const COMMENT_ITEM = /(?:^|\s)comment(?:-\d+)?(?:\s|$)/;
const COMMENT_ID = /^comment-\d+$/;
const DATE_CLASS =
  /(?:^|[\s_-])(?:entry-date|post-date|posted-on|published|pubdate|post-meta-date)(?:$|[\s_-])/;
const READ_MORE = /^(?:read more|continue reading|read the full|view more|learn more)\b/i;
const BODY_POST = /(?:^|\s)(?:single-post|postid-\d+|type-post)(?:\s|$)/;
const BODY_PAGE = /(?:^|\s)(?:page-template\S*|page-id-\d+|type-page)(?:\s|$)/;
const BODY_ATTACHMENT = /(?:^|\s)(?:attachment|attachmentid-\d+|single-attachment)(?:\s|$)/;
const BODY_ARCHIVE = /(?:^|\s)(?:archive|category-\S+|tag-\S+)(?:\s|$)/;
const BODY_PRODUCT = /(?:^|\s)(?:single-product|woocommerce-page|product-template-default)(?:\s|$)/;

function scanElement(name: string, element: DomNode, ownTrim: string, stats: PageStats): void {
  if (name === "a") {
    const href = (element.getAttribute?.("href") ?? "").slice(0, 200);
    if (/^tel:/i.test(href)) stats.tel++;
    else if (/^mailto:/i.test(href)) stats.mailto++;
    else if (SOCIAL_HOST.test(href)) stats.social++;
  } else if (name === "iframe") {
    if (MAP_SRC.test((element.getAttribute?.("src") ?? "").slice(0, 300))) stats.map++;
  } else if (name === "pre") {
    stats.code++;
  }
  if (name === "li" || name === "div" || name === "article" || name === "section") {
    const cls = (element.getAttribute?.("class") ?? "").toLowerCase().slice(0, 200);
    if (cls && COMMENT_ITEM.test(cls)) stats.commentItems++;
    else if (COMMENT_ID.test((element.getAttribute?.("id") ?? "").toLowerCase()))
      stats.commentItems++;
  }
  {
    const cls = (element.getAttribute?.("class") ?? "").toLowerCase().slice(0, 200);
    if (cls && DATE_CLASS.test(cls)) stats.dateClass++;
  }
  if (name === "a" && ownTrim.length > 0 && ownTrim.length < 30 && READ_MORE.test(ownTrim))
    stats.readMore++;
  if (ownTrim.length > 4) {
    const head = ownTrim.slice(0, 300);
    if (head.includes("@") && EMAIL_TEXT.test(head)) stats.emailText++;
    if (/\d{3}/.test(head) && hasPhone(head)) stats.phoneText++;
    if (PRICE_TERMS.test(head)) stats.priceTerms++;
    if (UPDATED.test(head)) stats.updated++;
  }
}

interface Folded {
  nodes: SkeletonNode[];
  textLen: number;
  links: number;
  price: boolean;
  buy: boolean;
}

class Budget {
  nodes = 0;
  dropped = 0;
  constructor(readonly max: number) {}
}

function tokens(element: DomNode): string {
  const id = element.getAttribute?.("id") ?? "";
  const cls = element.getAttribute?.("class") ?? "";
  return `${id} ${cls}`.toLowerCase().replace(/\s+/g, " ").slice(0, 400);
}

/** One cheap reject before the ordered kind patterns: most class strings match none. */
const KIND_HINT =
  /comment|respond|disqus|commento|breadcrumb|pag(?:ination|er|e-numbers)|pric|amount|cart|basket|buy|byline|author|posted|entry-meta|post-meta|share|sharing|social|related|recommend|you-may|more-posts|cookie|consent|gdpr|search|product/;

function kindOf(element: DomNode): string | null {
  const text = tokens(element);
  if (!text.trim() || !KIND_HINT.test(text)) return null;
  for (const [kind, pattern] of KINDS) if (pattern.test(text)) return kind;
  return null;
}

function clean(text: string, limit: number): string {
  // control characters and line/paragraph separators become spaces
  const flat = text
    .replace(/[\p{Cc}\u2028\u2029]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
  return flat.length > limit ? `${flat.slice(0, limit - 3)}...` : flat;
}

function signature(node: SkeletonNode): string {
  if (node.sig !== undefined) return node.sig;
  let out = `${node.tag}|${node.text === undefined ? "" : "t"}`;
  for (const [key, value] of node.attrs) out += `|${key}=${value}`;
  if (node.kids.length)
    out += `(${node.kids.map((kid) => `${signature(kid)}*${kid.n}`).join(",")})`;
  node.sig = out;
  return out;
}

/** Collapse consecutive siblings with an identical shape into one node with `n`. */
function collapse(nodes: SkeletonNode[]): SkeletonNode[] {
  const out: SkeletonNode[] = [];
  let previous = "";
  for (const node of nodes) {
    const sig = signature(node);
    const last = out[out.length - 1];
    if (last && sig === previous) last.n += node.n;
    else out.push(node);
    previous = sig;
  }
  return out;
}

/** Run-length collapse; a group that does not repeat is hoisted back into its parent. */
function runLength(nodes: SkeletonNode[]): SkeletonNode[] {
  const runs = collapse(nodes);
  if (!runs.some((node) => node.tag === "g" && node.n === 1)) return runs;
  return collapse(runs.flatMap((node) => (node.tag === "g" && node.n === 1 ? node.kids : [node])));
}

function ownText(element: DomNode): string {
  let out = "";
  for (const child of Array.from(element.childNodes ?? [])) {
    if (child.nodeType === 3) out += child.nodeValue ?? "";
  }
  return out;
}

function fold(element: DomNode, depth: number, options: Context): Folded {
  const empty: Folded = { nodes: [], textLen: 0, links: 0, price: false, buy: false };
  const name = (element.localName ?? "").toLowerCase();
  if (
    SKIP.has(name) ||
    element.hasAttribute?.("hidden") ||
    element.getAttribute?.("aria-hidden") === "true"
  )
    return empty;

  const folded: Folded = { nodes: [], textLen: 0, links: 0, price: false, buy: false };
  const own = ownText(element);
  const ownTrim = own.replace(/\s+/g, " ").trim();
  scanElement(name, element, ownTrim, options.stats);
  folded.textLen += ownTrim.length;
  if (ownTrim && PRICE.test(ownTrim.slice(0, 200))) folded.price = true;

  const role = (element.getAttribute?.("role") ?? "").toLowerCase();
  const tag = ROLE_TAG[role] && !LANDMARK.has(name) ? ROLE_TAG[role] : name;
  const kind = kindOf(element);

  // children
  const children: SkeletonNode[] = [];
  const foldChild = depth < options.maxDepth;
  for (const child of Array.from(element.childNodes ?? [])) {
    if (child.nodeType !== 1) continue;
    const result = fold(child, depth + 1, options);
    folded.textLen += result.textLen;
    folded.links += result.links;
    folded.price ||= result.price;
    folded.buy ||= result.buy;
    if (foldChild) children.push(...result.nodes);
  }

  if (name === "a") {
    folded.links += 1;
    if (BUY.test(ownTrim.slice(0, 80))) folded.buy = true;
    // an anchor wrapping blocks (a card link) hoists them; a plain link folds away
    return { ...folded, nodes: children };
  }
  if (INLINE.has(name)) return { ...folded, nodes: children };

  if (HEADING.test(name)) {
    const text = clean(element.textContent ?? "", MAX_TEXT);
    const node: SkeletonNode = { tag: name, attrs: [], kids: [], n: 1 };
    if (options.text === "headings" && text) node.text = text;
    else node.attrs.push(["len", bucket(folded.textLen)]);
    return { ...folded, nodes: [node] };
  }
  if (
    name === "p" ||
    name === "dd" ||
    name === "dt" ||
    name === "figcaption" ||
    (name === "li" && children.length === 0)
  ) {
    if (folded.textLen === 0 && children.length === 0) return { ...folded, nodes: [] };
    const attrs: [string, string | number][] = [["len", bucket(folded.textLen)]];
    if (folded.links) attrs.push(["links", Math.min(folded.links, 99)]);
    if (folded.price) attrs.push(["price", 1]);
    return {
      ...folded,
      nodes: [{ tag: name === "li" ? "li" : "p", attrs, kids: runLength(children), n: 1 }],
    };
  }
  if (name === "img" || name === "picture") {
    return { ...folded, nodes: [{ tag: "img", attrs: [], kids: [], n: 1 }] };
  }
  if (name === "video" || name === "audio" || name === "iframe" || name === "canvas") {
    return { ...folded, nodes: [{ tag: "media", attrs: [["kind", name]], kids: [], n: 1 }] };
  }
  if (
    name === "button" ||
    (name === "input" && element.getAttribute?.("type") !== "hidden") ||
    name === "select" ||
    name === "textarea"
  ) {
    const attrs: [string, string | number][] = [
      [
        "kind",
        name === "input"
          ? (element.getAttribute?.("type") ?? "text").toLowerCase().slice(0, 12)
          : name,
      ],
    ];
    if (BUY.test(ownTrim.slice(0, 80)) || kind === "cart") {
      attrs.push(["buy", 1]);
      folded.buy = true;
    }
    return {
      ...folded,
      nodes: [{ tag: name === "button" ? "button" : "field", attrs, kids: [], n: 1 }],
    };
  }
  if (name === "time") {
    return { ...folded, nodes: [{ tag: "time", attrs: [], kids: [], n: 1 }] };
  }
  if (name === "li" || name === "tr") {
    return {
      ...folded,
      nodes: [{ tag: name === "li" ? "li" : "row", attrs: [], kids: runLength(children), n: 1 }],
    };
  }

  const isList = name === "ul" || name === "ol" || name === "dl";
  const isTable = name === "table";
  const landmark = LANDMARK.has(name) || tag !== name;
  if (!landmark && !kind) {
    // A plain wrapper is hoisted, unless it groups several things: then it is a candidate card
    // (`g`), kept only when it repeats among its siblings (see `runLength`).
    if (children.length >= 2)
      return { ...folded, nodes: [{ tag: "g", attrs: [], kids: runLength(children), n: 1 }] };
    return { ...folded, nodes: children };
  }

  const attrs: [string, string | number][] = [];
  if (kind) attrs.push(["kind", kind]);
  const outTag = isList
    ? "list"
    : isTable
      ? "table"
      : landmark
        ? tag === name
          ? name
          : tag
        : "block";
  if (isList || isTable) {
    const rows = children.reduce((sum, kid) => sum + kid.n, 0);
    attrs.push([isTable ? "rows" : "items", Math.min(rows, 999)]);
  }
  if (name === "form") {
    const fields = countTag(children, "field");
    if (fields) attrs.push(["fields", Math.min(fields, 99)]);
  }
  if (isList || isTable) attrs.push(["links", Math.min(folded.links, 999)]);
  if (folded.buy) attrs.push(["buy", 1]);
  return { ...folded, nodes: [{ tag: outTag, attrs, kids: runLength(children), n: 1 }] };
}

function countTag(nodes: SkeletonNode[], tag: string): number {
  let total = 0;
  for (const node of nodes) total += (node.tag === tag ? node.n : 0) + countTag(node.kids, tag);
  return total;
}

function trim(nodes: SkeletonNode[], budget: Budget): SkeletonNode[] {
  const out: SkeletonNode[] = [];
  for (const node of nodes) {
    if (budget.nodes >= budget.max) {
      budget.dropped += node.n;
      continue;
    }
    budget.nodes += 1;
    out.push({ ...node, kids: trim(node.kids, budget) });
  }
  return out;
}

/** Build the skeleton tree of a parsed document. */
export function buildSkeleton(document: unknown, options: SkeletonOptions = {}): SkeletonNode {
  const stats: PageStats = {
    tel: 0,
    mailto: 0,
    emailText: 0,
    phoneText: 0,
    social: 0,
    map: 0,
    code: 0,
    priceTerms: 0,
    updated: 0,
    wpPost: 0,
    wpPage: 0,
    wpAttachment: 0,
    wpArchive: 0,
    wpProduct: 0,
    commentItems: 0,
    dateClass: 0,
    readMore: 0,
    h1EqTitle: 0,
    titleLen: 0,
  };
  const opts: Context = {
    text: options.text ?? "headings",
    maxNodes: options.maxNodes ?? DEFAULT_MAX_NODES,
    maxDepth: options.maxDepth ?? DEFAULT_MAX_DEPTH,
    stats,
  };
  const doc = document as DomNode & {
    documentElement?: DomNode;
    body?: DomNode;
    querySelector?(s: string): DomNode | null;
  };
  const root = doc.documentElement ?? (doc as DomNode);
  const body = doc.body ?? root;
  const budget = new Budget(opts.maxNodes);
  const result = fold(body, 0, opts);
  const bodyClass = (body.getAttribute?.("class") ?? "").toLowerCase().slice(0, 600);
  if (BODY_POST.test(bodyClass)) stats.wpPost = 1;
  if (BODY_PAGE.test(bodyClass)) stats.wpPage = 1;
  if (BODY_ATTACHMENT.test(bodyClass)) stats.wpAttachment = 1;
  if (BODY_ARCHIVE.test(bodyClass)) stats.wpArchive = 1;
  if (BODY_PRODUCT.test(bodyClass)) stats.wpProduct = 1;
  {
    const title = clean(doc.querySelector?.("title")?.textContent ?? "", 200).toLowerCase();
    const h1 = clean(doc.querySelector?.("h1")?.textContent ?? "", 200).toLowerCase();
    stats.titleLen = bucket(title.length);
    if (title && h1 && (title.includes(h1.slice(0, 40)) || h1.includes(title.slice(0, 40))))
      stats.h1EqTitle = 1;
  }
  const kids = trim(runLength(result.nodes), budget);
  if (budget.dropped) kids.push({ tag: "more", attrs: [["n", budget.dropped]], kids: [], n: 1 });
  const attrs: [string, string | number][] = [];
  const lang = (root.getAttribute?.("lang") ?? "").toLowerCase().slice(0, 8);
  if (lang) attrs.push(["lang", lang]);
  attrs.push(["len", bucket(result.textLen)]);
  for (const [key, value] of Object.entries(stats))
    if (value > 0) attrs.push([key, Math.min(value, 999)]);
  const node: SkeletonNode = { tag: "page", attrs, kids, n: 1 };
  if (opts.text === "headings") {
    const title = doc.querySelector?.("title");
    const text = clean(title?.textContent ?? "", MAX_TEXT);
    if (text) node.text = text;
  }
  return node;
}

const ESCAPES: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const escape = (text: string) => text.replace(/[&<>"]/g, (c) => ESCAPES[c]!);

function xml(node: SkeletonNode, depth: number, out: string[]): void {
  const pad = "  ".repeat(depth);
  const attrs = [...node.attrs, ...(node.n > 1 ? ([["n", node.n]] as [string, number][]) : [])]
    .map(([key, value]) => ` ${key}="${escape(String(value))}"`)
    .join("");
  if (node.tag === "page" && node.text !== undefined) {
    out.push(`${pad}<page${attrs}>`, `${pad}  <title>${escape(node.text)}</title>`);
    for (const kid of node.kids) xml(kid, depth + 1, out);
    out.push(`${pad}</page>`);
    return;
  }
  if (!node.kids.length) {
    out.push(
      node.text !== undefined
        ? `${pad}<${node.tag}${attrs}>${escape(node.text)}</${node.tag}>`
        : `${pad}<${node.tag}${attrs}/>`,
    );
    return;
  }
  out.push(`${pad}<${node.tag}${attrs}>`);
  for (const kid of node.kids) xml(kid, depth + 1, out);
  out.push(`${pad}</${node.tag}>`);
}

function compact(node: SkeletonNode, depth: number, out: string[]): void {
  const attrs = node.attrs
    .map(([key, value]) => ` ${key}=${String(value).replace(/\s+/g, "_")}`)
    .join("");
  const text = node.text !== undefined ? ` ${JSON.stringify(node.text)}` : "";
  out.push(`${"  ".repeat(depth)}${node.tag}${attrs}${node.n > 1 ? ` x${node.n}` : ""}${text}`);
  for (const kid of node.kids) compact(kid, depth + 1, out);
}

export function renderXml(root: SkeletonNode): string {
  const out: string[] = [];
  xml(root, 0, out);
  return out.join("\n");
}

export function renderCompact(root: SkeletonNode): string {
  const out: string[] = [];
  compact(root, 0, out);
  return out.join("\n");
}

/** One call: parsed document to text. */
export function skeleton(
  document: unknown,
  format: "xml" | "compact" = "xml",
  options: SkeletonOptions = {},
): string {
  const tree = buildSkeleton(document, options);
  return format === "xml" ? renderXml(tree) : renderCompact(tree);
}

/** Fixed order and fixed set: this is the tree-model input derived from a skeleton. */
export const SKELETON_FEATURES = [
  "nodes",
  "depth",
  "lenBucket",
  "h1",
  "h2",
  "h3",
  "h4plus",
  "p",
  "pLong",
  "pShort",
  "img",
  "list",
  "listItems",
  "listLinks",
  "table",
  "form",
  "field",
  "button",
  "time",
  "media",
  "group",
  "groupRuns",
  "maxRun",
  "header",
  "nav",
  "main",
  "article",
  "section",
  "aside",
  "footer",
  "kindComments",
  "kindBreadcrumb",
  "kindPagination",
  "kindPrice",
  "kindCart",
  "kindByline",
  "kindShare",
  "kindRelated",
  "kindCookie",
  "kindSearch",
  "kindProduct",
  "price",
  "buy",
  "linkRatio",
  "more",
  "tel",
  "mailto",
  "emailText",
  "phoneText",
  "social",
  "map",
  "code",
  "priceTerms",
  "updated",
  "wpPost",
  "wpPage",
  "wpAttachment",
  "wpArchive",
  "wpProduct",
  "commentItems",
  "dateClass",
  "readMore",
  "h1EqTitle",
  "titleLen",
] as const;

/** Page-level numbers from a skeleton tree, for models that do not read text. */
export function skeletonFeatures(
  root: SkeletonNode,
): Record<(typeof SKELETON_FEATURES)[number], number> {
  const f = Object.fromEntries(SKELETON_FEATURES.map((k) => [k, 0])) as Record<
    (typeof SKELETON_FEATURES)[number],
    number
  >;
  const lenOf = (node: SkeletonNode): number =>
    Number(node.attrs.find(([k]) => k === "len")?.[1] ?? 0);
  const attr = (node: SkeletonNode, key: string): number =>
    Number(node.attrs.find(([k]) => k === key)?.[1] ?? 0);
  const visit = (node: SkeletonNode, depth: number): void => {
    f.nodes += node.n;
    f.depth = Math.max(f.depth, depth);
    f.maxRun = Math.max(f.maxRun, node.n);
    const times = node.n;
    switch (node.tag) {
      case "h1":
        f.h1 += times;
        break;
      case "h2":
        f.h2 += times;
        break;
      case "h3":
        f.h3 += times;
        break;
      case "h4":
      case "h5":
      case "h6":
        f.h4plus += times;
        break;
      case "p":
      case "li": {
        if (node.tag === "p") {
          f.p += times;
          if (lenOf(node) >= 200) f.pLong += times;
          else f.pShort += times;
        }
        break;
      }
      case "img":
        f.img += times;
        break;
      case "list":
        f.list += times;
        f.listItems += attr(node, "items") * times;
        f.listLinks += attr(node, "links") * times;
        break;
      case "table":
        f.table += times;
        break;
      case "form":
        f.form += times;
        break;
      case "field":
        f.field += times;
        break;
      case "button":
        f.button += times;
        break;
      case "time":
        f.time += times;
        break;
      case "media":
        f.media += times;
        break;
      case "g":
        f.group += times;
        f.groupRuns += 1;
        break;
      case "header":
      case "nav":
      case "main":
      case "article":
      case "section":
      case "aside":
      case "footer":
        f[node.tag] += times;
        break;
      case "more":
        f.more += attr(node, "n");
        break;
    }
    const kind = node.attrs.find(([k]) => k === "kind")?.[1];
    if (typeof kind === "string") {
      const key =
        `kind${kind[0]!.toUpperCase()}${kind.slice(1)}` as (typeof SKELETON_FEATURES)[number];
      if (key in f) f[key] += times;
    }
    if (attr(node, "price")) f.price += times;
    if (attr(node, "buy")) f.buy += times;
    for (const kid of node.kids) visit(kid, depth + 1);
  };
  visit(root, 0);
  for (const key of [
    "tel",
    "mailto",
    "emailText",
    "phoneText",
    "social",
    "map",
    "code",
    "priceTerms",
    "updated",
    "wpPost",
    "wpPage",
    "wpAttachment",
    "wpArchive",
    "wpProduct",
    "commentItems",
    "dateClass",
    "readMore",
    "h1EqTitle",
    "titleLen",
  ] as const)
    f[key] = attr(root, key);
  f.lenBucket = lenOf(root);
  f.linkRatio = f.nodes ? Math.round((f.listLinks / f.nodes) * 100) / 100 : 0;
  return f;
}

/** Hashed word buckets for the title and headings: the text channel of the skeleton. */
export const TOKEN_BUCKETS = 256;
const MAX_TOKENS = 60;

function fnv1a(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/**
 * Sorted unique bucket ids of the words in the page title and headings (lowercased letter/number
 * runs of 2-24 characters, first 60 distinct). Empty for a `text: "none"` skeleton: no words, no tokens.
 */
export function skeletonTokens(root: SkeletonNode): number[] {
  const words = new Set<string>();
  const take = (text: string | undefined): void => {
    if (!text) return;
    for (const match of text.toLowerCase().matchAll(/[\p{L}\p{N}]{2,24}/gu)) {
      if (words.size >= MAX_TOKENS) return;
      words.add(match[0]);
    }
  };
  const visit = (node: SkeletonNode): void => {
    if (node.tag === "page" || /^h[1-6]$/.test(node.tag)) take(node.text);
    for (const kid of node.kids) visit(kid);
  };
  visit(root);
  return [...new Set([...words].map((w) => fnv1a(w) % TOKEN_BUCKETS))].sort((a, b) => a - b);
}

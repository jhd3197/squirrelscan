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

function fold(element: DomNode, depth: number, options: Required<SkeletonOptions>): Folded {
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
  const opts: Required<SkeletonOptions> = {
    text: options.text ?? "headings",
    maxNodes: options.maxNodes ?? DEFAULT_MAX_NODES,
    maxDepth: options.maxDepth ?? DEFAULT_MAX_DEPTH,
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
  const kids = trim(runLength(result.nodes), budget);
  if (budget.dropped) kids.push({ tag: "more", attrs: [["n", budget.dropped]], kids: [], n: 1 });
  const attrs: [string, string | number][] = [];
  const lang = (root.getAttribute?.("lang") ?? "").toLowerCase().slice(0, 8);
  if (lang) attrs.push(["lang", lang]);
  attrs.push(["len", bucket(result.textLen)]);
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

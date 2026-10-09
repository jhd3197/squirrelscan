// A deterministic table-of-contents detector (#2307).
//
// The section head cannot call table_of_contents at P >= 0.95 on the
// template-platform pages (2026-10-09: wrappers and anchor bars read as TOC), so
// this is the strict rule tried in its place. A container is a table of
// contents when ALL of these hold:
//
//   1. it holds >= 3 in-page anchors (`href="#id"`) whose target id sits on an
//      h2-h4 heading (the heading itself or an element inside it);
//   2. each of those anchors' text matches its heading's text, normalised,
//      with similarity >= 0.7 (character-bigram Dice);
//   3. it is positioned before or beside the main content: inside a <nav> or
//      <aside>, before the main element in document order, or within the first
//      third of the main element.
//
// The container reported is the lowest common ancestor of the qualifying
// anchors of one list (anchors are grouped by their outermost list ancestor,
// so a nested TOC is one container).

type El = Element & { parentElement: El | null; children: ArrayLike<El> };

const HEADING = new Set(["h2", "h3", "h4"]);
const LIST = new Set(["ul", "ol"]);
export const TOC_MIN_ANCHORS = 3;
export const TOC_MIN_SIMILARITY = 0.7;

/** Lower case, accents and leading numbering ("1.", "2.3", "iv.") stripped,
 *  punctuation dropped, whitespace collapsed. */
export function normaliseHeadingText(text: string): string {
  return text
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/^\s*(?:\d+(?:\.\d+)*|[ivxlc]+)[.):]?\s+/, "")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Dice coefficient over character bigrams; 1 for equal strings. */
export function textSimilarity(a: string, b: string): number {
  if (a === b) return a.length > 0 ? 1 : 0;
  if (a.length < 2 || b.length < 2) return 0;
  const grams = new Map<string, number>();
  for (let i = 0; i < a.length - 1; i++) {
    const g = a.slice(i, i + 2);
    grams.set(g, (grams.get(g) ?? 0) + 1);
  }
  let shared = 0;
  for (let i = 0; i < b.length - 1; i++) {
    const g = b.slice(i, i + 2);
    const n = grams.get(g) ?? 0;
    if (n > 0) {
      shared++;
      grams.set(g, n - 1);
    }
  }
  return (2 * shared) / (a.length - 1 + (b.length - 1));
}

function preorder(root: El): Map<El, number> {
  const order = new Map<El, number>();
  const stack: El[] = [root];
  while (stack.length > 0) {
    const el = stack.pop()!;
    order.set(el, order.size);
    const kids = Array.from(el.children);
    for (let i = kids.length - 1; i >= 0; i--) stack.push(kids[i]!);
  }
  return order;
}

function ancestors(el: El): El[] {
  const out: El[] = [];
  for (let cur: El | null = el; cur; cur = cur.parentElement) out.push(cur);
  return out;
}

function lowestCommonAncestor(els: El[]): El {
  let common = ancestors(els[0]!);
  for (const el of els.slice(1)) {
    const set = new Set(ancestors(el));
    common = common.filter((a) => set.has(a));
  }
  return common[0]!;
}

function within(el: El, names: Set<string>): boolean {
  for (let cur: El | null = el; cur; cur = cur.parentElement)
    if (names.has(cur.localName) || names.has(cur.getAttribute("role") ?? "")) return true;
  return false;
}

/** Table-of-contents containers in a parsed document (usually 0 or 1). */
export function findTableOfContents(document: unknown): Element[] {
  const doc = document as {
    body?: El | null;
    querySelector(s: string): El | null;
    querySelectorAll(s: string): ArrayLike<El>;
  };
  const body = doc.body;
  if (!body) return [];
  // id -> heading, for ids on an h2-h4 or on any element inside one.
  const headingById = new Map<string, El>();
  for (const heading of Array.from(doc.querySelectorAll("h2, h3, h4"))) {
    for (const el of [heading, ...Array.from(heading.querySelectorAll("[id]") as ArrayLike<El>)]) {
      const id = el.getAttribute("id");
      if (id && !headingById.has(id)) headingById.set(id, heading);
    }
  }
  if (headingById.size < TOC_MIN_ANCHORS) return [];

  const groups = new Map<El, El[]>();
  for (const anchor of Array.from(doc.querySelectorAll('a[href^="#"]'))) {
    const href = anchor.getAttribute("href") ?? "";
    let id = href.slice(1);
    try {
      id = decodeURIComponent(id);
    } catch {
      // keep the raw fragment
    }
    const heading = id ? headingById.get(id) : undefined;
    if (!heading || !HEADING.has(heading.localName) || heading.contains(anchor)) continue;
    const similarity = textSimilarity(
      normaliseHeadingText(anchor.textContent ?? ""),
      normaliseHeadingText(heading.textContent ?? ""),
    );
    if (similarity < TOC_MIN_SIMILARITY) continue;
    // Group by the OUTERMOST list ancestor, so a nested TOC stays one group;
    // an anchor in no list groups by its parent.
    let key: El | null = null;
    for (let cur = anchor.parentElement; cur && cur !== body; cur = cur.parentElement)
      if (LIST.has(cur.localName)) key = cur;
    key ??= anchor.parentElement ?? body;
    const group = groups.get(key);
    if (group) group.push(anchor as El);
    else groups.set(key, [anchor as El]);
  }

  const out: Element[] = [];
  // Computed once per page, not per group: a page with thousands of anchor
  // groups must not re-walk the document for each one.
  let order: Map<El, number> | null = null;
  let regionSpan: { start: number; size: number } | null = null;
  const main =
    doc.querySelector("main") ?? doc.querySelector('[role="main"]') ?? doc.querySelector("article");
  for (const anchors of groups.values()) {
    if (anchors.length < TOC_MIN_ANCHORS) continue;
    const container = lowestCommonAncestor(anchors);
    let placed = within(container, new Set(["nav", "aside", "navigation", "complementary"]));
    if (!placed) {
      order ??= preorder(body);
      if (!regionSpan) {
        const region = main ?? body;
        regionSpan = {
          start: order.get(region) ?? 0,
          size: region.querySelectorAll("*").length + 1,
        };
      }
      const at = order.get(container) ?? 0;
      placed = at < regionSpan.start || at - regionSpan.start <= regionSpan.size / 3;
    }
    if (placed) out.push(container);
  }
  return out;
}

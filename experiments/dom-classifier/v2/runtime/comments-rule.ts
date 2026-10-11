// A deterministic comment-section detector (#2307).
//
// On linking pages the section head misses almost every comment thread
// (2026-10-12 Opus check: 29 of 50 CC links sat in one, the head found 1). Two
// reasons: per-comment nodes are cut by the 400-candidate cap on big threads,
// and the comment-list node it does rank first rarely clears the strict 0.96
// threshold. Comment threads are almost always CMS-built and say so in their
// markup, so this rule reads that markup instead.
//
// An element is a comment section when its id or one of its class tokens:
//   1. holds a comment word: `comment` or `comments`, alone or fused with
//      list/box/area/wrap/block/section/thread/form (`commentlist`, `comments-area`,
//      `li-comment-12`, `ast-comment-list`, `wp-block-post-comments`), or is
//      WordPress's reply-form id `respond` or Disqus's `disqus_thread`; and
//   2. holds no stop word (`comments-link`, `comment-count`, `recent-comments`,
//      `has-comments`, `post-meta-comments`: links TO comments, widgets, flags).
// A match that contains the page's <h1> is a page wrapper, not a thread, and is
// refused. A thread inside a reviews container (WooCommerce stores product
// reviews as comments) is `reviews`. Policy: `comments` is the discussion
// thread under the content, or its reply form.

type El = Element & { parentElement: El | null };

export type CommentSection = "comments" | "reviews";

export interface CommentMatch {
  label: CommentSection;
  /** The nearest matching ancestor-or-self. */
  container: Element;
  /** Steps from the start element up to `container` (0 = itself). */
  hops: number;
  /** The id or class token that matched, lowercased, digits folded to N. */
  token: string;
}

const COMMENT_WORD = /^comments?(?:list|box|area|wrap|block|section|thread|form)?$/;
const WHOLE_IDS = new Set(["respond", "disqus_thread"]);
const STOP = new Set([
  "count",
  "counts",
  "counter",
  "link",
  "links",
  "number",
  "num",
  "icon",
  "icons",
  "btn",
  "button",
  "toggle",
  "bubble",
  "total",
  "meta",
  "recent",
  "latest",
  "popular",
  "top",
  "widget",
  "no",
  "has",
  "open",
  "closed",
  "disabled",
  "off",
  "rss",
  "feed",
]);
const REVIEW_WORD = /^(?:product)?reviews?$/;

function words(token: string): string[] {
  return token.split(/[-_]+/).filter(Boolean);
}

/** Does one id or class token name a comment section? Exported for tests. */
export function isCommentToken(token: string, kind: "id" | "class"): boolean {
  const t = token.toLowerCase();
  if (kind === "id" && WHOLE_IDS.has(t)) return true;
  const ws = words(t);
  return ws.some((w) => COMMENT_WORD.test(w)) && !ws.some((w) => STOP.has(w));
}

function isReviewToken(token: string): boolean {
  const ws = words(token.toLowerCase());
  return ws.some((w) => REVIEW_WORD.test(w)) && !ws.some((w) => STOP.has(w));
}

function tokensOf(el: El): Array<[string, "id" | "class"]> {
  const out: Array<[string, "id" | "class"]> = [];
  const id = el.getAttribute("id");
  if (id) out.push([id.trim(), "id"]);
  const cls = el.getAttribute("class");
  if (cls) for (const c of cls.split(/\s+/)) if (c) out.push([c, "class"]);
  return out;
}

const fold = (token: string) => token.toLowerCase().replace(/\d+/g, "N").slice(0, 60);

// One h1 lookup per container per document, however many links share it.
const holdsH1 = new WeakMap<object, boolean>();
function containsH1(el: El): boolean {
  let known = holdsH1.get(el);
  if (known === undefined) {
    known = el.querySelector("h1") !== null;
    holdsH1.set(el, known);
  }
  return known;
}

/** The comment section `start` sits in (itself or an ancestor), or null. */
export function commentSectionOf(start: Element): CommentMatch | null {
  let hops = 0;
  for (let el: El | null = start as El; el; el = el.parentElement, hops++) {
    const tag = el.localName;
    if (tag === "body" || tag === "html") return null;
    const hit = tokensOf(el).find(([token, kind]) => isCommentToken(token, kind));
    if (!hit) continue;
    // A wrapper that holds the page title holds the article too: no ancestor
    // above it can be a thread either.
    if (containsH1(el)) return null;
    let label: CommentSection = "comments";
    for (let up: El | null = el; up && up.localName !== "body"; up = up.parentElement)
      if (tokensOf(up).some(([token]) => isReviewToken(token))) {
        label = "reviews";
        break;
      }
    return { label, container: el, hops, token: fold(hit[0]) };
  }
  return null;
}

/** Does the anchor carry rel="ugc" (Google's user-generated-content marker,
 *  which WordPress puts on comment author links)? */
export function hasUgcRel(anchor: Element): boolean {
  return (anchor.getAttribute("rel") ?? "").toLowerCase().split(/\s+/).includes("ugc");
}

/** Outermost comment-section containers in a parsed document, for tagging a
 *  whole page (a nested match inside one is the same thread). */
export function findCommentSections(document: unknown): Element[] {
  const doc = document as { body?: El | null };
  const body = doc.body;
  if (!body) return [];
  const out: Element[] = [];
  const all = body.querySelectorAll("[id], [class]") as ArrayLike<El>;
  for (const el of Array.from(all)) {
    if (!tokensOf(el).some(([token, kind]) => isCommentToken(token, kind))) continue;
    if (out.some((outer) => outer.contains(el))) continue;
    const match = commentSectionOf(el);
    if (match && match.container === el) out.push(el);
  }
  return out;
}

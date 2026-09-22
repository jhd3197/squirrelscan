/**
 * Whether stored bytes hold a page at all, shared by the page-type queue
 * builder and the runner so both skip the same pages for the same reason.
 */
import { parseHTML } from "../../../packages/parser/src/dom.ts";

/**
 * HTML, or unknown, by the response content type the crawl recorded in
 * `pages.content_type`. A 200 can be a PNG, a zip or a JSON feed, and the
 * content store files those bytes as text/html all the same, so this is the
 * only place the difference survives. A missing type is left to the bytes.
 */
export function isHtmlType(type: string | null | undefined): boolean {
  const essence = type?.split(";")[0]?.trim().toLowerCase();
  return !essence || essence === "text/html" || essence === "application/xhtml+xml";
}

/** Why stored bytes cannot be labelled as a page. */
export type UnusableHtml = "emptyHtml" | "noDocumentElement";

/** Parsing a prefix settles almost every page without a full parse. */
const DOCUMENT_ELEMENT_PREFIX = 8192;

/**
 * Why this HTML has no page to label, or null when it has one.
 *
 * A 200 response with an empty body is stored under sha256("") and decodes to
 * "". A body that is only text, JSON, a doctype or comments parses to a
 * document with no element at all. Neither has a title or prose, so a model
 * call on it spends tokens on nothing; both are skipped by name instead.
 *
 * The prefix parse is sound in one direction only: the tokenizer streams, so an
 * element opened in the first bytes is opened in the whole document, while a
 * tag cut off at the prefix boundary is dropped rather than invented. A prefix
 * with no element therefore falls through to the full parse.
 */
export function unusableHtmlReason(html: string): UnusableHtml | null {
  if (html.trim().length === 0) return "emptyHtml";
  const hasRoot = (text: string) =>
    Boolean((parseHTML(text).document as { documentElement?: unknown }).documentElement);
  if (html.length > DOCUMENT_ELEMENT_PREFIX && hasRoot(html.slice(0, DOCUMENT_ELEMENT_PREFIX)))
    return null;
  return hasRoot(html) ? null : "noDocumentElement";
}

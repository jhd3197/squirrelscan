#!/usr/bin/env bun
/**
 * Text outlines for the link-position check (#2307): one packet per sampled
 * link, for a locked-down LLM judge that never sees our answer.
 *
 * The anchor is found by the nodeId link-position.ts recorded, through the
 * featurizer's own walk (`elementsOut`), so both scripts name the same
 * element. A packet holds: the page URL, title and heading outline; the link's
 * anchor text, target and position in the document; its ancestor path (tag,
 * role, id, class tokens, the span of the document each ancestor covers); the
 * text around it with the link marked; and the other links beside it.
 *
 *   bun run experiments/link-check-packets.ts --in <probe input dir> \
 *     --sample sample.jsonl --out packets.jsonl
 * sample.jsonl rows: {itemId, sha256, url, anchorNodeId, href, anchorText}.
 * Output is PRIVATE (page text). Page content is hostile: parsed, never run.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { featurizeDocument } from "../featurize.ts";
import { decode } from "./link-position.ts";

interface SampleRow {
  itemId: string;
  sha256: string;
  url: string;
  anchorNodeId: number;
  href: string;
  anchorText: string;
}

interface El {
  localName?: string;
  nodeType: number;
  parentElement: El | null;
  firstChild: El | null;
  nextSibling: El | null;
  data?: string;
  textContent?: string | null;
  getAttribute(name: string): string | null;
}

const MAX_PATH = 24;
const MAX_HEADINGS = 15;
const CONTEXT_CHARS = 320;
const MAX_SIBLING_LINKS = 15;

export function clean(text: string, max: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
}

function describe(el: El): string {
  const tag = (el.localName ?? "?").toLowerCase();
  const role = el.getAttribute("role");
  const id = el.getAttribute("id");
  const classes = (el.getAttribute("class") ?? "")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 5)
    .map((c) => c.slice(0, 40));
  return [
    tag,
    role ? `[role=${clean(role, 30)}]` : "",
    id ? `#${clean(id, 40)}` : "",
    classes.length ? `.${classes.join(".")}` : "",
  ].join("");
}

/** What a reader sees for the link: its text, else its accessible name or an
 *  image's alt text. */
function linkText(anchor: El & { querySelector?(s: string): El | null }): string {
  const text = clean(anchor.textContent ?? "", 160);
  if (text) return text;
  const named = anchor.getAttribute("aria-label") ?? anchor.getAttribute("title");
  if (named) return `${clean(named, 160)} (accessible name)`;
  const alt = anchor.querySelector?.("img[alt]")?.getAttribute("alt");
  return alt ? `${clean(alt, 160)} (image alt)` : "";
}

/** Text of `container` with the anchor's own text wrapped in ⟦LINK: …⟧.
 *  Explicit stack: hostile markup can nest deeper than the call stack. */
function markedText(container: El, anchor: El): string {
  const parts: string[] = [];
  const stack: Array<El | " "> = [container];
  while (stack.length > 0) {
    const node = stack.pop()!;
    if (node === " ") {
      parts.push(" ");
      continue;
    }
    if (node === anchor) {
      parts.push(` ⟦LINK: ${linkText(anchor).slice(0, 120)}⟧ `);
      continue;
    }
    if (node.nodeType === 3) {
      parts.push(node.data ?? "");
      continue;
    }
    if (node.nodeType !== 1) continue;
    const tag = (node.localName ?? "").toLowerCase();
    if (tag === "script" || tag === "style" || tag === "template" || tag === "noscript") continue;
    const children: El[] = [];
    for (let child = node.firstChild; child; child = child.nextSibling) children.push(child);
    stack.push(" ");
    for (let k = children.length - 1; k >= 0; k--) stack.push(children[k]!);
  }
  return parts.join("").replace(/\s+/g, " ").trim();
}

export function buildPacket(html: string, row: SampleRow): string | null {
  const { document } = parseHTML(html);
  const elements: unknown[] = [];
  featurizeDocument(document as never, row.url, { elementsOut: elements });
  const anchor = elements[row.anchorNodeId] as El | undefined;
  if (!anchor || (anchor.localName ?? "").toLowerCase() !== "a") return null;

  const total = elements.length;
  const indexOf = new Map<unknown, number>();
  elements.forEach((el, i) => indexOf.set(el, i));
  // Subtree sizes by one reverse fold over the preorder list.
  const parent = elements.map((el) => indexOf.get((el as El).parentElement) ?? -1);
  const size = Array.from({ length: total }, () => 1);
  for (let i = total - 1; i > 0; i--) if (parent[i]! >= 0) size[parent[i]!]! += size[i]!;
  const span = (i: number) =>
    `${Math.round((i / total) * 100)}-${Math.round(((i + size[i]! - 1) / total) * 100)}%`;

  const chain: El[] = [];
  for (let el: El | null = anchor; el; el = el.parentElement) {
    if (indexOf.has(el)) chain.unshift(el);
  }
  let pathLines = chain.map((el, depth) => {
    const i = indexOf.get(el)!;
    const marker = el === anchor ? "   <- THE LINK" : "";
    return `${"  ".repeat(Math.min(depth, 24))}${describe(el)} [spans ${span(i)} of the document]${marker}`;
  });
  if (pathLines.length > MAX_PATH) {
    const keep = MAX_PATH / 2;
    pathLines = [...pathLines.slice(0, keep), "  …", ...pathLines.slice(-keep)];
  }

  // The nearest ancestor with enough text to read, and the links beside it.
  let container: El = anchor;
  for (let el: El | null = anchor.parentElement; el; el = el.parentElement) {
    container = el;
    if (clean(el.textContent ?? "", 400).length >= 80 + clean(anchor.textContent ?? "", 400).length)
      break;
  }
  const text = markedText(container, anchor);
  const at = text.indexOf("⟦LINK:");
  const start = Math.max(0, at - CONTEXT_CHARS);
  const end = Math.min(text.length, text.indexOf("⟧", at) + 1 + CONTEXT_CHARS);
  const context = `${start > 0 ? "… " : ""}${text.slice(start, end)}${end < text.length ? " …" : ""}`;
  const siblings: string[] = [];
  const ci = indexOf.get(container);
  if (ci !== undefined) {
    for (let i = ci; i < ci + size[ci]! && siblings.length < MAX_SIBLING_LINKS; i++) {
      const el = elements[i] as El;
      if (el === anchor || (el.localName ?? "").toLowerCase() !== "a") continue;
      const t = clean(el.textContent ?? "", 60);
      if (t) siblings.push(t);
    }
  }

  const headings: string[] = [];
  for (const raw of elements) {
    const el = raw as El;
    const tag = (el.localName ?? "").toLowerCase();
    if (tag !== "h1" && tag !== "h2" && tag !== "h3") continue;
    const t = clean(el.textContent ?? "", 80);
    if (t) headings.push(`${tag}: ${t}`);
    if (headings.length >= MAX_HEADINGS) break;
  }
  const title = clean((document as unknown as { title?: string }).title ?? "", 120);
  const ai = indexOf.get(anchor)!;

  return [
    "PAGE",
    `url: ${clean(row.url, 200)}`,
    `title: ${title || "(none)"}`,
    `document size: ${total} elements`,
    "heading outline:",
    ...(headings.length ? headings.map((h) => `  ${h}`) : ["  (none)"]),
    "",
    "LINK",
    `anchor text: ${linkText(anchor) || "(empty: image or icon link)"}`,
    `target: ${clean(row.href, 200)}`,
    `position: element ${ai + 1} of ${total} (${Math.round((ai / total) * 100)}% into the document)`,
    "",
    "ANCESTOR PATH (outermost first)",
    ...pathLines,
    "",
    `TEXT AROUND THE LINK (inside ${describe(container)})`,
    clean(context, 2 * CONTEXT_CHARS + 200),
    "",
    `OTHER LINKS IN THAT ELEMENT (${siblings.length}${siblings.length === MAX_SIBLING_LINKS ? "+" : ""})`,
    ...(siblings.length ? siblings.map((s) => `  - ${s}`) : ["  (none)"]),
  ].join("\n");
}

function argument(name: string): string {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? undefined : Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

if (import.meta.main) {
  const input = argument("--in");
  const rows = readFileSync(argument("--sample"), "utf8")
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l) as SampleRow);
  const out: string[] = [];
  let missing = 0;
  for (const row of rows) {
    const html = decode(gunzipSync(readFileSync(`${input}/blobs/${row.sha256}.html.gz`)));
    const packet = buildPacket(html, row);
    if (packet === null) {
      missing++;
      continue;
    }
    out.push(JSON.stringify({ itemId: row.itemId, packet }));
  }
  writeFileSync(argument("--out"), `${out.join("\n")}\n`);
  console.log(JSON.stringify({ packets: out.length, missing }));
}

#!/usr/bin/env bun
/**
 * Score runtime/toc-rule.ts against Jev's table_of_contents labels (#2307).
 *
 * Labels are node-level (the featurizer's preorder nodeId) on each page's Jev
 * shortlist, under two granularities: v2 (tightest container) and v3 (region
 * level). For every page given (train/val only: the caller never passes a
 * holdout page) the rule's containers are mapped to nodeIds, and per corpus
 * slice:
 *   precision  rule containers that carry a label, TOC / labelled
 *   coverage   TOC-labelled nodes the rule names exactly; `near` also counts a
 *              rule container within two levels above or below the node
 * Unlabelled containers are counted, not judged.
 *
 *   bun run experiments/toc-rule-eval.ts --pages pages.jsonl --v2 a.jsonl,b.jsonl \
 *     --v3 c.jsonl --store x.db:y.db --out report.json
 * pages.jsonl rows: {pageId, slice}.
 */
import { Database } from "bun:sqlite";
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { SKIP_SUBTREE } from "../featurize.ts";
import { findTableOfContents } from "../runtime/toc-rule.ts";

function argument(name: string): string {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? undefined : Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

type El = Element & { parentElement: El | null; children: ArrayLike<El> };
type Labels = Map<string, Map<number, string>>;

function featurizerOrder(document: { body?: unknown }): El[] {
  const order: El[] = [];
  const body = document.body as El | null | undefined;
  const stack: El[] = body ? [body] : [];
  while (stack.length > 0) {
    const el = stack.pop()!;
    if (SKIP_SUBTREE.has(el.localName)) continue;
    order.push(el);
    const kids = Array.from(el.children).filter((c) => !SKIP_SUBTREE.has(c.localName));
    for (let i = kids.length - 1; i >= 0; i--) stack.push(kids[i]!);
  }
  return order;
}

function readLabels(paths: string): Labels {
  const out: Labels = new Map();
  for (const path of paths.split(",").filter(Boolean))
    for (const line of readFileSync(path, "utf8").split("\n")) {
      if (!line) continue;
      const row = JSON.parse(line) as { pageId: string; nodeId: number; section: string | null };
      if (!row.section) continue;
      let page = out.get(row.pageId);
      if (!page) out.set(row.pageId, (page = new Map()));
      page.set(row.nodeId, row.section);
    }
  return out;
}

function near(a: El, b: El): boolean {
  let cur: El | null = a;
  for (let d = 0; d <= 2 && cur; d++, cur = cur.parentElement) if (cur === b) return true;
  cur = b;
  for (let d = 0; d <= 2 && cur; d++, cur = cur.parentElement) if (cur === a) return true;
  return false;
}

const pages = readFileSync(argument("--pages"), "utf8")
  .split("\n")
  .filter(Boolean)
  .map((l) => JSON.parse(l) as { pageId: string; slice: string });
const truths = { v2: readLabels(argument("--v2")), v3: readLabels(argument("--v3")) };
const stores = argument("--store")
  .split(":")
  .filter(Boolean)
  .map((p) => new Database(p, { readonly: true }));
const selects = stores.map((db) =>
  db.query<{ content: Uint8Array }, [string, string]>(
    "SELECT content FROM content WHERE hash >= ? AND hash < ? LIMIT 2",
  ),
);
function htmlFor(prefix: string): string | null {
  for (const select of selects) {
    const rows = select.all(prefix, `${prefix}g`);
    if (rows.length === 1) return gunzipSync(rows[0]!.content).toString("utf8");
  }
  return null;
}

type Tally = {
  pages: number;
  pagesFired: number;
  fires: number;
  unlabelled: number;
  judged: number;
  correct: number;
  tocNodes: number;
  hitExact: number;
  hitNear: number;
};
const blank = (): Tally => ({
  pages: 0,
  pagesFired: 0,
  fires: 0,
  unlabelled: 0,
  judged: 0,
  correct: 0,
  tocNodes: 0,
  hitExact: 0,
  hitNear: 0,
});
const tally: Record<string, Tally> = {};
const misses: Array<Record<string, unknown>> = [];
let processed = 0;
for (const { pageId, slice } of pages) {
  const html = htmlFor(pageId);
  if (!html) continue;
  let document: unknown;
  try {
    document = parseHTML(html).document;
  } catch {
    continue;
  }
  const order = featurizerOrder(document as { body?: unknown });
  const index = new Map<El, number>(order.map((el, i) => [el, i]));
  const containers = findTableOfContents(document) as El[];
  // A container the featurizer does not visit maps to its nearest visited ancestor.
  const fired = containers.map((el) => {
    for (let cur: El | null = el; cur; cur = cur.parentElement)
      if (index.has(cur)) return { el: cur, node: index.get(cur)! };
    return null;
  });
  for (const [granularity, labels] of Object.entries(truths)) {
    const pageLabels = labels.get(pageId);
    if (!pageLabels) continue;
    const key = `${slice}/${granularity}`;
    const t = (tally[key] ??= blank());
    t.pages++;
    if (fired.some(Boolean)) t.pagesFired++;
    for (const f of fired) {
      if (!f) continue;
      t.fires++;
      const label = pageLabels.get(f.node);
      if (label === undefined) t.unlabelled++;
      else {
        t.judged++;
        if (label === "table_of_contents") t.correct++;
        else if (misses.length < 200)
          misses.push({ pageId, granularity, node: f.node, label, tag: f.el.localName });
      }
    }
    for (const [node, label] of pageLabels) {
      if (label !== "table_of_contents") continue;
      t.tocNodes++;
      if (fired.some((f) => f && f.node === node)) t.hitExact++;
      const el = order[node];
      if (el && fired.some((f) => f && near(f.el, el))) t.hitNear++;
    }
  }
  if (++processed % 2000 === 0) console.log(JSON.stringify({ processed }));
}
const summary = Object.fromEntries(
  Object.entries(tally).map(([key, t]) => [
    key,
    {
      ...t,
      precision: t.judged ? Number((t.correct / t.judged).toFixed(4)) : null,
      coverageExact: t.tocNodes ? Number((t.hitExact / t.tocNodes).toFixed(4)) : null,
      coverageNear: t.tocNodes ? Number((t.hitNear / t.tocNodes).toFixed(4)) : null,
    },
  ]),
);
writeFileSync(argument("--out"), `${JSON.stringify({ summary, misses }, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));

/**
 * Text packets for the Opus teacher, built from labeller captures (#2307).
 *
 * The gold sets hold captures, not stored HTML, so the page and node packets
 * come from the capture's node walk: the capture-sourced packet revision Jev's
 * capture path uses (tag, role, ancestor tags, position, children, subtree,
 * size, own text; no attributes). Plus the h1-h3 outline, from the same nodes.
 *
 *   bun run teacher-packets.ts --set <dom-classifier-data>/gold-v1
 *
 * Reads <set>/teacher/items.jsonl and <set>/captures; writes <set>/teacher/packets.jsonl,
 * keyed by capture page id, nodes keyed by capture node id. PRIVATE output.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { CapturedPage } from "../../labeler/types.ts";
import {
  cleanText,
  nodeEvidence,
  pageEvidence,
  serializeNodePacket,
  serializePagePacket,
} from "../../model-suggestions/jev-v5.ts";

const index = Bun.argv.indexOf("--set");
if (index < 0 || !Bun.argv[index + 1]) throw new Error("missing --set");
const set = Bun.argv[index + 1]!;
const MAX_OUTLINE = 30;

type Item = { pageId: string; nodeId: string | null };
const items = readFileSync(join(set, "teacher", "items.jsonl"), "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line) as Item);
const byPage = new Map<string, Set<string>>();
for (const item of items) {
  const nodes = byPage.get(item.pageId) ?? new Set<string>();
  if (item.nodeId) nodes.add(item.nodeId);
  byPage.set(item.pageId, nodes);
}
const rows: string[] = [];
const totals = { pages: 0, nodes: 0, missingCapture: 0, missingNode: 0 };
for (const [pageId, nodes] of byPage) {
  const path = join(set, "captures", `${pageId}.json`);
  if (!existsSync(path)) {
    totals.missingCapture++;
    continue;
  }
  const page = JSON.parse(readFileSync(path, "utf8")) as CapturedPage;
  const outline = page.nodes
    .filter((node) => ["h1", "h2", "h3"].includes(node.tag.toLowerCase()))
    .map((node) => `${node.tag.toLowerCase()}: ${cleanText(node.text ?? "", 100)}`)
    .filter((line) => line.length > 4)
    .slice(0, MAX_OUTLINE);
  const packets: Record<string, string> = {};
  for (const nodeId of nodes) {
    const node = page.nodes.find((candidate) => candidate.id === nodeId);
    if (!node) {
      totals.missingNode++;
      continue;
    }
    packets[nodeId] = serializeNodePacket(nodeEvidence(page, node));
    totals.nodes++;
  }
  rows.push(
    JSON.stringify({
      pageId,
      pagePacket: serializePagePacket(pageEvidence(page)),
      outline,
      nodes: packets,
    }),
  );
  totals.pages++;
}
writeFileSync(join(set, "teacher", "packets.jsonl"), `${rows.join("\n")}\n`, { mode: 0o600 });
console.log(JSON.stringify({ set, ...totals }));

/**
 * Prepare replay jobs and text packets for the visual adjudicator (#2307).
 *
 * For every page an item names: read the STORED html, mark each section target
 * by its featurizer index (the attribute the replay locates), and build the
 * bounded text the adjudicator reads beside the screenshot: the same page and
 * node packets Jev was given, plus a heading outline. Nothing renders here; the
 * jobs are rendered off this machine by `modal_render.py`.
 *
 *   bun run prepare-render.ts --items <items.jsonl> --stores a.db:b.db --out <dir>
 *
 * Writes <out>/jobs/<pageId>.html.gz, <out>/jobs.jsonl and <out>/packets.jsonl.
 * PRIVATE output (urls, page bytes, page text): the private data root only.
 */
import { Database } from "bun:sqlite";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";
import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { shape } from "../../extract-dom.ts";
import { REPLAY_MARKER } from "../../labeler/replay-capture.ts";
import { featurizerOrder } from "../../model-suggestions/featurizer-order.ts";
import {
  cleanText,
  nodeEvidenceFromHtml,
  pageEvidenceFromHtml,
  serializeNodePacket,
  serializePagePacket,
  type ExtractedCandidate,
} from "../../model-suggestions/jev-v5.ts";

const arg = (name: string) => {
  const index = Bun.argv.indexOf(name);
  if (index < 0 || !Bun.argv[index + 1]) throw new Error(`missing ${name}`);
  return Bun.argv[index + 1]!;
};
const itemsPath = arg("--items");
const out = arg("--out");
const stores = arg("--stores")
  .split(":")
  .map((path) => new Database(path, { readonly: true }));
const selects = stores.map((db) =>
  db.query<{ content: Uint8Array }, [string, string]>(
    "select content from content where hash >= ? and hash < ? limit 2",
  ),
);
function htmlFor(pageId: string): string | null {
  for (const select of selects) {
    const rows = select.all(pageId, `${pageId}g`);
    if (rows.length === 1) return new TextDecoder().decode(gunzipSync(rows[0]!.content));
  }
  return null;
}

/** Builders whose pages are client-rendered: without scripts there is nothing to judge. */
const SCRIPT_PLATFORMS = new Set(["wix", "framer"]);
const MAX_OUTLINE = 30;

type Item = {
  axis: "page" | "section";
  pageId: string;
  nodeId?: number;
  url: string;
  platform: string;
};
const items = readFileSync(itemsPath, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line) as Item);
const byPage = new Map<string, { url: string; platform: string; targets: Set<number> }>();
for (const item of items) {
  const page = byPage.get(item.pageId) ?? {
    url: item.url,
    platform: item.platform,
    targets: new Set(),
  };
  if (item.axis === "section" && item.nodeId !== undefined) page.targets.add(item.nodeId);
  byPage.set(item.pageId, page);
}

mkdirSync(join(out, "jobs"), { recursive: true, mode: 0o700 });
const jobsPath = join(out, "jobs.jsonl");
const packetsPath = join(out, "packets.jsonl");
const jobs: string[] = [];
const packets: string[] = [];
const totals = { pages: 0, missingHtml: 0, targets: 0, missingTargets: 0 };
type El = {
  localName: string;
  textContent: string | null;
  setAttribute(name: string, value: string): void;
  querySelectorAll(selector: string): ArrayLike<El>;
};

for (const [pageId, page] of byPage) {
  const html = htmlFor(pageId);
  if (!html) {
    totals.missingHtml++;
    continue;
  }
  const { document } = parseHTML(html);
  const order = featurizerOrder(document as never) as unknown as El[];
  const outline = Array.from((document as unknown as El).querySelectorAll("h1, h2, h3"))
    .map((heading) => `${heading.localName}: ${cleanText(heading.textContent ?? "", 100)}`)
    .filter((line) => line.length > 4)
    .slice(0, MAX_OUTLINE);
  const nodes: Record<string, { packet: string; tag: string; text: string }> = {};
  for (const n of [...page.targets].sort((a, b) => a - b)) {
    totals.targets++;
    const element = order[n];
    if (!element) {
      totals.missingTargets++;
      continue;
    }
    // Packet first: the marker attribute must not leak into the attribute line.
    const candidate = shape(element as never, page.url) as unknown as ExtractedCandidate;
    nodes[String(n)] = {
      packet: serializeNodePacket(nodeEvidenceFromHtml(candidate, `${pageId}#${n}`)),
      tag: element.localName,
      text: (element.textContent ?? "").replace(/\s+/g, " ").trim().slice(0, 60).toLowerCase(),
    };
  }
  for (const n of Object.keys(nodes)) order[Number(n)]!.setAttribute(REPLAY_MARKER, n);
  const marked = Object.keys(nodes).length
    ? (document as unknown as { toString(): string }).toString()
    : html;
  writeFileSync(join(out, "jobs", `${pageId}.html.gz`), gzipSync(marked), { mode: 0o600 });
  const pageEvidence = pageEvidenceFromHtml(html, page.url, pageId);
  jobs.push(
    JSON.stringify({
      pageId,
      url: page.url,
      platform: page.platform,
      scripts: SCRIPT_PLATFORMS.has(page.platform) ? "on" : "policy",
      markers: Object.keys(nodes),
      fingerprints: Object.fromEntries(
        Object.entries(nodes).map(([n, node]) => [n, { tag: node.tag, text: node.text }]),
      ),
    }),
  );
  packets.push(
    JSON.stringify({
      pageId,
      pagePacket: serializePagePacket(pageEvidence),
      outline,
      nodes: Object.fromEntries(Object.entries(nodes).map(([n, node]) => [n, node.packet])),
    }),
  );
  totals.pages++;
  if (totals.pages % 100 === 0) console.log(JSON.stringify({ stage: "progress", ...totals }));
}
writeFileSync(jobsPath, `${jobs.join("\n")}\n`, { mode: 0o600 });
writeFileSync(packetsPath, `${packets.join("\n")}\n`, { mode: 0o600 });
for (const db of stores) db.close();
console.log(
  JSON.stringify({ stage: "complete", ...totals, jobsPath, existed: existsSync(jobsPath) }),
);

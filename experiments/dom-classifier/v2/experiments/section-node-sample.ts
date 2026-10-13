#!/usr/bin/env bun
/**
 * Natural sample of candidate section nodes from corpus pages, with text packets (#2307).
 *
 *   bun run experiments/section-node-sample.ts --pages ids.jsonl --corpus <page-corpus> --urls url-paths.json \
 *       --out <dir> [--per-page 6] [--seed 7]
 *
 * For each page (ids.jsonl rows `{p}`, p = first 16 hex of the content sha256) the featurizer's candidate walk
 * picks the nodes; up to --per-page of them are sampled (landmarks and repeated-sibling containers get a share,
 * the rest is random among nodes with some content). Output, in the shape adjudicate_cli.py --modality text reads:
 *   <out>/items.jsonl    {itemId, axis: "section", pageId, nodeId, url, platform, lang, geometry}
 *   <out>/packets.jsonl  {pageId, pagePacket, outline, nodes: {nodeId: "runtime-dom-html-v1 ..."}}
 * The packet format is the one the labeller captures use (tag, role, ancestors, position, children, subtree, class
 * and id tokens, own text), built here from raw HTML so any corpus page can be labelled without a screenshot.
 * Words appear in the output: PRIVATE, never commit.
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { featurizeDocument } from "../featurize.ts";
import { decode } from "./link-position.ts";

function arg(name: string, fallback?: string): string {
  const i = Bun.argv.indexOf(name);
  const value = i < 0 ? fallback : Bun.argv[i + 1];
  if (value === undefined) throw new Error(`${name} is required`);
  return value;
}
const corpus = arg("--corpus");
const out = arg("--out");
const perPage = Number(arg("--per-page", "6"));
let seed = Number(arg("--seed", "7"));
const rand = () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296);
const urlPaths: Record<string, string> = JSON.parse(readFileSync(arg("--urls"), "utf8"));
mkdirSync(out, { recursive: true });

const ids = readFileSync(arg("--pages"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l).p as string);
const sha = new Map<string, string>();
const wanted = new Set(ids);
for (const line of readFileSync(`${corpus}/pages/manifest.jsonl`, "utf8").split("\n")) {
  if (!line) continue;
  try {
    const row = JSON.parse(line);
    const full = row.raw?.sha256 as string | undefined;
    if (full && wanted.has(full.slice(0, 16))) sha.set(full.slice(0, 16), full);
  } catch {
    // half-written last line
  }
}

interface El {
  localName?: string;
  tagName?: string;
  parentElement?: El | null;
  children?: ArrayLike<El>;
  getAttribute?(n: string): string | null;
  textContent?: string | null;
  querySelectorAll?(s: string): ArrayLike<El>;
}
const lower = (el: El) => (el.localName ?? el.tagName ?? "").toLowerCase();
const clean = (text: string | null | undefined, n: number) => (text ?? "").replace(/\s+/g, " ").trim().slice(0, n);
const tokens = (value: string | null | undefined) =>
  (value ?? "").toLowerCase().split(/[\s_]+/).filter((t) => t.length > 1).slice(0, 8).join(",") || "none";

function packetFor(el: El): string {
  const tag = lower(el);
  const parent = el.parentElement;
  const siblings = parent ? Array.from(parent.children ?? []) : [];
  const index = siblings.indexOf(el) + 1;
  const ancestors: string[] = [];
  for (let a = el.parentElement, i = 0; a && i < 4; a = a.parentElement, i++) ancestors.unshift(lower(a));
  const kids = Array.from(el.children ?? []);
  const childTags = [...new Set(kids.map(lower))].slice(0, 6).join(",") || "none";
  const counts = new Map<string, number>();
  for (const d of Array.from(el.querySelectorAll?.("*") ?? []).slice(0, 400)) counts.set(lower(d), (counts.get(lower(d)) ?? 0) + 1);
  const subtree = [...counts].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([t, n]) => `${t}*${n}`).join(",") || "none";
  const links = Array.from(el.querySelectorAll?.("a") ?? []).length;
  const same = siblings.filter((s) => lower(s) === tag).length;
  return [
    "runtime-dom-html-v1",
    `tag=${tag}`,
    `role=${clean(el.getAttribute?.("role"), 20) || "none"}`,
    `ancestors=${ancestors.join(">") || "none"}`,
    `position=${index} of ${siblings.length}`,
    `children=${childTags}`,
    `subtree=${subtree}`,
    "attributes=none",
    "src=none",
    "media=none",
    `links=${links}`,
    `repeatedSiblings=${same}`,
    `classTokens=${tokens(el.getAttribute?.("class"))}`,
    `idTokens=${tokens(el.getAttribute?.("id"))}`,
    `text=${clean(el.textContent, 300) || "[empty]"}`,
  ].join("\n");
}

const items: string[] = [];
const packets: string[] = [];
let pages = 0;
let nodesTotal = 0;
for (const id of ids) {
  const full = sha.get(id);
  if (!full) continue;
  try {
    const html = decode(gunzipSync(readFileSync(`${corpus}/pages/blobs/${full}.html.gz`)).subarray(0, 4_000_000));
    const { document } = parseHTML(html);
    const elements: unknown[] = [];
    const featurized = featurizeDocument(document as never, "https://x.invalid" + (urlPaths[id] ?? "/"), { elementsOut: elements });
    const candidates = featurized.nodes.filter((n) => {
      const el = elements[n.nodeId] as El | undefined;
      return el && clean(el.textContent, 40).length >= 10;
    });
    if (candidates.length === 0) continue;
    const landmark = candidates.filter((n) => ["nav", "aside", "footer", "header", "form", "section"].includes(n.tag) || n.role);
    const rest = candidates.filter((n) => !landmark.includes(n));
    const pick = (pool: typeof candidates, n: number) => {
      const copy = [...pool];
      const chosen: typeof candidates = [];
      while (chosen.length < n && copy.length) chosen.push(copy.splice(Math.floor(rand() * copy.length), 1)[0]!);
      return chosen;
    };
    const chosen = [...pick(landmark, Math.ceil(perPage / 3)), ...pick(rest, perPage)].slice(0, perPage);
    const nodePackets: Record<string, string> = {};
    for (const node of chosen) {
      const el = elements[node.nodeId] as El;
      nodePackets[String(node.nodeId)] = packetFor(el);
      items.push(
        JSON.stringify({
          itemId: `sec-${id}-${node.nodeId}`,
          axis: "section",
          pageId: id,
          nodeId: String(node.nodeId),
          url: `https://unknown.invalid${urlPaths[id] ?? "/"}`,
          platform: null,
          lang: "unknown",
          geometry: null,
        }),
      );
      nodesTotal++;
    }
    const title = clean(document.querySelector("title")?.textContent, 160);
    const headings = Array.from(document.querySelectorAll("h1,h2,h3"))
      .slice(0, 30)
      .map((h: El) => `${lower(h)}: ${clean(h.textContent, 100)}`)
      .filter((l: string) => l.length > 4);
    packets.push(
      JSON.stringify({
        pageId: id,
        pagePacket: `dom-page-text-v2\ntitle=${title}\npath=${urlPaths[id] ?? "/"}\ntext=${clean(document.body?.textContent, 600)}`,
        outline: headings,
        nodes: nodePackets,
      }),
    );
    pages++;
  } catch {
    // unreadable page
  }
}
writeFileSync(`${out}/items.jsonl`, items.join("\n") + "\n");
writeFileSync(`${out}/packets.jsonl`, packets.join("\n") + "\n");
console.log(JSON.stringify({ pages, items: nodesTotal, requested: ids.length }));

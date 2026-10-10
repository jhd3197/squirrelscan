#!/usr/bin/env bun
/**
 * Link-position probe (#2307): where on the page does each outbound link sit?
 *
 * For every page it runs the runtime (`classifyDocument`, the same contract the
 * CLI would get) and maps each link to
 *   - its nearest ancestor (or itself) with a confident REGION: a markup answer
 *     or a model answer over its class threshold; `unknown` when none is;
 *   - separately, its nearest ancestor with a SECTION purpose.
 * The element-to-nodeId map comes from the featurizer's own walk
 * (`elementsOut`), so it cannot drift from the ids the labels carry.
 *
 *   bun run experiments/link-position.ts --in <dir> --artifact classifier-v2.json \
 *     --out <dir> [--budget-ms 200] [--limit N]
 *
 * Input dir: `manifest.jsonl`, one page per row, and `blobs/<sha256>.html.gz`.
 *   {sha256, url, source?, year_bin?, tld_group?, state?, links?: [{href, anchor_text?, link_type?}]}
 * With `links`, those links are located on the page (by resolved href, then
 * anchor text). Without, every <a href> to another registrable domain is a
 * link. `url` is the page's original URL; Wayback-rewritten hrefs are unwrapped.
 *
 * Output (private: it holds URLs and anchor text): links.jsonl, pages.jsonl,
 * summary.json and summary.md, aggregated per source x year_bin x tld_group x
 * state. Page content is hostile: it is parsed, never executed or fetched.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { loadavg } from "node:os";
import { gunzipSync } from "node:zlib";

import { parse as parseDomain } from "tldts";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import {
  classifyDocument,
  tryLoadModel,
  type NodePrediction,
  type PageClassification,
} from "../runtime/predict.ts";

export interface ManifestLink {
  href: string;
  anchor_text?: string | null;
  link_type?: string | null;
}

export interface ManifestRow {
  sha256: string;
  url: string;
  source?: string;
  year_bin?: string | number;
  tld_group?: string;
  state?: string;
  links?: ManifestLink[] | null;
}

export type RegionGroup =
  | "chrome"
  | "content_block"
  | "main_content"
  | "dialog"
  | "other"
  | "unknown";

export interface LinkPosition {
  /** Index in the manifest's `links`, or among the page's outbound anchors. */
  i: number;
  href: string;
  anchorText: string;
  linkType: string | null;
  /** False when a manifest link was not found on the page. */
  matched: boolean;
  /** How many anchors carried the same resolved href. */
  candidates: number;
  /** How a manifest link was tied to its anchor: by href and anchor text, by
   *  href alone, or null (not found, or not a manifest link). */
  matchedBy: "text" | "href" | null;
  region: string;
  regionGroup: RegionGroup;
  regionSource: string | null;
  regionConfidence: number | null;
  regionNodeId: number | null;
  regionTag: string | null;
  /** Element steps from the anchor up to the region node (0 = the anchor). */
  regionHops: number | null;
  section: string | null;
  sectionConfidence: number | null;
  sectionNodeId: number | null;
  sectionHops: number | null;
  /** nodeId of the anchor itself, for building an outline around it. */
  anchorNodeId: number | null;
}

const CHROME = new Set(["site_header", "navigation", "footer", "sidebar"]);

export function regionGroup(label: string): RegionGroup {
  if (CHROME.has(label)) return "chrome";
  if (label === "content_block" || label === "main_content" || label === "dialog") return label;
  if (label === "unknown") return "unknown";
  return "other";
}

const WAYBACK = /^(?:https?:\/\/web\.archive\.org)?\/web\/\d{1,14}[a-z_]{0,4}\/(?=https?:|\/\/)/i;

/** Strip a Wayback Machine rewrite prefix, leaving the original URL. */
export function unwrapWayback(href: string): string {
  return href.replace(WAYBACK, "");
}

/** The URL relative hrefs resolve against: the page's `<base href>` when it is
 *  a usable http(s) URL, else the page URL. */
export function baseUrlOf(pageUrl: string, baseHref?: string | null): string {
  if (!baseHref) return pageUrl;
  return resolveHref(baseHref, pageUrl) ?? pageUrl;
}

/** Absolute http(s) URL for an href on a page, or null for anything else. */
export function resolveHref(href: string, pageUrl: string): string | null {
  const raw = unwrapWayback(href.trim());
  if (!raw || raw.startsWith("#")) return null;
  let url: URL;
  try {
    url = new URL(raw, pageUrl);
  } catch {
    return null;
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return null;
  // A relative href on a Wayback page resolves against archive.org: unwrap again.
  const unwrapped = unwrapWayback(url.href);
  if (unwrapped !== url.href) {
    try {
      url = new URL(unwrapped);
    } catch {
      return null;
    }
  }
  return url.href;
}

/** Comparison key: scheme-less, lower-case host without www., no fragment,
 *  no trailing slash. */
export function hrefKey(href: string): string {
  try {
    const url = new URL(href);
    const host = url.hostname.toLowerCase().replace(/^www\./, "");
    let end = url.pathname.length;
    while (end > 0 && url.pathname.charCodeAt(end - 1) === 47) end--;
    const path = url.pathname.slice(0, end);
    return `${host}${path}${url.search}`;
  } catch {
    return href.trim().toLowerCase();
  }
}

export function registrableDomain(url: string): string {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return parseDomain(host, { allowPrivateDomains: true }).domain ?? host;
  } catch {
    return "";
  }
}

/** Coarse TLD group for pages whose manifest does not carry one. */
export function tldGroup(url: string): string {
  let suffix = "";
  try {
    suffix = parseDomain(new URL(url).hostname, { allowPrivateDomains: false }).publicSuffix ?? "";
  } catch {
    return "other";
  }
  const last = suffix.split(".").pop() ?? "";
  if (last === "com") return "com";
  if (last === "org" || last === "net") return "org_net";
  if (/^[a-z]{2}$/.test(last)) return "cc";
  return "other";
}

function normaliseText(text: string): string {
  return text.replace(/\s+/g, " ").trim().toLowerCase();
}

interface DomElement {
  localName?: string;
  parentElement?: DomElement | null;
  textContent?: string | null;
  getAttribute(name: string): string | null;
}

/**
 * Locate links on a classified page. `elements` is the walk's element list
 * (index = nodeId) from `classifyDocument(..., {elementsOut})`. `manifestLinks`
 * as an array (even empty) means "these links only"; null or absent means every
 * outbound anchor. `baseHref` is the page's `<base href>`, if any.
 */
export function linkPositions(
  pageUrl: string,
  result: PageClassification,
  elements: readonly unknown[],
  manifestLinks?: readonly ManifestLink[] | null,
  baseHref?: string | null,
): LinkPosition[] {
  const base = baseUrlOf(pageUrl, baseHref);
  const idOf = new Map<unknown, number>();
  for (let i = 0; i < elements.length; i++) idOf.set(elements[i], i);
  const byId = new Map<number, NodePrediction>();
  for (const node of result.nodes) byId.set(node.nodeRef.nodeId, node);

  // Every walked anchor with an http(s) target, keyed by resolved href.
  const anchors = new Map<string, Array<{ el: DomElement; href: string; text: string }>>();
  const order: Array<{ el: DomElement; href: string; text: string }> = [];
  for (const raw of elements) {
    const el = raw as DomElement;
    if ((el.localName ?? "").toLowerCase() !== "a") continue;
    const attr = el.getAttribute("href");
    if (attr === null) continue;
    const href = resolveHref(attr, base);
    if (href === null) continue;
    const entry = { el, href, text: normaliseText(el.textContent ?? "") };
    const key = hrefKey(href);
    const list = anchors.get(key);
    if (list) list.push(entry);
    else anchors.set(key, [entry]);
    order.push(entry);
  }

  const out: LinkPosition[] = [];
  const locate = (
    i: number,
    entry: { el: DomElement; href: string; text: string } | null,
    href: string,
    anchorText: string,
    linkType: string | null,
    candidates: number,
    matchedBy: "text" | "href" | null = null,
  ): void => {
    const row: LinkPosition = {
      i,
      href,
      anchorText: anchorText.slice(0, 200),
      linkType,
      matched: entry !== null,
      candidates,
      matchedBy,
      region: "unknown",
      regionGroup: "unknown",
      regionSource: null,
      regionConfidence: null,
      regionNodeId: null,
      regionTag: null,
      regionHops: null,
      section: null,
      sectionConfidence: null,
      sectionNodeId: null,
      sectionHops: null,
      anchorNodeId: entry ? (idOf.get(entry.el) ?? null) : null,
    };
    let el: DomElement | null | undefined = entry?.el;
    let hops = 0;
    while (el && (row.regionNodeId === null || row.sectionNodeId === null)) {
      const id = idOf.get(el);
      const node = id === undefined ? undefined : byId.get(id);
      if (node) {
        if (row.regionNodeId === null && node.label !== "unknown") {
          row.region = node.label;
          row.regionGroup = regionGroup(node.label);
          row.regionSource = node.source;
          row.regionConfidence = node.confidence;
          row.regionNodeId = id!;
          row.regionTag = node.nodeRef.tag;
          row.regionHops = hops;
        }
        if (row.sectionNodeId === null && node.section) {
          row.section = node.section.label;
          row.sectionConfidence = node.section.confidence;
          row.sectionNodeId = id!;
          row.sectionHops = hops;
        }
      }
      el = el.parentElement;
      hops++;
    }
    out.push(row);
  };

  if (Array.isArray(manifestLinks)) {
    // A link listed twice (header and footer) takes two different anchors:
    // prefer an unused anchor with the same text, then any unused one.
    const used = new Set<DomElement>();
    manifestLinks.forEach((link, i) => {
      const href = resolveHref(link.href, base) ?? link.href;
      const list = anchors.get(hrefKey(href)) ?? [];
      const text = normaliseText(link.anchor_text ?? "");
      const byText = text ? list.filter((a) => a.text === text) : [];
      let entry = byText.find((a) => !used.has(a.el)) ?? null;
      let matchedBy: "text" | "href" | null = entry ? "text" : null;
      if (!entry) {
        entry = list.find((a) => !used.has(a.el)) ?? byText[0] ?? list[0] ?? null;
        matchedBy = entry ? (byText.includes(entry) ? "text" : "href") : null;
      }
      if (entry) used.add(entry.el);
      locate(
        i,
        entry,
        href,
        link.anchor_text ?? entry?.text ?? "",
        link.link_type ?? null,
        list.length,
        matchedBy,
      );
    });
    return out;
  }

  const own = registrableDomain(pageUrl);
  let i = 0;
  for (const entry of order) {
    const domain = registrableDomain(entry.href);
    if (!domain || domain === own || domain === "archive.org") continue;
    locate(i++, entry, entry.href, entry.text, null, anchors.get(hrefKey(entry.href))!.length);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

export interface PageRecord {
  sha256: string;
  key: string;
  ms: number;
  parseMs: number;
  missing: boolean;
  parseFailure: boolean;
  budgetFailure: boolean;
  nodes: number;
  pageType: string;
  pageTypeSource: string;
  degraded: string;
  links: number;
  matched: number;
}

export interface LinkRecord extends LinkPosition {
  sha256: string;
  url: string;
  key: string;
}

export function percentile(values: number[], p: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, index)]!;
}

function share(n: number, d: number): number | null {
  return d > 0 ? Math.round((n / d) * 1000) / 1000 : null;
}

const GROUPS: RegionGroup[] = [
  "chrome",
  "content_block",
  "main_content",
  "dialog",
  "other",
  "unknown",
];

export function summarise(pages: PageRecord[], links: LinkRecord[]): Record<string, unknown> {
  const keysOf = (key: string): string[] => {
    const [source] = key.split("|");
    return [key, `${source}|*|*|*`, "*|*|*|*"];
  };
  const groups = new Map<string, { pages: PageRecord[]; links: LinkRecord[] }>();
  const bucket = (key: string) => {
    let g = groups.get(key);
    if (!g) groups.set(key, (g = { pages: [], links: [] }));
    return g;
  };
  for (const page of pages) for (const k of keysOf(page.key)) bucket(k).pages.push(page);
  for (const link of links) for (const k of keysOf(link.key)) bucket(k).links.push(link);

  const out: Record<string, unknown> = {};
  for (const [key, g] of [...groups].sort(([a], [b]) => a.localeCompare(b))) {
    const parsed = g.pages.filter((p) => !p.missing && !p.parseFailure);
    const located = g.links.filter((l) => l.matched);
    const regionCounts = Object.fromEntries(GROUPS.map((x) => [x, 0])) as Record<
      RegionGroup,
      number
    >;
    const regionLabels: Record<string, number> = {};
    const sections: Record<string, number> = {};
    let withSection = 0;
    for (const link of located) {
      regionCounts[link.regionGroup]++;
      regionLabels[link.region] = (regionLabels[link.region] ?? 0) + 1;
      if (link.section) {
        withSection++;
        sections[link.section] = (sections[link.section] ?? 0) + 1;
      }
    }
    out[key] = {
      pages: g.pages.length,
      missingBlobs: g.pages.filter((p) => p.missing).length,
      parseFailures: g.pages.filter((p) => p.parseFailure).length,
      budgetFailures: g.pages.filter((p) => p.budgetFailure).length,
      msP50: percentile(
        parsed.map((p) => p.ms),
        50,
      ),
      msP95: percentile(
        parsed.map((p) => p.ms),
        95,
      ),
      pageTypeAnswered: share(parsed.filter((p) => p.pageType !== "unknown").length, parsed.length),
      pageTypeModel: share(
        parsed.filter((p) => p.pageType !== "unknown" && p.pageTypeSource === "model").length,
        parsed.length,
      ),
      links: g.links.length,
      linksLocated: located.length,
      regionAnswered: share(located.length - regionCounts.unknown, located.length),
      regionGroups: Object.fromEntries(
        GROUPS.map((x) => [x, share(regionCounts[x], located.length)]),
      ),
      regionLabels,
      sectionAny: share(withSection, located.length),
      sections,
    };
  }
  return out;
}

export function summaryMarkdown(summary: Record<string, unknown>): string {
  const lines = [
    "| source | year_bin | tld_group | state | pages | parse fail | budget fail | p50 / p95 ms | page type answered | links located | region answered | chrome | content_block | main_content | dialog | other | section any |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
  ];
  const fmt = (v: unknown) => (v === null || v === undefined ? "–" : String(v));
  for (const [key, raw] of Object.entries(summary)) {
    const s = raw as Record<string, any>;
    const [source, year, tld, state] = key.split("|");
    lines.push(
      `| ${source} | ${year} | ${tld} | ${state} | ${s.pages} | ${s.parseFailures} | ${s.budgetFailures} | ` +
        `${fmt(s.msP50)} / ${fmt(s.msP95)} | ${fmt(s.pageTypeAnswered)} | ${s.linksLocated} / ${s.links} | ` +
        `${fmt(s.regionAnswered)} | ${fmt(s.regionGroups.chrome)} | ${fmt(s.regionGroups.content_block)} | ` +
        `${fmt(s.regionGroups.main_content)} | ${fmt(s.regionGroups.dialog)} | ${fmt(s.regionGroups.other)} | ` +
        `${fmt(s.sectionAny)} |`,
    );
  }
  return `${lines.join("\n")}\n`;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function argument(name: string, fallback?: string): string {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? fallback : Bun.argv[index + 1];
  if (value === undefined || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

/** Decode with the charset the page declares near its top, else UTF-8. */
export function decode(bytes: Uint8Array): string {
  const head = new TextDecoder("latin1").decode(bytes.subarray(0, 4096));
  const declared = /<meta[^>]{0,200}charset=["']?\s*([A-Za-z0-9_:.-]{1,40})/i.exec(head)?.[1];
  if (declared) {
    try {
      return new TextDecoder(declared.toLowerCase()).decode(bytes);
    } catch {
      // Unknown label: fall through to UTF-8.
    }
  }
  return new TextDecoder("utf-8").decode(bytes);
}

async function main(): Promise<void> {
  const input = argument("--in");
  const outDir = argument("--out");
  const budgetMs = Number(argument("--budget-ms", "200"));
  const limit = Number(argument("--limit", "0"));
  const artifactText = readFileSync(argument("--artifact"), "utf8");
  const { model, reason } = tryLoadModel(JSON.parse(artifactText));
  if (!model) throw new Error(`artifact unusable: ${reason}`);
  mkdirSync(outDir, { recursive: true });

  const rows = readFileSync(`${input}/manifest.jsonl`, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as ManifestRow);
  const pages: PageRecord[] = [];
  const links: LinkRecord[] = [];
  const linkLines: string[] = [];
  // Wall-clock numbers depend on machine load: record it beside them.
  const run = {
    startedAt: new Date().toISOString(),
    artifactSha256: createHash("sha256").update(artifactText).digest("hex"),
    budgetMs,
    loadavgStart: loadavg().map((v) => Math.round(v * 100) / 100),
    loadavgEnd: [] as number[],
  };

  for (const row of limit > 0 ? rows.slice(0, limit) : rows) {
    const key = [
      row.source ?? "na",
      String(row.year_bin ?? "na"),
      row.tld_group ?? tldGroup(row.url),
      row.state ?? "na",
    ].join("|");
    const page: PageRecord = {
      sha256: row.sha256,
      key,
      ms: 0,
      parseMs: 0,
      missing: false,
      parseFailure: false,
      budgetFailure: false,
      nodes: 0,
      pageType: "unknown",
      pageTypeSource: "rules",
      degraded: "",
      links: 0,
      matched: 0,
    };
    pages.push(page);
    const blob = `${input}/blobs/${row.sha256}.html.gz`;
    if (!existsSync(blob)) {
      page.missing = true;
      continue;
    }
    try {
      const html = decode(gunzipSync(readFileSync(blob)));
      const started = performance.now();
      const { document } = parseHTML(html);
      const parsed = performance.now();
      const elements: unknown[] = [];
      const result = classifyDocument(document, row.url, {
        model,
        budgetMs,
        html,
        elementsOut: elements,
      });
      const done = performance.now();
      page.parseMs = Math.round((parsed - started) * 100) / 100;
      page.ms = Math.round((done - started) * 100) / 100;
      page.nodes = result.nodes.length;
      page.pageType = result.pageType.label;
      page.pageTypeSource = result.pageType.source;
      page.degraded = result.degraded;
      page.budgetFailure = result.degraded.startsWith("time budget");
      const baseHref = (document as unknown as { querySelector(s: string): DomElement | null })
        .querySelector("base[href]")
        ?.getAttribute("href");
      const positions = linkPositions(row.url, result, elements, row.links, baseHref);
      page.links = positions.length;
      page.matched = positions.filter((p) => p.matched).length;
      for (const position of positions) {
        const record: LinkRecord = { sha256: row.sha256, url: row.url, key, ...position };
        links.push(record);
        linkLines.push(JSON.stringify(record));
      }
    } catch (error) {
      page.parseFailure = true;
      page.degraded = String(error).slice(0, 200);
    }
  }

  run.loadavgEnd = loadavg().map((v) => Math.round(v * 100) / 100);
  writeFileSync(`${outDir}/run.json`, `${JSON.stringify(run, null, 2)}\n`);
  writeFileSync(`${outDir}/links.jsonl`, linkLines.length ? `${linkLines.join("\n")}\n` : "");
  writeFileSync(`${outDir}/pages.jsonl`, `${pages.map((p) => JSON.stringify(p)).join("\n")}\n`);
  const summary = summarise(pages, links);
  writeFileSync(`${outDir}/summary.json`, `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(`${outDir}/summary.md`, summaryMarkdown(summary));
  const all = summary["*|*|*|*"] as Record<string, unknown> | undefined;
  console.log(JSON.stringify({ pages: pages.length, links: links.length, run, all }));
}

if (import.meta.main) await main();

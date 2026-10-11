#!/usr/bin/env bun
/**
 * Per-stage runtime bench (#2307): where does a page's time go?
 *
 * For every page, R times (reps outer, pages inner, so no page is timed
 * back-to-back with itself): parse, then featurize alone, then the full
 * `classifyDocument` with no budget. Predict = classify - featurize on the same
 * parsed document. Each stage keeps its MIN over reps: the noise on a shared or
 * cloud box is one-sided, so the minimum is the stable estimate.
 *
 *   bun run experiments/perf-bench.ts --in name=<dir>[,name=<dir>...] \
 *     --artifact classifier-v2.json --out <dir> [--reps 3] [--limit N]
 *
 * Input dirs use the link probe's format (`manifest.jsonl` + `blobs/<sha256>.html.gz`).
 * A row with a `source` reports under `<name>:<source>`.
 * Output: pages.jsonl (one row per page, no URL) and summary.json / summary.md,
 * per set: p50/p90/p95/p99/max per stage, stage shares, the share of pages a
 * budget of 25/50/100/200 ms would finish, and cost by DOM size. Numbers from
 * different machines or containers do not compare: A/B in one run.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { cpus, loadavg } from "node:os";
import { gunzipSync } from "node:zlib";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import { featurizeDocument } from "../featurize.ts";
import { classifyDocument, tryLoadModel } from "../runtime/predict.ts";
import { decode } from "./link-position.ts";

interface Row {
  sha256: string;
  url: string;
  source?: string;
}

interface PageTiming {
  set: string;
  sha256: string;
  bytes: number;
  elements: number;
  nodes: number;
  parseMs: number;
  featurizeMs: number;
  predictMs: number;
  totalMs: number;
}

const BUDGETS = [25, 50, 100, 200] as const;
const SIZE_BUCKETS = [
  ["<1k", 1_000],
  ["1k-3k", 3_000],
  ["3k-10k", 10_000],
  [">=10k", Number.POSITIVE_INFINITY],
] as const;

function argument(name: string, fallback?: string): string {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? fallback : Bun.argv[index + 1];
  if (value === undefined || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

function quantile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return round(sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))]!);
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}

function cpuModel(): string {
  try {
    const line = readFileSync("/proc/cpuinfo", "utf8")
      .split("\n")
      .find((l) => l.startsWith("model name"));
    if (line) return line.split(":")[1]!.trim();
  } catch {
    // Not Linux: fall through.
  }
  return cpus()[0]?.model ?? "unknown";
}

function stats(values: number[]) {
  return {
    p50: quantile(values, 0.5),
    p90: quantile(values, 0.9),
    p95: quantile(values, 0.95),
    p99: quantile(values, 0.99),
    max: quantile(values, 1),
  };
}

function summarize(pages: PageTiming[]) {
  const sum = (key: keyof PageTiming) => pages.reduce((s, p) => s + (p[key] as number), 0);
  const total = sum("totalMs") || 1;
  const bySize = Object.fromEntries(
    SIZE_BUCKETS.map(([name, upper], i) => {
      const lower = i === 0 ? 0 : SIZE_BUCKETS[i - 1]![1];
      const inBucket = pages.filter((p) => p.elements >= lower && p.elements < upper);
      return [
        name,
        {
          pages: inBucket.length,
          totalP50: quantile(
            inBucket.map((p) => p.totalMs),
            0.5,
          ),
          totalP95: quantile(
            inBucket.map((p) => p.totalMs),
            0.95,
          ),
          parseP50: quantile(
            inBucket.map((p) => p.parseMs),
            0.5,
          ),
          featurizeP50: quantile(
            inBucket.map((p) => p.featurizeMs),
            0.5,
          ),
          predictP50: quantile(
            inBucket.map((p) => p.predictMs),
            0.5,
          ),
        },
      ];
    }),
  );
  return {
    pages: pages.length,
    elements: stats(pages.map((p) => p.elements)),
    nodes: stats(pages.map((p) => p.nodes)),
    parseMs: stats(pages.map((p) => p.parseMs)),
    featurizeMs: stats(pages.map((p) => p.featurizeMs)),
    predictMs: stats(pages.map((p) => p.predictMs)),
    totalMs: stats(pages.map((p) => p.totalMs)),
    timeShare: {
      parse: round(sum("parseMs") / total),
      featurize: round(sum("featurizeMs") / total),
      predict: round(sum("predictMs") / total),
    },
    finishedWithin: Object.fromEntries(
      BUDGETS.map((b) => [
        b,
        round(pages.filter((p) => p.totalMs <= b).length / Math.max(pages.length, 1)),
      ]),
    ),
    bySize,
  };
}

async function main(): Promise<void> {
  const sets = argument("--in")
    .split(",")
    .map((pair) => {
      const [name, dir] = pair.split("=");
      if (!name || !dir) throw new Error(`--in wants name=dir, got "${pair}"`);
      return { name, dir };
    });
  const outDir = argument("--out");
  const reps = Number(argument("--reps", "3"));
  const limit = Number(argument("--limit", "0"));
  const artifactText = readFileSync(argument("--artifact"), "utf8");
  const loadStarted = performance.now();
  const { model, reason } = tryLoadModel(JSON.parse(artifactText));
  const coldLoadMs = round(performance.now() - loadStarted);
  if (!model) throw new Error(`artifact unusable: ${reason}`);
  mkdirSync(outDir, { recursive: true });

  const work: { set: string; dir: string; row: Row }[] = [];
  for (const { name, dir } of sets) {
    const rows = readFileSync(`${dir}/manifest.jsonl`, "utf8")
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as Row)
      .filter((row) => existsSync(`${dir}/blobs/${row.sha256}.html.gz`));
    for (const row of limit > 0 ? rows.slice(0, limit) : rows) {
      work.push({ set: row.source ? `${name}:${row.source}` : name, dir, row });
    }
  }

  const best = new Map<number, PageTiming>();
  const started = new Date().toISOString();
  const loadStart = loadavg().map(round);
  for (let rep = 0; rep < reps; rep++) {
    for (const [i, { set, dir, row }] of work.entries()) {
      // Decoding is I/O, not runtime cost: it stays outside every timer.
      const html = decode(gunzipSync(readFileSync(`${dir}/blobs/${row.sha256}.html.gz`)));
      const t0 = performance.now();
      const { document } = parseHTML(html);
      const t1 = performance.now();
      const featurized = featurizeDocument(document as never, row.url);
      const t2 = performance.now();
      const result = classifyDocument(document, row.url, { model, budgetMs: 1e9, html });
      const t3 = performance.now();
      const timing: PageTiming = {
        set,
        sha256: row.sha256,
        bytes: html.length,
        elements: featurized.page.elementCount,
        nodes: result.nodes.length,
        parseMs: t1 - t0,
        featurizeMs: t2 - t1,
        // classifyDocument featurizes again: what it adds on top is the prediction.
        predictMs: Math.max(0, t3 - t2 - (t2 - t1)),
        totalMs: t1 - t0 + (t3 - t2),
      };
      const previous = best.get(i);
      best.set(
        i,
        previous
          ? {
              ...timing,
              parseMs: Math.min(previous.parseMs, timing.parseMs),
              featurizeMs: Math.min(previous.featurizeMs, timing.featurizeMs),
              predictMs: Math.min(previous.predictMs, timing.predictMs),
              totalMs: Math.min(previous.totalMs, timing.totalMs),
            }
          : timing,
      );
    }
    console.log(`rep ${rep + 1}/${reps}: ${work.length} pages`);
  }

  const pages = [...best.values()].map((p) => ({
    ...p,
    parseMs: round(p.parseMs),
    featurizeMs: round(p.featurizeMs),
    predictMs: round(p.predictMs),
    totalMs: round(p.totalMs),
  }));
  writeFileSync(`${outDir}/pages.jsonl`, `${pages.map((p) => JSON.stringify(p)).join("\n")}\n`);
  const summary = {
    run: {
      startedAt: started,
      artifactSha256: createHash("sha256").update(artifactText).digest("hex"),
      coldLoadMs,
      reps,
      cpu: cpuModel(),
      cpus: cpus().length,
      bun: Bun.version,
      loadavgStart: loadStart,
      loadavgEnd: loadavg().map(round),
    },
    sets: Object.fromEntries(
      [...new Set(work.map((w) => w.set))].map((name) => [
        name,
        summarize(pages.filter((p) => p.set === name)),
      ]),
    ),
  };
  writeFileSync(`${outDir}/summary.json`, `${JSON.stringify(summary, null, 2)}\n`);
  const lines = [
    `# perf bench ${started}`,
    "",
    `cpu ${summary.run.cpu} x${summary.run.cpus}, bun ${summary.run.bun}, ${reps} reps (min per stage), artifact ${summary.run.artifactSha256.slice(0, 12)}, cold load ${coldLoadMs} ms`,
    "",
    "| set | pages | total p50 / p95 / max | parse p95 | featurize p95 | predict p95 | share parse / feat / pred | within 25 / 50 / 100 / 200 ms |",
    "|---|---|---|---|---|---|---|---|",
  ];
  for (const [name, s] of Object.entries(summary.sets)) {
    lines.push(
      `| ${name} | ${s.pages} | ${s.totalMs.p50} / ${s.totalMs.p95} / ${s.totalMs.max} | ${s.parseMs.p95} | ${s.featurizeMs.p95} | ${s.predictMs.p95} | ${s.timeShare.parse} / ${s.timeShare.featurize} / ${s.timeShare.predict} | ${BUDGETS.map((b) => s.finishedWithin[b]).join(" / ")} |`,
    );
  }
  writeFileSync(`${outDir}/summary.md`, `${lines.join("\n")}\n`);
  console.log(lines.join("\n"));
}

if (import.meta.main) await main();

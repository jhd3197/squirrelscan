#!/usr/bin/env bun
/**
 * Per-stage runtime bench (#2307): where does a page's time go?
 *
 * For every page, R times (reps outer, pages inner, so no page is timed
 * back-to-back with itself): parse, then the full `classifyDocument` with no
 * budget, which reports featurize, page type, region and section through
 * `timingsOut`. Each stage keeps its MIN over reps: the noise on a shared or
 * cloud box is one-sided, so the minimum is the stable estimate. `cli` is what
 * the CLI pays: everything but the parse, which the audit has already done.
 *
 *   bun run experiments/perf-bench.ts --in name=<dir>[,name=<dir>...] \
 *     --artifact classifier-v2.json --out <dir> [--reps 3] [--limit N] [--arms 1,32]
 *
 * `--arms` A/Bs `scoringBatch` values in ONE process: every page runs every arm
 * in rotating order each rep, and an arm whose answers differ from the first
 * arm's is counted in `mismatches` (scoring is meant to be output-identical).
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
import { type ClassifyTimings, classifyDocument, tryLoadModel } from "../runtime/predict.ts";
import { decode } from "./link-position.ts";

interface Row {
  sha256: string;
  url: string;
  source?: string;
}

interface PageTiming {
  set: string;
  arm: string;
  sha256: string;
  bytes: number;
  elements: number;
  nodes: number;
  parseMs: number;
  featurizeMs: number;
  pageTypeMs: number;
  regionMs: number;
  sectionMs: number;
  cliMs: number;
  totalMs: number;
}

const STAGES = [
  "parseMs",
  "featurizeMs",
  "pageTypeMs",
  "regionMs",
  "sectionMs",
  "cliMs",
  "totalMs",
] as const;

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
          ...Object.fromEntries(
            STAGES.map((stage) => [
              `${stage}P50`,
              quantile(
                inBucket.map((p) => p[stage]),
                0.5,
              ),
            ]),
          ),
          cliMsP95: quantile(
            inBucket.map((p) => p.cliMs),
            0.95,
          ),
        },
      ];
    }),
  );
  const stageStats = Object.fromEntries(
    STAGES.map((stage) => [stage, stats(pages.map((p) => p[stage]))]),
  ) as Record<(typeof STAGES)[number], ReturnType<typeof stats>>;
  return {
    pages: pages.length,
    elements: stats(pages.map((p) => p.elements)),
    nodes: stats(pages.map((p) => p.nodes)),
    ...stageStats,
    timeShare: Object.fromEntries(
      STAGES.filter((stage) => stage !== "cliMs" && stage !== "totalMs").map((stage) => [
        stage.replace(/Ms$/, ""),
        round(sum(stage) / total),
      ]),
    ),
    // The runtime's budget clock starts inside classifyDocument: it sees cliMs.
    cliFinishedWithin: Object.fromEntries(
      BUDGETS.map((b) => [
        b,
        round(pages.filter((p) => p.cliMs <= b).length / Math.max(pages.length, 1)),
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
  const arms = argument("--arms", "default").split(",");
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

  const best = new Map<string, PageTiming>();
  const mismatches: Record<string, number> = Object.fromEntries(arms.map((a) => [a, 0]));
  const started = new Date().toISOString();
  const loadStart = loadavg().map(round);
  for (let rep = 0; rep < reps; rep++) {
    for (const [i, { set, dir, row }] of work.entries()) {
      // Decoding is I/O, not runtime cost: it stays outside every timer.
      const html = decode(gunzipSync(readFileSync(`${dir}/blobs/${row.sha256}.html.gz`)));
      const answers = new Map<string, string>();
      for (let a = 0; a < arms.length; a++) {
        const armIndex = (a + rep + i) % arms.length;
        const arm = arms[armIndex]!;
        const t0 = performance.now();
        const { document } = parseHTML(html);
        const t1 = performance.now();
        const timings: ClassifyTimings = {
          featurizeMs: 0,
          pageTypeMs: 0,
          regionMs: 0,
          sectionMs: 0,
        };
        const result = classifyDocument(document, row.url, {
          model,
          budgetMs: 1e9,
          html,
          timingsOut: timings,
          ...(arm === "default" ? {} : { scoringBatch: Number(arm) }),
        });
        const t2 = performance.now();
        if (rep === 0) answers.set(arm, JSON.stringify(result));
        const key = `${i}|${arm}`;
        const previous = best.get(key);
        const timing: PageTiming = {
          set,
          arm,
          sha256: row.sha256,
          bytes: html.length,
          // Counted once, outside every timer.
          elements:
            previous?.elements ??
            (
              document as unknown as { querySelectorAll(s: string): ArrayLike<unknown> }
            ).querySelectorAll("*").length,
          nodes: result.nodes.length,
          parseMs: t1 - t0,
          ...timings,
          cliMs: t2 - t1,
          totalMs: t2 - t0,
        };
        if (previous) {
          for (const stage of STAGES) timing[stage] = Math.min(previous[stage], timing[stage]);
        }
        best.set(key, timing);
      }
      for (const arm of arms.slice(1)) {
        if (rep === 0 && answers.get(arm) !== answers.get(arms[0]!)) mismatches[arm]!++;
      }
    }
    console.log(`rep ${rep + 1}/${reps}: ${work.length} pages x ${arms.length} arms`);
  }

  const pages = [...best.values()].map((p) => {
    const rounded = { ...p };
    for (const stage of STAGES) rounded[stage] = round(p[stage]);
    return rounded;
  });
  writeFileSync(`${outDir}/pages.jsonl`, `${pages.map((p) => JSON.stringify(p)).join("\n")}\n`);
  const summary = {
    run: {
      startedAt: started,
      artifactSha256: createHash("sha256").update(artifactText).digest("hex"),
      coldLoadMs,
      reps,
      arms,
      mismatches,
      cpu: cpuModel(),
      cpus: cpus().length,
      bun: Bun.version,
      loadavgStart: loadStart,
      loadavgEnd: loadavg().map(round),
    },
    sets: Object.fromEntries(
      [...new Set(work.map((w) => w.set))].flatMap((name) =>
        arms.map((arm) => [
          arms.length > 1 ? `${name} @${arm}` : name,
          summarize(pages.filter((p) => p.set === name && p.arm === arm)),
        ]),
      ),
    ),
  };
  writeFileSync(`${outDir}/summary.json`, `${JSON.stringify(summary, null, 2)}\n`);
  const lines = [
    `# perf bench ${started}`,
    "",
    `cpu ${summary.run.cpu} x${summary.run.cpus}, bun ${summary.run.bun}, ${reps} reps (min per stage), arms ${arms.join(",")} (answers differing from the first arm: ${JSON.stringify(mismatches)}), artifact ${summary.run.artifactSha256.slice(0, 12)}, cold load ${coldLoadMs} ms. cli = featurize + page type + region + section (the parse is the audit's).`,
    "",
    "| set | pages | cli p50 / p95 / max | featurize p95 | region p95 | section p95 | share parse / feat / pageType / region / section | cli within 25 / 50 / 100 / 200 ms |",
    "|---|---|---|---|---|---|---|---|",
  ];
  for (const [name, s] of Object.entries(summary.sets)) {
    const share = s.timeShare;
    lines.push(
      `| ${name} | ${s.pages} | ${s.cliMs.p50} / ${s.cliMs.p95} / ${s.cliMs.max} | ${s.featurizeMs.p95} | ${s.regionMs.p95} | ${s.sectionMs.p95} | ${share.parse} / ${share.featurize} / ${share.pageType} / ${share.region} / ${share.section} | ${BUDGETS.map((b) => s.cliFinishedWithin[b]).join(" / ")} |`,
    );
  }
  writeFileSync(`${outDir}/summary.md`, `${lines.join("\n")}\n`);
  console.log(lines.join("\n"));
}

if (import.meta.main) await main();

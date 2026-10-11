#!/usr/bin/env bun
/**
 * Does site-wide item collapse actually reduce report noise on real sites?
 *
 * Streams the PRIVATE frozen report snapshot one file at a time, renders each
 * report twice through the library (never by spawning the CLI), and writes ONLY
 * AGGREGATES to a private output directory: totals, per-rule savings and a
 * per-site distribution. No hostname, URL, page path, item id or any other
 * content from the corpus is read into the output or into this repository.
 *
 * Read-only on the corpus. Sequential, one report resident at a time, because
 * the snapshot is 1,462 files and a few are very large.
 *
 *   NOISE_SURVEY_CORPUS=<snapshot dir> NOISE_SURVEY_OUT=<private out dir> \
 *     bun run scripts/component-findings/noise-survey.ts
 */
import { mkdirSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";

import { renderLlm } from "../../packages/report/src/output/llm.ts";
import { renderMarkdown } from "../../packages/report/src/output/markdown.ts";
import type { AuditReport, CheckResult, ReportRuleResult } from "../../packages/report/src/types.ts";

/** Both paths are private and machine-local, so they come from the environment. */
function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`set ${name} (a private, machine-local path)`);
  return value;
}
const CORPUS = resolve(requiredEnv("NOISE_SURVEY_CORPUS"));
const OUT_DIR = resolve(requiredEnv("NOISE_SURVEY_OUT"));

/** repo#2320: an unbounded sitemap bug produced a few enormous reports. */
const MAX_BYTES = 5 * 1024 * 1024;
/** Deterministic sample if the full pass runs long; see `--sample`. */
const SAMPLE_SIZE = 300;
const TIME_BUDGET_MS = 10 * 60 * 1000;

interface SlimCheck {
  name: string;
  status: "fail" | "warn";
  message: string;
  affectedPages?: string[];
  items?: Array<{ id: string; label?: string; sourcePages?: string[] }>;
  details?: Record<string, unknown>;
}
interface SlimIssue {
  ruleId: string;
  name: string;
  description: string;
  solution?: string;
  severity: "error" | "warning" | "info";
  checks: SlimCheck[];
}
interface SlimReport {
  meta?: { baseUrl?: string; timestamp?: string; totalPages?: number };
  issues?: SlimIssue[];
}

/**
 * Minimal slim -> AuditReport conversion, enough to drive the renderers.
 *
 * Deliberately NOT importing the CLI's `convertSlimReport`: that one expands a
 * check into one per affected page, which would change the very line counts
 * being measured. Here each check stays one check, exactly as the snapshot
 * recorded it.
 */
function toAuditReport(slim: SlimReport): AuditReport | undefined {
  const issues = slim.issues;
  if (!issues || issues.length === 0) return undefined;
  const ruleResults: Record<string, ReportRuleResult> = {};
  for (const issue of issues) {
    const checks: CheckResult[] = issue.checks.map((check) => ({
      name: check.name,
      status: check.status,
      message: check.message,
      pages: check.affectedPages ?? [],
      items: check.items,
      details: check.details,
    }));
    ruleResults[issue.ruleId] = {
      meta: {
        id: issue.ruleId,
        name: issue.name,
        description: issue.description,
        solution: issue.solution,
        category: (issue.ruleId.split("/")[0] ?? "other") as ReportRuleResult["meta"]["category"],
        scope: "page",
        severity: issue.severity === "error" ? "error" : "warning",
        weight: 1,
      },
      checks,
    };
  }
  return {
    baseUrl: slim.meta?.baseUrl ?? "https://site.invalid",
    timestamp: slim.meta?.timestamp ?? "2026-09-17T00:00:00.000Z",
    totalPages: slim.meta?.totalPages ?? 0,
    passed: 0,
    warnings: 0,
    failed: 0,
    ruleResults,
    healthScore: {
      overall: 0, categories: [], groups: [], errorCount: 0, warningCount: 0, passedCount: 0,
    },
  } as unknown as AuditReport;
}

function size(text: string) {
  return {
    lines: text.split("\n").length,
    bytes: Buffer.byteLength(text),
    approxTokens: Math.round(Buffer.byteLength(text) / 4),
  };
}

/** Per-rule savings, attributed by rendering one rule at a time. */
function perRuleSavings(report: AuditReport, into: Map<string, { before: number; after: number; sites: number }>) {
  for (const [ruleId, rule] of Object.entries(report.ruleResults)) {
    const single = { ...report, ruleResults: { [ruleId]: rule } } as AuditReport;
    const before = size(renderLlm(single, { siteWideCollapse: false })).lines;
    const after = size(renderLlm(single)).lines;
    if (before === after) continue;
    const entry = into.get(ruleId) ?? { before: 0, after: 0, sites: 0 };
    entry.before += before;
    entry.after += after;
    entry.sites += 1;
    into.set(ruleId, entry);
  }
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p));
  return sorted[index]!;
}

const sampleMode = process.argv.includes("--sample");
const all = readdirSync(CORPUS).filter((f) => f.endsWith(".json")).sort();
const selected = sampleMode
  ? all.filter((_, i) => i % Math.max(1, Math.floor(all.length / SAMPLE_SIZE)) === 0).slice(0, SAMPLE_SIZE)
  : all;

const totals = {
  llm: { before: { lines: 0, bytes: 0, approxTokens: 0 }, after: { lines: 0, bytes: 0, approxTokens: 0 } },
  markdown: { before: { lines: 0, bytes: 0, approxTokens: 0 }, after: { lines: 0, bytes: 0, approxTokens: 0 } },
};
const perRule = new Map<string, { before: number; after: number; sites: number }>();
const perSiteLineSavingPct: number[] = [];
let read = 0;
let skippedLarge = 0;
let skippedUnusable = 0;
let collapsedSites = 0;
const started = Date.now();

for (const file of selected) {
  const path = join(CORPUS, file);
  let slim: SlimReport;
  try {
    if (statSync(path).size > MAX_BYTES) {
      skippedLarge++;
      continue;
    }
    slim = JSON.parse(await Bun.file(path).text()) as SlimReport;
  } catch {
    skippedUnusable++;
    continue;
  }
  const report = toAuditReport(slim);
  if (!report) {
    skippedUnusable++;
    continue;
  }
  read++;

  for (const [format, render] of [["llm", renderLlm], ["markdown", renderMarkdown]] as const) {
    const before = size(render(report, { siteWideCollapse: false }));
    const after = size(render(report));
    const bucket = totals[format];
    for (const key of ["lines", "bytes", "approxTokens"] as const) {
      bucket.before[key] += before[key];
      bucket.after[key] += after[key];
    }
    if (format === "llm") {
      if (after.lines < before.lines) collapsedSites++;
      perSiteLineSavingPct.push(before.lines === 0 ? 0 : (1 - after.lines / before.lines) * 100);
    }
  }
  perRuleSavings(report, perRule);

  if (!sampleMode && Date.now() - started > TIME_BUDGET_MS) {
    console.error(`time budget reached after ${read} reports; re-run with --sample`);
    break;
  }
}

const topRules = [...perRule.entries()]
  .map(([ruleId, v]) => ({
    ruleId,
    linesBefore: v.before,
    linesAfter: v.after,
    linesSaved: v.before - v.after,
    sitesAffected: v.sites,
  }))
  .sort((a, b) => b.linesSaved - a.linesSaved)
  .slice(0, 15);

const output = {
  generatedAt: new Date().toISOString(),
  corpus: {
    // The PATH is recorded (it is a private artifact in a private directory);
    // no site identity, URL or content from it is.
    path: CORPUS,
    filesAvailable: all.length,
    filesRead: read,
    skippedLargerThanBytes: { limit: MAX_BYTES, count: skippedLarge, reason: "repo#2320 unbounded sitemap" },
    skippedUnusable,
    sampled: sampleMode,
  },
  method:
    "Each report rendered twice through the report library with siteWideCollapse true/false. No CLI spawn, no detections or scores involved, corpus read-only.",
  totals,
  savings: {
    llmLinesPct: pct(totals.llm.before.lines, totals.llm.after.lines),
    llmTokensPct: pct(totals.llm.before.approxTokens, totals.llm.after.approxTokens),
    markdownLinesPct: pct(totals.markdown.before.lines, totals.markdown.after.lines),
    markdownTokensPct: pct(totals.markdown.before.approxTokens, totals.markdown.after.approxTokens),
  },
  perSiteLineSaving: {
    sitesWithAnySaving: collapsedSites,
    medianPct: round(percentile(perSiteLineSavingPct, 0.5)),
    p90Pct: round(percentile(perSiteLineSavingPct, 0.9)),
  },
  topRulesByLinesSaved: topRules,
};

function pct(before: number, after: number): number {
  return before === 0 ? 0 : round((1 - after / before) * 100);
}
function round(value: number): number {
  return Math.round(value * 10) / 10;
}

mkdirSync(OUT_DIR, { recursive: true });
const outFile = join(OUT_DIR, `site-wide-collapse-${sampleMode ? "sample" : "full"}.json`);
writeFileSync(outFile, `${JSON.stringify(output, null, 2)}\n`);
console.log(`wrote ${basename(outFile)}: ${read} reports, llm lines ${output.savings.llmLinesPct}% saved`);

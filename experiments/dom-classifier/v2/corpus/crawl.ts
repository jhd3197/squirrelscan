#!/usr/bin/env bun
/**
 * Drive `squirrel crawl` over the seed list, one site at a time, into an
 * ISOLATED store. Breadth for the classifier corpus: the live store's pages are
 * ~8% landmark-less, and the region head has no supervision for footer or
 * navigation on the themes that need it most.
 *
 * `crawl` and not `audit`: the crawl subcommand fetches and stores and runs no
 * rules. Measured on the same 15-page site, audit is 34.4s (24.3s of it CPU in
 * the 260 rules) and crawl is 2.15s. The rules produce nothing this corpus uses.
 *
 * Politeness, all of it delegated to the crawler the CLI ships, which is the
 * same code a squirrelscan user runs against their own site:
 *  - robots.txt is obeyed by the crawler, including crawl-delay;
 *  - a host is crawled ONCE, ever: `--resume` is never passed and a domain that
 *    appears in the ledger is skipped;
 *  - 403 / 429 / WAF-blocked marks the host dead and it is not retried;
 *  - at most `--concurrency` site crawls run at once, each with a small
 *    per-host worker pool.
 *
 * The ledger (`crawl-ledger.jsonl`) holds one row per attempted domain and is
 * the resume key, so an interrupted run continues rather than restarting.
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const dataRoot = process.env.DOM_CLASSIFIER_DATA_ROOT;
if (!dataRoot) throw new Error("DOM_CLASSIFIER_DATA_ROOT must be set");
const cliPath = process.env.SQUIRREL_CLI;
if (!cliPath) throw new Error("SQUIRREL_CLI must point at apps/cli/src/cli.ts");

const outDir = resolve(dataRoot, "corpus");
mkdirSync(outDir, { recursive: true, mode: 0o700 });
const ledgerPath = `${outDir}/crawl-ledger.jsonl`;
const seedsPath = `${outDir}/seeds.jsonl`;

function argument(name: string, fallback: string): string {
  const index = Bun.argv.indexOf(name);
  if (index < 0) return fallback;
  const value = Bun.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} needs a value`);
  return value;
}

const crawlHome = resolve(argument("--home", `${dataRoot}/crawl-home`));
const workers = Number(argument("--workers", "4"));
const maxPages = Number(argument("--max-pages", "15"));
const perSiteConcurrency = Number(argument("--site-concurrency", "4"));
const siteTimeoutMs = Number(argument("--site-timeout-ms", "75000"));
const budgetMs = Number(argument("--budget-minutes", "180")) * 60_000;
const limit = Number(argument("--limit", "0"));
if (!Number.isInteger(workers) || workers < 1 || workers > 4)
  throw new Error("--workers must be 1..4");

mkdirSync(crawlHome, { recursive: true, mode: 0o700 });

type Seed = { domain: string; url: string; suffix: string; source: string };
const seeds: Seed[] = readFileSync(seedsPath, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line) as Seed);

/** Every domain already attempted, whatever the outcome. A host is never hit
 *  twice across runs: a failure is a result, not a reason to try again. */
const attempted = new Set<string>();
if (existsSync(ledgerPath))
  for (const line of readFileSync(ledgerPath, "utf8").split("\n")) {
    if (!line) continue;
    try {
      attempted.add((JSON.parse(line) as { domain: string }).domain);
    } catch {
      /* a half-written final line just means that domain is retried once */
    }
  }

const pending = seeds.filter((seed) => !attempted.has(seed.domain));
const work = limit > 0 ? pending.slice(0, limit) : pending;

/** Outcome classes, counted and printed: a silent block rate is the failure
 *  mode that makes a corpus look bigger than it is. Declared as a closed shape
 *  rather than a Record, so every counter is a number and the summary cannot
 *  print `undefined` for a class that happened not to occur. */
type Outcome = "ok" | "empty" | "blocked" | "unreachable" | "timeout" | "error";
const totals: Record<Outcome, number> & { pages: number } = {
  ok: 0,
  empty: 0,
  blocked: 0,
  unreachable: 0,
  timeout: 0,
  error: 0,
  pages: 0,
};
const startedAt = Date.now();
let cursor = 0;
/** Sites actually started. `cursor` overshoots by up to `workers`, because each
 *  worker increments it once more to discover the queue is empty, and dividing
 *  the failure count by that understates it. */
let claimed = 0;
let stopped = false;

const BLOCK = /\b(403|429|forbidden|too many requests|rate.?limit|waf|blocked by)\b/i;
const UNREACHABLE =
  /\b(cannot reach|connection timed out|enotfound|econnrefused|certificate|dns|ssl|getaddrinfo)\b/i;
const PAGES = /Crawled\s+(\d+)\s+page|(\d+)\s+pages?\s+(?:crawled|in)\b/i;

function classify(
  stdout: string,
  stderr: string,
  code: number | null,
): {
  outcome: Outcome;
  pages: number;
} {
  const text = `${stdout}\n${stderr}`;
  const match = PAGES.exec(text);
  const pages = match ? Number(match[1] ?? match[2] ?? 0) : 0;
  // Pages first. A site that 403s a few URLs and serves the rest is a success
  // with a partial yield, not a block: classifying it as blocked would inflate
  // the block rate with sites this corpus actually got data from.
  if (pages > 0 && code === 0) return { outcome: "ok", pages };
  if (BLOCK.test(text)) return { outcome: "blocked", pages };
  if (UNREACHABLE.test(text)) return { outcome: "unreachable", pages };
  if (code !== 0) return { outcome: "error", pages };
  return { outcome: "empty", pages };
}

async function crawlOne(seed: Seed): Promise<void> {
  const began = Date.now();
  let outcome: Outcome = "error";
  let pages = 0;
  let note = "";
  try {
    const child = Bun.spawn(
      [
        "bun",
        "run",
        cliPath!,
        "crawl",
        seed.url,
        "-m",
        String(maxPages),
        "--concurrency",
        String(perSiteConcurrency),
      ],
      {
        // A fresh HOME is the isolation: a separate ~/.squirrel, a separate
        // content store, and no chance of writing to the live one.
        env: { ...process.env, HOME: crawlHome, NO_COLOR: "1", CI: "1" },
        stdin: "ignore",
        stdout: "pipe",
        stderr: "pipe",
        cwd: crawlHome,
      },
    );
    // `timedOut` is set by the timer, not inferred from elapsed time. Inferring
    // it recorded a site that finished on its own inside the last half-second
    // of the window as a timeout, exit code 0 and pages crawled and all.
    let timedOut = false;
    let hardKill: ReturnType<typeof setTimeout> | undefined;
    const timer = setTimeout(() => {
      timedOut = true;
      try {
        child.kill();
      } catch {
        /* already gone */
      }
      // SIGTERM can be caught. Without an escalation a child that ignores it
      // holds its worker forever, and the budget is only checked BETWEEN
      // sites, so the whole run would never end and never write its summary.
      hardKill = setTimeout(() => {
        try {
          child.kill(9);
        } catch {
          /* already gone */
        }
      }, 5_000);
    }, siteTimeoutMs);
    const [stdout, stderr, code] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ]);
    clearTimeout(timer);
    if (hardKill) clearTimeout(hardKill);
    // classify() runs even on a timeout: the child stores pages as it goes, so
    // a killed crawl still contributed. Recording 0 would permanently
    // understate the corpus, because the ledger is the resume key and the host
    // is never revisited.
    const classified = classify(stdout, stderr, code);
    pages = classified.pages;
    outcome = timedOut ? "timeout" : classified.outcome;
    if (outcome !== "ok") {
      const line = `${stdout}\n${stderr}`
        .split("\n")
        .find((l) => /error|warn|✗/.test(l))
        ?.trim();
      // The note is a diagnostic class, truncated: never a page of site text.
      note = (line ?? "").slice(0, 160);
    }
  } catch (error) {
    note = (error instanceof Error ? error.message : String(error)).slice(0, 160);
  }
  totals[outcome] += 1;
  totals.pages += pages;
  appendFileSync(
    ledgerPath,
    `${JSON.stringify({
      domain: seed.domain,
      suffix: seed.suffix,
      source: seed.source,
      outcome,
      pages,
      ms: Date.now() - began,
      note,
      at: new Date().toISOString(),
    })}\n`,
    { mode: 0o600 },
  );
}

async function worker(): Promise<void> {
  for (;;) {
    if (stopped) return;
    if (Date.now() - startedAt > budgetMs) {
      stopped = true;
      return;
    }
    const index = cursor++;
    if (index >= work.length) return;
    claimed++;
    await crawlOne(work[index]!);
    const done = Object.entries(totals)
      .filter(([key]) => key !== "pages")
      .reduce((sum, [, value]) => sum + value, 0);
    if (done % 100 === 0)
      console.log(
        JSON.stringify({
          stage: "progress",
          done,
          of: work.length,
          ...totals,
          minutes: Number(((Date.now() - startedAt) / 60_000).toFixed(1)),
          pagesPerMinute: Number((totals.pages / ((Date.now() - startedAt) / 60_000)).toFixed(1)),
        }),
      );
  }
}

console.log(
  JSON.stringify({
    stage: "start",
    seeds: seeds.length,
    alreadyAttempted: attempted.size,
    work: work.length,
    workers,
    maxPages,
    budgetMinutes: budgetMs / 60_000,
  }),
);
await Promise.all(Array.from({ length: Math.min(workers, work.length) }, worker));
const summary = {
  stage: "complete",
  ...totals,
  attempted: claimed,
  remaining: Math.max(0, work.length - claimed),
  minutes: Number(((Date.now() - startedAt) / 60_000).toFixed(1)),
  // Sums blocked + unreachable + timeout + error + empty: a FAILURE rate, of
  // which an outright block is one kind.
  failureRate:
    Number(
      (
        (totals.blocked + totals.unreachable + totals.timeout + totals.error + totals.empty) /
        Math.max(1, claimed)
      ).toFixed(4),
    ) || 0,
};
writeFileSync(`${outDir}/crawl-summary.json`, `${JSON.stringify(summary, null, 2)}\n`, {
  mode: 0o600,
});
console.log(JSON.stringify(summary));

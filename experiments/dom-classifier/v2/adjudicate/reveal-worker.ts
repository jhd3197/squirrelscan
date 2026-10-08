/**
 * Forced-reveal worker for a Modal container (#2307): one element screenshot per hidden target.
 *
 *   bun run reveal-worker.ts --jobs <jobs.json> --html <dir> --out <dir> [--allow-data] [--concurrency 3]
 *
 * <jobs.json> rows: {pageId, url, scripts: "on" | "policy", markers: [n...]}. Writes
 * <out>/<pageId>-n<N>.png for every revealed target and <out>/results.jsonl.
 */
import { appendFileSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadPlaywright } from "../../labeler/browser-runtime.ts";
import { replayLaunchOptions } from "../../labeler/replay-capture.ts";
import { revealTarget } from "./reveal-capture.ts";

const arg = (name: string, fallback?: string) => {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? fallback : Bun.argv[index + 1];
  if (value === undefined) throw new Error(`missing ${name}`);
  return value;
};
type Job = { pageId: string; url: string; scripts: "on" | "policy"; markers: string[] };
const jobs = JSON.parse(readFileSync(arg("--jobs"), "utf8")) as Job[];
const htmlDir = arg("--html");
const out = arg("--out");
const allowData = Bun.argv.includes("--allow-data");
const concurrency = Number(arg("--concurrency", "3"));
const work = jobs.flatMap((job) => job.markers.map((marker) => ({ job, marker })));

const { chromium } = await loadPlaywright();
const browser = await chromium.launch(replayLaunchOptions());
let cursor = 0;
async function worker() {
  for (;;) {
    const next = work[cursor++];
    if (!next) return;
    const { job, marker } = next;
    const started = Date.now();
    const html = readFileSync(join(htmlDir, `${job.pageId}.html`), "utf8");
    const scripts = job.scripts === "on" || allowData;
    const result = await revealTarget(
      browser,
      { html, url: job.url, marker },
      { scripts, allowData },
    );
    if (result.png) writeFileSync(join(out, `${job.pageId}-n${marker}.png`), result.png);
    const { png, ...row } = result;
    appendFileSync(
      join(out, "results.jsonl"),
      `${JSON.stringify({ pageId: job.pageId, ...row, scripts, allowData, png: Boolean(png), ms: Date.now() - started })}\n`,
    );
  }
}
await Promise.all(Array.from({ length: concurrency }, worker));
await browser.close();
process.exit(0);

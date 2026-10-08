/**
 * Historical replay of stored pages inside a Modal container (#2307).
 *
 * The labeller's hardened replay (`labeler/replay-capture.ts`) runs unchanged:
 * stored html is the document, stylesheets/images/fonts load from public hosts
 * through the egress guard, data requests and further navigations are blocked.
 * Script policy is gold-v1's (`gold-v1/scripts/render.ts`): scripts off; on for
 * client-rendered builders (Wix, Framer), and on for a scripts-off render whose
 * quality signals are blank (an SPA shell). Every row records `scriptReason`.
 *
 *   bun run render-worker.ts --jobs <jobs.json> --html <dir> --store <dir> --out <results.jsonl>
 *
 * <jobs.json> is an array of prepare-render.ts job rows; <dir>/<pageId>.html is the
 * marked document. Captures land in <store>/captures as `historical_replay` pages.
 */
import { appendFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { loadPlaywright } from "../../labeler/browser-runtime.ts";
import { captureReplayTarget, replayLaunchOptions } from "../../labeler/replay-capture.ts";
import { LabelStore } from "../../labeler/store.ts";
import { promoteDeferredStyles } from "./replay-html.ts";

const arg = (name: string, fallback?: string) => {
  const index = Bun.argv.indexOf(name);
  const value = index < 0 ? fallback : Bun.argv[index + 1];
  if (value === undefined) throw new Error(`missing ${name}`);
  return value;
};
type Job = {
  pageId: string;
  url: string;
  scripts: "on" | "policy";
  markers: string[];
  fingerprints: Record<string, { tag: string; text: string }>;
};
const jobs = JSON.parse(readFileSync(arg("--jobs"), "utf8")) as Job[];
const htmlDir = arg("--html");
const out = arg("--out");
const concurrency = Number(arg("--concurrency", "3"));
const store = new LabelStore(arg("--store"));
const blank = (quality: { visibleTextChars: number; visibleNodes: number }) =>
  quality.visibleTextChars < 200 || quality.visibleNodes < 30;

const { chromium } = await loadPlaywright();
let browser = await chromium.launch(replayLaunchOptions());
let cursor = 0;
let rendered = 0;
async function worker() {
  for (;;) {
    const job = jobs[cursor++];
    if (!job) return;
    const started = Date.now();
    try {
      const html = readFileSync(join(htmlDir, `${job.pageId}.html`), "utf8");
      const styled = promoteDeferredStyles(html);
      const replay = (scripts: boolean) =>
        captureReplayTarget(
          store,
          browser,
          {
            html: scripts ? html : styled.html,
            url: job.url,
            corpusRef: `cal:${job.pageId}`,
            markers: job.markers,
          },
          { scripts },
        );
      let scriptReason = job.scripts === "on" ? "platform" : "default_off";
      let result = await replay(job.scripts === "on");
      if (job.scripts !== "on" && blank(result.quality)) {
        result = await replay(true);
        scriptReason = "spa_shell";
      }
      const located: Record<string, object> = {};
      for (const [marker, location] of Object.entries(result.located)) {
        const node = location.nodeId
          ? result.page.nodes.find((item) => item.id === location.nodeId)
          : undefined;
        located[marker] = {
          ...location,
          tagMatches: Boolean(node && node.tag === job.fingerprints[marker]?.tag),
          rect: node?.rect ?? null,
        };
      }
      appendFileSync(
        out,
        `${JSON.stringify({
          pageId: job.pageId,
          captureId: result.page.id,
          scriptMode: result.page.scriptMode,
          scriptReason,
          promotedStyles: result.page.scriptMode === "disabled" ? styled.promoted : 0,
          renderSettled: result.page.renderSettled,
          width: result.page.width,
          height: result.page.height,
          heightCapped: result.page.heightCapped,
          quality: result.quality,
          located,
          ms: Date.now() - started,
        })}\n`,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message.slice(0, 200) : String(error);
      appendFileSync(
        out,
        `${JSON.stringify({ pageId: job.pageId, error: message, ms: Date.now() - started })}\n`,
      );
    }
    // A long-lived browser grows; recycle it between pages.
    if (++rendered % 40 === 0) {
      const old = browser;
      browser = await chromium.launch(replayLaunchOptions());
      setTimeout(() => void old.close().catch(() => undefined), 50_000);
    }
  }
}
await Promise.all(Array.from({ length: concurrency }, worker));
await browser.close();
process.exit(0);

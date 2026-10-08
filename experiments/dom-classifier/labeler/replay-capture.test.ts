import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Browser } from "playwright";
import {
  browserLaunchOptions,
  browserRuntimeUnavailableReason,
  loadPlaywright,
} from "./browser-runtime.ts";
import { captureReplayTarget, REPLAY_MARKER } from "./replay-capture.ts";
import { LabelStore } from "./store.ts";

const browserSkipReason = browserRuntimeUnavailableReason();
describe.skipIf(browserSkipReason !== null)(
  `historical replay (needs a browser)${browserSkipReason ? `: skipped, ${browserSkipReason}` : ""}`,
  () => {
    let browser!: Browser;
    let dataDir = "";
    beforeAll(async () => {
      dataDir = mkdtempSync(join(tmpdir(), "labeler-replay-"));
      const { chromium } = await loadPlaywright();
      browser = await chromium.launch(browserLaunchOptions());
    });
    afterAll(async () => {
      await browser?.close();
      rmSync(dataDir, { recursive: true, force: true });
    });

    // No network is touched: the document is served from the stored bytes and
    // the page references no subresources.
    const stored = `<!doctype html><html><head><title>Stored page</title></head><body>
      <header ${REPLAY_MARKER}="1" style="height:120px">Site header</header>
      <main style="height:2400px"><h1>Stored article</h1><p>Body text from the crawl.</p></main>
      <div ${REPLAY_MARKER}="2" style="position:fixed;bottom:0;left:0;width:100%;height:80px">We use cookies</div>
      <div ${REPLAY_MARKER}="3" style="display:none">Hidden until a script shows it</div>
      <script>
        document.body.dataset.ran = "yes";
        fetch("/api/live").then(() => { document.title = "refetched"; }).catch(() => {});
        setTimeout(() => { location.href = "https://elsewhere.example.test/"; }, 10);
      </script>
    </body></html>`;

    test("renders stored bytes, locates marked elements and blocks refetch and redirects", async () => {
      const store = new LabelStore(dataDir);
      const result = await captureReplayTarget(
        store,
        browser,
        {
          html: stored,
          url: "https://replay.example.test/articles/one",
          originalCrawledAt: "2026-09-24T00:00:00.000Z",
          corpusRef: "slice:abc123",
          markers: ["1", "2", "3", "4"],
        },
        { timeoutMs: 20_000 },
      );
      const page = result.page;
      expect(page).toMatchObject({
        title: "Stored page",
        url: "https://replay.example.test/articles/one",
        sourceKind: "historical_replay",
        assetMode: "hybrid",
        scriptMode: "enabled",
        originalCrawledAt: "2026-09-24T00:00:00.000Z",
        corpusRef: "slice:abc123",
      });
      expect(store.readPage(page.id).id).toBe(page.id);
      expect(result.located["1"]!.visible).toBe(true);
      expect(result.located["3"]).toEqual({ nodeId: null, visible: false, present: true });
      expect(result.located["4"]).toEqual({ nodeId: null, visible: false, present: false });
      // The fixed banner is drawn where it was measured: the bottom of the first screen.
      const banner = page.nodes.find((node) => node.id === result.located["2"]!.nodeId)!;
      expect(result.located["2"]!.visible).toBe(true);
      expect(banner.rect).toMatchObject({ y: 820, height: 80 });
      const header = page.nodes.find((node) => node.id === result.located["1"]!.nodeId)!;
      expect(header.tag).toBe("header");
      expect(result.quality.visibleTextChars).toBeGreaterThan(20);
    }, 40_000);

    test("can replay with scripts off", async () => {
      const store = new LabelStore(dataDir);
      const result = await captureReplayTarget(
        store,
        browser,
        { html: stored, url: "https://replay.example.test/articles/two", markers: ["1"] },
        { scripts: false, timeoutMs: 20_000 },
      );
      expect(result.page.scriptMode).toBe("disabled");
      expect(result.located["1"]!.visible).toBe(true);
    }, 40_000);

    test("refuses a private replay URL before rendering", async () => {
      const store = new LabelStore(dataDir);
      await expect(
        captureReplayTarget(store, browser, { html: stored, url: "https://127.0.0.1/" }),
      ).rejects.toThrow("Private network URL");
    });
  },
);

import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Browser } from "playwright";
import { browserRuntimeUnavailableReason, loadPlaywright } from "./browser-runtime.ts";
import {
  captureReplayTarget,
  promoteDeferredStyles,
  REPLAY_BROWSER_ARGS,
  REPLAY_MARKER,
  replayLaunchOptions,
} from "./replay-capture.ts";
import { LabelStore } from "./store.ts";

describe("replay launch options", () => {
  test("keep Chromium's OS sandbox on unless the caller opts out by name", () => {
    expect(replayLaunchOptions().chromiumSandbox).toBeTrue();
    expect(replayLaunchOptions({}).chromiumSandbox).toBeTrue();
    expect(replayLaunchOptions({ disableChromiumSandbox: false }).chromiumSandbox).toBeTrue();
    expect(replayLaunchOptions({ disableChromiumSandbox: true }).chromiumSandbox).toBeFalse();
    for (const options of [
      replayLaunchOptions(),
      replayLaunchOptions({ disableChromiumSandbox: true }),
    ])
      expect(options.args).toEqual([...REPLAY_BROWSER_ARGS]);
  });
});

/** This process's replay browsers and their renderers, by command line. */
function replayProcesses() {
  const rows = Bun.spawnSync(["ps", "-Ao", "pid=,ppid=,args="])
    .stdout.toString()
    .split("\n")
    .map((line) => /^\s*(\d+)\s+(\d+)\s+(.*)$/.exec(line))
    .filter((match) => match !== null)
    .map(([, pid, ppid, args]) => ({ pid: Number(pid), ppid: Number(ppid), args: args! }));
  const parent = new Map(rows.map((row) => [row.pid, row.ppid]));
  const descends = (pid: number, ancestors: Set<number>) => {
    for (let at = parent.get(pid); at !== undefined && at > 1; at = parent.get(at))
      if (ancestors.has(at)) return true;
    return false;
  };
  const browsers = rows.filter(
    (row) =>
      descends(row.pid, new Set([process.pid])) &&
      row.args.includes(REPLAY_BROWSER_ARGS[0]) &&
      !row.args.includes("--type="),
  );
  // Renderers are the browser's children on macOS, and a zygote's on Linux.
  const ids = new Set(browsers.map((row) => row.pid));
  const renderers = rows.filter(
    (row) => descends(row.pid, ids) && row.args.includes("--type=renderer"),
  );
  return { browsers, renderers };
}

const browserSkipReason = browserRuntimeUnavailableReason();
describe.skipIf(browserSkipReason !== null)(
  `historical replay (needs a browser)${browserSkipReason ? `: skipped, ${browserSkipReason}` : ""}`,
  () => {
    let browser!: Browser;
    let dataDir = "";
    beforeAll(async () => {
      dataDir = mkdtempSync(join(tmpdir(), "labeler-replay-"));
      const { chromium } = await loadPlaywright();
      browser = await chromium.launch(replayLaunchOptions());
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
        { scripts: true, timeoutMs: 20_000 },
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

    test("the replay browser and its renderers run inside Chromium's OS sandbox", async () => {
      const context = await browser.newContext();
      try {
        const page = await context.newPage();
        await page.setContent("<p>sandboxed</p>");
        const { browsers, renderers } = replayProcesses();
        expect(browsers.length).toBeGreaterThan(0);
        expect(renderers.length).toBeGreaterThan(0);
        for (const row of [...browsers, ...renderers])
          expect(row.args).not.toContain("--no-sandbox");
        // macOS renderers start under a seatbelt profile only when the sandbox is on.
        if (process.platform === "darwin")
          for (const row of renderers) expect(row.args).toContain("--seatbelt-client");
      } finally {
        await context.close();
      }
    });

    test("replays with scripts off unless asked", async () => {
      const store = new LabelStore(dataDir);
      const result = await captureReplayTarget(
        store,
        browser,
        { html: stored, url: "https://replay.example.test/articles/two", markers: ["1"] },
        { timeoutMs: 20_000 },
      );
      expect(result.page.scriptMode).toBe("disabled");
      expect(result.located["1"]!.visible).toBe(true);
    }, 40_000);

    test("a replayed page's scripts cannot reach a local service", async () => {
      // A stand-in for an unauthenticated local service: counts every HTTP hit
      // and every WebSocket upgrade it receives.
      const seen = { http: 0, sockets: 0 };
      const local = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch(request, server) {
          if (request.headers.get("upgrade") === "websocket" && server.upgrade(request)) {
            seen.sockets++;
            return undefined;
          }
          seen.http++;
          return new Response("ok", { headers: { "access-control-allow-origin": "*" } });
        },
        websocket: { message() {} },
      });
      try {
        const origin = `127.0.0.1:${local.port}`;
        const probe = `<!doctype html><html><head><title>Probe</title></head><body>
          <h1>Probe page</h1><p>${"Stored text. ".repeat(40)}</p>
          <img src="http://${origin}/img.png">
          <iframe src="http://${origin}/frame"></iframe>
          <link rel="stylesheet" href="http://${origin}/style.css">
          <script>
            const socket = new WebSocket("ws://${origin}/socket");
            socket.onopen = () => socket.send("hello");
            fetch("http://${origin}/fetch", { mode: "no-cors" }).catch(() => {});
            navigator.sendBeacon("http://${origin}/beacon", "x");
            window.open("http://${origin}/popup");
            new Worker(URL.createObjectURL(new Blob(['fetch("http://${origin}/worker").catch(() => {})'])));
          </script>
        </body></html>`;
        const store = new LabelStore(dataDir);
        await captureReplayTarget(
          store,
          browser,
          { html: probe, url: "https://replay.example.test/probe" },
          { scripts: true, timeoutMs: 20_000 },
        );
        await Bun.sleep(500);
        expect(seen).toEqual({ http: 0, sockets: 0 });
      } finally {
        local.stop(true);
      }
    }, 40_000);

    test("a page that never settles fails at the deadline instead of hanging", async () => {
      const store = new LabelStore(dataDir);
      const started = Date.now();
      await expect(
        captureReplayTarget(
          store,
          browser,
          {
            html: `<!doctype html><title>Busy</title><p>Busy page</p><script>setTimeout(() => { for (;;) {} }, 50);</script>`,
            url: "https://replay.example.test/busy",
          },
          { scripts: true, timeoutMs: 5_000 },
        ),
      ).rejects.toThrow("deadline");
      expect(Date.now() - started).toBeLessThan(15_000);
    }, 30_000);

    test("scripts-off replays apply stylesheets a handler would have switched on", () => {
      const stored = `<head>
        <link rel="preload" as="style" href="/a.css" onload="this.onload=null;this.rel='stylesheet'">
        <link rel="stylesheet" href="/b.css" media="print" onload="this.media='all'">
        <link rel="stylesheet" href="/c.css" media="none" onload="if(media!='all')media='all'">
        <link rel="stylesheet" href="/d.css" media="print">
        <link rel="preload" as="font" href="/e.woff2" onload="void 0">
      </head>`;
      const { html, promoted } = promoteDeferredStyles(stored);
      expect(promoted).toBe(3);
      expect(html).toContain('<link rel="stylesheet" as="style" href="/a.css">');
      expect(html).toContain('<link rel="stylesheet" href="/b.css" media="all">');
      expect(html).toContain('<link rel="stylesheet" href="/c.css" media="all">');
      // A real print stylesheet and a non-style preload are left alone.
      expect(html).toContain('<link rel="stylesheet" href="/d.css" media="print">');
      expect(html).toContain('<link rel="preload" as="font" href="/e.woff2" onload="void 0">');
    });

    test("refuses a private replay URL before rendering", async () => {
      const store = new LabelStore(dataDir);
      await expect(
        captureReplayTarget(store, browser, { html: stored, url: "https://127.0.0.1/" }),
      ).rejects.toThrow("Private network URL");
    });
  },
);

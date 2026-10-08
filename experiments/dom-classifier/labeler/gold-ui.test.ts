import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Browser } from "playwright";
import {
  browserLaunchOptions,
  browserRuntimeUnavailableReason,
  loadPlaywright,
} from "./browser-runtime.ts";
import { createLabelerServer } from "./server.ts";
import { LabelStore } from "./store.ts";
import type { CapturedPage } from "./types.ts";

// Same contract as ui-smoke: without Playwright the suite reports as skipped.
const browserSkipReason = browserRuntimeUnavailableReason();
describe.skipIf(browserSkipReason !== null)(
  `gold review UI (needs a browser)${browserSkipReason ? `: skipped, ${browserSkipReason}` : ""}`,
  () => {
    const labelerDir = import.meta.dir;
    const captureHash = `sha256:${"e".repeat(64)}`;
    const aboutPageId = "page_aaaaaaaaaaaaaaaaaaaaaa01";
    const pricingPageId = "page_aaaaaaaaaaaaaaaaaaaaaa02";
    const bannerNodeId = "node_aaaaaaaaaaaaaaaaaaaaaa01";
    const ids = {
      about: "msug_00000000-0000-4000-8000-0000000000a1",
      banner: "msug_00000000-0000-4000-8000-0000000000a2",
      pricing: "msug_00000000-0000-4000-8000-0000000000a3",
    };
    const png = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/sgAAAABJRU5ErkJggg==",
      "base64",
    );
    let browser!: Browser;
    let dataDir = "";
    let cohortDir = "";
    let store!: LabelStore;
    let server!: ReturnType<typeof Bun.serve>;

    const capture = (id: string, capturedAt: string): CapturedPage => ({
      id,
      url: `https://example.test/${id}`,
      title: `Gold ${id.slice(-2)}`,
      capturedAt,
      contentHash: captureHash,
      captureHash,
      width: 1440,
      height: 900,
      viewport: { width: 1440, height: 900, deviceScaleFactor: 1 },
      screenshotUrl: `/captures/${id}.png`,
      split: "training-review",
      sourceKind: "historical_replay",
      assetMode: "hybrid",
      nodes: [
        {
          id: bannerNodeId,
          parentId: null,
          tag: "div",
          role: "dialog",
          text: "We use cookies",
          selector: "html > body:nth-of-type(1) > div:nth-of-type(1)",
          rect: { x: 0, y: 700, width: 1440, height: 200 },
          depth: 2,
          suggestion: null,
        },
      ],
    });
    const row = (id: string, pageId: string, nodeId: string | null, fields: object) => ({
      schemaVersion: 1,
      id,
      pageId,
      nodeId,
      captureHash,
      provider: "typesafe",
      modelId: "gold-v1",
      modelRevision: "v1",
      promptRevision: "gold-v1",
      snapshotHash: captureHash,
      rawAnswers: null,
      provisional: true,
      createdAt: "2026-10-09T00:00:00.000Z",
      ...fields,
    });

    beforeAll(async () => {
      dataDir = mkdtempSync(join(tmpdir(), "labeler-gold-ui-"));
      cohortDir = mkdtempSync(join(tmpdir(), "labeler-gold-ui-cohort-"));
      mkdirSync(join(cohortDir, "captures"));
      for (const [id, at] of [
        [aboutPageId, "2026-10-09T00:00:02.000Z"],
        [pricingPageId, "2026-10-09T00:00:01.000Z"],
      ] as const) {
        writeFileSync(join(cohortDir, "captures", `${id}.json`), JSON.stringify(capture(id, at)));
        writeFileSync(join(cohortDir, "captures", `${id}.png`), png);
      }
      writeFileSync(
        join(cohortDir, "model-suggestions.jsonl"),
        [
          row(ids.about, aboutPageId, null, {
            mappedLabels: { pageTypeV8: ["about"] },
            alternatives: { pageTypeV8: ["contact"] },
          }),
          row(ids.banner, aboutPageId, bannerNodeId, {
            mappedLabels: { sectionPurposeV3: ["cookie_consent"] },
            alternatives: { sectionPurposeV3: ["none", "promo_banner"] },
          }),
          row(ids.pricing, pricingPageId, null, { mappedLabels: { pageTypeV8: ["pricing"] } }),
        ]
          .map((item) => JSON.stringify(item))
          .join("\n") + "\n",
      );
      const cohorts = [{ id: "gold-ui", dir: cohortDir }];
      store = new LabelStore(dataDir, cohorts);
      const existingPort = process.env.PORT;
      process.env.PORT = "0";
      server = createLabelerServer({ dataDir, staticDir: labelerDir, cohorts });
      if (existingPort === undefined) delete process.env.PORT;
      else process.env.PORT = existingPort;
      const { chromium } = await loadPlaywright();
      browser = await chromium.launch(browserLaunchOptions());
    });

    afterAll(async () => {
      await browser?.close();
      server?.stop(true);
      rmSync(dataDir, { recursive: true, force: true });
      rmSync(cohortDir, { recursive: true, force: true });
    });

    test("answers, fixes, skips and undoes gold questions from the keyboard", async () => {
      const page = await browser.newPage({ viewport: { width: 1260, height: 900 } });
      const posted = (path: string) =>
        page.waitForResponse(
          (response) => response.url().endsWith(path) && response.status() === 201,
        );
      const asked = (text: string) =>
        page.waitForFunction(
          (expected) => document.querySelector("#gold-question")?.textContent?.trim() === expected,
          text,
          { timeout: 10_000 },
        );
      try {
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        await page.goto(server.url.toString());
        await asked("Is this page an About page?");
        expect(errors).toEqual([]);
        expect(await page.locator("#gold-tab").getAttribute("aria-selected")).toBe("true");
        expect(await page.locator("#gold-definition").textContent()).toContain("story, mission");
        // The pane never names which annotator proposed the class.
        expect(await page.locator("#gold-pane").textContent()).not.toContain("Jev");

        let saved = posted("/api/page-annotations");
        await page.keyboard.press("Enter");
        await saved;
        await asked("Is the outlined element a cookie consent banner?");
        await page.locator(`.node-region.gold-target[data-node-id="${bannerNodeId}"]`).waitFor();
        expect(await page.locator("#gold-rule").isHidden()).toBe(false);

        await page.keyboard.press("x");
        await page.locator("#gold-fix");
        expect(await page.locator("#gold-fix-heading").textContent()).toBe(
          "Not Cookie consent. What is it instead?",
        );
        expect(await page.locator("#gold-alternatives button").first().textContent()).toBe(
          "1 None",
        );
        saved = posted("/api/annotations");
        await page.keyboard.press("1");
        await saved;
        await asked("Is this page a Pricing page?");

        await page.keyboard.press("j");
        await asked("Only skipped questions are left.");

        const undone = posted("/api/review-actions/undo");
        await page.keyboard.press("Meta+z");
        await undone;
        await asked("Is the outlined element a cookie consent banner?");

        await page.keyboard.press("x");
        saved = posted("/api/model-reviews");
        await page.keyboard.press("x");
        await saved;

        const queue = store.goldQueue();
        expect(queue.map((item) => item.status)).toEqual(["accepted", "rejected", "pending"]);
        expect(store.pageAnnotationsForPage(aboutPageId).at(-1)).toMatchObject({
          pageTypesV8: ["about"],
          modelSuggestionId: ids.about,
          modelReview: "accept",
          cohortId: "gold-ui",
        });
        // The undone fix stays in the journal, superseded by an undo record.
        expect(store.readAnnotations().at(-1)).toMatchObject({
          sectionPurposesV3: ["none"],
          modelReview: "correct",
        });
        expect(store.annotationsForPage(aboutPageId)).toHaveLength(0);
      } finally {
        await page.close();
      }
    }, 60_000);
  },
);

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
    const footerNodeId = "node_aaaaaaaaaaaaaaaaaaaaaa02";
    const ids = {
      about: "msug_00000000-0000-4000-8000-0000000000a1",
      banner: "msug_00000000-0000-4000-8000-0000000000a2",
      pricing: "msug_00000000-0000-4000-8000-0000000000a3",
      pricingBanner: "msug_00000000-0000-4000-8000-0000000000a4",
      pricingFooter: "msug_00000000-0000-4000-8000-0000000000a5",
      commentBlind: "msug_00000000-0000-4000-8000-0000000000a6",
      commentShown: "msug_00000000-0000-4000-8000-0000000000a7",
    };
    const commentNodeId = "node_aaaaaaaaaaaaaaaaaaaaaa03";
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
        {
          id: footerNodeId,
          parentId: null,
          tag: "footer",
          role: null,
          text: "Footer links",
          selector: "html > body:nth-of-type(1) > footer:nth-of-type(1)",
          rect: { x: 0, y: 500, width: 1440, height: 120 },
          depth: 2,
          suggestion: null,
        },
        {
          id: commentNodeId,
          parentId: null,
          tag: "li",
          role: null,
          text: "Great post, thanks",
          selector: "html > body:nth-of-type(1) > li:nth-of-type(1)",
          rect: { x: 0, y: 300, width: 900, height: 80 },
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
          // Blind: no suggestion and no alternatives reach the card.
          row(ids.pricing, pricingPageId, null, { mappedLabels: {}, blindAxis: "pageTypeV8" }),
          row(ids.pricingBanner, pricingPageId, bannerNodeId, {
            mappedLabels: { sectionPurposeV3: ["promo_banner"] },
          }),
          row(ids.pricingFooter, pricingPageId, footerNodeId, {
            mappedLabels: {},
            blindAxis: "sectionPurposeV3",
          }),
          row(ids.commentBlind, pricingPageId, commentNodeId, {
            mappedLabels: {},
            blindAxis: "sectionPurposeV3",
          }),
          row(ids.commentShown, pricingPageId, commentNodeId, {
            mappedLabels: { sectionPurposeV3: ["comments"] },
          }),
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

    test("Y, N and S on shown questions, the picker on blind ones, unsure and undo", async () => {
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
        // One clean card: no queue panel, and no annotator named anywhere.
        expect(await page.locator(".queue-panel").isVisible()).toBe(false);
        expect(await page.locator("#gold-pane").textContent()).not.toContain("Jev");

        // Every question has a short id on the card and in the address bar.
        expect(await page.locator("#gold-item-id").textContent()).toBe("gold-ui-0");
        expect(new URL(page.url()).hash).toBe("#gold-ui-0");

        // Y: yes, this class.
        let saved = posted("/api/page-annotations");
        await page.keyboard.press("y");
        await saved;
        await asked("Is the outlined element a cookie consent banner?");
        await page.locator(`.node-region.gold-target[data-node-id="${bannerNodeId}"]`).waitFor();

        // S: suggest the right class; type to filter, Enter takes the top match.
        await page.keyboard.press("s");
        await page.locator("#gold-fix").waitFor();
        expect(await page.locator("#gold-fix-heading").textContent()).toBe(
          "Not Cookie consent? Pick the right class",
        );
        await page.keyboard.type("no");
        expect(await page.locator("#gold-filter").textContent()).toBe("no");
        expect(await page.locator("#gold-alternatives button").first().textContent()).toBe(
          "1 None",
        );
        saved = posted("/api/annotations");
        await page.keyboard.press("Enter");
        await saved;

        // Blind page: no suggestion, the picker is open at once.
        await asked("What is this page?");
        expect(await page.locator("#gold-actions").isHidden()).toBe(true);
        expect(await page.locator("#gold-fix").isVisible()).toBe(true);
        await page.keyboard.type("pri");
        expect(await page.locator("#gold-definition").textContent()).toContain("Pricing");
        saved = posted("/api/page-annotations");
        await page.keyboard.press("Enter");
        await saved;

        // N: no, not this class, no fix.
        await asked("Is the outlined element the site's own promo or announcement bar?");
        saved = posted("/api/model-reviews");
        await page.keyboard.press("n");
        await saved;

        // Blind section: U is unsure.
        await asked("What is the outlined element?");
        saved = posted("/api/annotations");
        await page.keyboard.press("u");
        await saved;

        // Blind section: I asks which section the outlined ITEM belongs to.
        await page.waitForFunction(() =>
          document.querySelector("#gold-progress")?.textContent?.startsWith("5 /"),
        );
        await asked("What is the outlined element?");
        await page.keyboard.press("i");
        expect(await page.locator("#gold-fix-heading").textContent()).toBe(
          "An individual item of which section?",
        );
        await page.keyboard.type("com");
        saved = posted("/api/annotations");
        await page.keyboard.press("Enter");
        await saved;

        // Shown section: I records "an item of" the shown class directly.
        await asked("Is the outlined element a comments section?");
        expect(await page.locator("#gold-keys").textContent()).toContain("I an item of it");
        saved = posted("/api/annotations");
        await page.keyboard.press("i");
        await saved;
        await asked("Every gold question has an answer.");

        // Cmd-Z undoes the last answer, then J skips the reopened question.
        const undone = posted("/api/review-actions/undo");
        await page.keyboard.press("Meta+z");
        await undone;
        await asked("Is the outlined element a comments section?");
        await page.keyboard.press("j");
        await asked("Only skipped questions are left.");

        expect(store.goldQueue().map((item) => [item.blind, item.status])).toEqual([
          [false, "accepted"],
          [false, "corrected"],
          [true, "corrected"],
          [false, "rejected"],
          [true, "unsure"],
          [true, "corrected"],
          [false, "pending"],
        ]);
        expect(store.pageAnnotationsForPage(pricingPageId).at(-1)).toMatchObject({
          pageTypesV8: ["pricing"],
          modelSuggestionId: ids.pricing,
          modelReview: "correct",
        });
        const effective = store.effectiveAnnotations();
        expect(effective.find((row) => row.modelSuggestionId === ids.pricingFooter)).toMatchObject({
          decision: "unsure",
        });
        expect(effective.find((row) => row.modelSuggestionId === ids.commentBlind)).toMatchObject({
          sectionPurposesV3: ["none"],
          sectionItemOfV3: "comments",
          modelReview: "correct",
        });
        // The undone shown-item answer stays in the journal, undone.
        expect(store.readAnnotations().at(-1)).toMatchObject({
          sectionItemOfV3: "comments",
          modelSuggestionId: ids.commentShown,
        });
        expect(effective.some((row) => row.modelSuggestionId === ids.commentShown)).toBe(false);
        expect(errors).toEqual([]);
      } finally {
        await page.close();
      }
    }, 60_000);

    test("a deep link opens that question, answered or not; an unknown id says so", async () => {
      const page = await browser.newPage({ viewport: { width: 1260, height: 900 } });
      const asked = (text: string) =>
        page.waitForFunction(
          (expected) => document.querySelector("#gold-question")?.textContent?.trim() === expected,
          text,
          { timeout: 10_000 },
        );
      try {
        // Answered in the test above: it opens with its saved answer.
        await page.goto(`${server.url}#gold-ui-0`);
        await asked("Is this page an About page?");
        expect(await page.locator("#gold-status").textContent()).toContain("Saved: yes");
        // The long suggestion id works too, and moving updates the address bar.
        await page.goto(`${server.url}#${ids.pricing}`);
        await page.reload();
        await asked("What is this page?");
        expect(await page.locator("#gold-item-id").textContent()).toBe("gold-ui-2");
        expect(new URL(page.url()).hash).toBe("#gold-ui-2");
        // Pasting another id into the address bar jumps without a reload.
        await page.evaluate(() => {
          location.hash = "#gold-ui-1";
        });
        await asked("Is the outlined element a cookie consent banner?");
        // An unknown id, opened fresh, says so and shows the next open question.
        const fresh = await browser.newPage({ viewport: { width: 1260, height: 900 } });
        try {
          await fresh.goto(`${server.url}#nope-7`);
          await fresh.waitForFunction(() =>
            document.querySelector("#gold-status")?.textContent?.includes("No such item: nope-7"),
          );
        } finally {
          await fresh.close();
        }
      } finally {
        await page.close();
      }
    }, 60_000);
  },
);

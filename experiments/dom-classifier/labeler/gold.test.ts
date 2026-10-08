import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { handleRequest } from "./server.ts";
import { LabelStore } from "./store.ts";
import {
  PAGE_TYPES_V8,
  PAGE_TYPE_V8_QUESTIONS,
  SECTION_PURPOSES_V3,
  SECTION_PURPOSE_V3_QUESTIONS,
  type CapturedPage,
} from "./types.ts";

const tempDirectories: string[] = [];
afterEach(() => {
  for (const directory of tempDirectories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

function temp(prefix: string) {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  tempDirectories.push(directory);
  return directory;
}

const captureHash = `sha256:${"b".repeat(64)}`;
const goldPageId = "page_3333333333333333333333cc";
const bannerNodeId = "node_3333333333333333333333cc";
const primaryPageId = "page_1111111111111111111111aa";
const primaryNodeId = "node_1111111111111111111111aa";

function page(id: string, nodeId: string): CapturedPage {
  return {
    id,
    url: "https://example.test/about",
    title: "About us",
    capturedAt: "2026-10-09T00:00:00.000Z",
    contentHash: `sha256:${"a".repeat(64)}`,
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
        id: nodeId,
        parentId: null,
        tag: "div",
        role: "dialog",
        text: "We use cookies. Accept all",
        selector: "html > body:nth-of-type(1) > div:nth-of-type(1)",
        rect: { x: 0, y: 700, width: 1440, height: 200 },
        depth: 2,
        suggestion: null,
      },
    ],
  };
}

let sequence = 0;
function suggestion(fields: Record<string, unknown>) {
  sequence += 1;
  return {
    schemaVersion: 1,
    id: `msug_00000000-0000-4000-8000-${String(sequence).padStart(12, "0")}`,
    pageId: goldPageId,
    nodeId: null,
    captureHash,
    provider: "typesafe",
    modelId: "gold-v1",
    modelRevision: "v8.1",
    promptRevision: "dom-suggestions-v8.1",
    snapshotHash: captureHash,
    rawAnswers: null,
    provisional: true,
    createdAt: "2026-10-09T00:00:00.000Z",
    ...fields,
  };
}

/** A primary store with one existing label, plus a mounted gold cohort. */
function goldFixture(rows: Array<Record<string, unknown>>) {
  const dataDir = temp("labeler-gold-primary-");
  const seed = new LabelStore(dataDir);
  seed.ensureDirectories();
  writeFileSync(
    join(dataDir, "captures", `${primaryPageId}.json`),
    JSON.stringify(page(primaryPageId, primaryNodeId)),
  );
  writeFileSync(join(dataDir, "captures", `${primaryPageId}.png`), "png-bytes");
  seed.saveAnnotation({
    pageId: primaryPageId,
    nodeId: primaryNodeId,
    decision: "label",
    regions: ["footer"],
    boundary: "correct",
    clientRequestId: "existing_label_1",
  });
  const cohortDir = temp("labeler-gold-cohort-");
  mkdirSync(join(cohortDir, "captures"));
  writeFileSync(
    join(cohortDir, "captures", `${goldPageId}.json`),
    JSON.stringify(page(goldPageId, bannerNodeId)),
  );
  writeFileSync(join(cohortDir, "captures", `${goldPageId}.png`), "png-bytes");
  writeFileSync(
    join(cohortDir, "model-suggestions.jsonl"),
    `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`,
  );
  const store = new LabelStore(dataDir, [{ id: "gold-v1", dir: cohortDir }]);
  return { dataDir, store };
}

const pageQuestion = () =>
  suggestion({
    mappedLabels: { pageTypeV8: ["about"] },
    alternatives: { pageTypeV8: ["contact", "other"] },
  });
const sectionQuestion = () =>
  suggestion({
    nodeId: bannerNodeId,
    promptRevision: "section-purpose-v3",
    mappedLabels: { sectionPurposeV3: ["cookie_consent"] },
    alternatives: { sectionPurposeV3: ["none", "promo_banner"] },
  });

async function request(store: LabelStore, path: string, body?: unknown) {
  const token = "c".repeat(32);
  const response = await handleRequest(
    new Request(`http://127.0.0.1:4317${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: {
        host: "127.0.0.1:4317",
        origin: "http://127.0.0.1:4317",
        cookie: `labeler_csrf=${token}`,
        "x-labeler-csrf": token,
        "content-type": "application/json",
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
    store,
  );
  return { status: response.status, body: (await response.json()) as Record<string, any> };
}

describe("gold review axes", () => {
  test("every class has exactly one question and policy one-liner", () => {
    expect(PAGE_TYPE_V8_QUESTIONS.map((entry) => entry.label)).toEqual([...PAGE_TYPES_V8]);
    expect(SECTION_PURPOSE_V3_QUESTIONS.map((entry) => entry.label)).toEqual([
      ...SECTION_PURPOSES_V3,
    ]);
    for (const entry of [...PAGE_TYPE_V8_QUESTIONS, ...SECTION_PURPOSE_V3_QUESTIONS]) {
      expect(entry.question).toEndWith("?");
      expect(entry.definition.length).toBeGreaterThan(10);
    }
  });

  test("section classes are the head's: the v2 class list minus the three folded into none", () => {
    const listed = (
      JSON.parse(
        readFileSync(
          join(import.meta.dir, "..", "v2", "labels", "section-classes-v2.json"),
          "utf8",
        ),
      ) as { classes: Array<{ name: string }> }
    ).classes.map((entry) => entry.name);
    const folded = ["recipe_card", "event_details", "chat_widget"];
    expect(([...SECTION_PURPOSES_V3] as string[]).sort()).toEqual(
      listed.filter((name) => !folded.includes(name)).sort(),
    );
  });

  test("serves the catalogue and the queue in sidecar order with alternatives", async () => {
    const first = sectionQuestion();
    const second = pageQuestion();
    const { store } = goldFixture([first, second]);
    const pages = await request(store, "/api/pages");
    expect(pages.body.gold.pageTypeV8).toHaveLength(12);
    expect(pages.body.gold.sectionPurposeV3).toHaveLength(19);
    expect(pages.body.gold.sectionGranularity).toContain("fragment");
    const queue = await request(store, "/api/gold-queue");
    expect(queue.body.items).toEqual([
      {
        suggestionId: first.id,
        pageId: goldPageId,
        nodeId: bannerNodeId,
        captureHash,
        cohortId: "gold-v1",
        axis: "sectionPurposeV3",
        label: "cookie_consent",
        blind: false,
        alternatives: ["none", "promo_banner"],
        status: "pending",
      },
      {
        suggestionId: second.id,
        pageId: goldPageId,
        nodeId: null,
        captureHash,
        cohortId: "gold-v1",
        axis: "pageTypeV8",
        label: "about",
        blind: false,
        alternatives: ["contact", "other"],
        status: "pending",
      },
    ]);
  });

  test("OK saves the shown class on the gold axis and leaves other rows unchanged", async () => {
    const question = pageQuestion();
    const { dataDir, store } = goldFixture([question]);
    const journalBefore = readFileSync(join(dataDir, "annotations.jsonl"), "utf8");
    const saved = await request(store, "/api/page-annotations", {
      pageId: goldPageId,
      decision: "label",
      pageTypesV8: ["about"],
      captureHash,
      clientRequestId: "gold_accept_1",
      modelSuggestionId: question.id,
      modelReview: "accept",
    });
    expect(saved.status).toBe(201);
    expect(saved.body.annotation).toMatchObject({
      pageTypesV8: ["about"],
      pageTypes: [],
      modelReview: "accept",
      cohortId: "gold-v1",
    });
    expect(store.goldQueue()[0]!.status).toBe("accepted");
    expect(readFileSync(join(dataDir, "annotations.jsonl"), "utf8")).toBe(journalBefore);
    const existing = store.readAnnotations()[0]!;
    expect("sectionPurposesV3" in existing).toBeFalse();
  });

  test("an acceptance must carry the shown class", async () => {
    const question = pageQuestion();
    const { store } = goldFixture([question]);
    const wrong = await request(store, "/api/page-annotations", {
      pageId: goldPageId,
      decision: "label",
      pageTypesV8: ["contact"],
      captureHash,
      clientRequestId: "gold_accept_wrong",
      modelSuggestionId: question.id,
      modelReview: "accept",
    });
    expect(wrong.status).toBe(400);
    const two = await request(store, "/api/page-annotations", {
      pageId: goldPageId,
      decision: "label",
      pageTypesV8: ["about", "contact"],
      captureHash,
      clientRequestId: "gold_accept_two",
    });
    expect(two.status).toBe(400);
    expect(store.goldQueue()[0]!.status).toBe("pending");
  });

  test("a one-click fix is a correction bound to the question", async () => {
    const question = sectionQuestion();
    const { store } = goldFixture([question]);
    const fixed = await request(store, "/api/annotations", {
      pageId: goldPageId,
      nodeId: bannerNodeId,
      decision: "label",
      sectionPurposesV3: ["promo_banner"],
      boundary: "correct",
      captureHash,
      clientRequestId: "gold_fix_1",
      modelSuggestionId: question.id,
      modelReview: "correct",
    });
    expect(fixed.status).toBe(201);
    expect(fixed.body.annotation).toMatchObject({
      sectionPurposesV3: ["promo_banner"],
      regions: [],
      modelReview: "correct",
    });
    expect(store.goldQueue()[0]!.status).toBe("corrected");
    // A retry of the same request is idempotent.
    const retry = await request(store, "/api/annotations", {
      pageId: goldPageId,
      nodeId: bannerNodeId,
      decision: "label",
      sectionPurposesV3: ["promo_banner"],
      boundary: "correct",
      captureHash,
      clientRequestId: "gold_fix_1",
      modelSuggestionId: question.id,
      modelReview: "correct",
    });
    expect(retry.body.annotation.id).toBe(fixed.body.annotation.id);
  });

  test("Not OK on a page question is a rejection that can be undone", async () => {
    const question = pageQuestion();
    const { store } = goldFixture([question]);
    const rejected = await request(store, "/api/model-reviews", {
      pageId: goldPageId,
      nodeId: null,
      modelSuggestionId: question.id,
      review: "reject",
      captureHash,
      clientRequestId: "gold_reject_1",
    });
    expect(rejected.status).toBe(201);
    expect(store.goldQueue()[0]!.status).toBe("rejected");
    const undone = await request(store, "/api/review-actions/undo", {
      pageId: goldPageId,
      nodeId: null,
      captureHash,
      action: { kind: "model_review", id: rejected.body.modelReview.id },
    });
    expect(undone.status).toBe(201);
    expect(store.goldQueue()[0]!.status).toBe("pending");
  });

  test("the latest answer wins after a re-answer, and undo restores the earlier one", () => {
    const question = sectionQuestion();
    const { store } = goldFixture([question]);
    store.saveAnnotation({
      pageId: goldPageId,
      nodeId: bannerNodeId,
      decision: "label",
      sectionPurposesV3: ["cookie_consent"],
      boundary: "correct",
      captureHash,
      clientRequestId: "gold_first_answer",
      modelSuggestionId: question.id,
      modelReview: "accept",
    });
    const second = store.saveModelReview({
      pageId: goldPageId,
      nodeId: bannerNodeId,
      modelSuggestionId: question.id,
      review: "reject",
      captureHash,
      clientRequestId: "gold_second_answer",
    });
    expect(store.goldQueue()[0]!.status).toBe("rejected");
    store.undoReviewAction({
      pageId: goldPageId,
      nodeId: bannerNodeId,
      captureHash,
      actionKind: "model_review",
      actionId: second.id,
    });
    expect(store.goldQueue()[0]!.status).toBe("accepted");
  });

  test("ignores malformed gold rows instead of guessing", () => {
    const { store } = goldFixture([
      // Page type asked about a node.
      suggestion({ nodeId: bannerNodeId, mappedLabels: { pageTypeV8: ["about"] } }),
      // Two gold values.
      suggestion({ mappedLabels: { pageTypeV8: ["about", "contact"] } }),
      // Alternatives repeating the suggestion.
      suggestion({
        mappedLabels: { pageTypeV8: ["about"] },
        alternatives: { pageTypeV8: ["about"] },
      }),
      // A gold axis mixed with a descriptive one.
      suggestion({ mappedLabels: { pageTypeV8: ["about"], pageTypes: ["about"] } }),
      // Not a class.
      suggestion({ nodeId: bannerNodeId, mappedLabels: { sectionPurposeV3: ["chat_widget"] } }),
      pageQuestion(),
    ]);
    expect(store.goldQueue().map((item) => item.label)).toEqual(["about"]);
  });

  test("a blind question shows no class; it takes a picked class or unsure, never an accept", async () => {
    const blindPage = suggestion({ mappedLabels: {}, blindAxis: "pageTypeV8" });
    const blindNode = suggestion({
      nodeId: bannerNodeId,
      mappedLabels: {},
      blindAxis: "sectionPurposeV3",
    });
    const { store } = goldFixture([
      blindPage,
      blindNode,
      // Malformed: a blind row may not carry a class, alternatives, or the wrong target.
      suggestion({ mappedLabels: { pageTypeV8: ["about"] }, blindAxis: "pageTypeV8" }),
      suggestion({
        mappedLabels: {},
        blindAxis: "pageTypeV8",
        alternatives: { pageTypeV8: ["about"] },
      }),
      suggestion({ nodeId: bannerNodeId, mappedLabels: {}, blindAxis: "pageTypeV8" }),
    ]);
    const queue = await request(store, "/api/gold-queue");
    expect(
      queue.body.items.map((item: Record<string, unknown>) => [
        item.label,
        item.blind,
        item.alternatives,
      ]),
    ).toEqual([
      [null, true, []],
      [null, true, []],
    ]);
    const accept = await request(store, "/api/page-annotations", {
      pageId: goldPageId,
      decision: "label",
      pageTypesV8: ["about"],
      captureHash,
      clientRequestId: "blind_accept_1",
      modelSuggestionId: blindPage.id,
      modelReview: "accept",
    });
    expect(accept.status).toBe(400);
    const picked = await request(store, "/api/page-annotations", {
      pageId: goldPageId,
      decision: "label",
      pageTypesV8: ["contact"],
      captureHash,
      clientRequestId: "blind_pick_1",
      modelSuggestionId: blindPage.id,
      modelReview: "correct",
    });
    expect(picked.status).toBe(201);
    const unsure = await request(store, "/api/annotations", {
      pageId: goldPageId,
      nodeId: bannerNodeId,
      decision: "unsure",
      boundary: "unsure",
      captureHash,
      clientRequestId: "blind_unsure_1",
      modelSuggestionId: blindNode.id,
      modelReview: "correct",
    });
    expect(unsure.status).toBe(201);
    expect(store.goldQueue().map((item) => item.status)).toEqual(["corrected", "unsure"]);
  });

  test("gold answers count in the review stats like any model review", () => {
    const question = pageQuestion();
    const { store } = goldFixture([question]);
    expect(store.stats().suggestions.page).toBe(1);
    store.savePageAnnotation({
      pageId: goldPageId,
      decision: "label",
      pageTypesV8: ["about"],
      captureHash,
      clientRequestId: "gold_stats_1",
      modelSuggestionId: question.id,
      modelReview: "accept",
    });
    const stats = store.stats();
    expect(stats.suggestions.reviewed.accepted).toBe(1);
    expect(stats.byCohort["gold-v1"]!.labelledPages).toBe(1);
  });
});

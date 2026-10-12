#!/usr/bin/env bun
/**
 * Score artifacts on teacher-labelled CALIBRATION-domain items (#2307).
 *
 *   bun run experiments/eval-teacher-calibration.ts --items eval-calibration-items.jsonl \
 *     --jobs <render dir>/jobs --artifact a=artifact3.json --artifact b=r1.json [--json out.json]
 *
 * The items are the teacher-accepted labels (`adjudicate/teacher_overrides.py --eval-out`) on sealed
 * domains with role `calibration`: pages no training run sees, labelled by GPT and Opus, not by a
 * human, and not gold. Each artifact runs through `classifyPage` as shipped (guard, comments rule,
 * site_chrome fallback included). Page type: a prediction FIRES when the model answered
 * (`source: model`); precision is over fired items and coverage is fired / all. Section: a node fires
 * when it carries a section; "none" is the absence of one, so precision and coverage are over the
 * teacher's non-"none" items plus every fired node, and `noneKept` is the share of teacher-"none"
 * items the artifact correctly left without a section.
 * Page HTML is the render job file (the section target carries a replay marker attribute only).
 * Counts only: no URLs or page text are printed. Items carry the page's real URL (private file).
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

import { decode } from "./link-position.ts";
import { classifyPage, tryLoadModel } from "../runtime/predict.ts";

interface Item {
  itemId: string;
  axis: "page" | "section";
  pageId: string;
  nodeId: number | null;
  label: string;
  platform: string | null;
  /** The page's own URL: the page-type head reads it, and link features need the host. */
  url: string;
}

function argValues(name: string): string[] {
  const out: string[] = [];
  Bun.argv.forEach((value, index) => {
    if (value === name && Bun.argv[index + 1]) out.push(Bun.argv[index + 1]!);
  });
  return out;
}

const items = readFileSync(argValues("--items")[0]!, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line) as Item);
const jobs = argValues("--jobs")[0]!;
const artifacts = argValues("--artifact").map((spec) => {
  const [name, path] = spec.split("=") as [string, string];
  const { model, reason } = tryLoadModel(JSON.parse(readFileSync(path, "utf8")));
  if (!model) throw new Error(`${name}: ${reason}`);
  return { name, model };
});

const byPage = new Map<string, Item[]>();
for (const item of items) byPage.set(item.pageId, [...(byPage.get(item.pageId) ?? []), item]);

type Tally = Record<string, number>;
const tallies = new Map<string, Tally>(artifacts.map((a) => [a.name, {}]));
const bump = (name: string, key: string, by = 1) => {
  const t = tallies.get(name)!;
  t[key] = (t[key] ?? 0) + by;
};
let missing = 0;
for (const [pageId, pageItems] of byPage) {
  const file = `${jobs}/${pageId}.html.gz`;
  if (!existsSync(file)) {
    missing += pageItems.length;
    continue;
  }
  const html = decode(gunzipSync(readFileSync(file)));
  const url = pageItems[0]!.url;
  for (const { name, model } of artifacts) {
    const result = classifyPage(html, url, { model });
    for (const item of pageItems) {
      if (item.axis === "page") {
        bump(name, "page:n");
        const fired = result.pageType.source === "model";
        if (fired) {
          bump(name, "page:fired");
          if (result.pageType.label === item.label) bump(name, "page:firedCorrect");
        }
        if (result.pageType.label === item.label) bump(name, "page:anyCorrect");
      } else {
        const node = result.nodes.find((n) => n.nodeRef.nodeId === item.nodeId);
        const predicted = node?.section?.label ?? null;
        bump(name, "section:n");
        if (item.label === "none") {
          bump(name, "section:teacherNone");
          if (predicted === null) bump(name, "section:noneKept");
        } else {
          bump(name, "section:teacherSection");
        }
        if (predicted !== null) {
          bump(name, "section:fired");
          if (predicted === item.label) bump(name, "section:firedCorrect");
        }
      }
    }
  }
}

const rate = (a = 0, b = 0) => (b > 0 ? Math.round((a / b) * 1000) / 1000 : null);
const report = Object.fromEntries(
  artifacts.map(({ name }) => {
    const t = tallies.get(name)!;
    return [
      name,
      {
        page: {
          n: t["page:n"] ?? 0,
          coverage: rate(t["page:fired"], t["page:n"]),
          precisionFired: rate(t["page:firedCorrect"], t["page:fired"]),
          accuracyAll: rate(t["page:anyCorrect"], t["page:n"]),
        },
        section: {
          n: t["section:n"] ?? 0,
          fired: t["section:fired"] ?? 0,
          precisionFired: rate(t["section:firedCorrect"], t["section:fired"]),
          recallOfTeacherSections: rate(t["section:firedCorrect"], t["section:teacherSection"]),
          noneKept: rate(t["section:noneKept"], t["section:teacherNone"]),
        },
      },
    ];
  }),
);
console.log(
  JSON.stringify(
    { items: items.length, pages: byPage.size, missingHtml: missing, report },
    null,
    1,
  ),
);
const out = argValues("--json")[0];
if (out) writeFileSync(out, `${JSON.stringify({ items: items.length, report }, null, 1)}\n`);

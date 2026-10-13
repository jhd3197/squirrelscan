// The skeleton head (#2307): the format guards always run; the parity test runs when an exported head is
// available (SKELETON_HEAD_DIR = the exporter's output dir, which holds private-derived numbers and is never
// committed): the TypeScript walk must reproduce the exporter's float32 reference on its fixture rows.

import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";

import { parseHTML } from "../../../../packages/parser/src/dom.ts";
import {
  classifySkeleton,
  loadSkeletonHead,
  predictSkeletonRow,
  SkeletonHeadError,
} from "./skeleton-head.ts";

describe("loadSkeletonHead", () => {
  test("refuses anything that is not a skeleton head", () => {
    expect(() => loadSkeletonHead({ format: "squirrel-dom-classifier" })).toThrow(
      SkeletonHeadError,
    );
    expect(() =>
      loadSkeletonHead({
        format: "squirrel-skeleton-head",
        headVersion: 99,
        featureNames: [],
        head: {},
      }),
    ).toThrow(SkeletonHeadError);
    expect(() => loadSkeletonHead(null)).toThrow(SkeletonHeadError);
  });
});

const dir = process.env.SKELETON_HEAD_DIR;
const available =
  !!dir && existsSync(`${dir}/skeleton-head.json`) && existsSync(`${dir}/parity-fixture.json`);

interface Fixture {
  names: string[];
  classes: string[];
  rows: number[][];
  probs: number[][];
}

describe.skipIf(!available)("exported skeleton head", () => {
  const head = available
    ? loadSkeletonHead(JSON.parse(readFileSync(`${dir}/skeleton-head.json`, "utf8")))
    : null!;
  const fixture: Fixture = available
    ? JSON.parse(readFileSync(`${dir}/parity-fixture.json`, "utf8"))
    : null!;

  test("feature names and classes match the fixture", () => {
    expect(head.featureNames).toEqual(fixture.names);
    expect(head.head.classes).toEqual(fixture.classes);
  });

  test("the TypeScript walk reproduces the float32 reference", () => {
    let worst = 0;
    for (const [i, row] of fixture.rows.entries()) {
      const result = predictSkeletonRow(head, Float32Array.from(row));
      fixture.classes.forEach((name, c) => {
        worst = Math.max(worst, Math.abs(result.probabilities[name]! - fixture.probs[i]![c]!));
      });
    }
    expect(worst).toBeLessThan(1e-4);
  });

  test("classifies a page from its document", () => {
    const html = `<!doctype html><html lang="en"><head><title>How we ship</title></head><body class="single-post postid-9"><main><article><h1>How we ship</h1><time>1 June</time><p>${"word ".repeat(300)}</p></article></main></body></html>`;
    const result = classifySkeleton(parseHTML(html).document, head);
    expect(Object.keys(result.probabilities)).toEqual(fixture.classes);
    expect(result.confidence).toBeGreaterThan(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });
});

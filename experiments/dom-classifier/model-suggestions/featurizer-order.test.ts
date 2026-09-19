import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseHTML } from "../../../packages/parser/src/dom.ts";
import { FEATURIZER_SKIP_SUBTREE, featurizerOrder } from "./featurizer-order.ts";

function order(body: string): string[] {
  // linkedom only exposes document.body for a full document, and every page in
  // the store is one; a bare fragment would make these fixtures test nothing.
  const html = body.startsWith("<html") ? body : `<html><head></head>${body}</html>`;
  const { document } = parseHTML(html);
  return featurizerOrder(document as never).map((element) =>
    (element as unknown as { tagName: string }).tagName.toLowerCase(),
  );
}

describe("featurizerOrder", () => {
  test("is rooted at body, inclusive, and body is nodeId 0", () => {
    expect(order("<html><head><title>T</title></head><body><div><p>a</p></div></body></html>")).toEqual([
      "body",
      "div",
      "p",
    ]);
  });

  test("is preorder depth first, not reverse sibling order", () => {
    // A naive stack pops children in reverse; this is the bug the explicit
    // reverse push exists to avoid, and it would shift every later nodeId.
    const html = "<body><section><h2>a</h2><p>b</p></section><footer>c</footer></body>";
    expect(order(html)).toEqual(["body", "section", "h2", "p", "footer"]);
  });

  test("a skipped tag takes its whole subtree with it", () => {
    const html = "<body><div>a</div><script><b>x</b></script><svg><circle/></svg><p>b</p></body>";
    expect(order(html)).toEqual(["body", "div", "p"]);
  });

  test("skipping shifts the indices of everything after it", () => {
    // The point of reproducing the walk exactly: one extra or missing element
    // renumbers every later node, and a wrong node still produces a packet.
    const withScript = order("<body><div>a</div><script>x</script><p>b</p><span>c</span></body>");
    expect(withScript.indexOf("span")).toBe(3);
    const withoutScript = order("<body><div>a</div><p>b</p><span>c</span></body>");
    expect(withoutScript.indexOf("span")).toBe(3);
  });

  test("a document with no body yields nothing rather than guessing", () => {
    const { document } = parseHTML("<html></html>");
    const value = featurizerOrder({ body: document.body ?? null } as never);
    expect(Array.isArray(value)).toBe(true);
  });

  test("the skip set matches the featurizer's, which owns it", () => {
    // v2/featurize.ts is the source of truth. If it gains or loses a tag this
    // fails rather than letting the two copies drift apart silently.
    const source = readFileSync(join(import.meta.dir, "..", "v2", "featurize.ts"), "utf8");
    const declared = /const SKIP_SUBTREE = new Set\(\[([\s\S]*?)\]\)/.exec(source)?.[1];
    expect(declared).toBeTruthy();
    const tags = [...declared!.matchAll(/"([a-z]+)"/g)].map((match) => match[1]).sort();
    expect(tags).toEqual([...FEATURIZER_SKIP_SUBTREE].sort());
  });
});

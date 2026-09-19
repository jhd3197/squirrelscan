/**
 * The featurizer's element enumeration, reproduced so a `nodeId` resolves.
 *
 * `divsoup-eval-v1/sample.jsonl` addresses nodes by "the featurizer's preorder
 * element index". That index is **not** an index into `extractPage`'s candidate
 * array and not an index into `document.querySelectorAll("*")`: indexing either
 * one resolves roughly half the sample to the wrong element, silently, because
 * a wrong element still has a tag and still produces a plausible packet.
 *
 * The rule, from `v2/featurize.ts`: a preorder depth-first walk rooted at
 * `<body>` inclusive, where an element whose tag is in `SKIP_SUBTREE` is not
 * visited and neither are its descendants.
 *
 * `v2/featurize.ts` is the source of truth and is owned by the modelling lane.
 * `featurizer-order.test.ts` checks this copy against it, so the two cannot
 * drift without a test failing.
 */
import type { Element } from "linkedom";

/** Mirrors `SKIP_SUBTREE` in `v2/featurize.ts`. */
export const FEATURIZER_SKIP_SUBTREE: ReadonlySet<string> = new Set([
  "script",
  "style",
  "template",
  "svg",
  "math",
  "noscript",
  "head",
  "link",
  "meta",
  "title",
  "base",
]);

type ElementLike = {
  localName?: string;
  nodeName: string;
  firstChild: unknown;
  nextSibling: unknown;
  nodeType: number;
};

function tagOf(element: ElementLike): string {
  return (element.localName ?? element.nodeName.toLowerCase()).toLowerCase();
}

/**
 * Elements in the featurizer's order. Index `i` is that document's `nodeId` `i`.
 *
 * Children are pushed in reverse onto an explicit stack so they pop in document
 * order, which is what makes the traversal a strict preorder DFS rather than
 * the reverse-sibling order a naive stack produces.
 */
export function featurizerOrder(document: { body?: Element | null }): Element[] {
  const body = document.body as ElementLike | null | undefined;
  if (!body) return [];
  const order: Element[] = [];
  const stack: ElementLike[] = [body];
  while (stack.length > 0) {
    const element = stack.pop()!;
    if (FEATURIZER_SKIP_SUBTREE.has(tagOf(element))) continue;
    order.push(element as unknown as Element);
    const children: ElementLike[] = [];
    for (
      let child = element.firstChild as ElementLike | null;
      child;
      child = child.nextSibling as ElementLike | null
    ) {
      if (child.nodeType === 1 && !FEATURIZER_SKIP_SUBTREE.has(tagOf(child))) children.push(child);
    }
    for (let index = children.length - 1; index >= 0; index--) stack.push(children[index]!);
  }
  return order;
}

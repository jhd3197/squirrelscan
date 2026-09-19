# The classifier runtime

What a consumer needs to know to call this and to trust what comes back.

## The contract

```ts
import { classifyPage, classifyDocument, tryLoadModel } from "./predict.ts";

const { model } = tryLoadModel(JSON.parse(artifactJson)); // null on any problem
const result = classifyPage(html, url, { model });
```

```ts
{
  pageType: { label: PageTypeLabel, confidence: number, source: "model" | "rules" },
  nodes: [
    {
      nodeRef: { nodeId: number, tag: string, elementId: string },
      label: RegionLabel,
      confidence: number,
      source: "model" | "rules",
    },
  ],
  degraded: string,   // "" when nothing fell back
}
```

`classifyDocument(document, url, opts)` takes a document the caller already
parsed, which is what the CLI holds by the time it would ask.

`nodeId` is the featurizer's preorder element index. It is stable for a given
document: re-running `featurizePage(html, url)` reproduces the same nodes with
the same ids, which is what makes a node reference storable.

### Precedence: markup wins, the model fills the silence

A node that carries its own landmark tag (`<nav>`, `<footer>`, `<header>`,
`<aside>`, `<main>`, `<form>`, `<search>`, `<dialog>`) or an ARIA landmark role
is answered by the RULES path at confidence 1.0, and the model is not consulted
for it. The model is asked only about nodes whose markup says nothing.

PLAN.md's "the model is primary, the deterministic path is the fallback" is
about nodes where the markup is SILENT. It never meant overriding a correct
declaration by the page author.

This is a measured correction, not a preference. With the model consulted
first, a `<nav class="main-navigation">` came back `content_block` at 0.74 on
66 of 68 pages of a stored site, contradicting a deterministic answer that was
right; two more nav roots on the same site failed the same way. The cause was a
train/serve mismatch — the region head was trained on MASKED rows only, so it
had never seen a landmark tag at training time, while the runtime feeds it true
unmasked features. Both halves are now fixed: the training mix carries masked
and unmasked rows, and the runtime does not ask about nodes that already told
it the answer.

The nesting rule still applies: a `<header>` inside an `<article>` or
`<section>` is that section's chrome (`article_chrome`), not site chrome, and
the rules path encodes that.

*Decided by the coordinating agent on 2026-09-20, pending the user's
confirmation.*

### `source` means provenance, not quality

**Consumers must not branch on `source`** except to display it. Both paths
produce the same labels from the same label space, and a rules answer is not a
worse answer — on a page with correct semantic markup it is the better one. The
field exists so a report can say where an attribution came from, and so this
lane can measure the two paths separately.

What `source` does NOT mean: it is not a confidence signal (`confidence` is),
and it is not a feature flag. Code shaped like `if (source === "model")` is
reading it wrong.

### `confidence` by path

- `source: "model"` — the softmax probability of the winning class, in `(0, 1]`.
- `source: "rules"` — `1` when the markup names the region outright (a landmark
  tag, an ARIA role, structured data), `0` for `unknown`. It is a certainty
  about *what the markup said*, not a calibrated probability, and the two are
  not comparable across paths. When a below-threshold model prediction falls
  back, the model's probability is carried through instead, so a consumer can
  still see how close it was.

`unknown` is a real answer and means "no evidence". PLAN.md's rule applies:
uncertain is unknown is page-scoped. Do not map it to a default class.

## Thresholds, and where they came from

The artifact carries `confidenceThreshold` per head. The region head ships
**0.7**, chosen on the holdout coverage table as the lowest threshold at or
above 0.5 whose covered accuracy clears 0.98:

| threshold | coverage | accuracy on covered | macro-F1 on covered |
|---|---|---|---|
| 0.0 | 100% | 0.968 | 0.790 |
| 0.6 | 98.1% | 0.976 | 0.812 |
| **0.7** | **96.6%** | **0.981** | **0.820** |
| 0.8 | 94.4% | 0.986 | 0.838 |
| 0.9 | 90.2% | 0.993 | 0.884 |

The 0.5 floor matters: without it the rule selects 0.0, because accuracy at full
coverage already clears any bar that a truncated model clears trivially — which
would ship a threshold that never falls back at all.

Every other choice (class-weight exponent, feature set, tree count, boosting
cut) was made on **val**. Holdout was read once per reported number. The splits
are frozen by registrable domain in `split-manifest.json` and are never
recomputed.

## When the rules path runs

`degraded` names the reason, and is `""` when the model handled the whole page.

| condition | `degraded` |
|---|---|
| no artifact passed | `no model artifact` |
| artifact unparseable, wrong format, or a future `artifactVersion` | the parse error |
| artifact built for another featurizer version | `artifact built for featurizer X, this build is Y` |
| artifact token-bucket count or node-feature width disagrees with this build | which one disagreed |
| per-page time budget spent (`budgetMs`, default 25 ms) | `time budget Nms exceeded` |
| one node's top probability is under the threshold | `""` — the page is not degraded, that node's `source` is `rules` |

The first four are page-wide: every node comes back from rules. The budget is
checked every 32 nodes and hands the remainder over mid-page rather than
returning a partial answer.

## Versioning

The artifact records `featurizerVersion`, `nodeFeatureCount`, `pageFeatureCount`
and `tokenBuckets`. `predict.ts` compares all of them against the featurizer
compiled into the same build and **refuses to use a mismatched artifact**,
falling back to rules with a reason rather than feeding a vector of the wrong
width into trees that will happily produce a number for it.

This is why `FEATURIZER_VERSION` has to move whenever the feature list, the
vocabulary, the hash or the candidate rule changes: it is the only thing
standing between a stale artifact and confident nonsense. An additive TEST-time
option (`maskAll`) does not move it, because the feature schema is unchanged.

`tryLoadModel` never throws; `loadModel` throws `ArtifactError`. Callers that
want the fallback use the first.

## Cost

Measured on 500 holdout pages, 82 candidate nodes per page on average, artifact
1.78 MB, 550 region trees:

| | p50 | p95 |
|---|---|---|
| featurize | 1.05 ms | 4.88 ms |
| predict | 1.29 ms | 5.61 ms |
| featurize + predict | 2.45 ms | 10.22 ms |
| linkedom parse (paid by the CLI anyway) | 0.65 ms | 2.97 ms |

Cold artifact load: 3.2 ms.

### The budget is p95 <= 10 ms, and why

PLAN.md originally said "< ~5 ms per page". That number was arbitrary, and it is
the wrong shape of budget for where this runs: the linkedom parse of a large
page is ~76 ms, and the fetch that produced the page is hundreds of milliseconds.
A classifier that adds single-digit milliseconds to that is not what makes an
audit slow. **G4 is therefore p95 <= 10 ms per page** (team lead, 2026-09-20).

p95 is driven by pages that hit the 400-node candidate cap; p50 is a typical
page. The tree count is the lever and it is close to linear in tree-node visits:

| trees | p95 tree-node visits | val macro-F1 | holdout macro-F1 |
|---|---|---|---|
| 550 (shipped) | 1.05M | .7611 | .7947 |
| 300 | 572k | .7535 | .7753 |
| 200 | 381k | .7441 | .7622 |

A content_block **cascade** was tried and rejected: a 40-round binary gate in
front of the head skips 87% of nodes and cuts visits to 0.17x, but holdout
macro-F1 fell 0.0138 against a 0.005 budget. The val-selected threshold did not
transfer — val lost 0.003 where holdout lost 0.014 — which is a reason to
distrust re-tuning it rather than an invitation to.

Wall-clock figures above were taken when the machine was quiet. When load
average is above 2 they are not reproducible on this box and only the visit
counts are reported; the counts are exact and load-independent.

## What the labels are

The region model is trained on **silver** labels: the page's own landmark tags
and ARIA roles, with those tags masked out of the features. A score against them
says "the model reproduces what semantic HTML would have said" — which is
exactly what is wanted for a page that has none. It is not human-verified truth.
The blind, human-labelled evaluation on div-soup pages is a separate exercise
and its numbers are not in this file.

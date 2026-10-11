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
      section?: { label: string, confidence: number }, // section purpose
    },
  ],
  degraded: string,         // "" when nothing fell back
  platform: string | null,  // the platform bucket the thresholds were chosen for
}
```

`classifyDocument(document, url, opts)` takes a document the caller already
parsed, which is what the CLI holds by the time it would ask.

Options, all optional:

| option | meaning |
|---|---|
| `model` | the parsed artifact; omit for the rules path only |
| `budgetMs` | per-page wall-clock budget, default 500 ms (see the measurement note on `DEFAULT_BUDGET_MS`) |
| `scoringBatch` | rows scored per tree pass, default 32; any size gives bit-identical scores (`scoreRows`), only the speed differs |
| `timingsOut` | analysis only: receives featurize, page type, region and section milliseconds |
| `siteFeatures` | the page-type head's cross-page block, when the caller holds the site's link graph (see "Missing values and the cross-page block") |
| `html` | raw HTML, read only for the platform fingerprint; `classifyPage` passes its own |
| `platform` | a platform the caller already knows; wins over `html` |
| `elementsOut` | analysis only: an array that receives the walked elements in nodeId order, so a caller can map a DOM element to the node carrying its label (`experiments/link-position.ts`) |

### Three heads, one featurizer

The artifact carries up to three LightGBM heads over the same feature vectors:

- **region** (`label` on each node): site_header, footer, navigation, sidebar,
  main_content, content_block, article_chrome, form, search, dialog;
- **page type** (`pageType`): homepage, product_detail, product_listing,
  article, blog_post, about, contact, docs, legal, pricing, listing_index,
  other;
- **section purpose** (`section` on a node): faq_block, reviews, breadcrumbs,
  business_info, product_offer, author_byline, video_embed, ad,
  cookie_consent, newsletter_signup, social_share, related_content, comments,
  promo_banner, pagination, table_of_contents, account_login,
  locale_switcher, or `none`.

The section head scores every candidate node, declared landmarks included (a
declared `<nav>` can still hold the site's breadcrumbs). `section` is present
only when the artifact has a section head and the class clears its threshold.
Absent means `none`, abstained, or not scored because the time budget ran out.
It has no rules path: the rules fallback answers region and page type only.

`table_of_contents` abstains in the model (see the threshold table).
`toc-rule.ts` is a deterministic detector tried in its place. It is not wired
into `predict.ts`: it did not hold 0.95 on the new corpus slice either
(2026-10-09, train + val pages).

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
train/serve mismatch: the region head was trained on MASKED rows only, so it
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
worse answer: on a page with correct semantic markup it is the better one. The
field exists so a report can say where an attribution came from, and so this
lane can measure the two paths separately.

What `source` does NOT mean: it is not a confidence signal (`confidence` is),
and it is not a feature flag. Code shaped like `if (source === "model")` is
reading it wrong.

### `confidence` by path

- `source: "model"`: the softmax probability of the winning class, in `(0, 1]`.
- `source: "rules"`: `1` when the markup names the region outright (a landmark
  tag, an ARIA role, structured data), `0` for `unknown`. It is a certainty
  about *what the markup said*, not a calibrated probability, and the two are
  not comparable across paths. When a below-threshold model prediction falls
  back, the model's probability is carried through instead, so a consumer can
  still see how close it was.

`unknown` is a real answer and means "no evidence". PLAN.md's rule applies:
uncertain is unknown is page-scoped. Do not map it to a default class.

## Thresholds: the strict rule (current)

Every head carries **one threshold per class** in `classThresholds`, picked on
out-of-fold predictions from 5-fold cross-validation grouped by domain over
train + val pages. A class's threshold is the lowest value at or above the
head's floor (region 0.5, page type and section 0.3) that holds **precision >=
0.95 on both corpus slices**, the old corpus and the new corpus. A slice with
fewer than 20 fires at that threshold is not held to it. A class with too few
fires overall (region 50, page type 20, section 15), or that cannot hold both
slices, gets a threshold above 1 and **abstains**. The region head is judged on
the non-declared population: the nodes the runtime actually asks the model
about, since markup answers the rest.

Then the platform guard may raise or abstain a class per platform (next
section).

Pooled thresholds, picked on old + new pages together, were not safe: the new
corpus pages are easier, so they carried precision the old corpus didn't have
(2026-10-08 CV: region 4 and section 11 classes below 0.95 on the old corpus
slice).

### The current artifact (2026-10-09)

The candidate artifact is 9.03 MB (10 MB ceiling): region 1,600 trees (silver
labels, silver-v4 refit), page type 1,740 (Jev labels, policy v8.2, with the
cross-page block), section 1,900 (Jev labels, policy-section-v3 with TOC
tightest labels). Parity against LightGBM on val rows: argmax 1.0 for all
three heads; page type on 2,000 val pages with the block and with it NaN.
It carries no platform overrides yet: the guard is computed in the next
retrain, after the pending page-type relabel.

Each row below is that artifact's threshold and its out-of-fold precision /
coverage per slice (2026-10-09 CV, silver-v4 train + val, 5 folds). Coverage
is the share of the class's labelled rows the head names correctly. A
precision below 0.95 marks a slice with fewer than 20 fires (docs 15, ad 16 on
the new corpus), which the rule does not hold.

**Region** (non-declared nodes of the class-rebalanced region sample, so read
precision per class; accuracy when firing: old corpus .955, new corpus .964):

| class | threshold | old corpus P / cov | new corpus P / cov |
|---|---|---|---|
| site_header | 0.85 | .951 / .28 | .980 / .43 |
| footer | 0.82 | .950 / .65 | .969 / .68 |
| navigation | 0.94 | .953 / .32 | .958 / .44 |
| sidebar | 0.87 | .955 / .13 | .951 / .21 |
| content_block | 0.95 | .961 / .53 | .953 / .50 |
| dialog | 0.94 | .952 / .22 | .984 / .51 |
| main_content, article_chrome, form, search | above 1 | abstain | abstain |

**Page type** (policy v8.2; accuracy when firing: old corpus .965, new corpus
.975):

| class | threshold | old corpus P / cov | new corpus P / cov |
|---|---|---|---|
| homepage | 0.30 | .976 / .89 | .989 / .98 |
| product_detail | 0.94 | .957 / .08 | .983 / .67 |
| product_listing | 0.98 | .960 / .03 | .987 / .22 |
| article | 0.95 | .960 / .06 | .959 / .10 |
| blog_post | 0.74 | .952 / .71 | .956 / .74 |
| about | 0.77 | .972 / .58 | .952 / .55 |
| contact | 0.68 | .950 / .80 | .965 / .75 |
| docs | 0.87 | .952 / .21 | .933 / .09 |
| legal | 0.56 | .958 / .85 | .952 / .73 |
| pricing | 0.93 | .994 / .74 | .958 / .37 |
| listing_index | 0.97 | .965 / .02 | .984 / .13 |
| other | 0.91 | .955 / .07 | .984 / .16 |

**Section purpose** (candidate nodes of the section sample):

| class | threshold | old corpus P / cov | new corpus P / cov |
|---|---|---|---|
| faq_block | 0.99 | .986 / .31 | .952 / .15 |
| reviews | 0.98 | .954 / .05 | .964 / .04 |
| breadcrumbs | 0.73 | .959 / .94 | .950 / .93 |
| business_info | 0.97 | .989 / .20 | .965 / .12 |
| product_offer | 0.95 | .952 / .11 | .971 / .28 |
| ad | 0.97 | .990 / .24 | .938 / .07 |
| cookie_consent | 0.96 | .953 / .72 | .976 / .80 |
| newsletter_signup | 0.96 | .950 / .61 | .987 / .69 |
| social_share | 0.97 | .988 / .40 | .955 / .33 |
| related_content | 0.97 | .958 / .29 | .982 / .25 |
| comments | 0.96 | .954 / .61 | .992 / .74 |
| promo_banner | 0.98 | .955 / .19 | .955 / .42 |
| pagination | 0.84 | .958 / .86 | .955 / .89 |
| account_login | 0.95 | .967 / .49 | .960 / .60 |
| locale_switcher | 0.89 | .953 / .40 | .963 / .56 |
| author_byline, video_embed, table_of_contents | above 1 | abstain | abstain |

These are cross-validation numbers on LLM and silver labels. The final score
is read once, on fresh test slices that no choice above has seen.

### Below threshold

When the model's answer is below its class's threshold the node comes back
`label: "unknown"`, `source: "model"`, `confidence: 0`, carrying `rawLabel` and
`rawConfidence` for debugging **only**. Those two fields are below the precision
bar by definition and must never be displayed or used to name a fix. The source
is `"model"` and not `"rules"` because the model was asked and declined; calling
it `"rules"` would claim the deterministic path had looked and found nothing.

### Superseded: region thresholds on val (2026-09-20)

*Kept as history; not the current artifact.* The region head alone, each
threshold chosen on VAL as the lowest value at or above 0.5 reaching a
precision of at least 0.95 on non-declared nodes, a class with fewer than 50
val predictions abstaining:

| class | threshold | val P / R | holdout P / R |
|---|---|---|---|
| content_block | 0.50 | .981 / .887 | .980 / .885 |
| sidebar | 0.66 | .951 / .218 | .899 / .199 |
| site_header | 0.87 | .954 / .304 | .933 / .269 |
| footer | 0.87 | .952 / .649 | .956 / .662 |
| navigation | 0.90 | .955 / .461 | .951 / .436 |
| dialog | 0.95 | .958 / .079 | .867 / .058 |
| main_content, article_chrome, form, search | n/a | abstain | abstain |

Coverage on non-declared holdout nodes (2026-09-20): **76.5%** at **0.976**
accuracy when firing. Three classes fell below 0.95 on holdout, which is one
reason thresholds now come from cross-validation with a per-slice bar.

A single global threshold was tried first and was the wrong instrument. The rule
that picked it (lowest value whose covered accuracy clears 0.98) selected 0.95
once inherited nodes entered the training population, leaving 31.6% coverage:
two thirds of nodes unnamed, including classes the model is good at. Ten classes
whose precision curves differ by 40 points cannot share one number.

At the time every other choice (class-weight exponent, feature set, tree count,
boosting cut) was made on val and holdout was read once per reported number.
That holdout is now frozen; choices come from cross-validation. The splits are
frozen by registrable domain in `split-manifest.json` and are never recomputed.

## When the rules path runs

`degraded` names the reason, and is `""` when the model handled the whole page.

| condition | `degraded` |
|---|---|
| no artifact passed | `no model artifact` |
| artifact unparseable, wrong format, or a future `artifactVersion` | the parse error |
| artifact built for another featurizer version | `artifact built for featurizer X, this build is Y` |
| artifact token-bucket count or node-feature width disagrees with this build | which one disagreed |
| per-page time budget spent (`budgetMs`, default 500 ms) | `time budget Nms exceeded` |
| one node's top probability is under the threshold | `""`: the page is not degraded, that node's `source` is `rules` |

The first four are page-wide: every node comes back from rules. The budget is
checked every 32 nodes and hands the remainder over mid-page rather than
returning a partial answer. The section head runs after region and page type
under the same budget; once it is spent, the remaining nodes get no `section`.

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

### Missing values and the cross-page block (#2307)

A tree block is version 1 (plain `<=` splits) or version 2, which adds one flag
byte per split carrying LightGBM's missing-value handling: the missing type
(None, Zero or NaN) and `default_left`. The exporter writes version 2 only when
a head has such a split, so every earlier artifact is byte-identical and still
loads. A runtime that predates version 2 refuses those blocks with an
`ArtifactError` and falls back, instead of misreading them.

The walk for version 2 is LightGBM's own decision: on a split that does not
treat NaN as missing a NaN reads as 0; on a NaN split a NaN goes the default
way; on a Zero split a zero (and so a NaN) goes the default way; anything else
compares `<=`.

A page-type head trained with cross-page features (inbound same-site anchors,
template cluster) names them in `pageType.extraFeatureNames`. They follow
`[numeric, tokens]` in its vector. Pass them as `siteFeatures` when the caller
holds the site's link graph. Absent, null or the wrong length means "no
graph", and the block goes in as NaN. That is the input the head was trained
to handle: 30% of its training pages had the block dropped to NaN. The exporter
checks parity against LightGBM both ways, with the block and with it NaN.

### Per-platform thresholds: the platform guard (#2307)

A head may carry `platformClassThresholds`: platform bucket → class → threshold.
On a page of that platform, these replace the class thresholds; a value above 1
abstains. `retrain-all` computes them from strict CV. A (class, platform) cell
gets an override only when it fires ≥ 50 times at precision < 0.95. The
threshold is raised if that keeps at least half the platform's coverage;
otherwise the class abstains on that platform. Smaller cells keep the global
threshold.

The current artifact (2026-10-09) carries none; see "The current artifact".

`runtime/platform.ts` names the platform from raw HTML with lower-cased
substring checks and a bounded scan of generator metas. The buckets are
shopify, woocommerce, wordpress, wix, squarespace, webflow, nextjs and custom.
It matches the Python tagger the guard was computed with on all 9,147 silver-v4
val pages (2026-10-09, `experiments/platform-parity.ts`).

`classifyPage` fingerprints its own HTML. `classifyDocument` uses
`options.platform` or `options.html`, and falls back to the global thresholds
without either. The result reports the `platform` it used.

## Cost

**Current artifact (2026-10-09).** 500 new-corpus val pages, featurize +
predict, 3 alternating rounds against the then-shipped 8.61 MB artifact, on a
machine under memory pressure (swap about 15 GB):

| | p95 per round | median p95 | p50 |
|---|---|---|---|
| shipped (8.61 MB) | 60.1 / 57.9 / 56.5 ms | 57.9 ms | about 19 ms |
| current (9.03 MB) | 57.6 / 59.7 / 58.5 ms | 58.5 ms | about 19 ms |

Equal within noise: the extra page-type trees cost one row per page. These
wall-clock numbers are not comparable with the 2026-09-20 table below (quiet
machine, holdout pages, a 1.78 MB artifact). The three-head artifact has not
been measured on a quiet machine, so the p95 <= 10 ms budget (G4) is
unconfirmed for it.

*(2026-09-20, superseded artifact.)* Measured on 500 holdout pages, 82
candidate nodes per page on average, artifact 1.78 MB, 550 region trees:

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
page. The tree count is the lever and it is close to linear in tree-node visits
(2026-09-20, region head):

| trees | p95 tree-node visits | val macro-F1 | holdout macro-F1 |
|---|---|---|---|
| 550 (shipped) | 1.05M | .7611 | .7947 |
| 300 | 572k | .7535 | .7753 |
| 200 | 381k | .7441 | .7622 |

A content_block **cascade** was tried and rejected: a 40-round binary gate in
front of the head skips 87% of nodes and cuts visits to 0.17x, but holdout
macro-F1 fell 0.0138 against a 0.005 budget. The val-selected threshold did not
transfer (val lost 0.003 where holdout lost 0.014), which is a reason to
distrust re-tuning it rather than an invitation to.

The 2026-09-20 wall-clock figures were taken when the machine was quiet. When load
average is above 2 they are not reproducible on this box and only the visit
counts are reported; the counts are exact and load-independent.

## What the labels are

The region model is trained on **silver** labels: the page's own landmark tags
and ARIA roles, with those tags masked out of the features. A score against them
says "the model reproduces what semantic HTML would have said", which is
exactly what is wanted for a page that has none.

The page-type model is trained on **Jev** labels under policy v8.2, with Clef
as a second vote where Jev is unsure (a page they disagree on is dropped) and
the policy's URL rules applied. The section model is trained on Jev labels
under policy-section-v3, with `table_of_contents` from the tightest-node
labels and `ad` vs `promo_banner` decided by third-party ad markup.

None of these is human-verified truth. The human gold evaluations are a
separate exercise and their numbers are not in this file.

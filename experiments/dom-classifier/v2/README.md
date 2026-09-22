# DOM classifier v2: featurizer + silver dataset

Experiment code for #2307. Three files carry the whole contract:

| File | What it is |
|---|---|
| `featurize.ts` | The ONE featurizer. Training data and inference both go through it, so there is no train/serve skew to discover later. |
| `silver-labels.ts` | Free supervision from the page's own semantics: landmarks for region, structured data for page type. |
| `build-dataset.ts` | Streams the local content store into `nodes-*.jsonl.gz` / `pages-*.jsonl.gz` under the private data root. |

Nothing here is wired into the CLI. `experiments/` is not a workspace member and
nothing installs into it: `linkedom` is reached through
`../../../packages/parser/src/dom.ts` (the same parse entry the CLI uses,
attribute-case patch included) and `tldts` through a `node_modules` symlink into
`packages/crawler/node_modules`.

## The two rules that decide the design

**1. The label source is never a feature.** Page type is labelled from JSON-LD,
microdata and og:type, so `featurize.ts` counts no scripts, no meta tags and no
`itemtype`: a count of `<script>` elements is a proxy for "this page ships
JSON-LD", and a model that reads it has learned nothing about the page's shape.
`featurize.test.ts` adds a schema block to a page and asserts that not one
number moves.

**2. Masking has to be airtight.** Region is labelled from landmark tags and
ARIA roles, so the masked variant replaces a landmark tag with `div`, drops its
role (explicit *and* implicit), masks landmark tags in the ancestor chain, and
drops class/id tokens that literally spell a label word with p = 0.5, seeded by
`(nodeId, token)`. The test asserts that a masked `<footer>` and a masked `<div>`
with the same subtree produce byte-identical vectors, and — separately — that
they differ when unmasked, so the first assertion cannot pass vacuously.

There is one deliberate asymmetry, pinned by its own test: masking hides the tag
from the **vector**, not from **candidate selection**. A `<footer>` is always a
candidate; the same subtree under a `<div>` is one only if it clears the
"two element children or twenty characters" gate. That is a population
difference, not a leak, and it means a very small div-soup footer never becomes
a training row.

## Features

`featureSchema()` is the single source of truth and is written to
`feature-schema.json` beside every dataset. At `FEATURIZER_VERSION = v2.0.0`:

- **197 node features**: structure (depth, document-order position of the
  subtree's start and end, subtree size, sibling shape runs), link statistics
  (count, density, internal/external share, social, `tel:`/`mailto:`), text
  shape (headings, list items, paragraphs, average paragraph length), media and
  controls, accessibility hints, eleven text-pattern flags, four categoricals
  (tag, role, parent tag, grandparent tag, each from a fixed vocabulary where 0
  is out-of-vocabulary), and the 65-word curated class/id vocabulary as explicit
  binaries for the node and for its parent, and (v2.1.0) ten ancestor/sibling
  context features: distance to and text/element share of the nearest
  `<article>`/`<section>` ancestor, first/last child of it, sibling section
  count, the node's body-level block size ratio and position, list-item text
  share, and whether its own first child is a heading. Those ten were added
  because the confusion matrix showed article_chrome losing 55% of its nodes to
  content_block and sidebar losing 33%, neither of which is separable from a
  node's own statistics. None of them reads a landmark tag or role, so the mask
  is unaffected — `<article>` and `<section>` are not landmarks.
- **137 page features**: URL shape (path depth, date in path, numeric id, eleven
  slug flags), document shape (title/h1, text length, heading histogram, link
  density, forms, media, code blocks, tables), landmark counts, page-level
  pattern counts, and the curated vocabulary over the `<body>` class and id.
- **512 hashed token buckets**, sparse, alongside both vectors. FNV-1a 32-bit
  over the token string, `% 512`, with the prefixes `s:` (self), `p:` (parent)
  and `b:` (body). Golden values are pinned in `featurize.test.ts` so another
  runtime can be checked against five numbers instead of a whole dataset.

Cost: one forward walk, one reverse fold, one emission pass over at most 400
candidates. The 20k-element test page featurizes in well under the 300 ms bound;
the linkedom parse of the same page costs an order of magnitude more, and that
cost is already paid by the CLI before it calls this.

## Labels

Region: `site_header`, `footer`, `navigation`, `sidebar`, `main_content`,
`article_header`, `article_footer`, `search`, `form`, `dialog`, `content_block`,
`unlabelled`.

`article_header` and `article_footer` are TRAINING-ONLY classes outside
`taxonomy.json`. They exist because a `<header>` inside an `<article>` or
`<section>` is that section's own chrome, and labelling it `site_header` teaches
the region head the opposite of what it needs. `<main>` is a sectioning ROOT,
not sectioning content, so a `<header>` directly inside `<main>` is still site
chrome. A training run that does not want the two classes should merge or drop
them rather than have them silently folded in here.

Every node also carries `containerLabel`: the region of its nearest ancestor
landmark. The brief's rule (inside `<main>` → `content_block`, outside every
landmark → `unlabelled`) is what `label` holds; `containerLabel` is the strictly
richer answer, recorded so the training lane can choose without a rebuild.

### The third label source, and what it can and cannot reach

Silver labels come from the page's own landmarks, so a landmark-less page
produces no `footer` and no `navigation` row at ANY corpus size. Jev labels a
node from its evidence instead, so `train/jev_labels.py` maps its three axes
onto the ten region classes and `sample.py --jev-labels` adds them as a third
source. Where silver and Jev disagree, BOTH rows are kept, each with its own
weight, and the rate is reported: dropping either would be a silent decision
about which annotator is right on a population nobody has adjudicated.

`navigation` is the case that proves the axes have to be combined rather than
chosen between. It is not in the `regions` vocabulary at all, while
`navigation_menu` is one of the largest `componentTypes`, so a mapping that
read only `regions` would produce no navigation labels whatsoever.

**Measured ceiling.** On a 62-node probe, Jev mapped 79% of landmark-page nodes
and **11%** of div-soup nodes — the same shape of gap silver has. Reading the
packets explains it: the div-soup population here is mostly single-page-app
shells (`classTokens=app-container`, one node holding the whole application),
and `layout_container` with an empty regions axis is the right answer, because
the page has no header or footer to find. That is a fact about the corpus, not
a failure of the annotator, and no relabelling fixes it.

What does help is not sampling div-soup nodes uniformly.
`corpus/node-sample.ts` ranks them by
`max(1 - startPos, endPos) * (0.5 + min(linkDensity, 1))` — a header sits at the
start, a footer at the end, both are link-dense — and spends two thirds of each
page's budget at the top of that ranking. On the same pages and the same node
budget, div-soup chrome labels went from 3 to 6 and navigation from 2 to 5.

Page type: `article`, `product_detail`, `product_listing`, `docs`, `homepage`,
`pricing`, `contact`, `about`, `legal`, `listing_index`, `other`, `unlabelled`.
Provenance is recorded per page. Two schema types that map to different classes
is a conflict and is `unlabelled`, not a coin flip: those pages are exactly the
ones the model is meant to resolve later. `homepage` is the one label that needs
the URL as well as the schema, and the URL shape IS a feature, so those rows
carry `leak: true` and the source `jsonld+url_root`.

## Growing the corpus (`corpus/`)

The 2026-09-19 build read 26,252 pages from 3,278 domains, all of them pages a
squirrelscan user had already audited. `corpus/` adds more, into a SEPARATE
store, so the live one is never written to.

```sh
export DOM_CLASSIFIER_DATA_ROOT=/path/to/private/root
export SQUIRREL_CLI=/path/to/apps/cli/src/cli.ts
# outlinks.ts and seeds.ts read ONE store; everything downstream reads both.
export SQUIRREL_STORE=/path/to/squirrel-store
export SQUIRREL_STORES=/path/to/squirrel-store:$DOM_CLASSIFIER_DATA_ROOT/crawl-home/.squirrel

bun run corpus/outlinks.ts          # outbound domains, from the stored HTML
bun run corpus/seeds.ts             # one seed url per registrable domain
bun run corpus/crawl.ts --workers 4 --max-pages 15 --budget-minutes 180
bun run corpus/pagetype-queue.ts --verify-against <prior>/split-manifest.json
bun run corpus/node-sample.ts --dataset $DOM_CLASSIFIER_DATA_ROOT/silver-v2 \
  --nodes 40000 --exclude-pages <divsoup-eval page ids>
```

`SQUIRREL_STORES` matters from `pagetype-queue.ts` onward: with only
`SQUIRREL_STORE` set, the queue and the dataset are built from the live store
alone and every page the crawl just fetched is missing, with no error.

Four things in there are not obvious:

- **`squirrel crawl`, not `squirrel audit`.** The crawl subcommand fetches and
  stores and runs no rules. Measured on one 15-page site: audit 34.4s (24.3s of
  it CPU inside the 260 rules), crawl 2.15s. Nothing in this corpus reads a rule
  result.
- **The `links` table is empty** on almost every project database in the store
  (463 domains across 17,239 of them), so `outlinks.ts` harvests outbound hosts
  from the stored bytes with a regex instead. It is a *composition* fix as much
  as a size one: the project directories are squirrelscan's own users, which
  skews hard to `vercel.app`, `lovable.app` and `.ai`, and what those pages link
  to is where the `.gov`, the news sites and the shops come from.
- **Seeds are emitted round-robin across public suffix**, so truncating the list
  at any length keeps the whole list's TLD mix. Taking the first N of a sorted
  list would give N `.com` domains.
- **The split cannot drift.** `split = sha256(registrableDomain)[0:8] mod 100`
  is a pure function of the domain, so a domain cannot change split because the
  corpus grew. `pagetype-queue.ts --verify-against` checks that against the
  previous manifest and refuses to write if one domain disagrees.

## Building the dataset

```sh
DOM_CLASSIFIER_DATA_ROOT=/path/to/private/root \
SQUIRREL_STORES=/path/to/store-a:/path/to/store-b \
DOM_CLASSIFIER_DATASET_VERSION=silver-v2 \
bun run build-dataset.ts [--index-only] [--limit N] [--max-per-domain N]
```

No path is hardcoded. `SQUIRREL_STORES` is colon-separated and each entry is a
directory holding `content-store.db` and `projects/`; `SQUIRREL_STORE`
(singular) still names one. A content hash present in two stores is featurized
once — the hash is the sha256 of the exact bytes, so it is the same page.

`DOM_CLASSIFIER_DATASET_VERSION` names the output directory and defaults to
`silver-v1`. Every script in `train/` reads the same variable, so a larger
corpus is built beside the old one rather than over it: the frozen split
manifest and every model trained against it live in that directory.

Each store is opened `readonly` and read in short batches keyed by `rowid`,
because another process writes to it while this runs.

Output, all under `$DOM_CLASSIFIER_DATA_ROOT/$DOM_CLASSIFIER_DATASET_VERSION/`: `nodes-{train,val,holdout}.jsonl.gz`
(two rows per node, `m: 0` unmasked and `m: 1` masked), `pages-*.jsonl.gz`,
`feature-schema.json`, `split-manifest.json`, `stats.json`, and
`page-index.jsonl.gz` — the only file that holds a URL. Rows carry
`d`, a 16-hex domain hash, never the domain.

The split is 80/10/10 by registrable domain, `sha256(domain)[0:8] % 100`, and
`split-manifest.json` is written once. A later, larger build that would move a
domain between splits fails rather than contaminating the holdout.

That manifest lives inside the dataset-version directory, so a build of a NEW
version finds none and writes a fresh one — which disables the check in exactly
the case it was written for. Pass `--verify-against <prior>/split-manifest.json`
to check a new version's splits against the corpus the current holdout came
from. It is not a formality: the 2026-09-22 build verified 3,657 domains with
zero drift, which is what makes the old holdout still a holdout.

## Tests

```sh
bun test featurize.test.ts silver-labels.test.ts
```

`fixtures/` holds eight hand-written pages (landmark site, the same site as div
soup, article with nested chrome, product, product listing, docs, homepage,
microdata-only about page) and `expected.json`, their hand-checked labels.

## Training and the runtime

```sh
export DOM_CLASSIFIER_DATA_ROOT=/path/to/private/root
export SQUIRREL_STORE=/path/to/squirrel-store
V=$DOM_CLASSIFIER_DATA_ROOT/.venv/bin/python          # uv venv + lightgbm numpy scipy scikit-learn

$V train/sample.py --data-root $DOM_CLASSIFIER_DATA_ROOT --out $.../models/region-sample
$V train/train_region.py   --sample $.../models/region-sample --data-root . --out $.../models/region
$V train/train_pagetype.py --data-root . --out $.../models/pagetype
$V train/export_model.py   --models $.../models --data-root . --out $.../models/artifact

bun run train/holdout-tools.ts masked-all      # the fully-token-hidden holdout
$V train/eval_maskall.py --data-root . --models $.../models
bun run train/holdout-tools.ts bench --limit 500
bun run train/holdout-tools.ts divsoup --limit 300
```

Nothing is materialised in full. `sample.py` streams the 5.25M rows through one
reusable 50k-row dense chunk, converts each to CSR and keeps only that; peak RSS
across the whole pipeline is 660 MB, on a dataset that is ~17 GB dense.

### What is selected where

Every choice that could read the answer is made on VAL: the class-weight
exponent, the feature set, the tree size, the boosting cut. Holdout is touched
once per reported number. The splits themselves are frozen by registrable
domain in `split-manifest.json` and are never recomputed.

Two traps worth keeping in mind if these are re-run:

- **Do not stack class weights on downsampling.** The training sample already
  moves content_block from 89% of nodes to 50%. Adding full inverse-frequency
  weights on top rebalances twice, and the model then over-predicts every rare
  class: sidebar came out at recall 0.82 for precision 0.18. The exponent is a
  swept parameter for exactly this reason, and it chose 0.25, not 1.0.
- **Do not pick the boosting cut by val macro-F1 when the probabilities matter.**
  On the page-type head that rule chose 10 iterations (18 KB, against a 600 KB
  budget) because macro-F1 over 926 val pages is noisy enough to peak anywhere.
  A 10-round model's probabilities are squashed toward uniform, and the whole
  `p >= 0.9` product metric read zero across the entire corpus. Early stopping
  on weighted logloss is the criterion that is about probabilities.

### The runtime

`runtime/predict.ts` exposes `classifyPage(html, url, {model})` and
`classifyDocument(document, url, {model})`, returning
`{pageType: {label, confidence, source}, nodes: [{nodeRef, label, confidence, source}], degraded}`.
`source` is `"model"` or `"rules"`, and the shape is identical either way.

The rules path (`runtime/rules-fallback.ts`) runs when the artifact is absent,
unparseable or built for another featurizer version, when a prediction is under
the artifact's threshold, or when the per-page time budget is spent. It reads
landmarks, ARIA roles and structured data — the same reading `silver-labels.ts`
does, minus the corpus-level parts. A missing model degrades attribution, never
detection.

Float parity is by construction: `export_model.py` rounds thresholds to float32
and walks the rounded trees in a numpy reference evaluator, and `predict.ts`
implements the same walk with `Math.fround` on the feature value. The parity
test compares TS against that reference on 10k holdout rows.

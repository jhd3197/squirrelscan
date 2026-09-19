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

- **187 node features**: structure (depth, document-order position of the
  subtree's start and end, subtree size, sibling shape runs), link statistics
  (count, density, internal/external share, social, `tel:`/`mailto:`), text
  shape (headings, list items, paragraphs, average paragraph length), media and
  controls, accessibility hints, eleven text-pattern flags, four categoricals
  (tag, role, parent tag, grandparent tag, each from a fixed vocabulary where 0
  is out-of-vocabulary), and the 65-word curated class/id vocabulary as explicit
  binaries for the node and for its parent.
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

Page type: `article`, `product_detail`, `product_listing`, `docs`, `homepage`,
`pricing`, `contact`, `about`, `legal`, `listing_index`, `other`, `unlabelled`.
Provenance is recorded per page. Two schema types that map to different classes
is a conflict and is `unlabelled`, not a coin flip: those pages are exactly the
ones the model is meant to resolve later. `homepage` is the one label that needs
the URL as well as the schema, and the URL shape IS a feature, so those rows
carry `leak: true` and the source `jsonld+url_root`.

## Building the dataset

```sh
DOM_CLASSIFIER_DATA_ROOT=/path/to/private/root \
SQUIRREL_STORE=/path/to/squirrel-store \
bun run build-dataset.ts [--index-only] [--limit N] [--max-per-domain N]
```

No path is hardcoded; both environment variables are required. The content store
is opened `readonly` and read in short batches keyed by `rowid`, because another
process writes to it while this runs.

Output, all under `$DOM_CLASSIFIER_DATA_ROOT/silver-v1/`: `nodes-{train,val,holdout}.jsonl.gz`
(two rows per node, `m: 0` unmasked and `m: 1` masked), `pages-*.jsonl.gz`,
`feature-schema.json`, `split-manifest.json`, `stats.json`, and
`page-index.jsonl.gz` — the only file that holds a URL. Rows carry
`d`, a 16-hex domain hash, never the domain.

The split is 80/10/10 by registrable domain, `sha256(domain)[0:8] % 100`, and
`split-manifest.json` is written once. A later, larger build that would move a
domain between splits fails rather than contaminating the holdout.

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

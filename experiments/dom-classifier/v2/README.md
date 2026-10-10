# DOM classifier v2: featurizer + silver dataset

Experiment code for #2307. Three files carry the whole contract:

| File | What it is |
|---|---|
| `featurize.ts` | The ONE featurizer. Training data and inference both go through it, so there is no train/serve skew to discover later. |
| `silver-labels.ts` | Free supervision from the page's own semantics: landmarks for region, structured data for page type. |
| `build-dataset.ts` | Streams the local content store into `nodes-*.jsonl.gz` / `pages-*.jsonl.gz` under the private data root. |

Updated 2026-10-10. Read `../DEVELOPMENT.md` first: it holds the rules, the
glossary and which doc owns which topic. Sections marked *(superseded)* are
kept as history: the numbers in them are true for their date, not for the
current model.

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
with the same subtree produce byte-identical vectors, and, separately, that
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
  is unaffected: `<article>` and `<section>` are not landmarks.
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

**Page type and section now train mainly on Jev labels.** Since 2026-09-20
the page-type head trains on Jev labels, now under policy v8.2
(`labels/policy-v8.2.md`). A page's own schema adds a silver label only for
classes whose schema agreed with Jev on a noise sample, and only on train. The
section-purpose head trains on Jev section labels (policy-section-v3), plus
free labels on train for (class, source) pairs measured >= 0.85 precise
against Jev. Both are described in "Datasets and slices" below. Region still
trains on silver labels.

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
and **11%** of div-soup nodes, the same shape of gap silver has. Reading the
packets explains it: the div-soup population here is mostly single-page-app
shells (`classTokens=app-container`, one node holding the whole application),
and `layout_container` with an empty regions axis is the right answer, because
the page has no header or footer to find. That is a fact about the corpus, not
a failure of the annotator, and no relabelling fixes it.

What does help is not sampling div-soup uniformly, at either level.

**Which pages.** "div-soup" means fewer than two landmark kinds, which lumps
together two populations that are not alike. A page with ZERO kinds is usually
an app shell with no chrome to find. A page with ONE has real site chrome and
simply did not declare the rest of it (a `<nav>` but no `<footer>`), so that is
where an undeclared footer actually exists. `corpus/node-sample.ts` spends the
div-soup budget on the one-landmark tier first and reports both counts.

**Which nodes.** Within a page it ranks candidates by
`max(1 - startPos, endPos) * (0.5 + min(linkDensity, 1))` (a header sits at the
start, a footer at the end, both are link-dense) and spends two thirds of each
page's budget at the top of that ranking. On the same pages and the same node
budget, div-soup chrome labels went from 3 to 6 and navigation from 2 to 5.

### Research cohorts

`corpus/import-webclasseg25.ts` folds **WebClasSeg25** (CC BY 4.0, Zenodo
record 19183572) in as a tagged cohort. Every row carries
`cohort: "research/webclasseg25"` and `s: "train"`, and `sample.py
--extra-train-dataset` puts them in train and nowhere else, so val and holdout
stay on squirrelscan's own distribution.

It is worth the trouble for two reasons. Its functional-classification segments
are `header` / `nav` / `maincontent` / `footer`: the region axis itself, not an
approximation, labelled by three to four **people** per page. And its 844 pages
span 25 languages of European public-service sites, which is the CMS-shaped,
non-English web the crawl does not reach. Imported: 5,960 human-labelled nodes
(navigation 1,446, main_content 2,189, site_header 1,285, footer 1,040), of
which 416 sit on landmark-less pages where silver produces nothing at all.

The join is exact and that is the point: every element carries `hyu="N"`, every
segment names a `hyuIndex`, so a label lands by attribute equality rather than
an XPath or a re-derived index. Measured over the whole release, **6,217 of
6,218 segments resolved**.

A node's weight is votes for the winning label over the page's ANNOTATOR COUNT.
Annotators agree on the label 97.9% of the time; what actually varies is which
node they pick, since each was asked for "the largest visible coherent DOM
node" and only 51% of nodes were marked by two or more of them. Dividing by the
markers instead would score almost everything 1.0 and throw that away.

Three other corpora were checked and rejected, which is recorded here so nobody
re-does the work: **Webis-WebSeg-20** is CC BY 4.0 but its ground truth is
polygons with no class label, so it cannot produce a row in this taxonomy;
**SWDE** has no stated licence anywhere (dead CodePlex original, unlicensed
mirrors); **WebSRC** has no dataset licence and labels answer containment rather
than regions.

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

### The Common Crawl diversity corpus (2026-10-08)

The squirrelscan-user crawls are dev-site skewed. On 1,500-page samples of the
training stores (2026-10-08, html lang and platform asset fingerprints):
Next.js 27-31%, custom 40-43%, WordPress 5-8%, WooCommerce 2-4%, Shopify about
1%, English 75-85%. Cloud customers run mostly WordPress, WooCommerce and
Shopify. So a second corpus was drawn from Common Crawl, stratified by
platform, language and vertical
(`corpus/cc_corpus.py` subcommands, plus `cc-exclusions.ts` and
`cc-domains.ts`):

1. **Index, not crawl.** The CC columnar index is read over https, one row
   group at a time, row groups picked per TLD so the language mix is chosen
   rather than inherited. Per domain it keeps page counts, URL-shape counts
   and platform hints from the URLs (e.g. `/collections/` + `/products/` for
   Shopify).
2. **Fingerprint, then fill quotas.** A sample of candidate homepages is
   fetched and tagged by `platform_of` (lower-cased marker substrings and
   generator metas; the runtime twin is `runtime/platform.ts`). Domains are
   then picked per platform x language cell (non-English held to a target
   share inside each platform) with minimums per vertical (recipe, event,
   job, gov/edu, news, local business) and for smaller platforms.
3. **A page mix per domain by URL shape**: the homepage plus product, listing,
   article, about, contact, legal, pricing and the rare shapes when present.
4. **One WARC range request per page**, rate-limited pool-wide (the CC CDN
   blocks an address that goes too fast; a 403 is treated as a pause, not a
   failure). The HTML is decoded with its own charset and written to a
   squirrel-shaped store (`content-store.db` + `projects/<x>/project.db`), so
   `build-dataset.ts` and every downstream script read it unchanged.
5. **Every domain is new.** `cc-exclusions.ts` collects every registrable
   domain the corpus has seen (all earlier datasets, seeds, ledgers, cloud
   pages, plus every split manifest's domain hashes) and candidates in it are
   dropped. Cloud customer domains are excluded the same way.
6. **The same split rule.** `cc-domains.ts` computes each candidate's split
   with the exact `splitOf` of `build-dataset.ts` (tldts with private
   suffixes), and `build-dataset.ts --verify-against` checks the build
   against the previous split manifest. New domains in the holdout bucket
   are built but never merged into a training dataset: they are a stored
   test slice, scored once at the end.

Result (2026-10-09, new corpus): 2,915 new domains and 19,166 pages, of which
train 15,509, val 1,759 and holdout bucket 1,898. Platform mix of those
pages: Shopify 19.3%, WordPress 15.8%, WooCommerce 15.7%, Squarespace 7.5%,
Wix 7.1%, Webflow 5.0%, custom 19.4%.

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
once: the hash is the sha256 of the exact bytes, so it is the same page.

`DOM_CLASSIFIER_DATASET_VERSION` names the output directory and defaults to
`silver-v1`. Every script in `train/` reads the same variable, so a larger
corpus is built beside the old one rather than over it: the frozen split
manifest and every model trained against it live in that directory.

Each store is opened `readonly` and read in short batches keyed by `rowid`,
because another process writes to it while this runs.

Output, all under `$DOM_CLASSIFIER_DATA_ROOT/$DOM_CLASSIFIER_DATASET_VERSION/`: `nodes-{train,val,holdout}.jsonl.gz`
(two rows per node, `m: 0` unmasked and `m: 1` masked), `pages-*.jsonl.gz`,
`feature-schema.json`, `split-manifest.json`, `stats.json`, and
`page-index.jsonl.gz`, the only file that holds a URL. Rows carry
`d`, a 16-hex domain hash, never the domain.

The split is 80/10/10 by registrable domain, `sha256(domain)[0:8] % 100`, and
`split-manifest.json` is written once. A later, larger build that would move a
domain between splits fails rather than contaminating the holdout.

That manifest lives inside the dataset-version directory, so a build of a NEW
version finds none and writes a fresh one, which disables the check in exactly
the case it was written for. Pass `--verify-against <prior>/split-manifest.json`
to check a new version's splits against the corpus the current holdout came
from. It is not a formality: the 2026-09-22 build verified 3,657 domains with
zero drift, which is what makes the old holdout still a holdout.

## Datasets and slices

**silver-v4** = silver-v3 + the new corpus's train and val pages
(`corpus/merge_datasets.py`; it refuses to merge a holdout split and checks
that both inputs share one featurizer schema and agree on every domain's
split). Row files are concatenated as gzip members. silver-v3's holdout is
carried unchanged. Pool sizes (2026-10-09): train 75,494 and val 9,147 pages.

**Slices.** Every CV number is reported per slice:

- **corpus**: old corpus (squirrelscan-user crawls) vs new corpus (Common Crawl);
- **platform**: shopify, woocommerce, wordpress, wix, squarespace, webflow,
  nextjs, custom (everything else), from `cc_corpus.py tag-pages`, which tags
  old and new pages with the same function;
- **language**: en, non-en, unk (html lang, else a script or stopword vote).

**Page-type labels.** Jev under the page-type policy for every page; the new
corpus was labelled under v8.1, the old under v8. Where Jev is unsure
(confidence < 0.7) Clef gives a second vote, and a page where the two
disagree is dropped from training and from scoring. The policy's URL rules
(the site root and locale roots are homepages; a page under `/blog/` or a Wix
`/post/` page that Jev called docs, article or other is a blog_post) are
applied on every split (`--policy-rules --policy-version
v8.2`). Clef can't vote on every unsure page, so on unvoted ones the Jev label
stands.

**section-v4: section-head inputs.**

- *Section labels*: Jev under policy-section-v3 (region level) on a
  shortlist of candidate nodes per labelled page, old and new corpus, train
  and val. Since 2026-10-09 the pool is pure v3: the old train pages that
  once carried the tightest-node (v2) policy were relabelled.
- *TOC tightest labels*: a node that either policy calls `table_of_contents`
  takes its v2 (tightest-node) answer. Old pages and the new pages with a TOC
  candidate carry both labels.
- *Ad markers*: nodes with third-party ad markup (network token, known ad
  host, EasyList match). The promo/ad rule makes an ad/promo label `ad` on a
  marked node and `promo_banner` on an unmarked one, on every split.
- *Free labels* (`corpus/section-labels.ts`): train only, see Labels above.

The section sample is built by `train/section_sample.py`; the region sample by
`train/sample.py` under the same memory budget as before, so a larger corpus
buys a more diverse draw rather than more rows.

## Tests

```sh
bun test featurize.test.ts silver-labels.test.ts
```

`fixtures/` holds eight hand-written pages (landmark site, the same site as div
soup, article with nested chrome, product, product listing, docs, homepage,
microdata-only about page) and `expected.json`, their hand-checked labels.

## Training and the runtime

*(superseded, 2026-09-20: the original per-head commands, kept so early
results reproduce. Current training is "Strict cross-validation" below.)*

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

*(superseded in part.)* Through the v8.1 page-type experiments every choice
that could read the answer was made on VAL (the class-weight exponent, the
feature set, the tree size, the boosting cut) and holdout was touched once per
reported number. After those experiments the silver-v3 holdout was frozen.
Every choice and every threshold now comes from domain-grouped
cross-validation over train + val, and the final score is read once, on fresh
test slices. The splits themselves are frozen by registrable
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

### Strict cross-validation and slices (current, 2026-10-09)

Every head is measured by 5-fold cross-validation grouped by domain over
train + val, in two arms scored on the same rows:

- **old**: the old corpus and its sampling. Its models are never fit on a new
  page and its thresholds are picked on old pages only (`--fit-exclude` in the
  page-type trainer, a separate old sample in `cv_compare.py`).
- **new**: silver-v4.

A section run also carries the shipped section config as an arm, fit on its
own labels but scored on the current ones (`trainLabels`).

**Strict thresholds.** One threshold per class, the lowest that holds
precision >= 0.95 on BOTH corpus slices (a slice with fewer than 20 fires is
not held to it). A class that cannot reach it abstains. Then the **platform
guard**: a (class, platform) cell with >= 50 fires below 0.95 gets a higher
threshold for that platform if that keeps at least half its coverage, and
otherwise the class abstains on that platform. Thresholds picked on the pooled
out-of-fold rows were not safe: the easier new-corpus pages carried precision
that the old corpus didn't have (2026-10-08: region 4, section 11 classes
below 0.95 on the old slice).

| script | role |
|---|---|
| `train/train_pagetype_v8.py --cv-folds 5` | page-type CV; `--fit-exclude`, `--drop-pages`, `--policy-version`, `--platform-tags`, `--dump-selection`, `--dump-probs` |
| `train/cv_compare.py` | region and section CV, both arms on one row set; `--dump-probs` writes OOF probabilities for confident learning |
| `train/sample_groups.py` | per-row fold bucket and page index for a sample |
| `train/cv_slices.py` | slices, old-only recalibration of the old arm, strict thresholds, tables and a summary |
| `train/platform_guard.py` | the guard table and the adopted overrides |
| `train/refit_heads.py` | region or section refit on a whole pool, for export |
| `train/export_model.py` | artifact: missing-value splits, cross-page block names, platform overrides, parity fixtures from val rows |

None of them opens a holdout file.

### The one-command retrain

One script runs the whole loop. Region is not retrained: its last refit and
strict thresholds are reused.

1. page-type CV in both arms;
2. section CV in three arms (old, new, and the shipped config);
3. slices and strict thresholds;
4. the platform guard;
5. the section refit;
6. the export with the guard's overrides;
7. the runtime parity tests;
8. an alternating latency bench against the shipped artifact.

It writes a retrain report holding the CV summary, the guard table, the export
report and the bench. Its location and inputs are in the private runbook.

### The runtime

`runtime/predict.ts` exposes `classifyPage(html, url, {model})` and
`classifyDocument(document, url, {model})`. `runtime/README.md` owns the
contract, including the section head, the platform guard and the cross-page
block. `source` is `"model"` or `"rules"`, and the shape is identical either
way.

The rules path (`runtime/rules-fallback.ts`) runs when the artifact is absent,
unparseable or built for another featurizer version, when a prediction is under
the artifact's threshold, or when the per-page time budget is spent. It reads
landmarks, ARIA roles and structured data, the same reading `silver-labels.ts`
does, minus the corpus-level parts. A missing model degrades attribution, never
detection.

Float parity is by construction: `export_model.py` rounds thresholds to float32
and walks the rounded trees in a numpy reference evaluator, and `predict.ts`
implements the same walk with `Math.fround` on the feature value. The parity
test compares TS against that reference on 10k rows. Since 2026-10-09 those
rows are val rows, plus 2,000 val pages for the page-type head with and
without its cross-page block.

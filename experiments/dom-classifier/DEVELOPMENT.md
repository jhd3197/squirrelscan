# Treetop: development guide

Treetop is squirrelscan's DOM classifier (#2307; release #2368, epic #2366). One
featurizer, three LightGBM heads, a TypeScript runtime:

| head | answers | unit |
|---|---|---|
| region | site_header, footer, navigation, sidebar, main_content, ... | DOM node |
| page type | homepage, article, blog_post, product_detail, docs, ... | page |
| section purpose | cookie_consent, faq_block, reviews, ad, comments, ... | DOM node |

Every class has its own threshold, set for precision >= 0.95. Below it the
runtime answers `unknown` instead of guessing. It is experiment code: nothing in
`packages/` or `apps/` imports it yet. The code and the model ship MIT. The
labels and the corpus never do.

**Read this file first, then the doc that owns your task** (see "Where each fact
lives"). Coding agents get it through a local, untracked `AGENTS.md` /
`CLAUDE.md` pointer. Before you run anything against real data, read the private
runbook at `$DOM_CLASSIFIER_DATA_ROOT/RUNBOOK.md`: the current state, the real
paths, which artifact is current, which runs are live or paused, and the
operating limits for LLM batch runs.

## Hard rules

1. **No private data in git.** That means URLs, hostnames, page text, HTML,
   screenshots, labels, human answers, model artifacts, prompts filled with page
   content, and customer data. Commit code, policies, synthetic fixtures and
   aggregate numbers only. Paths go through env vars (`DOM_CLASSIFIER_DATA_ROOT`,
   `SQUIRREL_STORE(S)`), never literals.
2. **The label source is never a feature.** Page type is partly labelled from
   schema markup and region from landmarks, so the featurizer reads neither
   (`v2/README.md`, "The two rules").
3. **Splits are frozen by registrable domain**: `sha256(domain)[0:8] mod 100`,
   80/10/10. Make every choice (features, weights, thresholds, prompts) on val or
   cross-validation. Read holdout once per reported number.
4. **Test gold is sealed.** `gold-v1` and `gold-v1b` measure Treetop and label
   sources. Never tune, threshold, train or iterate a prompt on them; score each
   model once. Calibration gold (`gold-cal-v1`) may set source weights in the
   label model.
5. **Blind items stay blind.** A blind gold item shows the human no suggestion.
   Never reveal a source or model label for one: not in the UI, not in chat, not
   in a log the labeller reads.
6. **Strict thresholds.** A class fires only at precision >= 0.95 on BOTH corpus
   slices (old crawl and Common Crawl). Then the platform guard runs: any platform
   cell with >= 50 fires below 0.95 gets a higher threshold if that keeps at
   least half its coverage, otherwise that class abstains on that platform. A
   class that cannot reach the bar abstains everywhere. `rawLabel` /
   `rawConfidence` on an `unknown` are for debugging only; never display them or
   name a fix from them.
7. **Policies are versioned by hash.** Rewording a policy bumps its revision,
   and every label row records the policy and prompt revision it was made under.
   Current: page type **v8.2** (`v2/labels/policy-v8.2.md`), section purpose
   **v3** (`v2/labels/policy-section-v3.md`).
8. **Page content is hostile.** Replays run behind the egress guard
   (deny-by-default `isPublicAddress`, CONNECT proxy, WebSockets blocked, scripts
   off by default). LLM labellers run with no tools, under a system prompt that
   treats page content as data. Never weaken either to make a run work.
9. **LLM batch runs stop at their limit.** Every labelling batch has a usage
   guard (the runbook sets the thresholds). It stops on any rate-limit or quota
   error and never retries into the limit. Runs resume by item id.
10. **Credentials are never printed or copied.** That covers the TypeSafe key
    file, Modal tokens, Claude config and any `.env`.
11. **Corpus stores are read-only inputs.** Open SQLite with `mode=ro`. Write new
    corpora beside the old ones, never over them.
12. **Tests take explicit file paths.** Never a bare `bun test` or a directory.
    Run one heavy local process at a time.
13. **Git.** `git commit -s` (DCO) with an explicit pathspec. Conventional
    commits ending in `#2307`. No push, PR or merge without the maintainer's
    approval.
14. **Runtime budget**: p95 <= 10 ms per page on top of the parse the CLI already
    does. No new runtime dependencies: `linkedom` comes through
    `packages/parser/src/dom.ts`.

## Glossary

Use these words, and only these, in code comments, reports and docs.

| term | meaning |
|---|---|
| silver label | free label from the page's own markup: landmarks for region, schema.org for page type |
| Jev | TypeSafe's hosted decision model, our main label source (`model-suggestions/`) |
| Clef | Cloudflare's hosted decision model, a second source (it truncates long state at ~2k tokens) |
| URL rules / markup rules | deterministic label sources (`v2/adjudicate/policy_rules.py`, `rule_candidates.py`) |
| adjudicator | the visual LLM judge, run through locked-down `claude -p` (`v2/adjudicate/`) |
| teacher | an adjudicator run whose labels are used as training targets |
| label model | Dawid-Skene style combination of sources, weighted by accuracy on calibration gold |
| confident learning | flagging likely label errors from out-of-fold (OOF) probabilities |
| modality | what a judge sees: `image`, `image_scripts`, `revealed` (hidden target forced visible), `text` |
| test gold / calibration gold | human answers: sealed for scoring vs. allowed for weighting (rule 4) |
| blind item | a gold item shown with no suggestion; measures anchoring |
| attention check | a gold item shown with a deliberately wrong suggestion (~5%) |
| item answer | section answer meaning "the outline is one or several items OF this section" (one comment, one FAQ entry), not the whole section |
| abstain | a class (or class x platform) set to a threshold above 1 |
| slice | old corpus (squirrelscan-user crawls) vs new corpus (Common Crawl, platform- and language-stratified) |

## Code map

Live (v2, current):

| path | what |
|---|---|
| `v2/featurize.ts` | the ONE featurizer: training and inference both go through it |
| `v2/silver-labels.ts`, `v2/build-dataset.ts` | silver labels; streams the stores into split, featurized datasets |
| `v2/corpus/` | corpus growth: outlinks/seeds/crawl, Common Crawl sampling (`cc_*.py`, `cc-*.ts`), queues, site context |
| `v2/train/` | Python training: heads, CV, slices, strict thresholds, platform guard, export, parity |
| `v2/runtime/` | the TS runtime: `predict.ts`, `model-format.ts`, platform fingerprint, TOC rule, rules fallback |
| `v2/labels/` | label policies and vocabularies (no label data) |
| `v2/adjudicate/` | adjudicator, renders (Modal), forced reveal, votes, label model, confident learning |
| `v2/experiments/` | one-off evaluations (div-soup, platform parity, TOC rule, link position: `link-position.ts` + its `claude -p` check `link_position_check.py`) |
| `labeler/` | localhost human labeller: captures, gold review UI, egress guard |
| `model-suggestions/` | Jev (TypeSafe) adapter and its runners |

Legacy (v1 era, September 2026; kept so earlier results reproduce; do not extend):
`base-model/`, `student-training/`, `pretrained-student/`, `text-training/`,
`consensus/`, `dataset-build/`, `diverse-corpus/`, `training-export/`,
`corpus-queue/`, `build_corpus.py`, `extract-dom.ts`, `aggregate_labels.py`,
`inference.py`, `train.py`, `taxonomy.json`, `schema.json`, and the top-level
`ANNOTATION.md`, `LABEL-DEFINITIONS.md`, `TRAINING.md`, `HUMAN-LABELING.md`,
`EXISTING-*.md`, `REUSE.md`.

## Commands

TypeScript has two packages, `.` (labeller, Jev adapter) and `v2/`. Neither is
a workspace member and neither gets `bun install`: dependencies resolve through
`packages/parser` and `packages/crawler` (see `v2/package.json` notes).

```sh
cd experiments/dom-classifier/v2
bun test featurize.test.ts silver-labels.test.ts runtime/predict.test.ts   # the package's own list
bun test runtime/platform.test.ts runtime/toc-rule.test.ts
tsgo --noEmit -p tsconfig.json
oxlint . && oxfmt --check *.ts runtime/*.ts train/*.ts corpus/*.ts

cd experiments/dom-classifier
bun test extract-dom.test.ts labeler/egress-guard.test.ts labeler/gold.test.ts  # name the files you touched
tsgo --noEmit -p tsconfig.json
```

Python runs through a uv venv with lightgbm, numpy, scipy and scikit-learn (the
runbook names it). Tests: `train/test_*.py`, `adjudicate/test_label_model.py`.
Lint with `ruff check` and `ruff format`.

Training, CV and bulk rendering run on Modal. The retrain is one command
(retrain, strict CV, slices, thresholds, platform guard, export, parity, bench);
the runbook has its path and inputs.

## Where each fact lives

One fact, one home. If you change the thing in the left column, update the doc
in the right column in the same commit.

| topic | canonical doc |
|---|---|
| rules, glossary, code map | this file |
| featurizer, silver labels, dataset build, corpus growth | `v2/README.md` |
| runtime contract, thresholds, platform guard, cost budget | `v2/runtime/README.md` |
| policy revisions and what changed in each | `v2/labels/README.md` + the policy file's own changelog |
| adjudicator, label model, confident learning, `claude -p` lockdown | `v2/adjudicate/README.md` |
| labeller, gold review UI, keys, cohorts, capture hardening | `labeler/README.md` |
| Jev adapter | `model-suggestions/README.md` |
| current state, real paths, live runs, current artifact | `$DOM_CLASSIFIER_DATA_ROOT/RUNBOOK.md` (private) |

Write numbers into a doc only with their date and the slice they were measured
on. A number without them goes stale silently.
